"""This module contains provider of Bavaria data."""

import hashlib
import os
from xml.etree import ElementTree as ET

from pydtmdl.base.dtm import DTMProvider


class BavariaProvider(DTMProvider):
    """Provider of Bavaria Digital terrain model (DTM) 1m data.
    Data is provided by the 'Bayerische Vermessungsverwaltung' and available
    at https://geodaten.bayern.de/opengeodata/OpenDataDetail.html?pn=dgm1 under CC BY 4.0 license.
    """

    _code = "bavaria"
    _name = "Bavaria DGM1"
    _region = "DE"
    _icon = "🇩🇪󠁥󠁢󠁹󠁿"
    _resolution = 1.0
    _extents = [(50.56, 47.25, 13.91, 8.95)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tiff_path = os.path.join(self._tile_directory, "tiffs")
        os.makedirs(self.tiff_path, exist_ok=True)
        self.meta4_path = os.path.join(self._tile_directory, "meta4")
        os.makedirs(self.meta4_path, exist_ok=True)

    def download_tiles(self) -> list[str]:
        download_urls = self.get_meta_file_from_coords()
        all_tif_files = self.download_tif_files(download_urls, self.tiff_path)
        return all_tif_files

    @staticmethod
    def get_meta_file_name(north: float, south: float, east: float, west: float) -> str:
        """Generate a hashed file name for the .meta4 file.

        Arguments:
            north (float): Northern latitude.
            south (float): Southern latitude.
            east (float): Eastern longitude.
            west (float): Western longitude.

        Returns:
            str: Hashed file name.
        """
        coordinates = f"{north}_{south}_{east}_{west}"
        hash_object = hashlib.md5(coordinates.encode())
        hashed_file_name = "download_" + hash_object.hexdigest() + ".meta4"
        return hashed_file_name

    def get_meta_file_from_coords(self) -> list[str]:
        """Download .meta4 (xml format) file

        Returns:
            list: List of download URLs.
        """
        (north, south, east, west) = self.get_bbox()
        file_path = os.path.join(self.meta4_path, self.get_meta_file_name(north, south, east, west))
        if not os.path.exists(file_path):
            # Use unified download method
            data = (
                f"SRID=4326;POLYGON(({west} {south},{east} {south},"
                f"{east} {north},{west} {north},{west} {south}))"
            )
            self.download_file(
                url="https://geoservices.bayern.de/services/poly2metalink/metalink/dgm1",
                output_path=file_path,
                method="POST",
                data=data,
            )
        else:
            self.logger.debug("File already exists: %s", file_path)
        return self.extract_urls_from_xml(file_path)

    def extract_urls_from_xml(self, file_path: str) -> list[str]:
        """Extract URLs from the XML file.

        Arguments:
            file_path (str): Path to the XML file.

        Returns:
            list: List of URLs.
        """
        urls: list[str] = []
        root = ET.parse(file_path).getroot()
        namespace = {"ml": "urn:ietf:params:xml:ns:metalink"}

        for file in root.findall(".//ml:file", namespace):
            url = file.find("ml:url", namespace)
            if url is not None:
                urls.append(str(url.text))

        self.logger.debug("Received %s urls", len(urls))
        return urls
