"""This module contains provider of Czech data."""

from typing import List

import requests

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.utils import tile_bbox


class CzechProviderDMR5G(DTMProvider):
    """Provider of Czech data."""

    _code = "czech_dmr5g"
    _name = "Czech Republic (DMR5G)"
    _region = "CZ"
    _icon = "🇨🇿"
    _resolution = 2.0
    _extents = [
        (
            51.0576876059846754,
            48.4917065572081754,
            18.9775933665038821,
            12.0428143585602161,
        )
    ]
    _max_tile_size = 4096
    _url = (
        "https://ags.cuzk.cz/arcgis2/rest/services/dmr5g/ImageServer/exportImage"
    )

    def download_tiles(self) -> List[str]:
        """Download DTM tiles for Czech Republic."""
        bbox = self.get_bbox()
        grid_size = max(1, self.size // self._max_tile_size)
        tile_size = (self.size / grid_size) / 111000  # Convert to degrees

        raw_tiles = tile_bbox(bbox, tile_size)
        # Fix coordinate swapping from utils.tile_bbox
        tiles = [(t[1], t[3], t[0], t[2]) for t in raw_tiles]  # Reorder N,S,E,W correctly

        download_urls = []
        for i, (north, south, east, west) in enumerate(tiles):
            params = {
                "f": "json",
                "bbox": f"{west},{south},{east},{north}",
                "bboxSR": "4326",
                "imageSR": "4326",
                "format": "tiff",
                "pixelType": "F32",
                "size": f"{self._max_tile_size},{self._max_tile_size}",
            }

            response = requests.get(
                self.url, params=params, verify=None, timeout=60  # type: ignore
            )
            data = response.json()
            if "href" not in data:
                raise RuntimeError(f"No image URL in response for tile {i}")
            download_urls.append(data["href"])

        return self.download_tif_files(download_urls, self._tile_directory)
