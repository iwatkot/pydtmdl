"""This module contains provider of Czech data."""

import requests

from pydtmdl.base.dtm import DTMProvider


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
    _url = "https://ags.cuzk.cz/arcgis2/rest/services/dmr5g/ImageServer/exportImage"

    def download_tiles(self) -> list[str]:
        """Download DTM tiles for Czech Republic."""
        bbox = self.get_bbox()
        columns = max(1, -(-self.download_width_m // self._max_tile_size))
        rows = max(1, -(-self.download_height_m // self._max_tile_size))
        tiles = self.split_bbox(bbox, columns=columns, rows=rows)
        tile_width_m = self.download_width_m / columns
        tile_height_m = self.download_height_m / rows

        download_urls = []
        for i, (north, south, east, west) in enumerate(tiles):
            pixel_width, pixel_height = self.get_tile_pixel_dimensions(
                tile_width_m,
                tile_height_m,
                self._max_tile_size,
            )
            params = {
                "f": "json",
                "bbox": f"{west},{south},{east},{north}",
                "bboxSR": "4326",
                "imageSR": "4326",
                "format": "tiff",
                "pixelType": "F32",
                "size": f"{pixel_width},{pixel_height}",
            }

            response = requests.get(self.url, params=params, timeout=60)  # type: ignore
            response.raise_for_status()
            data = response.json()
            if "href" not in data:
                raise RuntimeError(f"No image URL in response for tile {i}")
            download_urls.append(data["href"])

        return self.download_tif_files(download_urls, self._tile_directory)
