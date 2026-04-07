"""This module contains provider of Lithuania data."""

import requests

from pydtmdl.base.dtm import DTMProvider


class LithuaniaProvider(DTMProvider):
    """Provider of Lithuania data."""

    _code = "lithuania"
    _name = "Lithuania"
    _region = "LT"
    _icon = "🇱🇹"
    _resolution = 1.0
    _extents = [
        (
            56.4501789128452,
            53.8901567283941,
            26.8198345671209,
            20.9312456789123,
        )
    ]
    _max_tile_size = 4096
    _url = (
        "https://utility.arcgis.com/usrsvcs/servers/fef66dec83c14b0295180ecafa662aa0/"
        "rest/services/DTM_LT2020/ImageServer/exportImage"
    )

    def download_tiles(self) -> list[str]:
        """Download DTM tiles for Lithuania."""
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
                "imageSR": "3346",
                "format": "tiff",
                "pixelType": "F32",
                "size": f"{pixel_width},{pixel_height}",
            }

            response = requests.get(self.url, params=params, timeout=60)  # type: ignore
            data = response.json()
            if "href" not in data:
                raise RuntimeError(f"No image URL in response for tile {i}")
            download_urls.append(data["href"])

        return self.download_tif_files(download_urls, self._tile_directory)
