"""This module contains provider of Japan GSI DEM data."""

from __future__ import annotations

import math
import os
from typing import cast

import numpy as np
import requests
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from pydtmdl.base.dtm import DTMProvider


class JapanGSIProvider(DTMProvider):
    """Provider of Japan GSI PNG elevation tile data."""

    _code = "japan_gsi"
    _name = "Japan GSI DEM"
    _region = "JP"
    _icon = "JP"
    _resolution = 10.0
    _extents = [(45.8, 20.2, 154.0, 122.9)]

    _url = "https://cyberjapandata.gsi.go.jp/xyz/dem_png/{z}/{x}/{y}.png"
    _source_crs = "EPSG:3857"
    _zoom = 14
    _tile_pixels = 256
    _world_extent = 20037508.342789244
    _nodata = -9999.0
    _shared_tile_subdirectory = "shared_dem_png_v1"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(
            self._source_tile_directory,
            self._shared_tile_subdirectory,
        )
        os.makedirs(self.shared_tiff_path, exist_ok=True)

    def download_tiles(self) -> list[str]:
        north, south, east, west = self.get_bbox()
        tiles = list(self._iter_required_tiles(north, south, east, west))

        def tile_fetcher(tile: tuple[int, int, int, float, float, float, float]) -> bytes:
            _, tile_x, tile_y, left, bottom, right, top = tile
            response = requests.get(
                self._url.format(z=self._zoom, x=tile_x, y=tile_y),
                timeout=120,
            )
            response.raise_for_status()
            return self._decode_tile(response.content, left, bottom, right, top)

        def file_name_generator(tile: tuple[int, int, int, float, float, float, float]) -> str:
            _, tile_x, tile_y, _, _, _, _ = tile
            return f"{self._zoom}_{tile_x}_{tile_y}.tif"

        return self.download_tiles_with_fetcher(
            cast(list[tuple[float, float, float, float]], tiles),
            self.shared_tiff_path,
            tile_fetcher,
            file_name_generator=file_name_generator,
        )

    def _iter_required_tiles(
        self,
        north: float,
        south: float,
        east: float,
        west: float,
    ) -> list[tuple[int, int, int, float, float, float, float]]:
        min_x, max_y = self._lonlat_to_tile_fraction(west, north)
        max_x, min_y = self._lonlat_to_tile_fraction(east, south)
        matrix_width = 2**self._zoom

        min_col = max(0, min(matrix_width - 1, math.floor(min_x)))
        max_col = max(0, min(matrix_width - 1, math.floor(max_x - 1e-9)))
        min_row = max(0, min(matrix_width - 1, math.floor(min_y)))
        max_row = max(0, min(matrix_width - 1, math.floor(max_y - 1e-9)))

        tiles: list[tuple[int, int, int, float, float, float, float]] = []
        tile_span = (self._world_extent * 2.0) / matrix_width
        for tile_y in range(min_row, max_row + 1):
            for tile_x in range(min_col, max_col + 1):
                left = -self._world_extent + (tile_x * tile_span)
                right = left + tile_span
                top = self._world_extent - (tile_y * tile_span)
                bottom = top - tile_span
                tiles.append((self._zoom, tile_x, tile_y, left, bottom, right, top))
        return tiles

    def _lonlat_to_tile_fraction(self, lon: float, lat: float) -> tuple[float, float]:
        lat = max(-85.05112878, min(85.05112878, lat))
        lat_rad = math.radians(lat)
        matrix_width = 2**self._zoom
        x = (lon + 180.0) / 360.0 * matrix_width
        y = (
            (1.0 - math.log(math.tan(lat_rad) + (1.0 / math.cos(lat_rad))) / math.pi)
            / 2.0
            * matrix_width
        )
        return x, y

    def _decode_tile(
        self,
        image_bytes: bytes,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> bytes:
        with MemoryFile(image_bytes) as input_memory:
            with input_memory.open() as source:
                rgb = source.read(indexes=[1, 2, 3]).astype(np.uint32)

        encoded = (rgb[0] * 65536) + (rgb[1] * 256) + rgb[2]
        data = encoded.astype(np.float32)
        nodata_mask = encoded == 2**23
        negative_mask = encoded > 2**23
        data[negative_mask] = (data[negative_mask] - 2**24) * 0.01
        data[~negative_mask] = data[~negative_mask] * 0.01
        data[nodata_mask] = self._nodata

        profile = {
            "driver": "GTiff",
            "crs": self._source_crs,
            "dtype": "float32",
            "transform": from_bounds(
                west=left,
                south=bottom,
                east=right,
                north=top,
                width=self._tile_pixels,
                height=self._tile_pixels,
            ),
            "count": 1,
            "width": self._tile_pixels,
            "height": self._tile_pixels,
            "nodata": self._nodata,
        }
        with MemoryFile() as output_memory:
            with output_memory.open(**profile) as destination:
                destination.write(data.astype(np.float32), 1)
            return output_memory.read()
