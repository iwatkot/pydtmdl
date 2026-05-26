"""WMTS helpers for imagery providers."""

from __future__ import annotations

import math
import os
from abc import abstractmethod
from typing import cast

import requests
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from pydtmdl.base.imagery import ImageryProvider


class WMTSImageryProvider(ImageryProvider):
    """Generic imagery provider for Google-style WMTS tile services."""

    _source_crs: str = "EPSG:3857"
    _tile_matrix_set: str = "google3857"
    _style: str = "normal"
    _zoom: int = 17
    _tile_pixels: int = 256
    _image_format: str = "image/jpeg"
    _shared_tile_subdirectory = "shared_wmts"
    _world_extent: float = 20037508.3428

    @abstractmethod
    def get_tile_url(self, tile_matrix: int, tile_row: int, tile_col: int) -> str:
        """Return the tile URL for the requested WMTS tile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(
            self._source_tile_directory,
            self._shared_tile_subdirectory,
        )
        os.makedirs(self.shared_tiff_path, exist_ok=True)
        self._transformer = Transformer.from_crs("EPSG:4326", self._source_crs, always_xy=True)

    def download_tiles(self) -> list[str]:
        left, bottom, right, top = self._get_projected_bbox()
        tiles = list(self._iter_required_tiles(left, bottom, right, top))

        def tile_fetcher(tile: tuple[int, int, int, float, float, float, float]) -> bytes:
            _, tile_row, tile_col, tile_left, tile_bottom, tile_right, tile_top = tile
            response = requests.get(
                self.get_tile_url(self._zoom, tile_row, tile_col),
                timeout=120,
            )
            response.raise_for_status()
            return self._georeference_tile(response.content, tile_left, tile_bottom, tile_right, tile_top)

        def file_name_generator(tile: tuple[int, int, int, float, float, float, float]) -> str:
            _, tile_row, tile_col, _, _, _, _ = tile
            return f"{self._zoom}_{tile_row}_{tile_col}.tif"

        return self.download_tiles_with_fetcher(
            cast(list[tuple[float, float, float, float]], tiles),
            self.shared_tiff_path,
            tile_fetcher,
            file_name_generator=file_name_generator,
        )

    def _get_projected_bbox(self) -> tuple[float, float, float, float]:
        north, south, east, west = self.get_bbox()
        corners = [
            self._transformer.transform(west, south),
            self._transformer.transform(west, north),
            self._transformer.transform(east, north),
            self._transformer.transform(east, south),
        ]
        xs = [x for x, _ in corners]
        ys = [y for _, y in corners]
        return min(xs), min(ys), max(xs), max(ys)

    def _iter_required_tiles(
        self,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> list[tuple[int, int, int, float, float, float, float]]:
        matrix_width = 2**self._zoom
        tile_span = (self._world_extent * 2.0) / matrix_width

        min_col = max(0, min(matrix_width - 1, math.floor((left + self._world_extent) / tile_span)))
        max_col = max(
            0,
            min(matrix_width - 1, math.floor(((right - 1e-9) + self._world_extent) / tile_span)),
        )
        min_row = max(0, min(matrix_width - 1, math.floor((self._world_extent - top) / tile_span)))
        max_row = max(
            0,
            min(matrix_width - 1, math.floor((self._world_extent - (bottom + 1e-9)) / tile_span)),
        )

        tiles: list[tuple[int, int, int, float, float, float, float]] = []
        for tile_row in range(min_row, max_row + 1):
            for tile_col in range(min_col, max_col + 1):
                tile_left = -self._world_extent + (tile_col * tile_span)
                tile_right = tile_left + tile_span
                tile_top = self._world_extent - (tile_row * tile_span)
                tile_bottom = tile_top - tile_span
                tiles.append(
                    (
                        self._zoom,
                        tile_row,
                        tile_col,
                        tile_left,
                        tile_bottom,
                        tile_right,
                        tile_top,
                    )
                )
        return tiles

    def _georeference_tile(
        self,
        image_bytes: bytes,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> bytes:
        with MemoryFile(image_bytes) as input_memory:
            with input_memory.open() as source:
                data = source.read()
                profile = {
                    "driver": "GTiff",
                    "crs": self._source_crs,
                    "dtype": source.dtypes[0],
                    "transform": from_bounds(
                        west=left,
                        south=bottom,
                        east=right,
                        north=top,
                        width=source.width,
                        height=source.height,
                    ),
                    "count": source.count,
                    "width": source.width,
                    "height": source.height,
                }

        with MemoryFile() as output_memory:
            with output_memory.open(**profile) as destination:
                destination.write(data)
            return output_memory.read()
