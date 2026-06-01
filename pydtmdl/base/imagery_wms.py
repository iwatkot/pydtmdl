"""WMS helpers for imagery providers."""

import os
from abc import abstractmethod
from math import ceil
from typing import Any

import rasterio
from owslib.wms import WebMapService
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from pydtmdl import utils
from pydtmdl.base.imagery import ImageryProvider


class WMSImageryProvider(ImageryProvider):
    """Generic imagery provider for WMS orthophoto services."""

    _wms_version = "1.3.0"
    _source_crs: str = "EPSG:25832"
    _tile_size: float = 1000
    _tile_pixels: int = 3000
    _image_format: str = "image/tiff"
    _shared_tile_subdirectory = "shared"

    @abstractmethod
    def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        """Return WMS GetMap parameters for a projected tile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(
            self._source_tile_directory,
            self._shared_tile_subdirectory,
        )
        os.makedirs(self.shared_tiff_path, exist_ok=True)

    def download_tiles(self) -> list[str]:
        bbox = self._transform_bbox_to_source_crs(self.get_bbox())
        tiles = utils.tile_bbox(bbox, self._tile_size)
        wms = WebMapService(self._url, version=self._wms_version, timeout=600)

        def wms_fetcher(tile: tuple[float, float, float, float]) -> Any:
            response = wms.getmap(**self.get_wms_parameters(tile))
            image_bytes = response.read()
            if self._requires_georeferencing():
                return self._georeference_wms_image(image_bytes, tile)
            return image_bytes

        files = self.download_tiles_with_fetcher(
            tiles,
            self.shared_tiff_path,
            wms_fetcher,
            file_name_generator=self._tile_file_name,
        )
        for file_path in files:
            self._ensure_tile_crs(file_path)
        return files

    def _wms_image_size(self) -> tuple[int, int]:
        """Return a tile image size capped by service limits and provider resolution."""
        resolution = self.resolution()
        if resolution is None or resolution <= 0:
            pixels = self._tile_pixels
        else:
            pixels = min(self._tile_pixels, max(1, ceil(self._tile_size / resolution)))
        return pixels, pixels

    def _tile_file_name(self, tile: tuple[float, float, float, float]) -> str:
        pixels, _ = self._wms_image_size()
        return f"{'_'.join(map(str, tile))}_{pixels}px.tif"

    def _transform_bbox_to_source_crs(
        self,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Transform a WGS84 bbox to the source CRS using x/y output ordering."""
        north, south, east, west = bbox
        transformer = Transformer.from_crs("EPSG:4326", self._source_crs, always_xy=True)
        corners = [
            transformer.transform(west, south),
            transformer.transform(west, north),
            transformer.transform(east, north),
            transformer.transform(east, south),
        ]
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        return max(ys), min(ys), max(xs), min(xs)

    def _requires_georeferencing(self) -> bool:
        image_format = self._image_format.lower()
        return image_format in {"image/jpeg", "image/png", "image/png8", "image/png32"}

    def _georeference_wms_image(
        self,
        image_bytes: bytes,
        tile: tuple[float, float, float, float],
    ) -> bytes:
        with MemoryFile(image_bytes) as input_memory:
            with input_memory.open() as source:
                data = source.read()
                transform = from_bounds(
                    west=tile[1],
                    south=tile[0],
                    east=tile[3],
                    north=tile[2],
                    width=source.width,
                    height=source.height,
                )
                profile = {
                    "driver": "GTiff",
                    "crs": self._source_crs,
                    "dtype": source.dtypes[0],
                    "transform": transform,
                    "count": source.count,
                    "width": source.width,
                    "height": source.height,
                }
                if source.nodata is not None:
                    profile["nodata"] = source.nodata

        with MemoryFile() as output_memory:
            with output_memory.open(**profile) as destination:
                destination.write(data)
            return output_memory.read()

    def _ensure_tile_crs(self, file_path: str) -> None:
        """Assign the requested source CRS when a WMS GeoTIFF omits EPSG metadata."""
        with rasterio.open(file_path, "r+") as dataset:
            if dataset.crs is None or dataset.crs.to_epsg() is None:
                dataset.crs = self._source_crs
