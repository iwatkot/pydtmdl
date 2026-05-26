"""WMS helpers for imagery providers."""

import os
from abc import abstractmethod
from typing import Any

import rasterio
from owslib.wms import WebMapService

from pydtmdl import utils
from pydtmdl.base.imagery import ImageryProvider


class WMSImageryProvider(ImageryProvider):
    """Generic imagery provider for WMS orthophoto services."""

    _wms_version = "1.3.0"
    _source_crs: str = "EPSG:25832"
    _tile_size: float = 1000
    _tile_pixels: int = 3000

    @abstractmethod
    def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        """Return WMS GetMap parameters for a projected tile."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(self._source_tile_directory, "shared")
        os.makedirs(self.shared_tiff_path, exist_ok=True)

    def download_tiles(self) -> list[str]:
        bbox = utils.transform_bbox(self.get_bbox(), self._source_crs)
        tiles = utils.tile_bbox(bbox, self._tile_size)
        wms = WebMapService(self._url, version=self._wms_version, timeout=600)

        def wms_fetcher(tile: tuple[float, float, float, float]) -> Any:
            return wms.getmap(**self.get_wms_parameters(tile))

        files = self.download_tiles_with_fetcher(tiles, self.shared_tiff_path, wms_fetcher)
        for file_path in files:
            self._ensure_tile_crs(file_path)
        return files

    def _ensure_tile_crs(self, file_path: str) -> None:
        """Assign the requested source CRS when a WMS GeoTIFF omits EPSG metadata."""
        with rasterio.open(file_path, "r+") as dataset:
            if dataset.crs is None or dataset.crs.to_epsg() is None:
                dataset.crs = self._source_crs
