"""Polish national orthophoto imagery provider."""

from __future__ import annotations

from typing import Any

from owslib.wms import WebMapService
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from pydtmdl import utils
from pydtmdl.base.imagery_wms import WMSImageryProvider


class PolandHighResolutionOrthophotoProvider(WMSImageryProvider):
    """Poland high-resolution orthophoto imagery from Geoportal/GUGiK."""

    _code = "poland_orto_highres"
    _name = "Poland high-resolution orthophoto"
    _region = "PL"
    _icon = "PL"
    _resolution = 0.25
    _dataset = "pl-orto-highres"
    _extents = [(54.94, 49.0, 24.15, 14.12)]

    _url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
    _source_crs = "EPSG:2180"
    _tile_size = 1000
    _tile_pixels = 3000
    _image_format = "image/jpeg"
    _layer = "Raster"

    def download_tiles(self) -> list[str]:
        bbox = utils.transform_bbox(self.get_bbox(), self._source_crs)
        tiles = utils.tile_bbox(bbox, self._tile_size)
        wms = WebMapService(self._url, version=self._wms_version, timeout=600)

        def wms_fetcher(tile: tuple[float, float, float, float]) -> Any:
            response = wms.getmap(**self.get_wms_parameters(tile))
            return self._georeference_wms_image(response.read(), tile)

        return self.download_tiles_with_fetcher(tiles, self.shared_tiff_path, wms_fetcher)

    def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        return {
            "layers": [self._layer],
            "crs": self._source_crs,
            "bbox": (tile[1], tile[0], tile[3], tile[2]),
            "size": (self._tile_pixels, self._tile_pixels),
            "format": self._image_format,
            "transparent": False,
        }

    def _georeference_wms_image(self, image_bytes: bytes, tile: tuple[float, float, float, float]) -> bytes:
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

        with MemoryFile() as output_memory:
            with output_memory.open(**profile) as destination:
                destination.write(data)
            return output_memory.read()
