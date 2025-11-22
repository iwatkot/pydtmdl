"""This module contains the base WMS provider."""

import os
from abc import abstractmethod
from typing import Any

from owslib.wms import WebMapService

from pydtmdl import utils
from pydtmdl.base.dtm import DTMProvider


# pylint: disable=too-many-locals
class WMSProvider(DTMProvider):
    """Generic provider of WMS sources."""

    _wms_version = "1.3.0"
    _source_crs: str = "EPSG:4326"
    _tile_size: float = 0.02

    @abstractmethod
    def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        """Get the parameters for the WMS request.

        Arguments:
            tile (tuple): The tile to download.

        Returns:
            dict: The parameters for the WMS request.
        """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(self._tile_directory, "shared")
        os.makedirs(self.shared_tiff_path, exist_ok=True)

    def download_tiles(self) -> list[str]:
        bbox = self.get_bbox()
        bbox = utils.transform_bbox(bbox, self._source_crs)
        tiles = utils.tile_bbox(bbox, self._tile_size)

        # Create WMS instance once
        wms = WebMapService(
            self._url,
            version=self._wms_version,
            # auth=Authentication(verify=False),
            timeout=600,
        )

        # Use unified download method with WMS data fetcher
        def wms_fetcher(tile: tuple[float, float, float, float]) -> Any:
            return wms.getmap(**self.get_wms_parameters(tile))

        return self.download_tiles_with_fetcher(tiles, self.shared_tiff_path, wms_fetcher)
