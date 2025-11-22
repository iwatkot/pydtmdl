"""This module contains the base WCS provider."""

import os
from abc import abstractmethod
from typing import Any

from owslib.wcs import WebCoverageService

from pydtmdl import utils
from pydtmdl.base.dtm import DTMProvider


# pylint: disable=too-many-locals
class WCSProvider(DTMProvider):
    """Generic provider of WCS sources."""

    _wcs_version = "2.0.1"
    _source_crs: str = "EPSG:4326"
    _tile_size: float = 0.02

    @abstractmethod
    def get_wcs_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        """Get the parameters for the WCS request.

        Arguments:
            tile (tuple): The tile to download.

        Returns:
            dict: The parameters for the WCS request.
        """

    def get_wcs_instance_parameters(self) -> dict:
        """Get the parameters for the WCS instance.

        Returns:
            dict: The parameters for the WCS instance.
        """
        return {
            "url": self._url,
            "version": self._wcs_version,
            "timeout": 120,
        }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(self._tile_directory, "shared")
        os.makedirs(self.shared_tiff_path, exist_ok=True)

    def download_tiles(self) -> list[str]:
        bbox = self.get_bbox()
        bbox = utils.transform_bbox(bbox, self._source_crs)
        tiles = utils.tile_bbox(bbox, self._tile_size)

        # Create WCS instance once
        params = self.get_wcs_instance_parameters()
        wcs = WebCoverageService(**params)

        # Use unified download method with WCS data fetcher
        def wcs_fetcher(tile: tuple[float, float, float, float]) -> Any:
            return wcs.getCoverage(**self.get_wcs_parameters(tile))

        return self.download_tiles_with_fetcher(tiles, self.shared_tiff_path, wcs_fetcher)
