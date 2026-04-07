"""This module contains provider of USGS data."""

import math

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider


class USGSWCSProvider(WCSProvider, DTMProvider):
    """Provider of USGS data."""

    _code = "usgs_wcs"
    _name = "USGS"
    _region = "USA"
    _icon = "🇺🇸"
    _resolution = 1.0
    _extents = [(50.0, 17.0, -64.0, -162.0)]

    _url = "https://elevation.nationalmap.gov/arcgis/services/3DEPElevation/ImageServer/WCSServer"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:3857"
    _tile_size = 1000
    _is_multipart = False

    def get_wcs_parameters(self, tile):
        width = max(1, math.ceil(abs(tile[3] - tile[1]) / (self.resolution() or 1.0)))
        height = max(1, math.ceil(abs(tile[2] - tile[0]) / (self.resolution() or 1.0)))
        return {
            "identifier": "DEP3Elevation",
            "bbox": (tile[1], tile[0], tile[3], tile[2]),
            "crs": "EPSG:3857",
            "width": width,
            "height": height,
            "format": "GeoTIFF",
        }
