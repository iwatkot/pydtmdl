"""This module contains provider of Flanders data."""

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider


class FlandersProvider(WCSProvider, DTMProvider):
    """Provider of Flanders data."""

    _code = "flanders"
    _name = "Flanders DHM II"
    _region = "BE"
    _icon = "🇧🇪"
    _resolution = 1.0
    _extents = [(51.5150730375579684, 50.6694992827160817, 5.9444417082210812, 2.5170092434134252)]

    _url = "https://geo.api.vlaanderen.be/el-dtm/wcs"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:4258"
    _tile_size = 0.02

    def get_wcs_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        return {
            "identifier": "EL.GridCoverage.DTM",
            "bbox": tile,
            "format": "GeoTIFF",
            "crs": "EPSG:4258",
            "width": 1000,
            "height": 1000,
            "timeout": 600,
        }
