"""This module contains provider of Baden-Württemberg data."""

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider


class BadenWurttembergProvider(WCSProvider, DTMProvider):
    """Provider of Baden-Württemberg data."""

    _code = "baden"
    _name = "Baden-Württemberg"
    _region = "DE"
    _icon = "🇩🇪"
    _resolution = 1.0
    _extents = [(49.79645444804715, 47.52877040346605, 10.54203149250156, 7.444081717803481)]

    _url = "https://owsproxy.lgl-bw.de/owsproxy/wcs/WCS_INSP_BW_Hoehe_Coverage_DGM1"
    _wcs_version = "2.0.1"
    _source_crs = "EPSG:25832"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {
            "identifier": ["EL.ElevationGridCoverage"],
            "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))],
            "format": "image/tiff",
        }
