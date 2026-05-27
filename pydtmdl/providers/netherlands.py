"""This module contains provider of Netherlands AHN4 data."""

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider


class NetherlandsAHN4Provider(WCSProvider, DTMProvider):
    """Provider of Netherlands AHN4 0.5 m DTM data."""

    _code = "netherlands_ahn4"
    _name = "Netherlands AHN4"
    _region = "NL"
    _icon = "🇳🇱"
    _resolution = 0.5
    _extents = [(53.54007281954217, 50.727949501589116, 7.272727495354396, 3.33354056165032)]

    _url = "https://service.pdok.nl/rws/ahn/wcs/v1_0"
    _wcs_version = "2.0.1"
    _source_crs = "EPSG:28992"
    _tile_size = 1000

    def get_wcs_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        return {
            "identifier": "dtm_05m",
            "subsets": [("x", str(tile[1]), str(tile[3])), ("y", str(tile[0]), str(tile[2]))],
            "format": "image/tiff",
        }
