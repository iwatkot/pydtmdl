"""This module contains provider of NRW data."""

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider


class NRWProvider(WCSProvider, DTMProvider):
    """Provider of NRW data."""

    _code = "NRW"
    _name = "North Rhine-Westphalia DGM1"
    _region = "DE"
    _icon = "🇩🇪󠁥󠁢󠁹󠁿"
    _resolution = 1.0
    _extents = [(52.6008271, 50.1506045, 9.5315425, 5.8923538)]

    _url = "https://www.wcs.nrw.de/geobasis/wcs_nw_dgm"
    _wcs_version = "2.0.1"
    _source_crs = "EPSG:25832"
    _tile_size = 1000

    def get_wcs_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        return {
            "identifier": ["nw_dgm"],
            "subsets": [("y", str(tile[0]), str(tile[2])), ("x", str(tile[1]), str(tile[3]))],
            "format": "image/tiff",
        }
