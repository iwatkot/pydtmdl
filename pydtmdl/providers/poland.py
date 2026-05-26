"""This module contains provider of Poland DTM data."""

import math

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider


class PolandDTM1MProvider(WCSProvider, DTMProvider):
    """Provider of Poland 1 m Digital Terrain Model data from Geoportal/GUGiK."""

    _code = "poland_dtm1m"
    _name = "Poland DTM 1 m"
    _region = "PL"
    _icon = "🇵🇱"
    _resolution = 1.0
    _extents = [(54.94, 49.0, 24.15, 14.12)]

    _url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/NMT/GRID1/WCS/DigitalTerrainModel"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:2180"
    _tile_size = 1000
    _is_multipart = False
    _coverage_identifier = "NMT-PL-EVRF2007-NH"

    def get_wcs_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        width = max(1, math.ceil(abs(tile[3] - tile[1]) / (self.resolution() or 1.0)))
        height = max(1, math.ceil(abs(tile[2] - tile[0]) / (self.resolution() or 1.0)))
        return {
            "identifier": self._coverage_identifier,
            "bbox": (tile[1], tile[0], tile[3], tile[2]),
            "crs": self._source_crs,
            "width": width,
            "height": height,
            "format": "GeoTIFF",
        }
