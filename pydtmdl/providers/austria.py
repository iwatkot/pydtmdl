"""
This module contains 1m DTM providers for all Austrian states
where the data is openly available (excluding Burgenland).
"""

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider


# --- Eastern Austria ---

class Vienna1MProvider(WCSProvider, DTMProvider):
    """Provider for Vienna's DTM data (1m)."""
    _code = "vienna1m"
    _name = "Vienna DGM1"
    _region = "AT-9"
    _icon = "🇦🇹"
    _resolution = 1.0
    _extents = [(48.32, 48.11, 16.57, 16.18)]
    _url = "https://www.wien.gv.at/wien-gis/wcs"
    _wcs_version = "2.0.1"
    _source_crs = "EPSG:31256"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {"identifier": ["GELAENDEMODELL"], "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))], "format": "image/tiff"}


class LowerAustria1MProvider(WCSProvider, DTMProvider):
    """Provider for Lower Austria's DTM data (1m)."""
    _code = "loweraustria1m"
    _name = "Lower Austria DGM1"
    _region = "AT-3"
    _icon = "🇦🇹"
    _resolution = 1.0
    _extents = [(49.02, 47.43, 17.07, 14.46)]
    _url = "https://maps.noel.gv.at/wcs"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:31256"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {"identifier": ["dgm"], "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))], "format": "image/tiff"}


# --- Southern Austria ---

class Styria1MProvider(WCSProvider, DTMProvider):
    """Provider for Styria's DTM data (1m)."""
    _code = "styria1m"
    _name = "Styria DGM1"
    _region = "AT-6"
    _icon = "🇦🇹"
    _resolution = 1.0
    _extents = [(47.90, 46.60, 16.20, 13.50)]
    _url = "https://gis.stmk.gv.at/wcs"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:31255"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {"identifier": ["dhm"], "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))], "format": "image/tiff"}


class Carinthia1MProvider(WCSProvider, DTMProvider):
    """Provider for Carinthia's DTM data (1m)."""
    _code = "carinthia1m"
    _name = "Carinthia DGM1"
    _region = "AT-2"
    _icon = "🇦🇹"
    _resolution = 1.0
    _extents = [(47.14, 46.37, 15.18, 12.65)]
    _url = "https://gis.ktn.gv.at/wcs"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:31255"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {"identifier": ["dgm_1m"], "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))], "format": "image/tiff"}


# --- Central & Western Austria ---

class UpperAustria1MProvider(WCSProvider, DTMProvider):
    """Provider for Upper Austria's DTM data (1m)."""
    _code = "upperaustria1m"
    _name = "Upper Austria DGM1"
    _region = "AT-4"
    _icon = "🇦🇹"
    _resolution = 1.0
    _extents = [(48.72, 47.43, 15.06, 12.74)]
    _url = "https://wcs.doris.at/wcs"
    _wcs_version = "2.0.1"
    _source_crs = "EPSG:31255"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {"identifier": ["gelaendemodell_1m"], "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))], "format": "image/tiff"}


class Salzburg1MProvider(WCSProvider, DTMProvider):
    """Provider for Salzburg's DTM data (1m)."""
    _code = "salzburg1m"
    _name = "Salzburg DGM1"
    _region = "AT-5"
    _icon = "🇦🇹"
    _resolution = 1.0
    _extents = [(48.05, 46.94, 13.79, 12.09)]
    _url = "https://www.salzburg.gv.at/sagis/wcs"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:31255"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {"identifier": ["hoehe"], "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))], "format": "image/tiff"}


class Tyrol1MProvider(WCSProvider, DTMProvider):
    """Provider for Tyrol's DTM data (1m)."""
    _code = "tyrol1m"
    _name = "Tyrol DGM1"
    _region = "AT-7"
    _icon = "🇦🇹"
    _resolution = 1.0
    _extents = [(47.75, 46.70, 12.98, 10.10)]
    _url = "https://gis.tirol.gv.at/arcgis/services/Service_Public/dgm/MapServer/WCSServer"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:31254"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {"identifier": ["1"], "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))], "format": "image/tiff"}


class Vorarlberg1MProvider(WCSProvider, DTMProvider):
    """Provider for Vorarlberg's DTM data (1m)."""
    _code = "vorarlberg1m"
    _name = "Vorarlberg DGM1"
    _region = "AT-8"
    _icon = "🇦🇹"
    _resolution = 1.0
    _extents = [(47.60, 46.84, 10.24, 9.53)]
    _url = "https://vogis.cnv.at/wcs"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:31254"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        return {"identifier": ["dhm"], "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))], "format": "image/tiff"}
      
