"""This module contains provider of Mecklenburg-Vorpommern data."""

from pydtmdl.base.dtm import DTMProvider, DTMProviderSettings
from pydtmdl.base.wcs import WCSProvider


class MecklenburgVorpommernProviderSettings(DTMProviderSettings):
    """Settings for the Mecklenburg-Vorpommern provider."""

    dataset: dict | str = {
        "mv_dgm": "Mecklenburg-Vorpommern DGM1",
        "mv_dgm5": "Mecklenburg-Vorpommern DGM5",
        "mv_dgm25": "Mecklenburg-Vorpommern DGM25",
    }


class MecklenburgVorpommernProvider(WCSProvider, DTMProvider):
    """Provider of Mecklenburg-Vorpommern data."""

    _code = "mv"
    _name = "Mecklenburg-Vorpommern"
    _region = "DE"
    _icon = "🇩🇪"
    _resolution = 1.0
    _settings = MecklenburgVorpommernProviderSettings
    _extents = [(54.8, 53, 14.5, 10.5)]

    _url = "https://www.geodaten-mv.de/dienste/dgm_wcs"
    _wcs_version = "2.0.1"
    _source_crs = "EPSG:25833"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        if not self.user_settings:
            raise ValueError("User settings are required for this provider.")
        if not self.user_settings.dataset:
            raise ValueError("Dataset is required for this provider.")
        return {
            "identifier": [self.user_settings.dataset],
            "subsets": [("x", str(tile[1]), str(tile[3])), ("y", str(tile[0]), str(tile[2]))],
            "format": "image/tiff",
        }
