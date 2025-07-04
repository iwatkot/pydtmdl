"""This module contains provider of Denmark data."""

from pydtmdl.base.dtm import DTMProvider, DTMProviderSettings
from pydtmdl.base.wcs import WCSProvider


class DenmarkProviderSettings(DTMProviderSettings):
    """Settings for the Denmark provider."""

    token: str = ""


class DenmarkProvider(WCSProvider, DTMProvider):
    """Provider of Denmark data."""

    _code = "denmark"
    _name = "Denmark"
    _region = "DK"
    _icon = "🇩🇰"
    _resolution = 0.4
    _settings = DenmarkProviderSettings
    _extents = [(57.7690657013977, 54.4354651516217, 15.5979112056959, 8.00830949937517)]

    _instructions = (
        "ℹ️ This provider requires an access token. See [here](https://confluence"
        ".sdfi.dk/display/MYD/How+to+create+a+user) for more information on "
        "how to create one, then enter it below in the settings field for token."
    )

    _url = "https://api.dataforsyningen.dk/dhm_wcs_DAF"
    _wcs_version = "1.0.0"
    _source_crs = "EPSG:25832"
    _tile_size = 1000

    def get_wcs_parameters(self, tile):
        if not self.user_settings:
            raise ValueError("User settings are required for this provider.")
        if not self.user_settings.token:
            raise ValueError("A token is required for this provider.")

        return {
            "identifier": "dhm_terraen",
            "bbox": (tile[1], tile[0], tile[3], tile[2]),
            "crs": "EPSG:25832",
            "width": 2500,
            "height": 2500,
            "format": "GTiff",
            "token": self.user_settings.token,
        }
