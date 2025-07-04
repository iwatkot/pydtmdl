"""This module contains provider of Canada data."""

from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider


class CanadaProvider(WCSProvider, DTMProvider):
    """Provider of Canada data."""

    _code = "canada"
    _name = "Canada HRDEM"
    _region = "CN"
    _icon = "🇨🇦"
    _resolution = 1.0
    _extents = [(76.49491845750764, 33.66564101989275, -26.69697497450798, -157.7322455868316)]
    _instructions = (
        "HRDEM coverage for Canada is limited. Make sure to check the "
        "[coverage map](https://geo.ca/imagery/high-resolution-digital"
        "-elevation-model-hrdem-canelevation-series/)."
    )

    _url = "https://datacube.services.geo.ca/ows/elevation"
    _wcs_version = "1.1.1"
    _source_crs = "EPSG:3979"
    _tile_size = 1000

    def get_wcs_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        return {
            "identifier": "dtm",
            "gridbasecrs": "urn:ogc:def:crs:EPSG::3979",
            "boundingbox": f"{tile[1]},{tile[0]},{tile[3]},{tile[2]},urn:ogc:def:crs:EPSG::3979",
            "format": "image/geotiff",
            "timeout": 600,
        }
