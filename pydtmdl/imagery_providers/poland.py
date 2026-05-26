"""Polish national orthophoto imagery provider."""

from pydtmdl.base.imagery_wms import WMSImageryProvider


class PolandHighResolutionOrthophotoProvider(WMSImageryProvider):
    """Poland high-resolution orthophoto imagery from Geoportal/GUGiK."""

    _code = "poland_orto_highres"
    _name = "Poland high-resolution orthophoto"
    _region = "PL"
    _icon = "PL"
    _resolution = 0.25
    _dataset = "pl-orto-highres"
    _extents = [(54.94, 49.0, 24.15, 14.12)]

    _url = "https://mapy.geoportal.gov.pl/wss/service/PZGIK/ORTO/WMS/HighResolution"
    _source_crs = "EPSG:2180"
    _tile_size = 1000
    _tile_pixels = 3000
    _image_format = "image/jpeg"
    _layer = "Raster"

    def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        return {
            "layers": [self._layer],
            "srs": self._source_crs,
            "bbox": (tile[1], tile[0], tile[3], tile[2]),
            "size": (self._tile_pixels, self._tile_pixels),
            "format": self._image_format,
            "transparent": False,
        }
