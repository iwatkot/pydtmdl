"""German state orthophoto imagery providers."""

from __future__ import annotations

from pydtmdl.base.imagery_wms import WMSImageryProvider


class GermanWMSOrthophotoProvider(WMSImageryProvider):
    """Base class for German WMS orthophoto imagery."""

    _region = "DE"
    _icon = "DE"
    _resolution = 0.25
    _dataset = "dop"
    _source_crs = "EPSG:25832"
    _tile_size = 1000
    _tile_pixels = 3000
    _image_format = "image/tiff"
    _wms_crs_parameter = "srs"
    _layer: str

    @classmethod
    def layer(cls) -> str:
        """Return the WMS layer name."""
        return cls._layer

    def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        return {
            "layers": [self.layer()],
            self._wms_crs_parameter: self._source_crs,
            "bbox": (tile[1], tile[0], tile[3], tile[2]),
            "size": self._wms_image_size(),
            "format": self._image_format,
            "transparent": False,
        }


class NRWImageryProvider(GermanWMSOrthophotoProvider):
    """North Rhine-Westphalia DOP RGB orthophoto imagery."""

    _code = "nrw_dop"
    _name = "North Rhine-Westphalia DOP RGB"
    _resolution = 0.1
    _dataset = "nw-dop-rgb"
    _extents = [(52.6008271, 50.1506045, 9.5315425, 5.8923538)]

    _url = "https://www.wms.nrw.de/geobasis/wms_nw_dop"
    _layer = "nw_dop_rgb"


class BavariaImageryProvider(GermanWMSOrthophotoProvider):
    """Bavaria DOP20 RGB orthophoto imagery."""

    _code = "bavaria_dop20"
    _name = "Bavaria DOP20 RGB"
    _resolution = 0.2
    _dataset = "by-dop20-rgb"
    _extents = [(50.56, 47.25, 13.91, 8.95)]

    _url = "https://geoservices.bayern.de/od/wms/dop/v1/dop20"
    _layer = "by_dop20c"


class HessenImageryProvider(GermanWMSOrthophotoProvider):
    """Hessen DOP20 RGB orthophoto imagery."""

    _code = "hessen_dop20"
    _name = "Hessen DOP20 RGB"
    _resolution = 0.2
    _dataset = "he-dop20-rgb"
    _extents = [(51.66698, 49.38533, 10.25780, 7.72773)]

    _url = "https://www.gds-srv.hessen.de/cgi-bin/lika-services/ogc-free-images.ows"
    _layer = "he_dop20_rgb"


class NiedersachsenImageryProvider(GermanWMSOrthophotoProvider):
    """Lower Saxony DOP20 RGB orthophoto imagery."""

    _code = "niedersachsen_dop20"
    _name = "Lower Saxony DOP20 RGB"
    _resolution = 0.2
    _dataset = "ni-dop20-rgb"
    _extents = [(54.148101, 51.153098, 11.754046, 6.505772)]

    _url = "https://opendata.lgln.niedersachsen.de/doorman/noauth/dop_wms"
    _layer = "ni_dop20"


class ThuringiaImageryProvider(GermanWMSOrthophotoProvider):
    """Thuringia DOP20 RGB orthophoto imagery."""

    _code = "thuringia_dop20"
    _name = "Thuringia DOP20 RGB"
    _resolution = 0.2
    _dataset = "th-dop20-rgb"
    _extents = [(51.5997, 50.2070, 12.69674, 9.8548)]

    _url = "https://www.geoproxy.geoportal-th.de/geoproxy/services/DOP20"
    _layer = "th_dop"
