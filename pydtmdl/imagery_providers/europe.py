"""European orthophoto imagery providers backed by official WMS services."""

from __future__ import annotations

from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from pydtmdl.base.imagery_wms import WMSImageryProvider


class EuropeanWMSImageryProvider(WMSImageryProvider):
    """Shared WMS request shape for European orthophoto services."""

    _tile_size = 1000
    _tile_pixels = 3000
    _shared_tile_subdirectory = "shared_v2"
    _wms_crs_parameter = "srs"
    _layer: str

    @classmethod
    def layer(cls) -> str:
        """Return the configured WMS layer name."""
        return cls._layer

    def _transform_bbox_to_source_crs(
        self,
        bbox: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        """Transform a WGS84 bbox to projected x/y bounds for European WMS services."""
        north, south, east, west = bbox
        transformer = Transformer.from_crs("EPSG:4326", self._source_crs, always_xy=True)
        corners = [
            transformer.transform(west, south),
            transformer.transform(west, north),
            transformer.transform(east, north),
            transformer.transform(east, south),
        ]
        xs = [corner[0] for corner in corners]
        ys = [corner[1] for corner in corners]
        return min(ys), max(ys), max(xs), min(xs)

    def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
        return {
            "layers": [self.layer()],
            self._wms_crs_parameter: self._source_crs,
            "bbox": tile,
            "size": self._wms_image_size(),
            "format": self._image_format,
            "transparent": False,
        }

    def _georeference_wms_image(
        self,
        image_bytes: bytes,
        tile: tuple[float, float, float, float],
    ) -> bytes:
        with MemoryFile(image_bytes) as input_memory:
            with input_memory.open() as source:
                data = source.read()
                transform = from_bounds(
                    west=tile[0],
                    south=tile[1],
                    east=tile[2],
                    north=tile[3],
                    width=source.width,
                    height=source.height,
                )
                profile = {
                    "driver": "GTiff",
                    "crs": self._source_crs,
                    "dtype": source.dtypes[0],
                    "transform": transform,
                    "count": source.count,
                    "width": source.width,
                    "height": source.height,
                }
                if source.nodata is not None:
                    profile["nodata"] = source.nodata

        with MemoryFile() as output_memory:
            with output_memory.open(**profile) as destination:
                destination.write(data)
            return output_memory.read()


class FranceBDOrthoImageryProvider(EuropeanWMSImageryProvider):
    """France IGN BD ORTHO orthophoto imagery."""

    _code = "france_bdortho"
    _name = "France IGN BD ORTHO"
    _region = "FR"
    _icon = "FR"
    _resolution = 0.2
    _dataset = "fr-bd-ortho"
    _extents = [(51.2, 41.2, 9.8, -5.3)]

    _url = "https://data.geopf.fr/wms-r/wms"
    _source_crs = "EPSG:3857"
    _image_format = "image/geotiff"
    _layer = "HR.ORTHOIMAGERY.ORTHOPHOTOS"


class SpainPNOAImageryProvider(EuropeanWMSImageryProvider):
    """Spain PNOA maximum actuality orthophoto imagery."""

    _code = "spain_pnoa"
    _name = "Spain PNOA maximum actuality"
    _region = "ES"
    _icon = "ES"
    _resolution = 0.25
    _dataset = "es-pnoa-ma"
    _extents = [(43.9, 35.8, 4.6, -9.5)]

    _url = "https://www.ign.es/wms-inspire/pnoa-ma"
    _source_crs = "EPSG:3857"
    _image_format = "image/tiff"
    _layer = "OI.OrthoimageCoverage"


class NetherlandsPDOKImageryProvider(EuropeanWMSImageryProvider):
    """Netherlands PDOK high-resolution RGB orthophoto imagery."""

    _code = "netherlands_luchtfoto_hr"
    _name = "Netherlands PDOK Luchtfoto RGB HR"
    _region = "NL"
    _icon = "NL"
    _resolution = 0.08
    _dataset = "nl-luchtfoto-rgb-hr"
    _extents = [(53.7, 50.7, 7.3, 3.2)]

    _url = "https://service.pdok.nl/hwh/luchtfotorgb/wms/v1_0"
    _source_crs = "EPSG:28992"
    _tile_pixels = 2048
    _image_format = "image/jpeg"
    _layer = "Actueel_orthoHR"


class LuxembourgOrthophotoImageryProvider(EuropeanWMSImageryProvider):
    """Luxembourg latest public orthophoto imagery."""

    _code = "luxembourg_orthophoto"
    _name = "Luxembourg orthophoto"
    _region = "LU"
    _icon = "LU"
    _resolution = 0.1
    _dataset = "lu-orthophoto"
    _extents = [(50.2, 49.4, 6.55, 5.73)]

    _url = "https://wms.geoportail.lu/public_map_layers/service"
    _source_crs = "EPSG:2169"
    _image_format = "image/jpeg"
    _layer = "3207"


class CopernicusVHR2021ImageryProvider(EuropeanWMSImageryProvider):
    """Copernicus pan-European very high resolution 2021 mosaic."""

    _code = "copernicus_vhr_2021"
    _name = "Copernicus VHR 2021"
    _region = "EU"
    _icon = "EU"
    _resolution = 2.0
    _dataset = "eu-copernicus-vhr-2021"
    _extents = [(72.0, 34.0, 45.0, -25.0)]

    _url = (
        "https://copernicus.discomap.eea.europa.eu/arcgis/services/"
        "GioLand/VHR_2021_LAEA/ImageServer/WMSServer"
    )
    _source_crs = "EPSG:3035"
    _image_format = "image/tiff"
    _layer = "VHR_2021_LAEA"


class WalloniaOrthophotoImageryProvider(EuropeanWMSImageryProvider):
    """Wallonia latest public orthophoto imagery."""

    _code = "wallonia_orthophoto"
    _name = "Wallonia orthophoto"
    _region = "BE"
    _icon = "BE"
    _resolution = 0.25
    _dataset = "be-wallonia-ortho-last"
    _extents = [(50.85, 49.45, 6.42, 2.75)]

    _url = "https://geoservices.wallonie.be/arcgis/services/IMAGERIE/ORTHO_LAST/MapServer/WMSServer"
    _source_crs = "EPSG:3857"
    _tile_pixels = 2048
    _image_format = "image/jpeg"
    _layer = "0"
