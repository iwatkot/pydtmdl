"""Austria orthophoto imagery provider backed by basemap.at WMTS."""

from __future__ import annotations

from pydtmdl.base.imagery_wmts import WMTSImageryProvider


class AustriaBasemapOrthophotoImageryProvider(WMTSImageryProvider):
    """Austria basemap.at orthophoto imagery."""

    _code = "austria_basemap_orthofoto"
    _name = "Austria basemap.at orthophoto"
    _region = "AT"
    _icon = "AT"
    _resolution = 1.0
    _dataset = "at-basemap-orthofoto"
    _extents = [(49.147828, 45.95086, 17.777012, 9.051669)]

    _zoom = 17
    _shared_tile_subdirectory = "shared_wmts_v1"
    _layer = "bmaporthofoto30cm"
    _resource_url_template = (
        "https://maps.wien.gv.at/basemap/bmaporthofoto30cm/"
        "{Style}/{TileMatrixSet}/{TileMatrix}/{TileRow}/{TileCol}.jpeg"
    )

    def get_tile_url(self, tile_matrix: int, tile_row: int, tile_col: int) -> str:
        return self._resource_url_template.format(
            Style=self._style,
            TileMatrixSet=self._tile_matrix_set,
            TileMatrix=tile_matrix,
            TileRow=tile_row,
            TileCol=tile_col,
        )
