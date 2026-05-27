"""Switzerland orthophoto imagery provider backed by swisstopo WMTS."""

from __future__ import annotations

from pydtmdl.base.imagery_wmts import WMTSImageryProvider


class SwitzerlandSWISSIMAGEImageryProvider(WMTSImageryProvider):
    """Switzerland SWISSIMAGE orthophoto imagery."""

    _code = "switzerland_swissimage"
    _name = "Switzerland SWISSIMAGE"
    _region = "CH"
    _icon = "CH"
    _resolution = 0.1
    _dataset = "ch-swisstopo-swissimage"
    _extents = [(47.8308275417, 45.7769477403, 10.4427014502, 6.02260949059)]

    _zoom = 18
    _shared_tile_subdirectory = "shared_wmts_swissimage_v1"
    _layer = "ch.swisstopo.swissimage"
    _resource_url_template = (
        "https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/"
        "default/current/3857/{TileMatrix}/{TileCol}/{TileRow}.jpeg"
    )

    def get_tile_url(self, tile_matrix: int, tile_row: int, tile_col: int) -> str:
        return self._resource_url_template.format(
            TileMatrix=tile_matrix,
            TileRow=tile_row,
            TileCol=tile_col,
        )
