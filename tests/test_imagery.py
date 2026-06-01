from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

from pydtmdl import DTMProvider, ImageryProvider
from pydtmdl.base.dtm import CropExtractionError, DownloadFailedError
from pydtmdl.base.imagery_wms import WMSImageryProvider
from pydtmdl.base.imagery_wmts import WMTSImageryProvider
from pydtmdl.imagery_providers.austria import AustriaBasemapOrthophotoImageryProvider
from pydtmdl.imagery_providers.europe import (
    CopernicusVHR2021ImageryProvider,
    FranceBDOrthoImageryProvider,
    LuxembourgOrthophotoImageryProvider,
    NetherlandsPDOKImageryProvider,
    SpainPNOAImageryProvider,
    WalloniaOrthophotoImageryProvider,
)
from pydtmdl.imagery_providers.germany import (
    BavariaImageryProvider,
    HessenImageryProvider,
    NiedersachsenImageryProvider,
    NRWImageryProvider,
    ThuringiaImageryProvider,
)
from pydtmdl.imagery_providers.naip import NAIPImageryProvider, NAIPImagerySettings
from pydtmdl.imagery_providers.sentinel2 import (
    Sentinel2L2AImageryProvider,
    Sentinel2L2AImagerySettings,
)
from pydtmdl.imagery_providers.switzerland import SwitzerlandSWISSIMAGEImageryProvider

_IMAGERY_PROVIDER_COUNTER = count()


def _next_imagery_code(prefix: str) -> str:
    return f"{prefix}_{next(_IMAGERY_PROVIDER_COUNTER)}"


def _write_rgb_raster(path: Path) -> Path:
    return _write_rgb_raster_with_bounds(path, bounds=(-0.05, -0.05, 0.05, 0.05))


def _write_rgb_raster_with_bounds(
    path: Path,
    *,
    bounds: tuple[float, float, float, float],
) -> Path:
    red = np.full((200, 200), 2200, dtype=np.uint16)
    green = np.full((200, 200), 1800, dtype=np.uint16)
    blue = np.full((200, 200), 1400, dtype=np.uint16)
    data = np.stack([red, green, blue])
    transform = from_bounds(*bounds, 200, 200)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=3,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(data)
    return path


def _write_partial_rgb_raster(path: Path) -> Path:
    red = np.full((200, 200), 2200, dtype=np.uint16)
    green = np.full((200, 200), 1800, dtype=np.uint16)
    blue = np.full((200, 200), 1400, dtype=np.uint16)
    data = np.stack([red, green, blue])
    data[:, 95:105, :] = 0
    transform = from_bounds(-0.05, -0.05, 0.05, 0.05, 200, 200)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=3,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(data)
    return path


def _write_rgbir_raster(path: Path) -> Path:
    red = np.full((200, 200), 180, dtype=np.uint8)
    green = np.full((200, 200), 150, dtype=np.uint8)
    blue = np.full((200, 200), 120, dtype=np.uint8)
    nir = np.full((200, 200), 200, dtype=np.uint8)
    data = np.stack([red, green, blue, nir])
    transform = from_bounds(-105.25, 39.99, -105.18, 40.06, 200, 200)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=4,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dataset:
        dataset.write(data)
    return path


def _make_rgb_jpeg_bytes() -> bytes:
    data = np.zeros((3, 24, 32), dtype=np.uint8)
    data[0, :, :] = 120
    data[1, :, :] = 90
    data[2, :, :] = 60

    with MemoryFile() as memory_file:
        with memory_file.open(
            driver="JPEG",
            width=32,
            height=24,
            count=3,
            dtype=data.dtype,
        ) as dataset:
            dataset.write(data)
        return memory_file.read()


def _write_scl_raster(path: Path) -> Path:
    return _write_scl_raster_with_bounds(path, bounds=(-0.05, -0.05, 0.05, 0.05), cloudy=True)


def _write_scl_raster_with_bounds(
    path: Path,
    *,
    bounds: tuple[float, float, float, float],
    cloudy: bool,
) -> Path:
    data = np.full((200, 200), 4, dtype=np.uint8)
    if cloudy:
        data[85:115, 85:115] = 9
    transform = from_bounds(*bounds, 200, 200)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(data, 1)
    return path


def _write_single_band_raster(
    path: Path,
    *,
    crs: str,
    bounds: tuple[float, float, float, float],
    value: int,
) -> Path:
    data = np.full((40, 40), value, dtype=np.uint8)
    transform = from_bounds(*bounds, 40, 40)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=40,
        width=40,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(data, 1)
    return path


def _make_static_imagery_provider(source_path: Path, code: str):
    calls = {"download": 0}

    class StaticImageryProvider(ImageryProvider):
        _code = code
        _name = f"Static imagery {code}"
        _region = "Test"
        _icon = "I"
        _resolution = 10.0
        _dataset = "test-imagery"
        _extents = [(1.0, -1.0, 1.0, -1.0)]

        def download_tiles(self) -> list[str]:
            calls["download"] += 1
            return [str(source_path)]

    return StaticImageryProvider, calls


def test_imagery_provider_extract_area_returns_rgb_result(tmp_path: Path):
    source_path = _write_rgb_raster(tmp_path / "imagery_source.tif")
    provider_code = _next_imagery_code("imagery")
    provider_class, _ = _make_static_imagery_provider(source_path, provider_code)

    result = ImageryProvider.extract_area(
        center=(0.0, 0.0),
        width_m=2000,
        height_m=1000,
        rotation_deg=15,
        provider_code=provider_class.code(),
        directory=str(tmp_path),
    )

    assert result.metadata.requested_provider == provider_code
    assert result.metadata.actual_provider == provider_code
    assert result.metadata.band_count == 3
    assert result.metadata.dataset == "test-imagery"
    assert result.data.shape[0] == 3
    assert result.data.dtype == np.uint16


def test_wms_imagery_provider_georeferences_jpeg_bytes(tmp_path: Path):
    class TestJPEGWMSImageryProvider(WMSImageryProvider):
        _code = _next_imagery_code("jpeg_wms")
        _name = "JPEG WMS"
        _region = "Test"
        _icon = "I"
        _resolution = 0.5
        _dataset = "test-jpeg-wms"
        _extents = [(1.0, -1.0, 1.0, -1.0)]
        _url = "https://example.invalid/wms"
        _source_crs = "EPSG:3857"
        _image_format = "image/jpeg"

        def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
            return {}

    provider = TestJPEGWMSImageryProvider(
        coordinates=(0.0, 0.0),
        width_m=1000,
        height_m=1000,
        directory=str(tmp_path),
    )
    tile = (10.0, 20.0, 30.0, 40.0)

    wrapped = provider._georeference_wms_image(_make_rgb_jpeg_bytes(), tile)

    with MemoryFile(wrapped) as memory_file:
        with memory_file.open() as dataset:
            assert dataset.driver == "GTiff"
            assert str(dataset.crs) == provider._source_crs
            assert dataset.count == 3
            assert dataset.bounds.left == pytest.approx(tile[1])
            assert dataset.bounds.bottom == pytest.approx(tile[0])
            assert dataset.bounds.right == pytest.approx(tile[3])
            assert dataset.bounds.top == pytest.approx(tile[2])


def test_tile_download_failure_threshold_aborts_provider(tmp_path: Path):
    class FailingWMSImageryProvider(WMSImageryProvider):
        _code = _next_imagery_code("failing_wms")
        _name = "Failing WMS"
        _region = "Test"
        _icon = "I"
        _resolution = 1.0
        _dataset = "test-failing-wms"
        _extents = [(1.0, -1.0, 1.0, -1.0)]
        _url = "https://example.invalid/wms"
        _max_retries = 1

        def get_wms_parameters(self, tile: tuple[float, float, float, float]) -> dict:
            return {}

    provider = FailingWMSImageryProvider(
        coordinates=(0.0, 0.0),
        width_m=1000,
        height_m=1000,
        directory=str(tmp_path),
        max_failed_tiles=2,
        max_failed_tile_ratio=0.5,
    )
    tiles = [
        (0.0, 0.0, 1.0, 1.0),
        (1.0, 1.0, 2.0, 2.0),
        (2.0, 2.0, 3.0, 3.0),
    ]

    with pytest.raises(DownloadFailedError, match="2/2 observed tiles failed"):
        provider.download_tiles_with_fetcher(
            tiles,
            str(tmp_path),
            lambda _tile: (_ for _ in ()).throw(RuntimeError("service unavailable")),
        )


def test_european_imagery_providers_are_registered():
    expected = {
        "austria_basemap_orthofoto": AustriaBasemapOrthophotoImageryProvider,
        "france_bdortho": FranceBDOrthoImageryProvider,
        "spain_pnoa": SpainPNOAImageryProvider,
        "netherlands_luchtfoto_hr": NetherlandsPDOKImageryProvider,
        "luxembourg_orthophoto": LuxembourgOrthophotoImageryProvider,
        "copernicus_vhr_2021": CopernicusVHR2021ImageryProvider,
        "switzerland_swissimage": SwitzerlandSWISSIMAGEImageryProvider,
        "wallonia_orthophoto": WalloniaOrthophotoImageryProvider,
    }

    for code, provider_class in expected.items():
        assert ImageryProvider.get_provider_by_code(code) is provider_class


def test_wms_imagery_provider_transforms_projected_bbox_in_xy_order(tmp_path: Path):
    provider = NetherlandsPDOKImageryProvider(
        coordinates=(52.3676, 4.9041),
        width_m=512,
        height_m=512,
        directory=str(tmp_path),
    )

    bbox = provider._transform_bbox_to_source_crs(provider.get_bbox())

    transformer = Transformer.from_crs("EPSG:4326", provider._source_crs, always_xy=True)
    x, y = transformer.transform(4.9041, 52.3676)

    assert bbox[3] < x < bbox[2]
    assert bbox[0] < y < bbox[1]

    tile = (bbox[3], bbox[0], bbox[2], bbox[1])
    assert provider.get_wms_parameters(tile)["bbox"] == tile


def test_european_wms_provider_georeferences_jpeg_bytes_in_xy_order(tmp_path: Path):
    provider = NetherlandsPDOKImageryProvider(
        coordinates=(52.3676, 4.9041),
        width_m=512,
        height_m=512,
        directory=str(tmp_path),
    )
    tile = (121840.0, 486488.0, 122352.0, 487000.0)

    wrapped = provider._georeference_wms_image(_make_rgb_jpeg_bytes(), tile)

    with MemoryFile(wrapped) as memory_file:
        with memory_file.open() as dataset:
            assert dataset.bounds.left == pytest.approx(tile[0])
            assert dataset.bounds.bottom == pytest.approx(tile[1])
            assert dataset.bounds.right == pytest.approx(tile[2])
            assert dataset.bounds.top == pytest.approx(tile[3])


def test_wallonia_orthophoto_builds_expected_wms_parameters(tmp_path: Path):
    provider = WalloniaOrthophotoImageryProvider(
        coordinates=(50.467, 4.867),
        width_m=512,
        height_m=512,
        directory=str(tmp_path),
    )

    bbox = provider._transform_bbox_to_source_crs(provider.get_bbox())
    transformer = Transformer.from_crs("EPSG:4326", provider._source_crs, always_xy=True)
    x, y = transformer.transform(4.867, 50.467)
    params = provider.get_wms_parameters((100.0, 200.0, 300.0, 400.0))

    assert bbox[3] < x < bbox[2]
    assert bbox[0] < y < bbox[1]
    assert params["layers"] == ["0"]
    assert params["srs"] == "EPSG:3857"
    assert params["bbox"] == (100.0, 200.0, 300.0, 400.0)
    assert params["size"] == (2048, 2048)
    assert params["format"] == "image/jpeg"


def test_copernicus_wms_tile_size_matches_advertised_resolution(tmp_path: Path):
    provider = CopernicusVHR2021ImageryProvider(
        coordinates=(45.2741, 20.2303),
        width_m=20480,
        height_m=20480,
        directory=str(tmp_path),
    )

    params = provider.get_wms_parameters((100.0, 200.0, 1100.0, 1200.0))

    assert params["size"] == (500, 500)
    assert provider._tile_file_name((100.0, 200.0, 1100.0, 1200.0)).endswith("_500px.tif")


def test_wmts_imagery_provider_georeferences_tile_bounds(tmp_path: Path):
    class TestWMTSImageryProvider(WMTSImageryProvider):
        _code = _next_imagery_code("wmts")
        _name = "WMTS"
        _region = "Test"
        _icon = "I"
        _resolution = 1.0
        _dataset = "test-wmts"
        _extents = [(1.0, -1.0, 1.0, -1.0)]

        def get_tile_url(self, tile_matrix: int, tile_row: int, tile_col: int) -> str:
            return "https://example.invalid/tile.jpeg"

    provider = TestWMTSImageryProvider(
        coordinates=(0.0, 0.0),
        width_m=512,
        height_m=512,
        directory=str(tmp_path),
    )

    wrapped = provider._georeference_tile(_make_rgb_jpeg_bytes(), 10.0, 20.0, 30.0, 40.0)

    with MemoryFile(wrapped) as memory_file:
        with memory_file.open() as dataset:
            assert str(dataset.crs) == provider._source_crs
            assert dataset.bounds.left == pytest.approx(10.0)
            assert dataset.bounds.bottom == pytest.approx(20.0)
            assert dataset.bounds.right == pytest.approx(30.0)
            assert dataset.bounds.top == pytest.approx(40.0)


def test_austria_wmts_tile_selection_contains_vienna(tmp_path: Path):
    provider = AustriaBasemapOrthophotoImageryProvider(
        coordinates=(48.2082, 16.3738),
        width_m=512,
        height_m=512,
        directory=str(tmp_path),
    )

    left, bottom, right, top = provider._get_projected_bbox()
    tiles = provider._iter_required_tiles(left, bottom, right, top)
    transformer = Transformer.from_crs("EPSG:4326", provider._source_crs, always_xy=True)
    x, y = transformer.transform(16.3738, 48.2082)

    assert tiles
    assert any(tile[3] <= x <= tile[5] and tile[4] <= y <= tile[6] for tile in tiles)


def test_switzerland_swissimage_wmts_tile_selection_contains_bern(tmp_path: Path):
    provider = SwitzerlandSWISSIMAGEImageryProvider(
        coordinates=(46.948, 7.4474),
        width_m=512,
        height_m=512,
        directory=str(tmp_path),
    )

    left, bottom, right, top = provider._get_projected_bbox()
    tiles = provider._iter_required_tiles(left, bottom, right, top)
    transformer = Transformer.from_crs("EPSG:4326", provider._source_crs, always_xy=True)
    x, y = transformer.transform(7.4474, 46.948)

    assert tiles
    assert any(tile[3] <= x <= tile[5] and tile[4] <= y <= tile[6] for tile in tiles)
    assert provider.get_tile_url(18, 91228, 136499).endswith("/18/136499/91228.jpeg")


def test_imagery_provider_rejects_partial_valid_coverage(tmp_path: Path):
    source_path = _write_partial_rgb_raster(tmp_path / "partial_imagery_source.tif")
    provider_code = _next_imagery_code("partial_imagery")
    provider_class, _ = _make_static_imagery_provider(source_path, provider_code)

    with pytest.raises(CropExtractionError, match="insufficient valid coverage"):
        ImageryProvider.extract_area(
            center=(0.0, 0.0),
            width_m=2000,
            height_m=2000,
            provider_code=provider_class.code(),
            directory=str(tmp_path),
            min_valid_coverage=0.95,
        )

    result = ImageryProvider.extract_area(
        center=(0.0, 0.0),
        width_m=2000,
        height_m=2000,
        provider_code=provider_class.code(),
        directory=str(tmp_path),
        min_valid_coverage=0.7,
    )

    assert result.metadata.actual_provider == provider_code


def test_dtm_provider_registry_excludes_imagery_providers(tmp_path: Path):
    source_path = _write_rgb_raster(tmp_path / "registry_source.tif")
    provider_code = _next_imagery_code("registry")
    provider_class, _ = _make_static_imagery_provider(source_path, provider_code)

    assert ImageryProvider.get_provider_by_code(provider_code) is provider_class
    assert DTMProvider.get_provider_by_code(provider_code) is None


def test_sentinel_provider_renders_local_assets_with_cloud_mask(tmp_path: Path, monkeypatch):
    red_path = tmp_path / "red.tif"
    green_path = tmp_path / "green.tif"
    blue_path = tmp_path / "blue.tif"
    scl_path = tmp_path / "scl.tif"
    _write_rgb_raster(red_path)
    _write_rgb_raster(green_path)
    _write_rgb_raster(blue_path)
    _write_scl_raster(scl_path)

    item = {
        "id": "sentinel-scene-1",
        "properties": {
            "datetime": "2024-07-21T10:00:00Z",
            "eo:cloud_cover": 2.0,
        },
        "assets": {
            "red": {"href": str(red_path)},
            "green": {"href": str(green_path)},
            "blue": {"href": str(blue_path)},
            "scl": {"href": str(scl_path)},
        },
    }

    def fake_search_items(self, settings):
        return [item]

    monkeypatch.setattr(Sentinel2L2AImageryProvider, "_search_items", fake_search_items)

    result = Sentinel2L2AImageryProvider.extract_area(
        center=(0.0, 0.0),
        width_m=2000,
        height_m=2000,
        provider_code=Sentinel2L2AImageryProvider.code(),
        user_settings=Sentinel2L2AImagerySettings(
            date_from="2024-07-01",
            date_to="2024-07-31",
            max_items=1,
        ),
        directory=str(tmp_path),
    )

    assert result.metadata.actual_provider == Sentinel2L2AImageryProvider.code()
    assert result.metadata.band_count == 3
    assert result.metadata.scene_ids == ["sentinel-scene-1"]
    assert result.metadata.dtype == "uint8"
    assert np.ma.isMaskedArray(result.data)
    assert np.ma.getmaskarray(result.data).any()


def test_sentinel_selection_prioritizes_roi_coverage_across_tiles(tmp_path: Path):
    provider = Sentinel2L2AImageryProvider(
        (45.1765, 20.0078),
        size=8192,
        width_m=8192,
        height_m=8192,
        directory=str(tmp_path),
    )
    north, south, east, west = provider.get_bbox()
    narrow_east = west + ((east - west) * 0.15)
    settings = Sentinel2L2AImagerySettings(max_items=2)
    features = [
        {
            "id": "S2A_34TDQ_20260501_0_L2A",
            "bbox": [west, south, narrow_east, north],
            "properties": {"eo:cloud_cover": 0.01, "datetime": "2026-05-01T09:00:00Z"},
        },
        {
            "id": "S2B_34TDQ_20260502_0_L2A",
            "bbox": [west, south, narrow_east, north],
            "properties": {"eo:cloud_cover": 0.02, "datetime": "2026-05-02T09:00:00Z"},
        },
        {
            "id": "S2C_34TDR_20260503_0_L2A",
            "bbox": [west, south, east, north],
            "properties": {"eo:cloud_cover": 0.10, "datetime": "2026-05-03T09:00:00Z"},
        },
        {
            "id": "S2D_34TDR_20260504_0_L2A",
            "bbox": [west, south, east, north],
            "properties": {"eo:cloud_cover": 0.11, "datetime": "2026-05-04T09:00:00Z"},
        },
    ]

    selected = provider._select_items(features, settings)

    assert [item["id"] for item in selected] == [
        "S2C_34TDR_20260503_0_L2A",
        "S2A_34TDQ_20260501_0_L2A",
    ]


def test_naip_provider_renders_local_rgb_asset(tmp_path: Path, monkeypatch):
    naip_path = tmp_path / "naip.tif"
    _write_rgbir_raster(naip_path)

    item = {
        "id": "naip-scene-1",
        "properties": {
            "datetime": "2023-09-25T16:00:00Z",
            "gsd": 0.3,
        },
        "assets": {
            "image": {"href": str(naip_path)},
        },
    }

    def fake_search_items(self, settings):
        return [item]

    monkeypatch.setattr(NAIPImageryProvider, "_search_items", fake_search_items)

    result = NAIPImageryProvider.extract_area(
        center=(40.03, -105.22),
        width_m=1500,
        height_m=1500,
        provider_code=NAIPImageryProvider.code(),
        user_settings=NAIPImagerySettings(
            date_from="2020-01-01",
            date_to="2026-12-31",
            max_items=1,
        ),
        directory=str(tmp_path),
    )

    assert result.metadata.actual_provider == NAIPImageryProvider.code()
    assert result.metadata.band_count == 3
    assert result.metadata.scene_ids == ["naip-scene-1"]
    assert result.metadata.dtype == "uint8"


def test_sentinel_scene_cache_isolated_per_request(tmp_path: Path, monkeypatch):
    bounds = (-0.5, -0.5, 0.5, 0.5)
    red_path = tmp_path / "sentinel_cache_red.tif"
    green_path = tmp_path / "sentinel_cache_green.tif"
    blue_path = tmp_path / "sentinel_cache_blue.tif"
    scl_path = tmp_path / "sentinel_cache_scl.tif"
    _write_rgb_raster_with_bounds(red_path, bounds=bounds)
    _write_rgb_raster_with_bounds(green_path, bounds=bounds)
    _write_rgb_raster_with_bounds(blue_path, bounds=bounds)
    _write_scl_raster_with_bounds(scl_path, bounds=bounds, cloudy=False)

    item = {
        "id": "sentinel-shared-scene",
        "properties": {
            "datetime": "2024-07-21T10:00:00Z",
            "eo:cloud_cover": 2.0,
        },
        "assets": {
            "red": {"href": str(red_path)},
            "green": {"href": str(green_path)},
            "blue": {"href": str(blue_path)},
            "scl": {"href": str(scl_path)},
        },
    }

    def fake_search_items(self, settings):
        return [item]

    monkeypatch.setattr(Sentinel2L2AImageryProvider, "_search_items", fake_search_items)

    first_result = Sentinel2L2AImageryProvider.extract_area(
        center=(0.0, 0.49),
        width_m=4000,
        height_m=4000,
        provider_code=Sentinel2L2AImageryProvider.code(),
        user_settings=Sentinel2L2AImagerySettings(
            date_from="2024-07-01",
            date_to="2024-07-31",
            max_items=1,
        ),
        directory=str(tmp_path),
    )

    second_result = Sentinel2L2AImageryProvider.extract_area(
        center=(0.0, 0.0),
        width_m=2000,
        height_m=2000,
        provider_code=Sentinel2L2AImageryProvider.code(),
        user_settings=Sentinel2L2AImagerySettings(
            date_from="2024-07-01",
            date_to="2024-07-31",
            max_items=1,
        ),
        directory=str(tmp_path),
        min_valid_coverage=0.95,
    )

    assert first_result.metadata.scene_ids == ["sentinel-shared-scene"]
    assert second_result.metadata.scene_ids == ["sentinel-shared-scene"]
    assert first_result.metadata.source_files != second_result.metadata.source_files
    assert np.ma.isMaskedArray(second_result.data)
    assert not np.ma.getmaskarray(second_result.data).all()


def test_naip_scene_cache_isolated_per_request(tmp_path: Path, monkeypatch):
    naip_path = tmp_path / "naip_cache.tif"
    _write_rgbir_raster(naip_path)

    item = {
        "id": "naip-shared-scene",
        "properties": {
            "datetime": "2023-09-25T16:00:00Z",
            "gsd": 0.3,
        },
        "assets": {
            "image": {"href": str(naip_path)},
        },
    }

    def fake_search_items(self, settings):
        return [item]

    monkeypatch.setattr(NAIPImageryProvider, "_search_items", fake_search_items)

    first_result = NAIPImageryProvider.extract_area(
        center=(40.03, -105.183),
        width_m=4000,
        height_m=4000,
        provider_code=NAIPImageryProvider.code(),
        user_settings=NAIPImagerySettings(
            date_from="2020-01-01",
            date_to="2024-12-31",
            max_items=1,
        ),
        directory=str(tmp_path),
    )

    second_result = NAIPImageryProvider.extract_area(
        center=(40.03, -105.22),
        width_m=1500,
        height_m=1500,
        provider_code=NAIPImageryProvider.code(),
        user_settings=NAIPImagerySettings(
            date_from="2020-01-01",
            date_to="2024-12-31",
            max_items=1,
        ),
        directory=str(tmp_path),
        min_valid_coverage=0.95,
    )

    assert first_result.metadata.scene_ids == ["naip-shared-scene"]
    assert second_result.metadata.scene_ids == ["naip-shared-scene"]
    assert first_result.metadata.source_files != second_result.metadata.source_files
    assert np.ma.isMaskedArray(second_result.data)
    assert not np.ma.getmaskarray(second_result.data).all()


def test_merge_geotiff_reprojects_mixed_crs_inputs(tmp_path: Path):
    tile_32630 = _write_single_band_raster(
        tmp_path / "tile_32630.tif",
        crs="EPSG:32630",
        bounds=(500000.0, 5650000.0, 500120.0, 5650120.0),
        value=50,
    )
    tile_32631 = _write_single_band_raster(
        tmp_path / "tile_32631.tif",
        crs="EPSG:32631",
        bounds=(300000.0, 5650000.0, 300120.0, 5650120.0),
        value=100,
    )

    provider_code = _next_imagery_code("merge")
    provider_class, _ = _make_static_imagery_provider(tile_32630, provider_code)
    provider = provider_class(
        coordinates=(51.14667, 1.34241),
        width_m=100,
        height_m=100,
        directory=str(tmp_path),
    )

    merged_path, merged_crs = provider.merge_geotiff([str(tile_32630), str(tile_32631)])

    assert Path(merged_path).exists()
    with rasterio.open(merged_path) as dataset:
        assert dataset.crs is not None
        assert dataset.count == 1
        assert str(dataset.crs) == str(merged_crs)
        assert dataset.read(1).max() > 0


def test_german_imagery_providers_are_available_by_region():
    cases = [
        ((51.46, 7.01), NRWImageryProvider.code()),
        ((48.14, 11.58), BavariaImageryProvider.code()),
        ((50.11, 8.68), HessenImageryProvider.code()),
        ((52.37, 9.73), NiedersachsenImageryProvider.code()),
        ((50.98, 11.03), ThuringiaImageryProvider.code()),
    ]

    for center, provider_code in cases:
        providers = {provider.code() for provider in ImageryProvider.get_list(center)}

        assert provider_code in providers


def test_german_imagery_providers_build_expected_wms_parameters(tmp_path: Path):
    cases = [
        (NRWImageryProvider, "nw_dop_rgb"),
        (BavariaImageryProvider, "by_dop20c"),
        (HessenImageryProvider, "he_dop20_rgb"),
        (NiedersachsenImageryProvider, "ni_dop20"),
        (ThuringiaImageryProvider, "th_dop"),
    ]

    for provider_class, layer in cases:
        provider = provider_class(
            coordinates=(51.0, 8.0),
            width_m=512,
            height_m=512,
            directory=str(tmp_path),
        )

        params = provider.get_wms_parameters((100.0, 200.0, 300.0, 400.0))

        assert params["layers"] == [layer]
        assert params["srs"] == "EPSG:25832"
        assert params["bbox"] == (200.0, 100.0, 400.0, 300.0)
        assert params["size"] == (3000, 3000)
        assert params["format"] == "image/tiff"
