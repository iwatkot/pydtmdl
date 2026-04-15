from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from pydtmdl import DTMProvider, ImageryProvider
from pydtmdl.imagery_providers.naip import NAIPImageryProvider, NAIPImagerySettings
from pydtmdl.imagery_providers.sentinel2 import (
    Sentinel2L2AImageryProvider,
    Sentinel2L2AImagerySettings,
)

_IMAGERY_PROVIDER_COUNTER = count()


def _next_imagery_code(prefix: str) -> str:
    return f"{prefix}_{next(_IMAGERY_PROVIDER_COUNTER)}"


def _write_rgb_raster(path: Path) -> Path:
    red = np.full((200, 200), 2200, dtype=np.uint16)
    green = np.full((200, 200), 1800, dtype=np.uint16)
    blue = np.full((200, 200), 1400, dtype=np.uint16)
    data = np.stack([red, green, blue])
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


def _write_scl_raster(path: Path) -> Path:
    data = np.full((200, 200), 4, dtype=np.uint8)
    data[85:115, 85:115] = 9
    transform = from_bounds(-0.05, -0.05, 0.05, 0.05, 200, 200)
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
    assert result.data.shape[0] == 3
    assert np.ma.isMaskedArray(result.data)
