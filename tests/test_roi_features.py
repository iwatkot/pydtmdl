from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np
import pytest
import rasterio
import requests
from rasterio.transform import from_bounds

from pydtmdl import DownloadFailedError, DTMProvider, ProviderUnavailableError

_PROVIDER_COUNTER = count()


def _next_code(prefix: str) -> str:
    return f"{prefix}_{next(_PROVIDER_COUNTER)}"


def _write_test_raster(path: Path) -> Path:
    data = np.arange(400 * 400, dtype=np.float32).reshape(400, 400)
    transform = from_bounds(-0.05, -0.05, 0.05, 0.05, 400, 400)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)
    return path


def _make_static_provider(source_path: Path, code: str):
    calls = {"download": 0}

    class StaticProvider(DTMProvider):
        _code = code
        _name = f"Static {code}"
        _region = "Test"
        _icon = "T"
        _resolution = 10.0
        _extents = [(1.0, -1.0, 1.0, -1.0)]

        def download_tiles(self) -> list[str]:
            calls["download"] += 1
            return [str(source_path)]

    return StaticProvider, calls


def _make_failing_provider(code: str):
    class FailingProvider(DTMProvider):
        _code = code
        _name = f"Failing {code}"
        _region = "Test"
        _icon = "F"
        _resolution = 10.0
        _extents = [(1.0, -1.0, 1.0, -1.0)]

        def download_tiles(self) -> list[str]:
            raise requests.exceptions.RequestException("primary provider unavailable")

    return FailingProvider


def test_provider_size_backwards_compatibility(tmp_path: Path):
    source_path = _write_test_raster(tmp_path / "source_square.tif")
    provider_code = _next_code("compat")
    provider_class, _ = _make_static_provider(source_path, provider_code)

    provider = provider_class((0.0, 0.0), size=1024, directory=str(tmp_path))

    assert provider.size == 1024
    assert provider.width_m == 1024
    assert provider.height_m == 1024
    assert provider.rotation_deg == 0.0


def test_provider_width_only_preserves_square_behavior(tmp_path: Path):
    source_path = _write_test_raster(tmp_path / "source_width_only.tif")
    provider_code = _next_code("width_only")
    provider_class, _ = _make_static_provider(source_path, provider_code)

    provider = provider_class((0.0, 0.0), width_m=1024, directory=str(tmp_path))

    assert provider.size == 1024
    assert provider.width_m == 1024
    assert provider.height_m == 1024
    assert provider.rotation_deg == 0.0


def test_extract_area_returns_structured_metadata_for_rotated_rectangle(tmp_path: Path):
    source_path = _write_test_raster(tmp_path / "source_rotated.tif")
    provider_code = _next_code("rotated")
    provider_class, _ = _make_static_provider(source_path, provider_code)

    result = DTMProvider.extract_area(
        center=(0.0, 0.0),
        width_m=2000,
        height_m=1000,
        rotation_deg=30,
        provider_code=provider_class.code(),
        directory=str(tmp_path),
    )

    assert result.metadata.requested_provider == provider_code
    assert result.metadata.actual_provider == provider_code
    assert result.metadata.width_m == 2000
    assert result.metadata.height_m == 1000
    assert result.metadata.rotation_deg == 30
    assert result.metadata.cache_hit is False
    assert result.metadata.shape[0] > 0
    assert result.metadata.shape[1] > 0
    assert result.metadata.shape[0] != result.metadata.shape[1]
    assert result.metadata.source_files == [str(source_path)]
    assert Path(result.metadata.output_path).exists()
    assert np.ma.isMaskedArray(result.data)
    assert np.ma.getmaskarray(result.data).any()


def test_get_result_uses_stable_cache_key_and_reports_cache_hit(tmp_path: Path):
    source_path = _write_test_raster(tmp_path / "source_cache.tif")
    provider_code = _next_code("cache")
    provider_class, calls = _make_static_provider(source_path, provider_code)

    first = provider_class((0.0, 0.0), size=1024, directory=str(tmp_path)).get_result()
    second = provider_class((0.0, 0.0), size=1024, directory=str(tmp_path)).get_result()

    assert first.metadata.cache_hit is False
    assert second.metadata.cache_hit is True
    assert first.metadata.cache_key == second.metadata.cache_key
    assert first.metadata.cache_path == second.metadata.cache_path
    assert calls["download"] == 1


def test_extract_area_uses_fallback_provider_and_reports_primary_failure(tmp_path: Path):
    source_path = _write_test_raster(tmp_path / "source_fallback.tif")
    primary_code = _next_code("primary")
    fallback_code = _next_code("fallback")
    _make_failing_provider(primary_code)
    fallback_provider, _ = _make_static_provider(source_path, fallback_code)

    result = DTMProvider.extract_area(
        center=(0.0, 0.0),
        width_m=1500,
        height_m=1000,
        rotation_deg=15,
        provider_code=primary_code,
        fallback_provider_code=fallback_provider.code(),
        directory=str(tmp_path),
    )

    assert result.metadata.requested_provider == primary_code
    assert result.metadata.actual_provider == fallback_code
    assert result.metadata.fallback_used is True
    assert result.metadata.primary_failure_reason is not None
    assert result.metadata.primary_failure_reason.error_type == "download_failed"


def test_extract_area_raises_machine_readable_error_without_fallback(tmp_path: Path):
    primary_code = _next_code("error")
    _make_failing_provider(primary_code)

    with pytest.raises(DownloadFailedError) as exc_info:
        DTMProvider.extract_area(
            center=(0.0, 0.0),
            width_m=1200,
            height_m=800,
            provider_code=primary_code,
            directory=str(tmp_path),
        )

    assert exc_info.value.to_details().error_type == "download_failed"


def test_extract_area_invalid_provider_code_reports_requested_code(tmp_path: Path):
    with pytest.raises(ProviderUnavailableError) as exc_info:
        DTMProvider.extract_area(
            center=(0.0, 0.0),
            width_m=1200,
            provider_code="missing_provider",
            directory=str(tmp_path),
        )

    assert exc_info.value.provider_code == "missing_provider"
    assert exc_info.value.to_details().error_type == "provider_unavailable"
    assert "missing_provider" in str(exc_info.value)
