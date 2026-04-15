from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from pydtmdl import extract_area_from_image


def _write_single_band_raster(path: Path) -> Path:
    data = np.arange(400 * 400, dtype=np.float32).reshape(400, 400)
    transform = from_bounds(-0.05, -0.05, 0.05, 0.05, 400, 400)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=400,
        width=400,
        count=1,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)
    return path


def _write_rgb_raster(path: Path) -> Path:
    red = np.full((200, 200), 180, dtype=np.uint8)
    green = np.full((200, 200), 140, dtype=np.uint8)
    blue = np.full((200, 200), 100, dtype=np.uint8)
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


def test_extract_area_from_image_single_band_returns_2d_result(tmp_path: Path):
    source_path = _write_single_band_raster(tmp_path / "single_band.tif")

    result = extract_area_from_image(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=2000,
        height_m=1000,
        rotation_deg=30,
        directory=str(tmp_path),
    )

    assert result.metadata.actual_provider == "local_raster"
    assert result.metadata.band_count == 1
    assert result.data.ndim == 2
    assert np.ma.isMaskedArray(result.data)
    assert Path(result.metadata.output_path).exists()


def test_extract_area_from_image_multiband_returns_cached_rgb_stack(tmp_path: Path):
    source_path = _write_rgb_raster(tmp_path / "rgb.tif")

    first = extract_area_from_image(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=1500,
        height_m=1500,
        directory=str(tmp_path),
    )
    second = extract_area_from_image(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=1500,
        height_m=1500,
        directory=str(tmp_path),
    )

    assert first.metadata.band_count == 3
    assert first.data.shape[0] == 3
    assert first.metadata.cache_hit is False
    assert second.metadata.cache_hit is True
    assert first.metadata.cache_key == second.metadata.cache_key
