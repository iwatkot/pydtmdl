from __future__ import annotations

from itertools import count
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from pydtmdl import DTMProvider, extract_area_from_dtm, extract_area_from_image, extract_area_from_imagery
from pydtmdl import extract_project_dtm, extract_project_dtm_from_file, extract_project_imagery_from_file

_PROVIDER_COUNTER = count()


def _next_provider_code(prefix: str) -> str:
    return f"{prefix}_{next(_PROVIDER_COUNTER)}"


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


def _write_single_band_raster_with_bounds(
    path: Path,
    *,
    bounds: tuple[float, float, float, float],
    crs: str,
) -> Path:
    data = np.arange(200 * 200, dtype=np.float32).reshape(200, 200)
    transform = from_bounds(*bounds, 200, 200)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=200,
        width=200,
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
        nodata=-9999.0,
    ) as dataset:
        dataset.write(data, 1)
    return path


def test_extract_area_from_dtm_single_band_returns_2d_result(tmp_path: Path):
    source_path = _write_single_band_raster(tmp_path / "single_band.tif")

    result = extract_area_from_dtm(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=2000,
        height_m=1000,
        rotation_deg=30,
        directory=str(tmp_path),
    )

    assert result.metadata.actual_provider == "local_dtm"
    assert result.data.ndim == 2
    assert np.ma.isMaskedArray(result.data)
    assert Path(result.metadata.output_path).exists()


def test_extract_area_from_imagery_multiband_returns_cached_rgb_stack(tmp_path: Path):
    source_path = _write_rgb_raster(tmp_path / "rgb.tif")

    first = extract_area_from_imagery(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=1500,
        height_m=1500,
        directory=str(tmp_path),
    )
    second = extract_area_from_imagery(
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


def test_extract_area_from_image_alias_matches_imagery_wrapper(tmp_path: Path):
    source_path = _write_rgb_raster(tmp_path / "alias_rgb.tif")

    aliased = extract_area_from_image(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=1000,
        height_m=1000,
        directory=str(tmp_path),
    )
    direct = extract_area_from_imagery(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=1000,
        height_m=1000,
        directory=str(tmp_path),
    )

    assert aliased.metadata.actual_provider == "local_raster"
    assert aliased.metadata.cache_key == direct.metadata.cache_key
    assert aliased.data.shape == direct.data.shape


def test_extract_project_dtm_from_file_writes_full_and_preview_pngs(tmp_path: Path):
    source_path = _write_single_band_raster(tmp_path / "project_dtm.tif")

    result = extract_project_dtm_from_file(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=1200,
        height_m=800,
        directory=str(tmp_path),
        source_buffer_m=128,
        max_edge=64,
        target_resolution_m=10,
    )

    assert result.full_format == "png"
    assert result.preview_format == "png"
    assert result.full_shape == (80, 120)
    assert max(result.preview_shape) == 64
    assert result.pixel_value_max == 65535
    assert Path(result.full_output_path).exists()
    assert Path(result.preview_output_path).exists()
    assert result.source_manifest.source_buffer_m == 128

    with rasterio.open(result.full_output_path) as src:
        assert src.driver == "PNG"
        assert src.dtypes[0] == "uint16"
        assert src.shape == result.full_shape


def test_extract_project_dtm_from_file_reuses_full_png_when_it_fits_preview_cap(tmp_path: Path):
    source_path = _write_single_band_raster(tmp_path / "project_dtm_small.tif")

    result = extract_project_dtm_from_file(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=120,
        height_m=80,
        directory=str(tmp_path),
        source_buffer_m=128,
        max_edge=256,
        target_resolution_m=10,
    )

    assert result.full_shape == (8, 12)
    assert result.preview_shape == result.full_shape
    assert result.preview_output_path == result.full_output_path


def test_extract_project_imagery_from_file_writes_capped_jpeg(tmp_path: Path):
    source_path = _write_rgb_raster(tmp_path / "project_rgb.tif")

    result = extract_project_imagery_from_file(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=1000,
        height_m=500,
        directory=str(tmp_path),
        source_buffer_m=128,
        max_edge=50,
        target_resolution_m=5,
    )

    assert result.preview_format == "jpg"
    assert result.preview_mime_type == "image/jpeg"
    assert max(result.preview_shape) == 50
    assert Path(result.preview_output_path).exists()
    with rasterio.open(result.preview_output_path) as src:
        assert src.driver == "JPEG"
        assert src.count == 3
        assert src.dtypes[0] == "uint8"


def test_extract_project_imagery_from_file_cleanup_temp_files_removes_preview_tiff(tmp_path: Path):
    source_path = _write_rgb_raster(tmp_path / "project_rgb_cleanup.tif")

    result = extract_project_imagery_from_file(
        image_path=str(source_path),
        center=(0.0, 0.0),
        width_m=1000,
        height_m=500,
        directory=str(tmp_path),
        source_buffer_m=128,
        max_edge=50,
        target_resolution_m=5,
        cleanup_temp_files=True,
    )

    assert Path(result.preview_output_path).exists()
    assert not (Path(result.cache_path) / "assets" / "imagery-preview.tmp.tif").exists()


def test_extract_project_dtm_cleanup_temp_files_removes_intermediates_only(tmp_path: Path):
    provider_code = _next_provider_code("cleanup")
    source_tiles: list[Path] = []

    class CleanupProvider(DTMProvider):
        _code = provider_code
        _name = f"Cleanup {provider_code}"
        _region = "Test"
        _icon = "T"
        _resolution = 10.0
        _extents = [(1.0, -1.0, 1.0, -1.0)]
        _output_crs = "EPSG:4326"

        def download_tiles(self) -> list[str]:
            if not source_tiles:
                source_tiles.extend(
                    [
                        _write_single_band_raster_with_bounds(
                            Path(self._source_tile_directory) / "tile_left.tif",
                            bounds=(-2000.0, -1500.0, 0.0, 1500.0),
                            crs="EPSG:3857",
                        ),
                        _write_single_band_raster_with_bounds(
                            Path(self._source_tile_directory) / "tile_right.tif",
                            bounds=(0.0, -1500.0, 2000.0, 1500.0),
                            crs="EPSG:3857",
                        ),
                    ]
                )
            return [str(path) for path in source_tiles]

    result = extract_project_dtm(
        center=(0.0, 0.0),
        width_m=1000,
        height_m=1000,
        provider_code=provider_code,
        directory=str(tmp_path),
        source_buffer_m=0,
        max_edge=64,
        target_resolution_m=10,
        cleanup_temp_files=True,
    )

    cache_path = Path(result.cache_path)

    assert Path(result.full_output_path).exists()
    assert Path(result.preview_output_path).exists()
    assert all(path.exists() for path in source_tiles)
    assert not (cache_path / "merged.tif").exists()
    assert not (cache_path / "reprojected.tif").exists()
    assert not (cache_path / "assets" / "dtm-full.tmp.tif").exists()
    assert not (cache_path / "assets" / "dtm-preview.tmp.tif").exists()
