"""Final asset generation APIs for application-ready pydtmdl outputs."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Type, cast

import numpy as np
import rasterio
from affine import Affine
from pydantic import BaseModel
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.warp import reproject

from pydtmdl.base.dtm import (
    DTMErrorDetails,
    DTMProvider,
    DTMProviderError,
    DTMProviderSettings,
    ProviderUnavailableError,
)
from pydtmdl.base.imagery import ImageryProvider
from pydtmdl.base.local_raster import LocalDTMProvider, LocalRasterProvider, LocalRasterSettings


@dataclass(frozen=True, slots=True)
class ProjectGrid:
    """Metric project grid used for MapToPlay-ready derivatives."""

    crs: CRS
    transform: Affine
    width: int
    height: int
    resolution_m: float


class SourceManifest(BaseModel):
    """Metadata needed to audit or recreate generated assets."""

    provider_code: str
    fallback_provider_code: str | None = None
    provider_settings: dict[str, Any] = {}
    center: tuple[float, float]
    requested_width_m: int
    requested_height_m: int
    source_width_m: int
    source_height_m: int
    source_buffer_m: float
    rotation_deg: float
    source_resolution: float | None = None
    source_files: list[str] = []
    cache_key: str | None = None
    created_by_pydtmdl_version: str | None = None


class DTMAssetResult(BaseModel):
    """Application-ready DTM outputs.

    GeoTIFFs are intentionally not exposed as public assets here. Any temporary
    raster used to build these PNGs remains an internal pydtmdl implementation detail.
    """

    requested_provider: str
    requested_provider_name: str | None = None
    actual_provider: str
    actual_provider_name: str | None = None
    fallback_used: bool = False
    primary_failure_reason: DTMErrorDetails | None = None

    preview_output_path: str
    preview_format: str = "png"
    preview_mime_type: str = "image/png"
    preview_dtype: str = "uint16"
    preview_shape: tuple[int, int]
    preview_resolution: float

    full_output_path: str
    full_format: str = "png"
    full_mime_type: str = "image/png"
    full_dtype: str = "uint16"
    full_shape: tuple[int, int]
    full_resolution: float

    processed_output_crs: str
    pixel_value_max: int = 65535
    min: float
    max: float
    processed_min: int = 0
    processed_max: int = 65535
    nodata: None = None

    source_resolution: float | None = None
    source_width_m: int
    source_height_m: int
    source_buffer_m: float
    source_files: list[str]
    source_manifest: SourceManifest

    center: tuple[float, float]
    width_m: int
    height_m: int
    rotation_deg: float
    cache_hit: bool = False
    cache_key: str | None = None
    cache_path: str | None = None


class ImageryAssetResult(BaseModel):
    """Application-ready imagery preview output."""

    requested_provider: str
    requested_provider_name: str | None = None
    actual_provider: str
    actual_provider_name: str | None = None
    fallback_used: bool = False
    primary_failure_reason: DTMErrorDetails | None = None

    preview_output_path: str
    preview_format: str = "jpg"
    preview_mime_type: str = "image/jpeg"
    preview_dtype: str = "uint8"
    preview_shape: tuple[int, int]
    preview_resolution: float
    processed_output_crs: str

    source_resolution: float | None = None
    source_width_m: int
    source_height_m: int
    source_buffer_m: float
    source_files: list[str]
    source_manifest: SourceManifest

    center: tuple[float, float]
    width_m: int
    height_m: int
    rotation_deg: float
    cache_hit: bool = False
    cache_key: str | None = None
    cache_path: str | None = None


def _auto_utm_crs(center: tuple[float, float]) -> CRS:
    lat, lon = center
    zone = math.floor((lon + 180.0) / 6.0) + 1
    epsg = (32600 if lat >= 0 else 32700) + zone
    return CRS.from_epsg(epsg)


def _build_project_grid(
    *,
    center: tuple[float, float],
    width_m: int,
    height_m: int,
    rotation_deg: float,
    output_width: int,
    output_height: int,
    destination_crs: str,
) -> ProjectGrid:
    crs = _auto_utm_crs(center) if destination_crs == "auto-utm" else CRS.from_user_input(destination_crs)
    center_lat, center_lon = center
    center_x, center_y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(
        center_lon,
        center_lat,
    )
    pixel_width_m = width_m / output_width
    pixel_height_m = height_m / output_height
    angle = math.radians(rotation_deg)
    cos_angle = math.cos(angle)
    sin_angle = math.sin(angle)
    transform = Affine(
        pixel_width_m * cos_angle,
        -pixel_height_m * sin_angle,
        center_x + (-width_m / 2 * cos_angle) - (-height_m / 2 * sin_angle),
        -pixel_width_m * sin_angle,
        -pixel_height_m * cos_angle,
        center_y - (-width_m / 2 * sin_angle) - (-height_m / 2 * cos_angle),
    )
    return ProjectGrid(
        crs=crs,
        transform=transform,
        width=output_width,
        height=output_height,
        resolution_m=max(pixel_width_m, pixel_height_m),
    )


def _bounded_shape(
    *,
    width_m: int,
    height_m: int,
    resolution_m: float,
    max_edge: int | None,
    max_pixels: int | None,
) -> tuple[int, int, float]:
    width = max(1, int(round(width_m / resolution_m)))
    height = max(1, int(round(height_m / resolution_m)))
    scale = 1.0
    if max_edge is not None and max(width, height) > max_edge:
        scale = max(scale, max(width, height) / max_edge)
    if max_pixels is not None and width * height > max_pixels:
        scale = max(scale, math.sqrt((width * height) / max_pixels))
    if scale > 1.0:
        width = max(1, int(math.floor(width / scale)))
        height = max(1, int(math.floor(height / scale)))
    return width, height, max(width_m / width, height_m / height)


def _resolve_source_buffer(width_m: int, height_m: int, source_buffer_m: float | None) -> float:
    if source_buffer_m is not None:
        return float(source_buffer_m)
    return float(min(max(max(width_m, height_m) * 0.04, 128.0), 512.0))


def _resampling(name: str) -> Resampling:
    try:
        return Resampling[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported resampling method: {name}") from exc


def _reproject_to_temp_tiff(
    *,
    source_path: str,
    output_path: str,
    grid: ProjectGrid,
    band_indexes: list[int],
    dtype: str,
    resampling: Resampling,
) -> None:
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nodata = np.nan if np.issubdtype(np.dtype(dtype), np.floating) else 0
    with rasterio.open(source_path) as src:
        profile = {
            "driver": "GTiff",
            "height": grid.height,
            "width": grid.width,
            "count": len(band_indexes),
            "dtype": dtype,
            "crs": grid.crs,
            "transform": grid.transform,
            "nodata": nodata,
            "tiled": True,
            "blockxsize": 256,
            "blockysize": 256,
            "compress": "deflate",
        }
        with rasterio.open(output_path, "w", **profile) as dst:
            for output_index, source_index in enumerate(band_indexes, start=1):
                reproject(
                    source=rasterio.band(src, source_index),
                    destination=rasterio.band(dst, output_index),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=grid.transform,
                    dst_crs=grid.crs,
                    dst_nodata=nodata,
                    resampling=resampling,
                )


def _scan_min_max(path: str) -> tuple[float, float]:
    minimum: float | None = None
    maximum: float | None = None
    with rasterio.open(path) as src:
        for _, window in src.block_windows(1):
            data = src.read(1, window=window, masked=True)
            data = np.ma.masked_invalid(data)
            if data.count() == 0:
                continue
            block_min = float(data.min())
            block_max = float(data.max())
            minimum = block_min if minimum is None else min(minimum, block_min)
            maximum = block_max if maximum is None else max(maximum, block_max)
    if minimum is None or maximum is None:
        raise ValueError("The processed DTM does not contain any valid data.")
    return minimum, maximum


def _write_normalized_png(
    *,
    source_path: str,
    output_path: str,
    height_min: float,
    height_max: float,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as src:
        with rasterio.open(
            output,
            "w",
            driver="PNG",
            width=src.width,
            height=src.height,
            count=1,
            dtype="uint16",
        ) as dst:
            span = height_max - height_min
            for _, window in src.block_windows(1):
                data = src.read(1, window=window, masked=True)
                data = np.ma.masked_invalid(data)
                filled = np.asarray(data.filled(height_min), dtype=np.float32)
                if math.isclose(span, 0.0):
                    normalized = np.zeros(filled.shape, dtype=np.uint16)
                else:
                    scaled = ((filled - height_min) / span) * 65535.0
                    normalized = np.rint(np.clip(scaled, 0, 65535)).astype(np.uint16)
                dst.write(normalized, 1, window=window)


def _write_jpeg_preview(source_path: str, output_path: str, quality: int) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(source_path) as src:
        band_count = min(src.count, 3)
        data = src.read(list(range(1, band_count + 1)), masked=True)
        if band_count == 1:
            data = np.repeat(data, 3, axis=0)
        elif band_count == 2:
            data = np.concatenate([data, data[1:2]], axis=0)

        if np.asarray(data).dtype == np.uint8:
            rgb = np.asarray(np.ma.array(data).filled(0), dtype=np.uint8)
        else:
            rgb = np.zeros(data.shape, dtype=np.uint8)
            for band_index in range(data.shape[0]):
                band = np.ma.masked_invalid(data[band_index])
                if band.count() == 0:
                    continue
                values = band.compressed()
                low, high = np.percentile(values, [2, 98])
                if math.isclose(float(low), float(high)):
                    rgb[band_index] = 0
                else:
                    scaled = ((band.filled(low) - low) / (high - low)) * 255.0
                    rgb[band_index] = np.rint(np.clip(scaled, 0, 255)).astype(np.uint8)

        with rasterio.open(
            output,
            "w",
            driver="JPEG",
            width=src.width,
            height=src.height,
            count=3,
            dtype="uint8",
            photometric="YCBCR",
            quality=quality,
        ) as dst:
            dst.write(rgb[:3])


def _provider_settings_payload(provider: DTMProvider) -> dict[str, Any]:
    get_payload = getattr(provider, "get_cache_settings_payload", None)
    if callable(get_payload):
        return cast(dict[str, Any], get_payload())
    return {}


def _source_manifest(
    *,
    provider: DTMProvider,
    fallback_provider_code: str | None,
    center: tuple[float, float],
    width_m: int,
    height_m: int,
    source_width_m: int,
    source_height_m: int,
    source_buffer_m: float,
    rotation_deg: float,
    source_files: list[str],
) -> SourceManifest:
    return SourceManifest(
        provider_code=provider.code() or "unknown",
        fallback_provider_code=fallback_provider_code,
        provider_settings=_provider_settings_payload(provider),
        center=center,
        requested_width_m=width_m,
        requested_height_m=height_m,
        source_width_m=source_width_m,
        source_height_m=source_height_m,
        source_buffer_m=source_buffer_m,
        rotation_deg=rotation_deg,
        source_resolution=provider.resolution(),
        source_files=source_files,
        cache_key=provider.cache_key,
    )


def _select_provider_class(
    base_class: Any,
    center: tuple[float, float],
    width_m: int,
    height_m: int,
    rotation_deg: float,
    provider_code: str | None,
) -> Type[DTMProvider]:
    if base_class is DTMProvider:
        provider_class = (
            base_class.get_provider_by_code(provider_code)
            if provider_code
            else base_class.get_best(center, width_m=width_m, height_m=height_m, rotation_deg=rotation_deg)
        )
    else:
        provider_class = base_class
    if provider_class is None:
        if provider_code:
            raise ProviderUnavailableError(
                f"No provider is available for the requested provider_code: {provider_code!r}.",
                provider_code=provider_code,
            )
        raise ProviderUnavailableError("No provider is available for the requested geometry.")
    return provider_class


def _select_imagery_provider_class(
    center: tuple[float, float],
    width_m: int,
    height_m: int,
    rotation_deg: float,
    provider_code: str | None,
) -> Type[DTMProvider]:
    provider_class = (
        ImageryProvider.get_provider_by_code(provider_code)
        if provider_code
        else ImageryProvider.get_best(
            center,
            width_m=width_m,
            height_m=height_m,
            rotation_deg=rotation_deg,
        )
    )
    if provider_class is None:
        if provider_code:
            raise ProviderUnavailableError(
                f"No imagery provider is available for the requested provider_code: {provider_code!r}.",
                provider_code=provider_code,
            )
        raise ProviderUnavailableError("No imagery provider is available for the requested geometry.")
    return cast(Type[DTMProvider], provider_class)


def _instantiate_provider(
    provider_class: Type[DTMProvider],
    *,
    center: tuple[float, float],
    source_width_m: int,
    source_height_m: int,
    rotation_deg: float,
    user_settings: DTMProviderSettings | None,
    directory: str,
    logger: Any,
    min_valid_coverage: float | None,
    max_failed_tiles: int | None = None,
    max_failed_tile_ratio: float = 1.0,
) -> DTMProvider:
    return provider_class(
        center,
        size=max(source_width_m, source_height_m),
        user_settings=user_settings,
        directory=directory,
        logger=logger,
        width_m=source_width_m,
        height_m=source_height_m,
        rotation_deg=rotation_deg,
        min_valid_coverage=min_valid_coverage,
        max_failed_tiles=max_failed_tiles,
        max_failed_tile_ratio=max_failed_tile_ratio,
    )


def _prepare_provider_source(
    provider: DTMProvider,
) -> tuple[str, list[str], bool]:
    if os.path.exists(provider._result_tiff_path):  # pylint: disable=protected-access
        return provider._result_tiff_path, provider._get_cached_source_files(), True  # pylint: disable=protected-access
    source_files = provider._download_source_tiles()  # pylint: disable=protected-access
    prepared = provider._prepare_source_tile(source_files)  # pylint: disable=protected-access
    return prepared, source_files, False


def _provider_temp_paths(provider: DTMProvider) -> tuple[str, str]:
    cache_path = Path(provider.cache_path)
    return (str(cache_path / "merged.tif"), str(cache_path / "reprojected.tif"))


def _cleanup_temp_files(paths: list[str]) -> None:
    for path in {candidate for candidate in paths if candidate}:
        try:
            Path(path).unlink()
        except FileNotFoundError:
            continue


def _extract_project_dtm_with_provider(
    provider: DTMProvider,
    *,
    requested_provider_code: str,
    requested_provider_name: str | None,
    fallback_provider_code: str | None,
    fallback_used: bool,
    primary_failure: DTMErrorDetails | None,
    center: tuple[float, float],
    width_m: int,
    height_m: int,
    rotation_deg: float,
    source_width_m: int,
    source_height_m: int,
    source_buffer_m: float,
    directory: str,
    output_basename: str,
    max_edge: int,
    max_pixels: int | None,
    target_resolution_m: float | None,
    destination_crs: str,
    resampling: str,
    cleanup_temp_files: bool,
) -> DTMAssetResult:
    source_path, source_files, cache_hit = _prepare_provider_source(provider)
    source_resolution = target_resolution_m or provider.resolution() or 1.0
    full_width, full_height, full_resolution = _bounded_shape(
        width_m=width_m,
        height_m=height_m,
        resolution_m=source_resolution,
        max_edge=None,
        max_pixels=None,
    )
    preview_width, preview_height, preview_resolution = _bounded_shape(
        width_m=width_m,
        height_m=height_m,
        resolution_m=source_resolution,
        max_edge=max_edge,
        max_pixels=max_pixels,
    )
    full_grid = _build_project_grid(
        center=center,
        width_m=width_m,
        height_m=height_m,
        rotation_deg=rotation_deg,
        output_width=full_width,
        output_height=full_height,
        destination_crs=destination_crs,
    )
    preview_grid = _build_project_grid(
        center=center,
        width_m=width_m,
        height_m=height_m,
        rotation_deg=rotation_deg,
        output_width=preview_width,
        output_height=preview_height,
        destination_crs=destination_crs,
    )
    asset_dir = Path(directory) / (provider.code() or "unknown") / provider.cache_key / "assets"
    full_tiff = str(asset_dir / f"{output_basename}-full.tmp.tif")
    preview_tiff = str(asset_dir / f"{output_basename}-preview.tmp.tif")
    full_png = str(asset_dir / f"{output_basename}-full.png")
    preview_png = str(asset_dir / f"{output_basename}-preview.png")
    resolved_resampling = _resampling(resampling)
    temp_paths = [*_provider_temp_paths(provider), full_tiff, preview_tiff]

    try:
        _reproject_to_temp_tiff(
            source_path=source_path,
            output_path=full_tiff,
            grid=full_grid,
            band_indexes=[1],
            dtype="float32",
            resampling=resolved_resampling,
        )
        height_min, height_max = _scan_min_max(full_tiff)
        _write_normalized_png(
            source_path=full_tiff,
            output_path=full_png,
            height_min=height_min,
            height_max=height_max,
        )
        if preview_width == full_width and preview_height == full_height:
            preview_png = full_png
        else:
            _reproject_to_temp_tiff(
                source_path=source_path,
                output_path=preview_tiff,
                grid=preview_grid,
                band_indexes=[1],
                dtype="float32",
                resampling=resolved_resampling,
            )
            _write_normalized_png(
                source_path=preview_tiff,
                output_path=preview_png,
                height_min=height_min,
                height_max=height_max,
            )
        manifest = _source_manifest(
            provider=provider,
            fallback_provider_code=fallback_provider_code,
            center=center,
            width_m=width_m,
            height_m=height_m,
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            source_buffer_m=source_buffer_m,
            rotation_deg=rotation_deg,
            source_files=source_files,
        )
        return DTMAssetResult(
            requested_provider=requested_provider_code,
            requested_provider_name=requested_provider_name,
            actual_provider=provider.code() or "unknown",
            actual_provider_name=provider.name(),
            fallback_used=fallback_used,
            primary_failure_reason=primary_failure,
            preview_output_path=preview_png,
            preview_shape=(preview_height, preview_width),
            preview_resolution=preview_resolution,
            full_output_path=full_png,
            full_shape=(full_height, full_width),
            full_resolution=full_resolution,
            processed_output_crs=full_grid.crs.to_string(),
            min=height_min,
            max=height_max,
            source_resolution=provider.resolution(),
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            source_buffer_m=source_buffer_m,
            source_files=source_files,
            source_manifest=manifest,
            center=center,
            width_m=width_m,
            height_m=height_m,
            rotation_deg=rotation_deg,
            cache_hit=cache_hit,
            cache_key=provider.cache_key,
            cache_path=provider.cache_path,
        )
    finally:
        if cleanup_temp_files:
            _cleanup_temp_files(temp_paths)


def _extract_project_imagery_with_provider(
    provider: DTMProvider,
    *,
    requested_provider_code: str,
    requested_provider_name: str | None,
    fallback_provider_code: str | None,
    fallback_used: bool,
    primary_failure: DTMErrorDetails | None,
    center: tuple[float, float],
    width_m: int,
    height_m: int,
    rotation_deg: float,
    source_width_m: int,
    source_height_m: int,
    source_buffer_m: float,
    output_basename: str,
    max_edge: int,
    max_pixels: int | None,
    target_resolution_m: float | None,
    destination_crs: str,
    resampling: str,
    jpeg_quality: int,
    cleanup_temp_files: bool,
) -> ImageryAssetResult:
    source_path, source_files, cache_hit = _prepare_provider_source(provider)
    source_resolution = target_resolution_m or provider.resolution() or 1.0
    preview_width, preview_height, preview_resolution = _bounded_shape(
        width_m=width_m,
        height_m=height_m,
        resolution_m=source_resolution,
        max_edge=max_edge,
        max_pixels=max_pixels,
    )
    grid = _build_project_grid(
        center=center,
        width_m=width_m,
        height_m=height_m,
        rotation_deg=rotation_deg,
        output_width=preview_width,
        output_height=preview_height,
        destination_crs=destination_crs,
    )
    asset_dir = Path(provider.cache_path) / "assets"
    preview_tiff = str(asset_dir / f"{output_basename}-preview.tmp.tif")
    preview_jpg = str(asset_dir / f"{output_basename}-preview.jpg")
    temp_paths = [*_provider_temp_paths(provider), preview_tiff]

    try:
        with rasterio.open(source_path) as src:
            band_indexes = list(range(1, min(src.count, 3) + 1))
        _reproject_to_temp_tiff(
            source_path=source_path,
            output_path=preview_tiff,
            grid=grid,
            band_indexes=band_indexes,
            dtype="float32",
            resampling=_resampling(resampling),
        )
        _write_jpeg_preview(preview_tiff, preview_jpg, jpeg_quality)
        manifest = _source_manifest(
            provider=provider,
            fallback_provider_code=fallback_provider_code,
            center=center,
            width_m=width_m,
            height_m=height_m,
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            source_buffer_m=source_buffer_m,
            rotation_deg=rotation_deg,
            source_files=source_files,
        )
        return ImageryAssetResult(
            requested_provider=requested_provider_code,
            requested_provider_name=requested_provider_name,
            actual_provider=provider.code() or "unknown",
            actual_provider_name=provider.name(),
            fallback_used=fallback_used,
            primary_failure_reason=primary_failure,
            preview_output_path=preview_jpg,
            preview_shape=(preview_height, preview_width),
            preview_resolution=preview_resolution,
            processed_output_crs=grid.crs.to_string(),
            source_resolution=provider.resolution(),
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            source_buffer_m=source_buffer_m,
            source_files=source_files,
            source_manifest=manifest,
            center=center,
            width_m=width_m,
            height_m=height_m,
            rotation_deg=rotation_deg,
            cache_hit=cache_hit,
            cache_key=provider.cache_key,
            cache_path=provider.cache_path,
        )
    finally:
        if cleanup_temp_files:
            _cleanup_temp_files(temp_paths)


def extract_project_dtm(
    *,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None = None,
    rotation_deg: float = 0.0,
    provider_code: str | None = None,
    fallback_provider_code: str | None = None,
    user_settings: DTMProviderSettings | None = None,
    fallback_user_settings: DTMProviderSettings | None = None,
    directory: str = os.path.join(os.getcwd(), "tiles"),
    logger: Any = logging.getLogger(__name__),
    min_valid_coverage: float | None = None,
    source_buffer_m: float | None = None,
    output_basename: str = "dtm",
    max_edge: int = 8192,
    max_pixels: int | None = None,
    target_resolution_m: float | None = None,
    destination_crs: str = "auto-utm",
    resampling: str = "bilinear",
    cleanup_temp_files: bool = False,
) -> DTMAssetResult:
    """Generate full and editor-preview DTM PNG assets for a project."""
    resolved_height = height_m if height_m is not None else width_m
    buffer_m = _resolve_source_buffer(width_m, resolved_height, source_buffer_m)
    source_width_m = int(math.ceil(width_m + buffer_m * 2))
    source_height_m = int(math.ceil(resolved_height + buffer_m * 2))
    provider_class = _select_provider_class(
        DTMProvider,
        center,
        source_width_m,
        source_height_m,
        rotation_deg,
        provider_code,
    )
    provider = _instantiate_provider(
        provider_class,
        center=center,
        source_width_m=source_width_m,
        source_height_m=source_height_m,
        rotation_deg=rotation_deg,
        user_settings=user_settings,
        directory=directory,
        logger=logger,
        min_valid_coverage=min_valid_coverage,
    )
    requested_provider_code = provider.code() or provider_code or "unknown"
    requested_provider_name = provider.name()
    try:
        return _extract_project_dtm_with_provider(
            provider,
            requested_provider_code=requested_provider_code,
            requested_provider_name=requested_provider_name,
            fallback_provider_code=fallback_provider_code,
            fallback_used=False,
            primary_failure=None,
            center=center,
            width_m=width_m,
            height_m=resolved_height,
            rotation_deg=rotation_deg,
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            source_buffer_m=buffer_m,
            directory=directory,
            output_basename=output_basename,
            max_edge=max_edge,
            max_pixels=max_pixels,
            target_resolution_m=target_resolution_m,
            destination_crs=destination_crs,
            resampling=resampling,
            cleanup_temp_files=cleanup_temp_files,
        )
    except DTMProviderError as primary_error:
        if not fallback_provider_code:
            raise
        fallback_class = DTMProvider.get_provider_by_code(fallback_provider_code)
        if fallback_class is None:
            raise ProviderUnavailableError(
                f"Fallback provider '{fallback_provider_code}' is not available.",
                provider_code=fallback_provider_code,
            ) from primary_error
        fallback_provider = _instantiate_provider(
            fallback_class,
            center=center,
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            rotation_deg=rotation_deg,
            user_settings=fallback_user_settings,
            directory=directory,
            logger=logger,
            min_valid_coverage=min_valid_coverage,
        )
        return _extract_project_dtm_with_provider(
            fallback_provider,
            requested_provider_code=requested_provider_code,
            requested_provider_name=requested_provider_name,
            fallback_provider_code=fallback_provider_code,
            fallback_used=True,
            primary_failure=primary_error.to_details(),
            center=center,
            width_m=width_m,
            height_m=resolved_height,
            rotation_deg=rotation_deg,
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            source_buffer_m=buffer_m,
            directory=directory,
            output_basename=output_basename,
            max_edge=max_edge,
            max_pixels=max_pixels,
            target_resolution_m=target_resolution_m,
            destination_crs=destination_crs,
            resampling=resampling,
            cleanup_temp_files=cleanup_temp_files,
        )


def extract_project_imagery(
    *,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None = None,
    rotation_deg: float = 0.0,
    provider_code: str | None = None,
    fallback_provider_code: str | None = None,
    user_settings: DTMProviderSettings | None = None,
    fallback_user_settings: DTMProviderSettings | None = None,
    directory: str = os.path.join(os.getcwd(), "tiles"),
    logger: Any = logging.getLogger(__name__),
    min_valid_coverage: float | None = None,
    max_failed_tiles: int | None = None,
    max_failed_tile_ratio: float = 1.0,
    source_buffer_m: float | None = None,
    output_basename: str = "imagery",
    max_edge: int = 8192,
    max_pixels: int | None = None,
    target_resolution_m: float | None = None,
    destination_crs: str = "auto-utm",
    resampling: str = "bilinear",
    jpeg_quality: int = 90,
    cleanup_temp_files: bool = False,
) -> ImageryAssetResult:
    """Generate an editor-preview imagery JPEG asset for a project."""
    resolved_height = height_m if height_m is not None else width_m
    buffer_m = _resolve_source_buffer(width_m, resolved_height, source_buffer_m)
    source_width_m = int(math.ceil(width_m + buffer_m * 2))
    source_height_m = int(math.ceil(resolved_height + buffer_m * 2))
    provider_class = _select_imagery_provider_class(
        center,
        source_width_m,
        source_height_m,
        rotation_deg,
        provider_code,
    )
    provider = _instantiate_provider(
        provider_class,
        center=center,
        source_width_m=source_width_m,
        source_height_m=source_height_m,
        rotation_deg=rotation_deg,
        user_settings=user_settings,
        directory=directory,
        logger=logger,
        min_valid_coverage=min_valid_coverage,
        max_failed_tiles=max_failed_tiles,
        max_failed_tile_ratio=max_failed_tile_ratio,
    )
    requested_provider_code = provider.code() or provider_code or "unknown"
    requested_provider_name = provider.name()
    try:
        return _extract_project_imagery_with_provider(
            provider,
            requested_provider_code=requested_provider_code,
            requested_provider_name=requested_provider_name,
            fallback_provider_code=fallback_provider_code,
            fallback_used=False,
            primary_failure=None,
            center=center,
            width_m=width_m,
            height_m=resolved_height,
            rotation_deg=rotation_deg,
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            source_buffer_m=buffer_m,
            output_basename=output_basename,
            max_edge=max_edge,
            max_pixels=max_pixels,
            target_resolution_m=target_resolution_m,
            destination_crs=destination_crs,
            resampling=resampling,
            jpeg_quality=jpeg_quality,
            cleanup_temp_files=cleanup_temp_files,
        )
    except DTMProviderError as primary_error:
        if not fallback_provider_code:
            raise
        fallback_class = ImageryProvider.get_provider_by_code(fallback_provider_code)
        if fallback_class is None:
            raise ProviderUnavailableError(
                f"Fallback imagery provider '{fallback_provider_code}' is not available.",
                provider_code=fallback_provider_code,
            ) from primary_error
        fallback_provider = _instantiate_provider(
            cast(Type[DTMProvider], fallback_class),
            center=center,
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            rotation_deg=rotation_deg,
            user_settings=fallback_user_settings,
            directory=directory,
            logger=logger,
            min_valid_coverage=min_valid_coverage,
            max_failed_tiles=max_failed_tiles,
            max_failed_tile_ratio=max_failed_tile_ratio,
        )
        return _extract_project_imagery_with_provider(
            fallback_provider,
            requested_provider_code=requested_provider_code,
            requested_provider_name=requested_provider_name,
            fallback_provider_code=fallback_provider_code,
            fallback_used=True,
            primary_failure=primary_error.to_details(),
            center=center,
            width_m=width_m,
            height_m=resolved_height,
            rotation_deg=rotation_deg,
            source_width_m=source_width_m,
            source_height_m=source_height_m,
            source_buffer_m=buffer_m,
            output_basename=output_basename,
            max_edge=max_edge,
            max_pixels=max_pixels,
            target_resolution_m=target_resolution_m,
            destination_crs=destination_crs,
            resampling=resampling,
            jpeg_quality=jpeg_quality,
            cleanup_temp_files=cleanup_temp_files,
        )


def extract_project_dtm_from_file(
    *,
    image_path: str,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None = None,
    rotation_deg: float = 0.0,
    directory: str = os.path.join(os.getcwd(), "tiles"),
    logger: Any = logging.getLogger(__name__),
    min_valid_coverage: float | None = None,
    source_buffer_m: float | None = None,
    output_basename: str = "dtm",
    max_edge: int = 8192,
    max_pixels: int | None = None,
    target_resolution_m: float | None = None,
    destination_crs: str = "auto-utm",
    resampling: str = "bilinear",
    cleanup_temp_files: bool = False,
) -> DTMAssetResult:
    """Generate DTM PNG assets from a user-provided GeoTIFF source."""
    source_path = os.path.abspath(image_path)
    return _extract_project_dtm_from_class(
        provider_class=LocalDTMProvider,
        source_path=source_path,
        center=center,
        width_m=width_m,
        height_m=height_m,
        rotation_deg=rotation_deg,
        directory=directory,
        logger=logger,
        min_valid_coverage=min_valid_coverage,
        source_buffer_m=source_buffer_m,
        output_basename=output_basename,
        max_edge=max_edge,
        max_pixels=max_pixels,
        target_resolution_m=target_resolution_m,
        destination_crs=destination_crs,
        resampling=resampling,
        cleanup_temp_files=cleanup_temp_files,
    )


def _extract_project_dtm_from_class(
    *,
    provider_class: Type[DTMProvider],
    source_path: str,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None,
    rotation_deg: float,
    directory: str,
    logger: Any,
    min_valid_coverage: float | None,
    source_buffer_m: float | None,
    output_basename: str,
    max_edge: int,
    max_pixels: int | None,
    target_resolution_m: float | None,
    destination_crs: str,
    resampling: str,
    cleanup_temp_files: bool,
) -> DTMAssetResult:
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Raster file not found: {source_path}")
    resolved_height = height_m if height_m is not None else width_m
    buffer_m = _resolve_source_buffer(width_m, resolved_height, source_buffer_m)
    source_width_m = int(math.ceil(width_m + buffer_m * 2))
    source_height_m = int(math.ceil(resolved_height + buffer_m * 2))
    provider = _instantiate_provider(
        provider_class,
        center=center,
        source_width_m=source_width_m,
        source_height_m=source_height_m,
        rotation_deg=rotation_deg,
        user_settings=LocalRasterSettings(source_path=source_path),
        directory=directory,
        logger=logger,
        min_valid_coverage=min_valid_coverage,
    )
    return _extract_project_dtm_with_provider(
        provider,
        requested_provider_code=provider.code() or "unknown",
        requested_provider_name=provider.name(),
        fallback_provider_code=None,
        fallback_used=False,
        primary_failure=None,
        center=center,
        width_m=width_m,
        height_m=resolved_height,
        rotation_deg=rotation_deg,
        source_width_m=source_width_m,
        source_height_m=source_height_m,
        source_buffer_m=buffer_m,
        directory=directory,
        output_basename=output_basename,
        max_edge=max_edge,
        max_pixels=max_pixels,
        target_resolution_m=target_resolution_m,
        destination_crs=destination_crs,
        resampling=resampling,
        cleanup_temp_files=cleanup_temp_files,
    )


def extract_project_imagery_from_file(
    *,
    image_path: str,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None = None,
    rotation_deg: float = 0.0,
    directory: str = os.path.join(os.getcwd(), "tiles"),
    logger: Any = logging.getLogger(__name__),
    source_buffer_m: float | None = None,
    output_basename: str = "imagery",
    max_edge: int = 8192,
    max_pixels: int | None = None,
    target_resolution_m: float | None = None,
    destination_crs: str = "auto-utm",
    resampling: str = "bilinear",
    jpeg_quality: int = 90,
    cleanup_temp_files: bool = False,
) -> ImageryAssetResult:
    """Generate a capped JPEG imagery preview from a user-provided GeoTIFF source."""
    source_path = os.path.abspath(image_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Raster file not found: {source_path}")
    resolved_height = height_m if height_m is not None else width_m
    buffer_m = _resolve_source_buffer(width_m, resolved_height, source_buffer_m)
    source_width_m = int(math.ceil(width_m + buffer_m * 2))
    source_height_m = int(math.ceil(resolved_height + buffer_m * 2))
    provider = _instantiate_provider(
        LocalRasterProvider,
        center=center,
        source_width_m=source_width_m,
        source_height_m=source_height_m,
        rotation_deg=rotation_deg,
        user_settings=LocalRasterSettings(source_path=source_path),
        directory=directory,
        logger=logger,
        min_valid_coverage=None,
    )
    return _extract_project_imagery_with_provider(
        provider,
        requested_provider_code=provider.code() or "unknown",
        requested_provider_name=provider.name(),
        fallback_provider_code=None,
        fallback_used=False,
        primary_failure=None,
        center=center,
        width_m=width_m,
        height_m=resolved_height,
        rotation_deg=rotation_deg,
        source_width_m=source_width_m,
        source_height_m=source_height_m,
        source_buffer_m=buffer_m,
        output_basename=output_basename,
        max_edge=max_edge,
        max_pixels=max_pixels,
        target_resolution_m=target_resolution_m,
        destination_crs=destination_crs,
        resampling=resampling,
        jpeg_quality=jpeg_quality,
        cleanup_temp_files=cleanup_temp_files,
    )
