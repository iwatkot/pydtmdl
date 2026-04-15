"""Helpers for extracting ROIs from user-provided georeferenced rasters."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import rasterio
from pydantic import Field
from pyproj import Transformer

from pydtmdl.base.dtm import (
    CropExtractionError,
    DTMExtractionResult,
    DTMProvider,
    DTMProviderSettings,
    ReprojectionFailedError,
)
from pydtmdl.base.imagery import (
    ImageryExtractionResult,
    ImageryProvider,
    ImageryProviderSettings,
)


class LocalRasterSettings(ImageryProviderSettings):
    """Settings for extracting an ROI from a user-provided raster file."""

    source_path: str = Field(min_length=1)


class _BaseLocalRasterMixin:
    """Shared validation and cache helpers for local raster wrappers."""

    def resolved_settings(self) -> DTMProviderSettings | None:
        """Return the fully resolved settings object used by the provider."""
        settings_model = self.settings()  # type: ignore[attr-defined]
        if settings_model is None:
            return None
        user_settings = self.user_settings  # type: ignore[attr-defined]
        if user_settings is None:
            return settings_model()
        if isinstance(user_settings, settings_model):
            return user_settings
        return settings_model.model_validate(user_settings.model_dump())

    def _resolve_local_settings(self) -> LocalRasterSettings:
        """Resolve and validate the local raster settings."""
        settings = self.resolved_settings()
        if settings is None:
            raise FileNotFoundError("A source_path is required for local raster extraction.")
        resolved = LocalRasterSettings.model_validate(settings.model_dump())
        source_path = os.path.abspath(resolved.source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Raster file not found: {source_path}")
        return LocalRasterSettings(source_path=source_path)

    def get_cache_settings_payload(self) -> dict[str, Any]:
        """Include source file identity in the stable cache key."""
        resolved = self._resolve_local_settings()
        stat_result = os.stat(resolved.source_path)
        return {
            "source_path": resolved.source_path,
            "source_size": stat_result.st_size,
            "source_mtime_ns": stat_result.st_mtime_ns,
        }

    def _estimate_resolution(self, dataset: rasterio.DatasetReader) -> float | None:
        """Estimate raster resolution in meters per pixel around the raster center."""
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
        pixel_center_x, pixel_center_y = dataset.xy(dataset.height // 2, dataset.width // 2)
        sample_lon, sample_lat = transformer.transform(pixel_center_x, pixel_center_y)
        local_transformer = Transformer.from_crs(
            dataset.crs,
            self._build_local_crs((sample_lat, sample_lon)),  # type: ignore[attr-defined]
            always_xy=True,
        )

        origin_x, origin_y = local_transformer.transform(pixel_center_x, pixel_center_y)
        east_x, east_y = local_transformer.transform(
            pixel_center_x + abs(dataset.transform.a),
            pixel_center_y,
        )
        south_x, south_y = local_transformer.transform(
            pixel_center_x,
            pixel_center_y + abs(dataset.transform.e),
        )
        x_resolution_m = ((east_x - origin_x) ** 2 + (east_y - origin_y) ** 2) ** 0.5
        y_resolution_m = ((south_x - origin_x) ** 2 + (south_y - origin_y) ** 2) ** 0.5
        estimated = max(x_resolution_m, y_resolution_m)
        return round(estimated, 6) if estimated > 0 else None

    def _validate_source_raster(self) -> tuple[str, rasterio.DatasetReader]:
        """Open and validate the source raster, returning its path and dataset."""
        resolved = self._resolve_local_settings()
        dataset = rasterio.open(resolved.source_path)
        if dataset.crs is None:
            dataset.close()
            raise ReprojectionFailedError(
                "Source raster does not define a CRS.",
                provider_code=self.code(),  # type: ignore[attr-defined]
                provider_name=self.name(),  # type: ignore[attr-defined]
            )
        return resolved.source_path, dataset


class LocalRasterProvider(_BaseLocalRasterMixin, ImageryProvider):
    """Imagery wrapper for an existing local georeferenced raster."""

    _code = "local_raster"
    _name = "Local Raster"
    _region = "Local"
    _icon = "R"
    _dataset = "local-raster"

    def download_tiles(self) -> list[str]:
        """Return the existing local raster path after validating its metadata."""
        source_path, dataset = self._validate_source_raster()
        with dataset:
            self._resolution = self._estimate_resolution(dataset)
        return [source_path]


class LocalDTMProvider(_BaseLocalRasterMixin, DTMProvider):
    """DTM wrapper for an existing single-band georeferenced raster."""

    _code = "local_dtm"
    _name = "Local DTM Raster"
    _region = "Local"
    _icon = "R"

    def build_cache_key(self) -> str:
        """Build a stable cache key that includes the source file identity."""
        payload = {
            "cache_version": self._cache_version,
            "provider": self.code(),
            "center": [round(self.coordinates[0], 8), round(self.coordinates[1], 8)],
            "width_m": self.width_m,
            "height_m": self.height_m,
            "rotation_deg": round(self.rotation_deg, 6),
            "output_crs": self._output_crs,
            "resolution": self.resolution(),
            "settings": self.get_cache_settings_payload(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    def download_tiles(self) -> list[str]:
        """Return the existing local raster path after validating it as single-band input."""
        source_path, dataset = self._validate_source_raster()
        with dataset:
            if dataset.count != 1:
                raise CropExtractionError(
                    "Local DTM extraction requires a single-band raster.",
                    provider_code=self.code(),
                    provider_name=self.name(),
                )
            self._resolution = self._estimate_resolution(dataset)
        return [source_path]


def extract_area_from_imagery(
    image_path: str,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None = None,
    rotation_deg: float = 0.0,
    directory: str = os.path.join(os.getcwd(), "tiles"),
    logger: Any = logging.getLogger(__name__),
) -> ImageryExtractionResult:
    """Extract a rotated ROI from a user-provided georeferenced imagery raster."""
    source_path = os.path.abspath(image_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Raster file not found: {source_path}")

    return LocalRasterProvider.extract_area(
        center=center,
        width_m=width_m,
        height_m=height_m,
        rotation_deg=rotation_deg,
        user_settings=LocalRasterSettings(source_path=source_path),
        directory=directory,
        logger=logger,
    )


def extract_area_from_dtm(
    image_path: str,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None = None,
    rotation_deg: float = 0.0,
    directory: str = os.path.join(os.getcwd(), "tiles"),
    logger: Any = logging.getLogger(__name__),
) -> DTMExtractionResult:
    """Extract a rotated ROI from a user-provided single-band DTM raster."""
    source_path = os.path.abspath(image_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Raster file not found: {source_path}")

    return LocalDTMProvider.extract_area(
        center=center,
        width_m=width_m,
        height_m=height_m,
        rotation_deg=rotation_deg,
        user_settings=LocalRasterSettings(source_path=source_path),
        directory=directory,
        logger=logger,
    )


def extract_area_from_image(
    image_path: str,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None = None,
    rotation_deg: float = 0.0,
    directory: str = os.path.join(os.getcwd(), "tiles"),
    logger: Any = logging.getLogger(__name__),
) -> ImageryExtractionResult:
    """Backward-compatible alias for extract_area_from_imagery()."""
    return extract_area_from_imagery(
        image_path=image_path,
        center=center,
        width_m=width_m,
        height_m=height_m,
        rotation_deg=rotation_deg,
        directory=directory,
        logger=logger,
    )
