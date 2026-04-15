"""Imagery provider abstractions parallel to the DTM provider workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Type, cast

import numpy as np
import rasterio
from affine import Affine
from pydantic import BaseModel
from rasterio.enums import Resampling
from rasterio.warp import reproject

from pydtmdl.base.dtm import (
    AuthConfigMissingError,
    CropExtractionError,
    DTMErrorDetails,
    DTMProvider,
    DTMProviderError,
    DTMProviderSettings,
    OutsideCoverageError,
    ProviderUnavailableError,
)


class ImageryProviderSettings(DTMProviderSettings):
    """Base class for imagery provider settings models."""


class ImageryResultMetadata(BaseModel):
    """Structured metadata returned with imagery extraction results."""

    requested_provider: str
    requested_provider_name: str | None = None
    actual_provider: str
    actual_provider_name: str | None = None
    resolution: float | None = None
    output_path: str
    output_crs: str
    shape: tuple[int, int]
    dtype: str
    nodata: float | int | None = None
    cache_hit: bool = False
    cache_key: str
    cache_path: str
    fallback_used: bool = False
    primary_failure_reason: DTMErrorDetails | None = None
    source_files: list[str]
    center: tuple[float, float]
    width_m: int
    height_m: int
    rotation_deg: float = 0.0
    band_count: int = 0
    dataset: str | None = None
    scene_ids: list[str] = []


@dataclass(slots=True)
class ImageryExtractionResult:
    """Return type for structured imagery extraction calls."""

    data: np.ndarray
    metadata: ImageryResultMetadata


class ImageryProvider(DTMProvider):
    """Base class for raster imagery providers."""

    _cache_version: int = 1
    _settings: Type[DTMProviderSettings] | None = ImageryProviderSettings
    _dataset: str | None = None

    @classmethod
    def settings_required(cls) -> bool:
        """Only require imagery settings when the model has mandatory fields."""
        settings_model = cls._settings
        if settings_model is None or settings_model in (
            DTMProviderSettings,
            ImageryProviderSettings,
        ):
            return False
        return any(field.is_required() for field in settings_model.model_fields.values())

    @classmethod
    def dataset(cls) -> str | None:
        """Dataset identifier exposed by the provider."""
        return cls._dataset

    @classmethod
    def _all_provider_classes(cls) -> list[Type[ImageryProvider]]:
        """Collect all imagery provider classes from the imagery inheritance tree."""
        providers: list[Type[ImageryProvider]] = []
        seen: set[type[Any]] = set()
        stack: list[type[Any]] = [ImageryProvider]

        while stack:
            provider_class = stack.pop()
            for child in provider_class.__subclasses__():
                if child in seen:
                    continue
                seen.add(child)
                providers.append(cast(Type[ImageryProvider], child))
                stack.append(child)

        return providers

    @classmethod
    def get_non_base_providers(cls) -> list[Type[ImageryProvider]]:
        """Get all non-base imagery providers."""
        return [
            provider for provider in cls._all_provider_classes() if not provider.__subclasses__()
        ]

    def resolved_settings(self) -> DTMProviderSettings | None:
        """Return the fully resolved settings object used by the provider."""
        settings_model = self.settings()
        if settings_model is None:
            return None
        if self.user_settings is None:
            return settings_model()
        if isinstance(self.user_settings, settings_model):
            return self.user_settings
        return settings_model.model_validate(self.user_settings.model_dump())

    def get_cache_settings_payload(self) -> dict[str, Any]:
        """Return settings that influence the imagery cache key."""
        settings = self.resolved_settings()
        if settings is None:
            return {}
        return settings.model_dump(mode="json")

    def build_cache_key(self) -> str:
        """Build a stable cache key that includes imagery-specific settings."""
        payload = {
            "cache_version": self._cache_version,
            "provider": self.code(),
            "dataset": self.dataset(),
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

    @classmethod
    def extract_area(
        cls,
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
    ) -> ImageryExtractionResult:
        """High-level extraction API for imagery providers."""
        resolved_height = height_m if height_m is not None else width_m

        if cls is ImageryProvider:
            provider_class = (
                cls.get_provider_by_code(provider_code)
                if provider_code
                else cls.get_best(
                    center,
                    width_m=width_m,
                    height_m=resolved_height,
                    rotation_deg=rotation_deg,
                )
            )
        else:
            provider_class = cls

        if provider_class is None:
            if provider_code:
                raise ProviderUnavailableError(
                    f"No imagery provider is available for the requested provider_code: {provider_code!r}.",
                    provider_code=provider_code,
                )
            raise ProviderUnavailableError(
                "No imagery provider is available for the requested geometry."
            )

        provider = provider_class(
            center,
            size=width_m if resolved_height == width_m else max(width_m, resolved_height),
            user_settings=user_settings,
            directory=directory,
            logger=logger,
            width_m=width_m,
            height_m=resolved_height,
            rotation_deg=rotation_deg,
        )
        return cast(
            ImageryExtractionResult,
            provider.get_result(
                fallback_provider_code=fallback_provider_code,
                fallback_user_settings=fallback_user_settings,
            ),
        )

    def _resample_rotated_roi(self, tile_path: str) -> tuple[np.ma.MaskedArray, dict[str, Any]]:
        """Resample a rotated multi-band ROI onto an aligned output grid."""
        local_crs = self._build_local_crs(self.coordinates)
        to_local, _ = self._build_local_transformers(self.coordinates)

        with rasterio.open(tile_path) as src:
            self.logger.debug("Opened tile, shape: %s, dtype: %s.", src.shape, src.dtypes[0])
            nodata = self._get_output_nodata(src)

            center_lat, center_lon = self.coordinates
            center_x, center_y = to_local.transform(center_lon, center_lat)
            east_x, east_y = to_local.transform(center_lon + abs(src.transform.a), center_lat)
            south_x, south_y = to_local.transform(center_lon, center_lat + abs(src.transform.e))
            x_resolution_m = math.hypot(east_x - center_x, east_y - center_y)
            y_resolution_m = math.hypot(south_x - center_x, south_y - center_y)
            if x_resolution_m <= 0 or y_resolution_m <= 0:
                raise CropExtractionError(
                    "Failed to resolve the rotated ROI output resolution.",
                    provider_code=self.code(),
                    provider_name=self.name(),
                )

            width_px = max(1, math.ceil(self.width_m / x_resolution_m))
            height_px = max(1, math.ceil(self.height_m / y_resolution_m))
            pixel_width_m = self.width_m / width_px
            pixel_height_m = self.height_m / height_px

            angle_rad = math.radians(-self.rotation_deg)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            half_width = self.width_m / 2.0
            half_height = self.height_m / 2.0

            top_left_x = (-half_width * cos_angle) - (half_height * sin_angle)
            top_left_y = (-half_width * sin_angle) + (half_height * cos_angle)
            transform = Affine(
                pixel_width_m * cos_angle,
                pixel_height_m * sin_angle,
                top_left_x,
                pixel_width_m * sin_angle,
                -pixel_height_m * cos_angle,
                top_left_y,
            )

            destination = np.full(
                (src.count, height_px, width_px),
                nodata,
                dtype=np.dtype(src.dtypes[0]),
            )
            validity = np.zeros((height_px, width_px), dtype=np.uint8)

            for band_index in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, band_index),
                    destination=destination[band_index - 1],
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=transform,
                    dst_crs=local_crs,
                    dst_nodata=nodata,
                    resampling=Resampling.bilinear,
                )

            reproject(
                source=src.dataset_mask(),
                destination=validity,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=transform,
                dst_crs=local_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

            data = np.ma.array(destination, mask=np.broadcast_to(validity == 0, destination.shape))
            metadata = src.meta.copy()
            metadata.update(
                {
                    "driver": "GTiff",
                    "height": height_px,
                    "width": width_px,
                    "count": src.count,
                    "transform": transform,
                    "crs": local_crs,
                    "nodata": nodata,
                }
            )

        if data.size == 0:
            raise CropExtractionError(
                "The requested geometry does not contain any data.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        if np.ma.isMaskedArray(data) and data.count() == 0:
            raise CropExtractionError(
                "The cropped ROI does not contain any valid data.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        return data, metadata

    def _build_result_from_output(
        self,
        output_path: str,
        requested_provider_code: str,
        requested_provider_name: str | None,
        fallback_used: bool,
        primary_failure: DTMErrorDetails | None,
        cache_hit: bool,
        source_files: list[str] | None,
    ) -> ImageryExtractionResult:
        """Create a structured imagery result from a cached or freshly written raster."""
        cached_metadata = self._load_cached_metadata() if cache_hit else None
        with rasterio.open(output_path) as src:
            data = src.read(masked=True)
            if data.size == 0 or (np.ma.isMaskedArray(data) and data.count() == 0):
                raise CropExtractionError(
                    "The output raster does not contain any valid data.",
                    provider_code=self.code(),
                    provider_name=self.name(),
                )
            metadata = ImageryResultMetadata(
                requested_provider=requested_provider_code,
                requested_provider_name=requested_provider_name,
                actual_provider=self.code() or "unknown",
                actual_provider_name=self.name(),
                resolution=self.resolution(),
                output_path=output_path,
                output_crs=src.crs.to_string() if src.crs else self._output_crs,
                shape=(src.height, src.width),
                dtype=src.dtypes[0],
                nodata=src.nodata,
                cache_hit=cache_hit,
                cache_key=self.cache_key,
                cache_path=self.cache_path,
                fallback_used=fallback_used,
                primary_failure_reason=primary_failure,
                source_files=source_files or [],
                center=self.coordinates,
                width_m=self.width_m,
                height_m=self.height_m,
                rotation_deg=self.rotation_deg,
                band_count=src.count,
                dataset=self.dataset() or (cached_metadata.dataset if cached_metadata else None),
                scene_ids=getattr(self, "_scene_ids", [])
                or (cached_metadata.scene_ids if cached_metadata else []),
            )
        return ImageryExtractionResult(data=data, metadata=metadata)

    def _write_result_metadata(self, metadata: ImageryResultMetadata) -> None:
        """Persist imagery metadata next to the cached raster output."""
        with open(self._metadata_path, "w", encoding="utf-8") as metadata_file:
            metadata_file.write(metadata.model_dump_json(indent=2))

    def _get_cached_source_files(self) -> list[str]:
        """Read imagery source file paths from cached metadata if available."""
        cached_metadata = self._load_cached_metadata()
        return cached_metadata.source_files if cached_metadata else []

    def _load_cached_metadata(self) -> ImageryResultMetadata | None:
        """Load imagery metadata from the cache directory if it exists and is valid."""
        if not os.path.exists(self._metadata_path):
            return None
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as metadata_file:
                return ImageryResultMetadata.model_validate_json(metadata_file.read())
        except (OSError, ValueError) as e:
            self.logger.warning(
                "Failed to read cached metadata from %s: %s", self._metadata_path, e
            )
            return None
