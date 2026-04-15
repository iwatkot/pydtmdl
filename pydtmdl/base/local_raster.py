"""Helpers for extracting ROIs from user-provided georeferenced rasters."""

from __future__ import annotations

import logging
import os
from typing import Any

import rasterio
from pydantic import Field
from pyproj import Transformer

from pydtmdl.base.dtm import ReprojectionFailedError
from pydtmdl.base.imagery import (
    ImageryExtractionResult,
    ImageryProvider,
    ImageryProviderSettings,
)


class LocalRasterSettings(ImageryProviderSettings):
    """Settings for extracting an ROI from a user-provided raster file."""

    source_path: str = Field(min_length=1)


class LocalRasterProvider(ImageryProvider):
    """Provider wrapper for an existing local georeferenced raster."""

    _code = "local_raster"
    _name = "Local Raster"
    _region = "Local"
    _icon = "R"
    _settings = LocalRasterSettings
    _dataset = "local-raster"

    def get_cache_settings_payload(self) -> dict[str, Any]:
        """Include source file identity in the stable cache key."""
        settings = self.resolved_settings()
        if settings is None:
            return {}
        resolved = LocalRasterSettings.model_validate(settings.model_dump())
        source_path = os.path.abspath(resolved.source_path)
        stat_result = os.stat(source_path)
        return {
            "source_path": source_path,
            "source_size": stat_result.st_size,
            "source_mtime_ns": stat_result.st_mtime_ns,
        }

    def download_tiles(self) -> list[str]:
        """Return the existing local raster path after validating its metadata."""
        settings = self.resolved_settings()
        if settings is None:
            raise FileNotFoundError("A source_path is required for local raster extraction.")
        resolved = LocalRasterSettings.model_validate(settings.model_dump())
        source_path = os.path.abspath(resolved.source_path)

        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Raster file not found: {source_path}")

        with rasterio.open(source_path) as dataset:
            if dataset.crs is None:
                raise ReprojectionFailedError(
                    "Source raster does not define a CRS.",
                    provider_code=self.code(),
                    provider_name=self.name(),
                )
            self._resolution = self._estimate_resolution(dataset)

        return [source_path]

    def _estimate_resolution(self, dataset: rasterio.DatasetReader) -> float | None:
        """Estimate raster resolution in meters per pixel around the requested center."""
        transformer = Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True)
        pixel_center_x, pixel_center_y = dataset.xy(dataset.height // 2, dataset.width // 2)
        sample_lon, sample_lat = transformer.transform(pixel_center_x, pixel_center_y)
        local_transformer = Transformer.from_crs(
            dataset.crs, self._build_local_crs((sample_lat, sample_lon)), always_xy=True
        )

        origin_x, origin_y = local_transformer.transform(pixel_center_x, pixel_center_y)
        east_x, east_y = local_transformer.transform(
            pixel_center_x + abs(dataset.transform.a), pixel_center_y
        )
        south_x, south_y = local_transformer.transform(
            pixel_center_x, pixel_center_y + abs(dataset.transform.e)
        )
        x_resolution_m = ((east_x - origin_x) ** 2 + (east_y - origin_y) ** 2) ** 0.5
        y_resolution_m = ((south_x - origin_x) ** 2 + (south_y - origin_y) ** 2) ** 0.5
        estimated = max(x_resolution_m, y_resolution_m)
        return round(estimated, 6) if estimated > 0 else None


def extract_area_from_image(
    image_path: str,
    center: tuple[float, float],
    width_m: int,
    height_m: int | None = None,
    rotation_deg: float = 0.0,
    directory: str = os.path.join(os.getcwd(), "tiles"),
    logger: Any = logging.getLogger(__name__),
) -> ImageryExtractionResult:
    """Extract a rotated ROI from a user-provided georeferenced raster image.

    The input raster must be readable by rasterio and define a CRS. Single-band
    rasters are returned as 2D masked arrays, while multiband rasters preserve
    their band-first shape.
    """
    source_path = os.path.abspath(image_path)
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"Raster file not found: {source_path}")

    result = LocalRasterProvider.extract_area(
        center=center,
        width_m=width_m,
        height_m=height_m,
        rotation_deg=rotation_deg,
        user_settings=LocalRasterSettings(source_path=source_path),
        directory=directory,
        logger=logger,
    )
    if result.metadata.band_count == 1:
        return ImageryExtractionResult(data=result.data[0], metadata=result.metadata)
    return result
