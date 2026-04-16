"""NAIP imagery provider using public STAC metadata and public COG assets."""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import rasterio
import requests
from pydantic import Field
from pyproj import Transformer
from rasterio.windows import from_bounds

from pydtmdl.base.dtm import OutsideCoverageError
from pydtmdl.base.imagery import ImageryProvider, ImageryProviderSettings


def _default_naip_date_to() -> str:
    return datetime.now(UTC).date().isoformat()


def _default_naip_date_from() -> str:
    return (datetime.now(UTC).date() - timedelta(days=365 * 5)).isoformat()


class NAIPImagerySettings(ImageryProviderSettings):
    """Optional tuning knobs for NAIP imagery search."""

    collection: str = "naip"
    date_from: str = Field(default_factory=_default_naip_date_from)
    date_to: str = Field(default_factory=_default_naip_date_to)
    search_limit: int = 24
    max_items: int = 12
    stac_api_url: str = "https://planetarycomputer.microsoft.com/api/stac/v1/search"


class NAIPImageryProvider(ImageryProvider):
    """Provider of NAIP orthophoto RGB imagery for the contiguous United States."""

    _code = "naip"
    _name = "NAIP RGB"
    _region = "USA"
    _icon = "US"
    _resolution = 0.6
    _dataset = "naip"
    _settings = NAIPImagerySettings
    _extents = [(49.5, 24.0, -66.0, -125.0)]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(self._tile_directory, "shared")
        os.makedirs(self.shared_tiff_path, exist_ok=True)
        self._scene_ids: list[str] = []

    def download_tiles(self) -> list[str]:
        started_at = time.perf_counter()
        settings = self.resolved_settings()
        if settings is None:
            raise OutsideCoverageError("NAIP settings could not be resolved.")
        settings = NAIPImagerySettings.model_validate(settings.model_dump())

        self.logger.debug(
            "NAIP download started center=%s size=%smx%sm rotation=%.2f date_range=%s..%s search_limit=%s max_items=%s",
            self.coordinates,
            self.width_m,
            self.height_m,
            self.rotation_deg,
            settings.date_from,
            settings.date_to,
            settings.search_limit,
            settings.max_items,
        )

        items = self._search_items(settings)
        if not items:
            raise OutsideCoverageError(
                "No NAIP scenes matched the requested geometry and filters.",
                provider_code=self.code(),
                provider_name=self.name(),
            )

        rendered_tiles: list[str] = []
        rendered_scene_ids: list[str] = []
        for item in items:
            output_path = os.path.join(self.shared_tiff_path, f"{item['id']}.tif")
            scene_cached = os.path.exists(output_path)
            rendered_path = self._render_item(item, output_path)
            if rendered_path is None:
                continue
            rendered_tiles.append(rendered_path)
            rendered_scene_ids.append(item["id"])
            self.logger.debug(
                "NAIP scene prepared scene_id=%s cache_hit=%s path=%s",
                item["id"],
                scene_cached,
                rendered_path,
            )

        if not rendered_tiles:
            raise OutsideCoverageError(
                "The selected NAIP scenes did not produce any valid imagery for the ROI.",
                provider_code=self.code(),
                provider_name=self.name(),
            )

        self._scene_ids = rendered_scene_ids
        self.logger.debug(
            "NAIP download finished rendered=%s elapsed=%.2fs",
            len(rendered_tiles),
            time.perf_counter() - started_at,
        )
        return rendered_tiles

    def _search_items(self, settings: NAIPImagerySettings) -> list[dict[str, Any]]:
        started_at = time.perf_counter()
        north, south, east, west = self.get_bbox()
        payload = {
            "collections": [settings.collection],
            "bbox": [west, south, east, north],
            "datetime": f"{settings.date_from}T00:00:00Z/{settings.date_to}T23:59:59Z",
            "limit": settings.search_limit,
        }
        self.logger.debug(
            "NAIP STAC search url=%s bbox=[%.6f, %.6f, %.6f, %.6f] datetime=%s limit=%s",
            settings.stac_api_url,
            west,
            south,
            east,
            north,
            payload["datetime"],
            settings.search_limit,
        )
        response = requests.post(settings.stac_api_url, json=payload, timeout=60)
        response.raise_for_status()
        features = response.json().get("features", [])
        self.logger.debug(
            "NAIP STAC search returned=%s elapsed=%.2fs",
            len(features),
            time.perf_counter() - started_at,
        )
        features.sort(
            key=lambda feature: (
                feature.get("properties", {}).get("datetime", ""),
                -(feature.get("properties", {}).get("gsd", float("inf")) or float("inf")),
            ),
            reverse=True,
        )
        selected = features[: settings.max_items]
        self.logger.debug("NAIP scenes selected=%s", len(selected))
        return selected

    def _render_item(self, item: dict[str, Any], output_path: str) -> str | None:
        scene_id = item.get("id", "unknown")
        started_at = time.perf_counter()
        if os.path.exists(output_path):
            self.logger.debug("NAIP scene cache hit scene_id=%s", scene_id)
            return output_path

        image_asset = item.get("assets", {}).get("image")
        if not image_asset or "href" not in image_asset:
            self.logger.debug("NAIP scene has no image asset scene_id=%s", scene_id)
            return None

        with rasterio.open(image_asset["href"]) as src:
            if src.count < 3:
                self.logger.debug(
                    "NAIP scene has insufficient bands scene_id=%s count=%s",
                    scene_id,
                    src.count,
                )
                return None

            bounds = self._project_bbox_to_dataset(src)
            window = from_bounds(*bounds, src.transform)
            window = window.round_offsets().round_lengths()

            self.logger.debug(
                "NAIP scene read scene_id=%s src_crs=%s window=%s",
                scene_id,
                src.crs,
                window,
            )

            rgb = src.read((1, 2, 3), window=window, boundless=True)
            validity = src.read_masks(1, window=window, boundless=True)
            transform = src.window_transform(window)
            metadata = src.meta.copy()

        rgb = np.ma.array(rgb, mask=np.broadcast_to(validity == 0, rgb.shape))

        if rgb.size == 0 or (np.ma.isMaskedArray(rgb) and rgb.count() == 0):
            self.logger.debug("NAIP scene produced empty ROI scene_id=%s", scene_id)
            return None

        metadata.update(
            {
                "driver": "GTiff",
                "height": rgb.shape[1],
                "width": rgb.shape[2],
                "count": 3,
                "dtype": str(rgb.dtype),
                "transform": transform,
                "nodata": 0,
                "compress": "deflate",
            }
        )
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(rgb.filled(0))

        self.logger.debug(
            "NAIP scene rendered scene_id=%s shape=%sx%s elapsed=%.2fs",
            scene_id,
            rgb.shape[1],
            rgb.shape[2],
            time.perf_counter() - started_at,
        )

        return output_path

    def _project_bbox_to_dataset(
        self, dataset: rasterio.DatasetReader
    ) -> tuple[float, float, float, float]:
        north, south, east, west = self.get_bbox()
        transformer = Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
        projected_points = [
            transformer.transform(west, south),
            transformer.transform(west, north),
            transformer.transform(east, south),
            transformer.transform(east, north),
        ]
        xs = [point[0] for point in projected_points]
        ys = [point[1] for point in projected_points]
        return min(xs), min(ys), max(xs), max(ys)
