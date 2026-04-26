"""Sentinel-2 L2A imagery provider using public STAC metadata and COG assets."""

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
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

from pydtmdl.base.dtm import OutsideCoverageError
from pydtmdl.base.imagery import ImageryProvider, ImageryProviderSettings


def _default_date_to() -> str:
    return datetime.now(UTC).date().isoformat()


def _default_date_from() -> str:
    return (datetime.now(UTC).date() - timedelta(days=365)).isoformat()


class Sentinel2L2AImagerySettings(ImageryProviderSettings):
    """Optional tuning knobs for Sentinel-2 imagery search and rendering."""

    collection: str = "sentinel-2-l2a"
    date_from: str = Field(default_factory=_default_date_from)
    date_to: str = Field(default_factory=_default_date_to)
    max_cloud_cover: float = 20.0
    search_limit: int = 24
    max_items: int = 6
    reflectance_min: int = 0
    reflectance_max: int = 4000
    gamma: float = 1.0
    stac_api_url: str = "https://earth-search.aws.element84.com/v1/search"


class Sentinel2L2AImageryProvider(ImageryProvider):
    """Provider of Sentinel-2 L2A RGB imagery rendered from public COG assets."""

    _code = "sentinel2_l2a"
    _name = "Sentinel-2 L2A RGB"
    _region = "Global"
    _icon = "S"
    _resolution = 10.0
    _dataset = "sentinel-2-l2a"
    _settings = Sentinel2L2AImagerySettings

    _bad_scl_classes = {0, 1, 3, 8, 9, 10, 11}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(self._source_tile_directory, "shared")
        os.makedirs(self.shared_tiff_path, exist_ok=True)
        self._scene_ids: list[str] = []

    def download_tiles(self) -> list[str]:
        started_at = time.perf_counter()
        settings = self.resolved_settings()
        if settings is None:
            raise OutsideCoverageError("Sentinel-2 settings could not be resolved.")
        settings = Sentinel2L2AImagerySettings.model_validate(settings.model_dump())

        self.logger.debug(
            "Sentinel-2 download started center=%s size=%smx%sm rotation=%.2f date_range=%s..%s cloud<%s search_limit=%s max_items=%s",
            self.coordinates,
            self.width_m,
            self.height_m,
            self.rotation_deg,
            settings.date_from,
            settings.date_to,
            settings.max_cloud_cover,
            settings.search_limit,
            settings.max_items,
        )

        items = self._search_items(settings)
        if not items:
            raise OutsideCoverageError(
                "No Sentinel-2 scenes matched the requested geometry and filters.",
                provider_code=self.code(),
                provider_name=self.name(),
            )

        rendered_tiles: list[str] = []
        rendered_scene_ids: list[str] = []
        for item in items:
            output_path = os.path.join(self.shared_tiff_path, f"{item['id']}.tif")
            scene_cached = os.path.exists(output_path)
            rendered_path = self._render_item(item, settings, output_path)
            if rendered_path is None:
                continue
            rendered_tiles.append(rendered_path)
            rendered_scene_ids.append(item["id"])
            self.logger.debug(
                "Sentinel-2 scene prepared scene_id=%s cache_hit=%s path=%s",
                item["id"],
                scene_cached,
                rendered_path,
            )

        if not rendered_tiles:
            raise OutsideCoverageError(
                "The selected Sentinel-2 scenes did not produce any valid imagery for the ROI.",
                provider_code=self.code(),
                provider_name=self.name(),
            )

        self._scene_ids = rendered_scene_ids
        self.logger.debug(
            "Sentinel-2 download finished rendered=%s elapsed=%.2fs",
            len(rendered_tiles),
            time.perf_counter() - started_at,
        )
        return rendered_tiles

    def _search_items(self, settings: Sentinel2L2AImagerySettings) -> list[dict[str, Any]]:
        started_at = time.perf_counter()
        north, south, east, west = self.get_bbox()
        payload = {
            "collections": [settings.collection],
            "bbox": [west, south, east, north],
            "datetime": f"{settings.date_from}T00:00:00Z/{settings.date_to}T23:59:59Z",
            "limit": settings.search_limit,
            "query": {
                "eo:cloud_cover": {
                    "lt": settings.max_cloud_cover,
                }
            },
        }
        self.logger.debug(
            "Sentinel-2 STAC search url=%s bbox=[%.6f, %.6f, %.6f, %.6f] datetime=%s cloud<%s limit=%s",
            settings.stac_api_url,
            west,
            south,
            east,
            north,
            payload["datetime"],
            settings.max_cloud_cover,
            settings.search_limit,
        )
        response = requests.post(settings.stac_api_url, json=payload, timeout=60)
        response.raise_for_status()
        features = response.json().get("features", [])
        self.logger.debug(
            "Sentinel-2 STAC search returned=%s elapsed=%.2fs",
            len(features),
            time.perf_counter() - started_at,
        )
        features.sort(
            key=lambda feature: (
                feature.get("properties", {}).get("eo:cloud_cover", float("inf")),
                feature.get("properties", {}).get("datetime", ""),
            )
        )
        selected = features[: settings.max_items]
        self.logger.debug("Sentinel-2 scenes selected=%s", len(selected))
        return selected

    def _render_item(
        self,
        item: dict[str, Any],
        settings: Sentinel2L2AImagerySettings,
        output_path: str,
    ) -> str | None:
        scene_id = item.get("id", "unknown")
        started_at = time.perf_counter()
        if os.path.exists(output_path):
            self.logger.debug("Sentinel-2 scene cache hit scene_id=%s", scene_id)
            return output_path

        assets = item.get("assets", {})
        for asset_name in ("red", "green", "blue", "scl"):
            if asset_name not in assets or "href" not in assets[asset_name]:
                self.logger.debug(
                    "Sentinel-2 scene missing asset scene_id=%s asset=%s",
                    scene_id,
                    asset_name,
                )
                return None

        red_href = assets["red"]["href"]
        green_href = assets["green"]["href"]
        blue_href = assets["blue"]["href"]
        scl_href = assets["scl"]["href"]

        with rasterio.open(red_href) as red_src:
            bounds = self._project_bbox_to_dataset(red_src)
            window = from_bounds(*bounds, red_src.transform)
            window = window.round_offsets().round_lengths()

            self.logger.debug(
                "Sentinel-2 scene read scene_id=%s src_crs=%s window=%s",
                scene_id,
                red_src.crs,
                window,
            )

            red = red_src.read(1, window=window, boundless=True, masked=True)
            transform = red_src.window_transform(window)
            metadata = red_src.meta.copy()

        if red.size == 0:
            self.logger.debug("Sentinel-2 scene produced empty ROI scene_id=%s", scene_id)
            return None

        with rasterio.open(green_href) as green_src:
            green = green_src.read(1, window=window, boundless=True, masked=True)
        with rasterio.open(blue_href) as blue_src:
            blue = blue_src.read(1, window=window, boundless=True, masked=True)
        with rasterio.open(scl_href) as scl_src:
            scl_bounds = self._project_bbox_to_dataset(scl_src)
            scl_window = from_bounds(*scl_bounds, scl_src.transform)
            scl_window = scl_window.round_offsets().round_lengths()
            scl = scl_src.read(
                1,
                window=scl_window,
                boundless=True,
                masked=True,
                out_shape=red.shape,
                resampling=Resampling.nearest,
            )

        rgb = np.stack([red, green, blue]).astype(np.float32)
        invalid_mask = (
            np.ma.getmaskarray(red)
            | np.ma.getmaskarray(green)
            | np.ma.getmaskarray(blue)
            | np.ma.getmaskarray(scl)
            | np.isin(np.asarray(scl.filled(0), dtype=np.uint8), tuple(self._bad_scl_classes))
        )

        stretch = max(1, settings.reflectance_max - settings.reflectance_min)
        rgb = ((rgb - settings.reflectance_min) / stretch).astype(np.float32)
        rgb = np.clip(rgb, 0.0, 1.0).astype(np.float32)
        if settings.gamma > 0 and not np.isclose(settings.gamma, 1.0):
            rgb = np.power(rgb, 1.0 / settings.gamma).astype(np.float32)
        rgb = np.round(rgb * 255.0).astype(np.uint8)
        rgb = np.ma.array(rgb, mask=np.broadcast_to(invalid_mask, rgb.shape))

        if rgb.count() == 0:
            self.logger.debug("Sentinel-2 scene masked out scene_id=%s", scene_id)
            return None

        metadata.update(
            {
                "driver": "GTiff",
                "height": rgb.shape[1],
                "width": rgb.shape[2],
                "count": 3,
                "dtype": "uint8",
                "transform": transform,
                "nodata": 0,
                "compress": "deflate",
            }
        )
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(rgb.filled(0))

        self.logger.debug(
            "Sentinel-2 scene rendered scene_id=%s shape=%sx%s elapsed=%.2fs",
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
