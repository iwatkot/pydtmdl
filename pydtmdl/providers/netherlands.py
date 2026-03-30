import os
import json
from datetime import datetime, timedelta

import numpy as np
import requests
import rasterio
from pyproj import Transformer
from rasterio.fill import fillnodata
from rasterio.merge import merge as rasterio_merge
from rasterio.warp import calculate_default_transform, reproject, Resampling

from pydtmdl.base.dtm import DTMProvider


class NetherlandsProvider(DTMProvider):
    """Provider for AHN5 DTM 0.5 m (PDOK kaartbladindex).

    Source: https://www.pdok.nl/introductie/-/article/actueel-hoogtebestand-nederland-ahn
    Free and openly accessible — no authentication required.

    Gaps (buildings, bridges, etc.) are filled using GDAL's fillnodata() algorithm
    """

    _code = "netherlands"
    _name = "Netherlands AHN5"
    _region = "NL"
    _icon = "🇳🇱"
    _resolution = 0.5
    _source_crs = "EPSG:28992"  # Rijksdriehoekstelsel — native CRS of AHN5
    _extents = [(53.6, 50.75, 7.2, 3.2)]

    _index_url = (
        "https://service.pdok.nl/rws/ahn/atom/downloads/dtm_05m/kaartbladindex.json"
    )
    _index_max_age_days = 7
    _fill_search_distance = 200
    _fill_smoothing_iterations = 5

    @property
    def _index_cache_path(self) -> str:
        return os.path.join(self._tile_directory, "ahn5_kaartbladindex.json")

    def _index_is_stale(self) -> bool:
        path = self._index_cache_path
        if not os.path.exists(path):
            self.logger.debug("[index] Cache file does not exist: %s", path)
            return True
        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        stale = age > timedelta(days=self._index_max_age_days)
        self.logger.debug(
            "[index] Cache age: %.1f days (max %d) — %s",
            age.total_seconds() / 86400,
            self._index_max_age_days,
            "STALE" if stale else "fresh",
        )
        return stale

    def _load_tile_index(self) -> list[dict]:
        self.logger.debug("[index] Checking kaartbladindex cache…")

        if self._index_is_stale():
            self.logger.debug("[index] Downloading kaartbladindex from %s", self._index_url)
            resp = requests.get(self._index_url, timeout=60)
            resp.raise_for_status()
            os.makedirs(self._tile_directory, exist_ok=True)
            with open(self._index_cache_path, "wb") as fh:
                fh.write(resp.content)
            self.logger.debug(
                "[index] Saved kaartbladindex (%d bytes) → %s",
                len(resp.content),
                self._index_cache_path,
            )
        else:
            self.logger.debug("[index] Using cached kaartbladindex: %s", self._index_cache_path)

        self.logger.debug("[index] Parsing kaartbladindex JSON…")
        with open(self._index_cache_path, encoding="utf-8") as fh:
            data = json.load(fh)

        tiles = []
        for feature in data.get("features", []):
            props = feature["properties"]
            coords = feature["geometry"]["coordinates"][0]
            x_min, y_min = coords[0]
            x_max, y_max = coords[2]
            tiles.append(
                {
                    "kaartbladNr": props.get("kaartbladNr"),
                    "url": props.get("url"),
                    "bbox_rd": (x_min, y_min, x_max, y_max),
                }
            )

        self.logger.debug("[index] Parsed %d tile entries from kaartbladindex.", len(tiles))
        return tiles

    def _bbox_rd(self) -> tuple[float, float, float, float]:
        north, south, east, west = self.get_bbox()
        self.logger.debug(
            "[bbox] Request bbox WGS84 → N=%.5f S=%.5f E=%.5f W=%.5f",
            north, south, east, west,
        )
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
        xs, ys = transformer.transform(
            [west, east, west, east],
            [south, south, north, north],
        )
        bbox = min(xs), min(ys), max(xs), max(ys)
        self.logger.debug(
            "[bbox] Request bbox RD28992 → xmin=%.1f ymin=%.1f xmax=%.1f ymax=%.1f",
            *bbox,
        )
        return bbox

    def _intersecting_urls(self, tile_index: list[dict]) -> list[str]:
        req_xmin, req_ymin, req_xmax, req_ymax = self._bbox_rd()
        self.logger.debug("[select] Scanning %d index tiles for intersection…", len(tile_index))

        urls = []
        for tile in tile_index:
            t_xmin, t_ymin, t_xmax, t_ymax = tile["bbox_rd"]
            if req_xmax < t_xmin or req_xmin > t_xmax:
                continue
            if req_ymax < t_ymin or req_ymin > t_ymax:
                continue
            self.logger.debug(
                "[select] ✓ Tile %s intersects request bbox",
                tile.get("kaartbladNr", "?"),
            )
            urls.append(tile["url"])

        self.logger.debug(
            "[select] Selected %d / %d tiles.", len(urls), len(tile_index)
        )
        return urls

    def _download_tiles_with_logging(self, urls: list[str], raw_dir: str) -> list[str]:
        total = len(urls)
        downloaded = []

        for idx, url in enumerate(urls, start=1):
            fname = os.path.basename(url)
            out_path = os.path.join(raw_dir, fname)

            if os.path.exists(out_path):
                self.logger.debug(
                    "[download] (%d/%d) Cache hit — skipping: %s",
                    idx, total, fname,
                )
                downloaded.append(out_path)
                continue

            self.logger.debug(
                "[download] (%d/%d) Fetching: %s", idx, total, url
            )
            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()
                with open(out_path, "wb") as fh:
                    fh.write(resp.content)
                self.logger.debug(
                    "[download] (%d/%d) Saved %.1f KB → %s",
                    idx, total, len(resp.content) / 1024, out_path,
                )
                downloaded.append(out_path)
            except Exception as exc:
                self.logger.error(
                    "[download] (%d/%d) FAILED for %s: %s", idx, total, url, exc
                )

        self.logger.debug(
            "[download] Done. %d/%d tiles available.", len(downloaded), total
        )
        return downloaded

    def _fill_gaps(self, src_path: str, tile_idx: int, tile_total: int) -> str:
        filled_name = os.path.basename(src_path).replace(".tif", "_filled.tif")
        filled_dir = os.path.join(self._tile_directory, "filled")
        os.makedirs(filled_dir, exist_ok=True)
        out_path = os.path.join(filled_dir, filled_name)

        tag = "[fill] (%d/%d) %s" % (tile_idx, tile_total, os.path.basename(src_path))

        if os.path.exists(out_path):
            self.logger.debug("%s — cache hit, skipping.", tag)
            return out_path

        self.logger.debug("%s — opening raster…", tag)
        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            profile = src.profile.copy()
            shape = data.shape

        self.logger.debug(
            "%s — size: %d×%d px | nodata value: %s",
            tag, shape[0], shape[1], nodata,
        )

        self.logger.debug("%s — building validity mask…", tag)
        if nodata is not None:
            valid_mask = (data != nodata).astype(np.uint8)
        else:
            valid_mask = (~np.isnan(data)).astype(np.uint8)

        total_px = data.size
        hole_px = int((valid_mask == 0).sum())
        hole_pct = 100.0 * hole_px / total_px if total_px else 0.0
        self.logger.debug(
            "%s — holes: %d px out of %d total (%.1f%%)",
            tag, hole_px, total_px, hole_pct,
        )

        if hole_px == 0:
            self.logger.debug("%s — no holes found, copying as-is.", tag)
            profile.update(dtype=rasterio.float32, nodata=None)
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(data, 1)
            self.logger.debug("%s — written (no-op fill) → %s", tag, out_path)
            return out_path

        data[valid_mask == 0] = 0.0

        self.logger.debug(
            "%s — running fillnodata (search_distance=%d px, smoothing_iterations=%d)…",
            tag, self._fill_search_distance, self._fill_smoothing_iterations,
        )
        filled = fillnodata(
            data,
            mask=valid_mask,
            max_search_distance=self._fill_search_distance,
            smoothing_iterations=self._fill_smoothing_iterations,
        )

        remaining_nan = int(np.isnan(filled).sum())
        self.logger.debug(
            "%s — fillnodata complete | remaining NaN pixels: %d",
            tag, remaining_nan,
        )

        self.logger.debug("%s — writing filled tile → %s", tag, out_path)
        profile.update(dtype=rasterio.float32, nodata=None)
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filled.astype(np.float32), 1)

        self.logger.debug("%s — done.", tag)
        return out_path

    def merge_geotiff(self, input_files: list[str]) -> tuple[str, str]:

        merged_path = os.path.join(self._tile_directory, "merged.tif")
        reprojected_path = os.path.join(self._tile_directory, "reprojected.tif")

        self.logger.debug("[merge] Opening %d filled tile(s) for merge…", len(input_files))
        for i, f in enumerate(input_files, start=1):
            self.logger.debug("[merge]   %d. %s", i, os.path.basename(f))

        datasets = [rasterio.open(f) for f in input_files]
        src_crs = datasets[0].crs or self._source_crs

        self.logger.debug("[merge] Running rasterio.merge…")
        mosaic, out_transform = rasterio_merge(datasets, nodata=0)

        self.logger.debug(
            "[merge] Mosaic: %d band(s) × %d rows × %d cols | dtype=%s",
            mosaic.shape[0], mosaic.shape[1], mosaic.shape[2], mosaic.dtype,
        )

        out_meta = datasets[0].meta.copy()
        out_meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            count=mosaic.shape[0],
            dtype=mosaic.dtype,
        )

        self.logger.debug("[merge] Writing merged.tif → %s", merged_path)
        with rasterio.open(merged_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        for ds in datasets:
            ds.close()

        self.logger.debug("[merge] merged.tif written successfully.")
        dst_crs = "EPSG:4326"
        self.logger.debug(
            "[reproject] Reprojecting %s → %s…", self._source_crs, dst_crs
        )

        with rasterio.open(merged_path) as src:
            self.logger.debug(
                "[reproject] Source: %d×%d px | CRS: %s",
                src.width, src.height, src.crs,
            )
            transform, width, height = calculate_default_transform(
                src.crs or src_crs, dst_crs, src.width, src.height, *src.bounds
            )
            self.logger.debug(
                "[reproject] Target: %d×%d px | transform: %s", width, height, transform
            )

            kwargs = src.meta.copy()
            kwargs.update(crs=dst_crs, transform=transform, width=width, height=height)

            self.logger.debug("[reproject] Writing reprojected.tif → %s", reprojected_path)
            with rasterio.open(reprojected_path, "w", **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs or src_crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

        self.logger.debug("[reproject] reprojected.tif written successfully.")
        self.logger.debug(
            "[merge_geotiff] Complete. merged=%s | reprojected=%s",
            merged_path, reprojected_path,
        )

        return reprojected_path, dst_crs

    def download_tiles(self) -> list[str]:
        self.logger.debug("[pipeline] === NetherlandsProvider.download_tiles() START ===")

        self.logger.debug("[pipeline] Step 1/4 — Loading kaartbladindex…")
        tile_index = self._load_tile_index()
        self.logger.debug(
            "[pipeline] Step 1/4 — Index loaded: %d entries.", len(tile_index)
        )

        self.logger.debug("[pipeline] Step 2/4 — Selecting intersecting tiles…")
        urls = self._intersecting_urls(tile_index)
        if not urls:
            raise RuntimeError(
                "No AHN5 tiles found for the requested area. "
                "Verify that your coordinates lie within the Netherlands "
                "(bbox: N=53.6, S=50.75, E=7.2, W=3.2)."
            )
        self.logger.debug(
            "[pipeline] Step 2/4 — %d tile(s) selected.", len(urls)
        )

        self.logger.debug("[pipeline] Step 3/4 — Downloading raw tiles…")
        raw_dir = os.path.join(self._tile_directory, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        raw_tiles = self._download_tiles_with_logging(urls, raw_dir)
        self.logger.debug(
            "[pipeline] Step 3/4 — %d/%d tile(s) ready after download.",
            len(raw_tiles), len(urls),
        )

        total = len(raw_tiles)
        self.logger.debug(
            "[pipeline] Step 4/4 — Gap filling %d tile(s)…", total
        )
        filled_tiles = []
        for idx, tile_path in enumerate(raw_tiles, start=1):
            self.logger.debug(
                "[pipeline] Step 4/4 — Processing tile %d/%d: %s",
                idx, total, os.path.basename(tile_path),
            )
            try:
                filled = self._fill_gaps(tile_path, idx, total)
                filled_tiles.append(filled)
                self.logger.debug(
                    "[pipeline] Step 4/4 — Tile %d/%d complete → %s",
                    idx, total, os.path.basename(filled),
                )
            except Exception as exc:
                self.logger.error(
                    "[pipeline] Step 4/4 — Gap fill FAILED for tile %d/%d (%s): %s "
                    "— falling back to raw tile.",
                    idx, total, os.path.basename(tile_path), exc,
                )
                filled_tiles.append(tile_path)

        self.logger.debug(
            "[pipeline] Step 4/4 — Filling complete: %d/%d tile(s) filled.",
            len(filled_tiles), total,
        )
        self.logger.debug(
            "[pipeline] === download_tiles() END — returning %d tile path(s) ===",
            len(filled_tiles),
        )
        return filled_tiles
