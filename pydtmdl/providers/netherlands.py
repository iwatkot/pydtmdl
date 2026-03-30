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


def _log(logger, level: str, msg: str, *args):
    formatted = msg % args if args else msg
    getattr(logger, level)(formatted)
    print(f"{level.upper()}: {formatted}")


class NetherlandsProvider(DTMProvider):

    _code = "netherlands"
    _name = "Netherlands AHN5"
    _region = "NL"
    _icon = "🇳🇱"
    _resolution = 0.5
    _source_crs = "EPSG:28992"
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
            _log(self.logger, "info", "[index] Cache file does not exist: %s", path)
            return True

        age = datetime.now() - datetime.fromtimestamp(os.path.getmtime(path))
        stale = age > timedelta(days=self._index_max_age_days)

        _log(
            self.logger,
            "debug",
            "[index] Cache age: %.1f days (max %d) — %s",
            age.total_seconds() / 86400,
            self._index_max_age_days,
            "STALE" if stale else "fresh",
        )
        return stale

    def _load_tile_index(self) -> list[dict]:
        _log(self.logger, "debug", "[index] Checking kaartbladindex cache…")

        if self._index_is_stale():
            _log(self.logger, "info", "Downloading kaartbladindex...")
            resp = requests.get(self._index_url, timeout=60)
            resp.raise_for_status()

            os.makedirs(self._tile_directory, exist_ok=True)
            with open(self._index_cache_path, "wb") as fh:
                fh.write(resp.content)

            _log(
                self.logger,
                "debug",
                "[index] Saved kaartbladindex (%d bytes) → %s",
                len(resp.content),
                self._index_cache_path,
            )
        else:
            _log(self.logger, "debug", "[index] Using cached kaartbladindex: %s", self._index_cache_path)

        _log(self.logger, "debug", "[index] Parsing kaartbladindex JSON…")

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

        _log(self.logger, "debug", "[index] Parsed %d tile entries.", len(tiles))
        return tiles

    def _bbox_rd(self) -> tuple[float, float, float, float]:
        north, south, east, west = self.get_bbox()

        _log(
            self.logger,
            "debug",
            "[bbox] WGS84 → N=%.5f S=%.5f E=%.5f W=%.5f",
            north,
            south,
            east,
            west,
        )

        transformer = Transformer.from_crs("EPSG:4326", "EPSG:28992", always_xy=True)
        xs, ys = transformer.transform(
            [west, east, west, east],
            [south, south, north, north],
        )

        bbox = min(xs), min(ys), max(xs), max(ys)

        _log(
            self.logger,
            "debug",
            "[bbox] RD28992 → xmin=%.1f ymin=%.1f xmax=%.1f ymax=%.1f",
            *bbox,
        )
        return bbox

    def _intersecting_urls(self, tile_index: list[dict]) -> list[str]:
        req_xmin, req_ymin, req_xmax, req_ymax = self._bbox_rd()

        _log(self.logger, "debug", "[select] Scanning %d tiles…", len(tile_index))

        urls = []
        for tile in tile_index:
            t_xmin, t_ymin, t_xmax, t_ymax = tile["bbox_rd"]

            if req_xmax < t_xmin or req_xmin > t_xmax:
                continue
            if req_ymax < t_ymin or req_ymin > t_ymax:
                continue

            _log(
                self.logger,
                "debug",
                "[select] ✓ Tile %s intersects",
                tile.get("kaartbladNr", "?"),
            )
            urls.append(tile["url"])

        _log(self.logger, "debug", "[select] Selected %d tiles.", len(urls))
        return urls

    def _download_tiles_with_logging(self, urls: list[str], raw_dir: str) -> list[str]:
        total = len(urls)
        downloaded = []

        for idx, url in enumerate(urls, start=1):
            fname = os.path.basename(url)
            out_path = os.path.join(raw_dir, fname)

            if os.path.exists(out_path):
                _log(self.logger, "debug", "[download] (%d/%d) Cache hit: %s", idx, total, fname)
                downloaded.append(out_path)
                continue

            _log(self.logger, "info", "[download] (%d/%d) Fetching %s", idx, total, url)

            try:
                resp = requests.get(url, timeout=120)
                resp.raise_for_status()

                with open(out_path, "wb") as fh:
                    fh.write(resp.content)

                _log(
                    self.logger,
                    "debug",
                    "[download] (%d/%d) Saved %.1f KB",
                    idx,
                    total,
                    len(resp.content) / 1024,
                )

                downloaded.append(out_path)

            except Exception as exc:
                _log(self.logger, "error", "[download] FAILED %s: %s", url, exc)

        _log(self.logger, "debug", "[download] Done: %d/%d", len(downloaded), total)
        return downloaded

    def _fill_gaps(self, src_path: str, tile_idx: int, tile_total: int) -> str:
        filled_name = os.path.basename(src_path).replace(".tif", "_filled.tif")
        filled_dir = os.path.join(self._tile_directory, "filled")
        os.makedirs(filled_dir, exist_ok=True)
        out_path = os.path.join(filled_dir, filled_name)

        tag = "[fill] (%d/%d) %s" % (tile_idx, tile_total, os.path.basename(src_path))

        if os.path.exists(out_path):
            _log(self.logger, "debug", "%s — cache hit, skipping.", tag)
            return out_path

        _log(self.logger, "debug", "%s — opening raster…", tag)

        with rasterio.open(src_path) as src:
            data = src.read(1).astype(np.float32)
            nodata = src.nodata
            profile = src.profile.copy()

        valid_mask = (data != nodata).astype(np.uint8) if nodata is not None else (~np.isnan(data)).astype(np.uint8)
        data[valid_mask == 0] = 0.0

        filled = fillnodata(
            data,
            mask=valid_mask,
            max_search_distance=self._fill_search_distance,
            smoothing_iterations=self._fill_smoothing_iterations,
        )

        profile.update(dtype=rasterio.float32, nodata=None)

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(filled.astype(np.float32), 1)

        _log(self.logger, "debug", "%s — done.", tag)
        return out_path

    def merge_geotiff(self, input_files: list[str]) -> tuple[str, str]:

        merged_path = os.path.join(self._tile_directory, "merged.tif")
        reprojected_path = os.path.join(self._tile_directory, "reprojected.tif")

        for p in [merged_path, reprojected_path]:
            if os.path.exists(p):
                _log(self.logger, "debug", "Removing old file: %s", p)
                os.remove(p)

        datasets = [rasterio.open(f) for f in input_files]
        mosaic, out_transform = rasterio_merge(datasets, nodata=0)

        out_meta = datasets[0].meta.copy()
        out_meta.update(
            driver="GTiff",
            height=mosaic.shape[1],
            width=mosaic.shape[2],
            transform=out_transform,
            count=mosaic.shape[0],
            dtype=mosaic.dtype,
        )

        with rasterio.open(merged_path, "w", **out_meta) as dst:
            dst.write(mosaic)

        for ds in datasets:
            ds.close()

        dst_crs = "EPSG:4326"

        with rasterio.open(merged_path) as src:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )

            kwargs = src.meta.copy()
            kwargs.update(crs=dst_crs, transform=transform, width=width, height=height)

            with rasterio.open(reprojected_path, "w", **kwargs) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )

        _log(self.logger, "info", "Merge + reprojection complete")

        return reprojected_path, dst_crs

    def download_tiles(self) -> list[str]:
        _log(self.logger, "debug", "[pipeline] START")

        tile_index = self._load_tile_index()
        urls = self._intersecting_urls(tile_index)

        if not urls:
            raise RuntimeError("No AHN5 tiles found.")

        raw_dir = os.path.join(self._tile_directory, "raw")
        os.makedirs(raw_dir, exist_ok=True)

        raw_tiles = self._download_tiles_with_logging(urls, raw_dir)

        filled_tiles = [self._fill_gaps(p, i + 1, len(raw_tiles)) for i, p in enumerate(raw_tiles)]
        self.merge_geotiff(filled_tiles)

        _log(self.logger, "debug", "[pipeline] DONE")

        return filled_tiles
