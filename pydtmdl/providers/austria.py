"""This module contains provider of Austria data."""

from __future__ import annotations

import math
import os
from typing import Iterator

import rasterio
from pyproj import Transformer
from rasterio.errors import RasterioIOError
from rasterio.merge import merge
from rasterio.windows import bounds as window_bounds

from pydtmdl.base.dtm import DTMProvider


class AustriaProvider(DTMProvider):
    """Provider of Austria BEV ALS DTM 1 m data (Stichtag 15.09.2024)."""

    _code = "austria"
    _name = "Austria BEV ALS DTM 1 m (2024)"
    _region = "AT"
    _icon = "🇦🇹"
    _resolution = 1.0

    _url = (
        "https://data.bev.gv.at/download/ALS/DTM/20240915/"
        "ALS_DTM_CRS3035RES50000mN{northing}E{easting}.tif"
    )

    _tile_size = 50000

    # Exact 50 km row envelopes derived from the official BEV ALS tile overview.
    _extents = [
        (46.427973, 45.95086, 14.927047, 14.242775),
        (46.963155, 46.332589, 16.275078, 9.724317),
        (47.40955, 46.741019, 16.986371, 9.059909),
        (47.859505, 47.188906, 17.047139, 9.051669),
        (48.287201, 47.591824, 17.777012, 12.390241),
        (48.736741, 48.084226, 17.172811, 12.411463),
        (49.147828, 48.531669, 17.237805, 13.790724),
    ]

    # Official 2024 BEV tile coverage: 55 tiles.
    _available_tiles = {
        2550000: (4650000,),
        2600000: (
            4300000,
            4350000,
            4400000,
            4450000,
            4500000,
            4550000,
            4600000,
            4650000,
            4700000,
            4750000,
        ),
        2650000: (
            4250000,
            4300000,
            4350000,
            4400000,
            4450000,
            4500000,
            4550000,
            4600000,
            4650000,
            4700000,
            4750000,
            4800000,
        ),
        2700000: (
            4250000,
            4300000,
            4350000,
            4400000,
            4450000,
            4500000,
            4550000,
            4600000,
            4650000,
            4700000,
            4750000,
            4800000,
        ),
        2750000: (
            4500000,
            4550000,
            4600000,
            4650000,
            4700000,
            4750000,
            4800000,
            4850000,
        ),
        2800000: (
            4500000,
            4550000,
            4600000,
            4650000,
            4700000,
            4750000,
            4800000,
        ),
        2850000: (
            4600000,
            4650000,
            4700000,
            4750000,
            4800000,
        ),
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.shared_tiff_path = os.path.join(self._tile_directory, "shared")
        os.makedirs(self.shared_tiff_path, exist_ok=True)

        self._transformer = Transformer.from_crs(
            "EPSG:4326", "EPSG:3035", always_xy=True
        )

    def _get_projected_bbox(self) -> tuple[float, float, float, float]:
        """Return the requested bbox as left, bottom, right, top in EPSG:3035."""
        north, south, east, west = self.get_bbox()

        corners = [
            self._transformer.transform(west, south),
            self._transformer.transform(west, north),
            self._transformer.transform(east, north),
            self._transformer.transform(east, south),
        ]

        xs = [x for x, _ in corners]
        ys = [y for _, y in corners]

        return min(xs), min(ys), max(xs), max(ys)

    def _iter_required_tiles(
        self, left: float, bottom: float, right: float, top: float
    ) -> Iterator[tuple[int, int]]:
        """Yield all official BEV tiles intersecting the requested projected bbox."""
        min_easting = math.floor(left / self._tile_size) * self._tile_size
        max_easting = math.floor((right - 1e-9) / self._tile_size) * self._tile_size
        min_northing = math.floor(bottom / self._tile_size) * self._tile_size
        max_northing = math.floor((top - 1e-9) / self._tile_size) * self._tile_size

        for northing in range(
            min_northing, max_northing + self._tile_size, self._tile_size
        ):
            for easting in self._available_tiles.get(northing, ()):
                if min_easting <= easting <= max_easting:
                    yield northing, easting

    def _open_remote(self, url: str):
        """Open a remote BEV COG, with a fallback for GDAL builds that need /vsicurl/."""
        last_error = None

        for candidate in (url, f"/vsicurl/{url}"):
            try:
                return rasterio.open(candidate)
            except RasterioIOError as error:
                last_error = error

        if last_error is not None:
            raise last_error

        raise RasterioIOError(f"Failed to open remote dataset: {url}")

    def _materialize_tile_window(
        self,
        northing: int,
        easting: int,
        left: float,
        bottom: float,
        right: float,
        top: float,
    ) -> str | None:
        """Read only the requested part of one BEV tile and save it as a local GeoTIFF."""
        tile_left = easting
        tile_bottom = northing
        tile_right = easting + self._tile_size
        tile_top = northing + self._tile_size

        intersection = (
            max(left, tile_left),
            max(bottom, tile_bottom),
            min(right, tile_right),
            min(top, tile_top),
        )

        if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
            return None

        url = self.formatted_url(northing=northing, easting=easting)

        with self._open_remote(url) as src:
            window = src.window(*intersection).round_offsets().round_lengths()

            if window.width <= 0 or window.height <= 0:
                return None

            data = src.read(1, window=window)

            if data.size == 0:
                return None

            transform = src.window_transform(window)
            snapped_bounds = window_bounds(window, src.transform)

            file_name = (
                f"ALS_DTM_20240915_N{northing}E{easting}_"
                f"{int(snapped_bounds[0])}_{int(snapped_bounds[1])}_"
                f"{int(snapped_bounds[2])}_{int(snapped_bounds[3])}.tif"
            )
            file_path = os.path.join(self.shared_tiff_path, file_name)

            if os.path.isfile(file_path):
                return file_path

            profile = {
                "driver": "GTiff",
                "width": data.shape[1],
                "height": data.shape[0],
                "count": 1,
                "dtype": data.dtype,
                "crs": src.crs,
                "transform": transform,
                "compress": "deflate",
            }

            if src.nodata is not None:
                profile["nodata"] = src.nodata

            with rasterio.open(file_path, "w", **profile) as dst:
                dst.write(data, 1)

        return file_path

    def download_tiles(self) -> list[str]:
        """Return one local GeoTIFF covering the requested area in BEV source CRS."""
        left, bottom, right, top = self._get_projected_bbox()

        file_name = (
            f"ALS_DTM_20240915_"
            f"{int(left)}_{int(bottom)}_{int(right)}_{int(top)}.tif"
        )
        file_path = os.path.join(self._tile_directory, file_name)

        if os.path.isfile(file_path):
            return [file_path]

        local_tiles: list[str] = []

        for northing, easting in self._iter_required_tiles(left, bottom, right, top):
            tile_path = self._materialize_tile_window(
                northing=northing,
                easting=easting,
                left=left,
                bottom=bottom,
                right=right,
                top=top,
            )
            if tile_path is not None:
                local_tiles.append(tile_path)

        if not local_tiles:
            return []

        datasets = [rasterio.open(path) for path in local_tiles]

        try:
            mosaic, out_transform = merge(datasets)
            out_meta = datasets[0].meta.copy()
            out_meta.update(
                {
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_transform,
                    "count": mosaic.shape[0],
                    "compress": "deflate",
                }
            )

            if datasets[0].nodata is None:
                out_meta.pop("nodata", None)

            with rasterio.open(file_path, "w", **out_meta) as dst:
                dst.write(mosaic)
        finally:
            for dataset in datasets:
                dataset.close()

        return [file_path]
