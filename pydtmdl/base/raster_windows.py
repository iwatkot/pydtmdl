"""Rasterio window helpers for bounded provider reads."""

from __future__ import annotations

import math
from typing import Any

from rasterio.errors import WindowError
from rasterio.windows import Window, from_bounds


Bounds = tuple[float, float, float, float]


def intersect_dataset_bounds(dataset: Any, bounds: Bounds) -> Bounds | None:
    """Return bounds clipped to the dataset extent, or None when disjoint."""
    west, south, east, north = bounds
    dataset_bounds = dataset.bounds
    clipped = (
        max(west, float(dataset_bounds.left)),
        max(south, float(dataset_bounds.bottom)),
        min(east, float(dataset_bounds.right)),
        min(north, float(dataset_bounds.top)),
    )
    if clipped[0] >= clipped[2] or clipped[1] >= clipped[3]:
        return None
    return clipped


def window_from_bounds_clamped(dataset: Any, bounds: Bounds) -> Window | None:
    """Build an integer read window clipped to a dataset's pixel extent."""
    clipped = intersect_dataset_bounds(dataset, bounds)
    if clipped is None:
        return None

    raw_window = from_bounds(*clipped, dataset.transform)
    col_start = math.floor(raw_window.col_off)
    row_start = math.floor(raw_window.row_off)
    col_stop = math.ceil(raw_window.col_off + raw_window.width)
    row_stop = math.ceil(raw_window.row_off + raw_window.height)
    window = Window(
        col_start,
        row_start,
        col_stop - col_start,
        row_stop - row_start,
    )

    try:
        window = window.intersection(Window(0, 0, dataset.width, dataset.height))
    except WindowError:
        return None

    if window.width <= 0 or window.height <= 0:
        return None

    return Window(
        int(window.col_off),
        int(window.row_off),
        int(window.width),
        int(window.height),
    )
