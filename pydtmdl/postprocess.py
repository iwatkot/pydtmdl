"""Standalone post-processing helpers for DTM and imagery arrays.

These utilities are intentionally decoupled from the extraction pipeline so callers can
apply additional transforms only when needed (e.g. game-engine preparation).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio


@dataclass(slots=True)
class PostprocessMetadata:
    """Metadata describing the applied post-processing operations."""

    input_dtype: str
    output_dtype: str
    shape: tuple[int, ...]
    valid_pixel_count: int
    masked_pixel_count: int
    input_min: float | None
    input_max: float | None
    output_min: float | None
    output_max: float | None
    normalize_to_dtype: bool
    target_dtype: str
    apply_zero_floor: bool
    zero_floor_value: float | None


@dataclass(slots=True)
class PngExportMetadata:
    """Metadata describing a written single-channel PNG file."""

    output_path: str
    dtype: str
    shape: tuple[int, int]
    nodata_fill_value: int


def _to_masked_array(data: np.ndarray | np.ma.MaskedArray) -> np.ma.MaskedArray:
    masked = np.ma.array(data, copy=True)
    if np.issubdtype(masked.dtype, np.floating):
        masked = np.ma.masked_invalid(masked)
    return masked


def _min_max(masked: np.ma.MaskedArray) -> tuple[float | None, float | None]:
    if masked.count() == 0:
        return None, None
    values = masked.compressed()
    return float(values.min()), float(values.max())


def _normalize_array(masked: np.ma.MaskedArray, dtype: np.dtype[Any]) -> np.ma.MaskedArray:
    if masked.count() == 0:
        return np.ma.array(np.zeros(masked.shape, dtype=dtype), mask=np.ma.getmaskarray(masked))

    source = np.ma.array(masked, dtype=np.float64, copy=True)
    min_value, max_value = _min_max(source)
    assert min_value is not None
    assert max_value is not None

    out_min: float
    out_max: float
    if np.issubdtype(dtype, np.integer):
        int_info = np.iinfo(dtype)
        out_min, out_max = float(int_info.min), float(int_info.max)
    elif np.issubdtype(dtype, np.floating):
        float_info = np.finfo(dtype)
        out_min, out_max = float(float_info.min), float(float_info.max)
    else:
        raise ValueError(f"Unsupported target dtype: {dtype}")

    if np.isclose(max_value, min_value):
        normalized = np.ma.zeros(source.shape, dtype=np.float64)
    else:
        normalized = np.ma.array((source - min_value) / (max_value - min_value), copy=False)

    scaled = np.ma.array((normalized * (out_max - out_min)) + out_min, copy=False)
    clipped = np.ma.array(np.ma.clip(scaled, out_min, out_max), copy=False)

    if np.issubdtype(dtype, np.integer):
        clipped = np.ma.array(np.ma.round(clipped), copy=False)

    return np.ma.array(clipped, dtype=dtype, copy=False)


def _apply_zero_floor(
    masked: np.ma.MaskedArray,
    zero_floor_value: float | None,
) -> tuple[np.ma.MaskedArray, float | None]:
    if masked.count() == 0:
        return masked, zero_floor_value

    floor_value = float(masked.min()) if zero_floor_value is None else float(zero_floor_value)
    shifted = np.ma.array(np.ma.maximum(masked.astype(np.float64) - floor_value, 0.0), copy=False)
    return shifted, floor_value


def _restore_output_type(
    original: np.ndarray | np.ma.MaskedArray,
    processed: np.ma.MaskedArray,
    *,
    output_dtype: np.dtype[Any],
) -> np.ndarray | np.ma.MaskedArray:
    casted = np.ma.array(processed, dtype=output_dtype, copy=False)
    if np.ma.isMaskedArray(original):
        return casted
    return np.asarray(casted.filled(0))


def _cast_clip_array(array: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        clipped = np.clip(array, info.min, info.max)
        return np.rint(clipped).astype(dtype)
    return array.astype(dtype)


def _postprocess(
    data: np.ndarray | np.ma.MaskedArray,
    *,
    normalize_to_dtype: bool,
    target_dtype: str,
    apply_zero_floor: bool,
    zero_floor_value: float | None,
    normalize_per_band: bool,
) -> tuple[np.ndarray | np.ma.MaskedArray, PostprocessMetadata]:
    masked = _to_masked_array(data)
    input_dtype = str(np.asarray(data).dtype)
    input_min, input_max = _min_max(masked)

    processed = masked
    used_floor_value: float | None = None

    if apply_zero_floor:
        processed, used_floor_value = _apply_zero_floor(processed, zero_floor_value)

    resolved_dtype = np.dtype(target_dtype) if normalize_to_dtype else np.asarray(data).dtype

    if normalize_to_dtype:
        if processed.ndim == 3 and normalize_per_band:
            normalized_bands = [
                _normalize_array(processed[band_index], resolved_dtype)
                for band_index in range(processed.shape[0])
            ]
            processed = np.ma.array(normalized_bands)
        else:
            processed = _normalize_array(processed, resolved_dtype)

    output = _restore_output_type(data, processed, output_dtype=np.dtype(resolved_dtype))

    output_masked = np.ma.array(output, copy=False)
    output_min, output_max = _min_max(output_masked)

    metadata = PostprocessMetadata(
        input_dtype=input_dtype,
        output_dtype=str(np.asarray(output).dtype),
        shape=tuple(int(dim) for dim in np.asarray(output).shape),
        valid_pixel_count=int(output_masked.count()),
        masked_pixel_count=int(np.asarray(output_masked).size - output_masked.count()),
        input_min=input_min,
        input_max=input_max,
        output_min=output_min,
        output_max=output_max,
        normalize_to_dtype=normalize_to_dtype,
        target_dtype=str(np.dtype(target_dtype)),
        apply_zero_floor=apply_zero_floor,
        zero_floor_value=used_floor_value,
    )
    return output, metadata


def postprocess_dtm(
    data: np.ndarray | np.ma.MaskedArray,
    *,
    normalize_to_dtype: bool = False,
    target_dtype: str = "uint16",
    apply_zero_floor: bool = False,
    zero_floor_value: float | None = None,
) -> tuple[np.ndarray | np.ma.MaskedArray, PostprocessMetadata]:
    """Post-process a DTM array without modifying the extraction pipeline.

    Args:
        data: Input DTM array.
        normalize_to_dtype: Normalize valid values into target dtype range.
        target_dtype: Output dtype for normalization (default: uint16).
        apply_zero_floor: Shift values down by floor and clamp negatives to zero.
        zero_floor_value: Explicit floor value; if None, uses current valid minimum.

    Returns:
        Tuple of (processed array, metadata).
    """
    return _postprocess(
        data,
        normalize_to_dtype=normalize_to_dtype,
        target_dtype=target_dtype,
        apply_zero_floor=apply_zero_floor,
        zero_floor_value=zero_floor_value,
        normalize_per_band=False,
    )


def postprocess_imagery(
    data: np.ndarray | np.ma.MaskedArray,
    *,
    normalize_to_dtype: bool = False,
    target_dtype: str = "uint16",
    apply_zero_floor: bool = False,
    zero_floor_value: float | None = None,
    normalize_per_band: bool = True,
) -> tuple[np.ndarray | np.ma.MaskedArray, PostprocessMetadata]:
    """Post-process an imagery array without modifying the extraction pipeline.

    Args:
        data: Input imagery array (typically bands-first).
        normalize_to_dtype: Normalize valid values into target dtype range.
        target_dtype: Output dtype for normalization (default: uint16).
        apply_zero_floor: Shift values down by floor and clamp negatives to zero.
        zero_floor_value: Explicit floor value; if None, uses current valid minimum.
        normalize_per_band: If True and array is 3D, normalize each band independently.

    Returns:
        Tuple of (processed array, metadata).
    """
    return _postprocess(
        data,
        normalize_to_dtype=normalize_to_dtype,
        target_dtype=target_dtype,
        apply_zero_floor=apply_zero_floor,
        zero_floor_value=zero_floor_value,
        normalize_per_band=normalize_per_band,
    )


def export_single_channel_png(
    data: np.ndarray | np.ma.MaskedArray,
    output_path: str,
    *,
    output_dtype: str = "uint16",
    nodata_fill_value: int = 0,
) -> PngExportMetadata:
    """Write a single-channel PNG from a 2D array.

    Args:
        data: Input 2D array.
        output_path: Destination PNG path.
        output_dtype: PNG pixel dtype. Supported values: uint8, uint16.
        nodata_fill_value: Fill value for masked pixels.

    Returns:
        Export metadata.
    """
    dtype = np.dtype(output_dtype)
    if dtype not in (np.dtype("uint8"), np.dtype("uint16")):
        raise ValueError("PNG export supports only uint8 and uint16 dtypes.")

    masked = np.ma.array(data, copy=False)
    if masked.ndim != 2:
        raise ValueError("PNG export expects a 2D single-channel array.")

    array = masked.filled(nodata_fill_value).astype(np.float64)
    array = _cast_clip_array(array, dtype)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output,
        "w",
        driver="PNG",
        width=array.shape[1],
        height=array.shape[0],
        count=1,
        dtype=str(dtype),
    ) as dst:
        dst.write(array, 1)

    return PngExportMetadata(
        output_path=str(output),
        dtype=str(dtype),
        shape=(int(array.shape[0]), int(array.shape[1])),
        nodata_fill_value=int(nodata_fill_value),
    )


def postprocess_dtm_to_png(
    data: np.ndarray | np.ma.MaskedArray,
    output_path: str,
    *,
    normalize_to_dtype: bool = False,
    target_dtype: str = "uint16",
    apply_zero_floor: bool = False,
    zero_floor_value: float | None = None,
    png_dtype: str = "uint16",
    nodata_fill_value: int = 0,
) -> tuple[np.ndarray | np.ma.MaskedArray, PostprocessMetadata, PngExportMetadata]:
    """Post-process DTM data and export it as a single-channel PNG.

    Returns:
        Tuple of (processed array, postprocess metadata, PNG export metadata).
    """
    processed, metadata = postprocess_dtm(
        data,
        normalize_to_dtype=normalize_to_dtype,
        target_dtype=target_dtype,
        apply_zero_floor=apply_zero_floor,
        zero_floor_value=zero_floor_value,
    )
    png_metadata = export_single_channel_png(
        processed,
        output_path,
        output_dtype=png_dtype,
        nodata_fill_value=nodata_fill_value,
    )
    return processed, metadata, png_metadata
