from __future__ import annotations

import numpy as np
import rasterio

from pydtmdl import (
    export_single_channel_png,
    postprocess_dtm,
    postprocess_dtm_to_png,
    postprocess_imagery,
)


def test_postprocess_dtm_applies_zero_floor_shift():
    data = np.array(
        [
            [35000, 45000],
            [55000, 34000],
        ],
        dtype=np.uint16,
    )

    processed, metadata = postprocess_dtm(
        data,
        apply_zero_floor=True,
        zero_floor_value=35000,
    )

    expected = np.array(
        [
            [0, 10000],
            [20000, 0],
        ],
        dtype=np.uint16,
    )

    assert isinstance(processed, np.ndarray)
    assert processed.dtype == np.uint16
    assert np.array_equal(processed, expected)
    assert metadata.apply_zero_floor is True
    assert metadata.zero_floor_value == 35000.0


def test_postprocess_dtm_normalizes_to_uint16_range():
    data = np.array(
        [
            [100.0, 150.0],
            [200.0, 125.0],
        ],
        dtype=np.float32,
    )

    processed, metadata = postprocess_dtm(
        data,
        normalize_to_dtype=True,
        target_dtype="uint16",
    )

    assert isinstance(processed, np.ndarray)
    assert processed.dtype == np.uint16
    assert processed.min() == 0
    assert processed.max() == np.iinfo(np.uint16).max
    assert metadata.normalize_to_dtype is True
    assert metadata.output_dtype == "uint16"


def test_postprocess_imagery_normalizes_each_band_independently():
    band_1 = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
    band_2 = np.array([[1000.0, 1200.0], [1400.0, 1600.0]], dtype=np.float32)
    band_3 = np.array([[2.0, 2.0], [2.0, 2.0]], dtype=np.float32)
    imagery = np.stack([band_1, band_2, band_3])

    processed, metadata = postprocess_imagery(
        imagery,
        normalize_to_dtype=True,
        target_dtype="uint16",
        normalize_per_band=True,
    )

    assert isinstance(processed, np.ndarray)
    assert processed.dtype == np.uint16
    assert processed[0].min() == 0
    assert processed[0].max() == np.iinfo(np.uint16).max
    assert processed[1].min() == 0
    assert processed[1].max() == np.iinfo(np.uint16).max
    assert processed[2].min() == 0
    assert processed[2].max() == 0
    assert metadata.shape == (3, 2, 2)


def test_export_single_channel_png_writes_uint16(tmp_path):
    data = np.array(
        [
            [0, 1000],
            [65535, 42],
        ],
        dtype=np.uint16,
    )
    output_path = tmp_path / "dtm.png"

    metadata = export_single_channel_png(data, str(output_path), output_dtype="uint16")

    assert output_path.exists()
    assert metadata.dtype == "uint16"
    with rasterio.open(output_path) as src:
        assert src.count == 1
        assert src.dtypes[0] == "uint16"
        loaded = src.read(1)
    assert np.array_equal(loaded, data)


def test_postprocess_dtm_to_png_returns_processed_and_file(tmp_path):
    data = np.array(
        [
            [35000, 45000],
            [55000, 34000],
        ],
        dtype=np.uint16,
    )
    output_path = tmp_path / "postprocessed_dtm.png"

    processed, post_metadata, png_metadata = postprocess_dtm_to_png(
        data,
        str(output_path),
        apply_zero_floor=True,
        zero_floor_value=35000,
        normalize_to_dtype=True,
        target_dtype="uint16",
        png_dtype="uint16",
    )

    assert output_path.exists()
    assert processed.dtype == np.uint16
    assert post_metadata.normalize_to_dtype is True
    assert png_metadata.output_path == str(output_path)
    with rasterio.open(output_path) as src:
        assert src.count == 1
        assert src.dtypes[0] == "uint16"
