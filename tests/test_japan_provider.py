from __future__ import annotations

import numpy as np
import pytest
import rasterio
from rasterio.io import MemoryFile

from pydtmdl.providers.japan import JapanGSIProvider


def _make_gsi_png(red: int, green: int, blue: int) -> bytes:
    data = np.zeros((3, 2, 2), dtype=np.uint8)
    data[0, :, :] = red
    data[1, :, :] = green
    data[2, :, :] = blue
    with MemoryFile() as memory_file:
        with memory_file.open(driver="PNG", width=2, height=2, count=3, dtype="uint8") as dataset:
            dataset.write(data)
        return memory_file.read()


def test_japan_gsi_decodes_positive_png_elevation(tmp_path):
    provider = JapanGSIProvider(
        coordinates=(35.6812, 139.7671),
        width_m=256,
        height_m=256,
        directory=str(tmp_path),
    )

    geotiff = provider._decode_tile(_make_gsi_png(0, 48, 57), 0.0, 1.0, 2.0, 3.0)

    with MemoryFile(geotiff) as memory_file:
        with memory_file.open() as dataset:
            data = dataset.read(1)
            assert str(dataset.crs) == "EPSG:3857"
            assert dataset.nodata == pytest.approx(provider._nodata)
            assert data[0, 0] == pytest.approx(123.45)


def test_japan_gsi_selects_tiles_containing_tokyo(tmp_path):
    provider = JapanGSIProvider(
        coordinates=(35.6812, 139.7671),
        width_m=512,
        height_m=512,
        directory=str(tmp_path),
    )

    tiles = provider._iter_required_tiles(*provider.get_bbox())

    assert tiles
    assert all(tile[0] == provider._zoom for tile in tiles)
    assert any(tile[1] == 14552 and tile[2] == 6451 for tile in tiles)
