# Postprocessing Guide

This document describes standalone postprocessing helpers in pydtmdl.

The postprocessing API is intentionally separate from extraction so you can:

- keep provider downloads and caching unchanged
- apply game-specific transforms only in selected pipelines
- normalize DTM and imagery outputs into a consistent numeric range
- optionally export postprocessed DTM to single-channel PNG

## Available APIs

The following functions are exported from pydtmdl:

- postprocess_dtm
- postprocess_imagery
- export_single_channel_png
- postprocess_dtm_to_png

Metadata models:

- PostprocessMetadata
- PngExportMetadata

## Postprocess DTM

Use this to normalize DTM arrays and/or shift terrain to a zero floor.

```python
from pydtmdl import postprocess_dtm

processed, metadata = postprocess_dtm(
    dtm_array,
    normalize_to_dtype=True,
    target_dtype="uint16",
    apply_zero_floor=True,
    zero_floor_value=35000,
)
```

### Parameters

- normalize_to_dtype: when True, rescales valid values to target dtype range
- target_dtype: output dtype name, for example uint16
- apply_zero_floor: when True, subtracts floor and clamps negatives to zero
- zero_floor_value: explicit floor value; when None, valid minimum is used

### Result

Returns tuple:

- processed array
- PostprocessMetadata with min/max, dtype, counts, and applied options

## Postprocess Imagery

Use this for imagery arrays (typically bands-first).

```python
from pydtmdl import postprocess_imagery

processed, metadata = postprocess_imagery(
    imagery_array,
    normalize_to_dtype=True,
    target_dtype="uint16",
    normalize_per_band=True,
)
```

### Notes

- normalize_per_band=True normalizes each band independently for 3D arrays
- set normalize_per_band=False to normalize all bands together

## Export to PNG

Use export_single_channel_png when you already have a 2D array.

```python
from pydtmdl import export_single_channel_png

png_metadata = export_single_channel_png(
    processed_dtm,
    "output/dtm.png",
    output_dtype="uint16",
    nodata_fill_value=0,
)
```

### PNG constraints

- input must be a single-channel 2D array
- output dtype supports uint8 and uint16
- PNG is not georeferenced

## One-step helper: postprocess and export

Use postprocess_dtm_to_png to do both operations in one call.

```python
from pydtmdl import postprocess_dtm_to_png

processed, post_meta, png_meta = postprocess_dtm_to_png(
    dtm_array,
    "output/dtm.png",
    normalize_to_dtype=True,
    target_dtype="uint16",
    apply_zero_floor=True,
    zero_floor_value=35000,
    png_dtype="uint16",
)
```

## Production recommendations

For stable cross-provider DTM output in terrain pipelines:

- normalize_to_dtype=True
- target_dtype="uint16"
- apply_zero_floor=True
- use a consistent zero_floor_value policy:
  - fixed global baseline for strict comparability
  - or per-tile minimum (zero_floor_value=None) for local contrast

Keep in mind that normalization aligns numeric range, not source quality. Different providers can still differ in detail, artifacts, and effective resolution.
