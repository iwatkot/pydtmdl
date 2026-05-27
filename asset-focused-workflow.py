"""Run a provider-backed final asset workflow for MapToPlay-style outputs.

This script intentionally prints only final JPG/PNG asset paths. GeoTIFFs are
internal pydtmdl cache/temp files and are not part of the app-facing contract.
"""

from __future__ import annotations

from pathlib import Path

from pydtmdl import extract_project_dtm, extract_project_imagery


CENTER = (48.137154, 11.576124)  # Munich, Bavaria
WIDTH_M = 600
HEIGHT_M = 600
ROTATION_DEG = 0.0
OUTPUT_DIR = Path("asset-focused-output")

IMAGERY_REQUESTS = [
    {
        "provider_code": "bavaria_dop20",
        "output_basename": "bavaria-imagery",
        "max_edge": 512,
        "target_resolution_m": None,
    },
    {
        "provider_code": "sentinel2_l2a",
        "output_basename": "sentinel2-imagery",
        "max_edge": 512,
        "target_resolution_m": 10,
    },
]


def print_dtm_result(result) -> None:
    print("DTM")
    print(f"  provider: {result.actual_provider} ({result.actual_provider_name})")
    print(f"  full PNG:    {result.full_output_path}")
    print(f"  full shape:  {result.full_shape}, resolution: {result.full_resolution:.3f} m/px")
    print(f"  preview PNG: {result.preview_output_path}")
    print(
        f"  preview shape: {result.preview_shape}, "
        f"resolution: {result.preview_resolution:.3f} m/px"
    )
    print(f"  shared file: {result.preview_output_path == result.full_output_path}")
    print(f"  height range: {result.min:.3f} to {result.max:.3f}")


def print_imagery_result(result) -> None:
    print("Imagery")
    print(f"  provider: {result.actual_provider} ({result.actual_provider_name})")
    print(f"  preview JPG: {result.preview_output_path}")
    print(
        f"  preview shape: {result.preview_shape}, "
        f"resolution: {result.preview_resolution:.3f} m/px"
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # max_edge is intentionally small here so the demo usually produces two DTM
    # PNG paths. Raise it above the full shape to see preview/full share one file.
    dtm = extract_project_dtm(
        center=CENTER,
        width_m=WIDTH_M,
        height_m=HEIGHT_M,
        rotation_deg=ROTATION_DEG,
        provider_code="bavaria",
        directory=str(OUTPUT_DIR),
        max_edge=256,
        output_basename="bavaria-dtm",
    )
    print_dtm_result(dtm)

    for request in IMAGERY_REQUESTS:
        try:
            imagery = extract_project_imagery(
                center=CENTER,
                width_m=WIDTH_M,
                height_m=HEIGHT_M,
                rotation_deg=ROTATION_DEG,
                provider_code=request["provider_code"],
                directory=str(OUTPUT_DIR),
                max_edge=request["max_edge"],
                target_resolution_m=request["target_resolution_m"],
                output_basename=request["output_basename"],
            )
            print()
            print_imagery_result(imagery)
            break
        except Exception as exc:  # pylint: disable=broad-exception-caught
            print()
            print(f"Imagery provider {request['provider_code']} failed: {exc}")
    else:
        raise RuntimeError("No imagery provider produced a preview JPEG.")


if __name__ == "__main__":
    main()
