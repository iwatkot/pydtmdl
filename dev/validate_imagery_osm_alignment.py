"""Create real imagery + OSM overlay artifacts for MapToPlay alignment checks.

The script intentionally uses pydtmdl's public project imagery API and invokes
MapToPlay's OSM worker helper as a subprocess, so the generated overlay follows
the same project-local vector transform as production imports.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw

from pydtmdl import extract_project_imagery
from pydtmdl.imagery_providers.naip import NAIPImagerySettings


MAPTOPLAY_ROOT = Path(r"C:\Coding\projectorbis")
OSM_PREPARE_SCRIPT = MAPTOPLAY_ROOT / "apps" / "worker" / "scripts" / "run_osm_prepare.py"


CASES: dict[str, dict[str, Any]] = {
    "naip-mississippi-river": {
        "name": "naip-osm-alignment-mississippi-river",
        "provider_code": "naip",
        "center": (35.92701060489532, -89.57458378845217),
        "width_m": 4096,
        "height_m": 4096,
        "playable_width_m": 4096,
        "playable_height_m": 4096,
        "rotation_deg": 12.5,
        "preview_max_edge": 1536,
        "min_source_files": 2,
        "provider_settings": {
            "search_limit": 48,
            "max_items": 8,
            "date_from": "2021-07-22",
        },
    },
    "copernicus-townsend-farm": {
        "name": "copernicus-osm-alignment-townsend-farm",
        "provider_code": "copernicus_vhr_2021",
        "center": (51.68827703786368, -1.7481606214009844),
        "width_m": 6144,
        "height_m": 6144,
        "playable_width_m": 2048,
        "playable_height_m": 2048,
        "rotation_deg": 0.0,
        "preview_max_edge": 1536,
        "min_source_files": 2,
    },
}


def local_to_pixel(
    point: tuple[float, float],
    *,
    width_m: float,
    height_m: float,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    x, y = point
    image_width, image_height = image_size
    return (
        (x + width_m / 2.0) * image_width / width_m,
        (y + height_m / 2.0) * image_height / height_m,
    )


def ring_to_pixels(
    ring: list[list[float]],
    *,
    width_m: float,
    height_m: float,
    image_size: tuple[int, int],
    scale: int,
) -> list[tuple[float, float]]:
    return [
        (
            pixel_x * scale,
            pixel_y * scale,
        )
        for pixel_x, pixel_y in (
            local_to_pixel((float(x), float(y)), width_m=width_m, height_m=height_m, image_size=image_size)
            for x, y in ring
        )
    ]


def line_to_pixels(
    coordinates: list[list[float]],
    *,
    width_m: float,
    height_m: float,
    image_size: tuple[int, int],
    scale: int,
) -> list[tuple[float, float]]:
    return ring_to_pixels(
        coordinates,
        width_m=width_m,
        height_m=height_m,
        image_size=image_size,
        scale=scale,
    )


def feature_style(feature: dict[str, Any]) -> tuple[tuple[int, int, int, int], int]:
    tags = feature.get("tags") or {}
    if "highway" in tags:
        return (255, 230, 0, 245), 4
    if "railway" in tags:
        return (255, 255, 255, 245), 3
    if "waterway" in tags:
        return (0, 210, 255, 245), 3
    if tags.get("building"):
        return (255, 85, 70, 220), 2
    if tags.get("natural") == "water" or tags.get("water"):
        return (0, 160, 255, 90), 2
    if tags.get("landuse") in {"forest", "orchard"} or tags.get("natural") == "wood":
        return (70, 220, 80, 80), 2
    return (255, 0, 180, 170), 2


def draw_osm_overlay(
    *,
    preview_path: Path,
    vector_path: Path,
    output_path: Path,
    transparent_overlay_path: Path,
    width_m: float,
    height_m: float,
) -> dict[str, int]:
    base = Image.open(preview_path).convert("RGB")
    scale = 3
    large_size = (base.width * scale, base.height * scale)
    large_base = base.resize(large_size, Image.Resampling.LANCZOS).convert("RGBA")
    transparent = Image.new("RGBA", large_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(transparent, "RGBA")

    payload = json.loads(vector_path.read_text(encoding="utf-8"))
    stats = {
        "drawn_polygons": 0,
        "drawn_lines": 0,
        "drawn_points": 0,
    }

    # Draw broad/filled features first, then lines on top.
    polygon_features = [f for f in payload["features"] if f["geometryType"] == "Polygon"]
    line_features = [f for f in payload["features"] if f["geometryType"] == "LineString"]
    point_features = [f for f in payload["features"] if f["geometryType"] == "Point"]

    for feature in polygon_features:
        color, width = feature_style(feature)
        rings = feature.get("geometry") or []
        if not rings:
            continue
        exterior = ring_to_pixels(
            rings[0],
            width_m=width_m,
            height_m=height_m,
            image_size=base.size,
            scale=scale,
        )
        if len(exterior) >= 3:
            fill = (color[0], color[1], color[2], min(color[3], 85))
            draw.polygon(exterior, outline=color, fill=fill)
            for interior in rings[1:]:
                hole = ring_to_pixels(
                    interior,
                    width_m=width_m,
                    height_m=height_m,
                    image_size=base.size,
                    scale=scale,
                )
                if len(hole) >= 3:
                    draw.polygon(hole, fill=(0, 0, 0, 0))
            stats["drawn_polygons"] += 1

    # Dark underlay makes light OSM strokes readable on bright roads/crops.
    for feature in line_features:
        coordinates = feature.get("geometry") or []
        if len(coordinates) < 2:
            continue
        _, width = feature_style(feature)
        points = line_to_pixels(
            coordinates,
            width_m=width_m,
            height_m=height_m,
            image_size=base.size,
            scale=scale,
        )
        draw.line(points, fill=(0, 0, 0, 190), width=max(1, (width + 2) * scale), joint="curve")

    for feature in line_features:
        coordinates = feature.get("geometry") or []
        if len(coordinates) < 2:
            continue
        color, width = feature_style(feature)
        points = line_to_pixels(
            coordinates,
            width_m=width_m,
            height_m=height_m,
            image_size=base.size,
            scale=scale,
        )
        draw.line(points, fill=color, width=max(1, width * scale), joint="curve")
        stats["drawn_lines"] += 1

    for feature in point_features:
        geometry = feature.get("geometry")
        if not geometry or len(geometry) < 2:
            continue
        pixel_x, pixel_y = local_to_pixel(
            (float(geometry[0]), float(geometry[1])),
            width_m=width_m,
            height_m=height_m,
            image_size=base.size,
        )
        radius = 3 * scale
        center_x = pixel_x * scale
        center_y = pixel_y * scale
        draw.ellipse(
            (center_x - radius, center_y - radius, center_x + radius, center_y + radius),
            fill=(255, 0, 180, 230),
            outline=(0, 0, 0, 180),
            width=scale,
        )
        stats["drawn_points"] += 1

    composited = Image.alpha_composite(large_base, transparent)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composited.resize(base.size, Image.Resampling.LANCZOS).convert("RGB").save(output_path)
    transparent.resize(base.size, Image.Resampling.LANCZOS).save(transparent_overlay_path)
    return stats


def build_user_settings(case: dict[str, Any]) -> Any:
    if case["provider_code"] == "naip":
        settings = dict(case.get("provider_settings") or {})
        settings.setdefault("date_to", datetime.now(UTC).date().isoformat())
        return NAIPImagerySettings(**settings)
    return None


def parse_args() -> dict[str, Any]:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        choices=sorted(CASES),
        default="naip-mississippi-river",
        help="Alignment case to render.",
    )
    args = parser.parse_args()
    return CASES[args.case]


def run_osm_prepare(output_dir: Path, case: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    result_path = output_dir / "osm-result.json"
    payload = {
        "providerCode": "osm-api",
        "sourceMode": "provider",
        "center": {"lat": case["center"][0], "lng": case["center"][1]},
        "width": case["width_m"],
        "height": case["height_m"],
        "playableWidth": case["playable_width_m"],
        "playableHeight": case["playable_height_m"],
        "rotation": case["rotation_deg"],
        "outputDirectory": str(output_dir),
        "downloadTimeoutSeconds": 120,
        "downloadMarginMeters": 1024,
        "downloadMaxAttempts": 3,
        "downloadRetryDelaySeconds": 10,
        "downloadRequestSpacingSeconds": 1,
        "enablePreprocessing": True,
        "cutterHalfWidthMeters": 5,
        "resultPath": str(result_path),
    }
    completed = subprocess.run(
        [sys.executable, str(OSM_PREPARE_SCRIPT), json.dumps(payload)],
        cwd=str(MAPTOPLAY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    (output_dir / "osm-prepare.stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "osm-prepare.stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(
            f"MapToPlay OSM prepare failed with exit code {completed.returncode}. "
            f"See {output_dir / 'osm-prepare.stderr.txt'}"
        )
    return result_path, json.loads(result_path.read_text(encoding="utf-8"))


def main() -> None:
    case = parse_args()

    if not OSM_PREPARE_SCRIPT.exists():
        raise FileNotFoundError(f"MapToPlay OSM script not found: {OSM_PREPARE_SCRIPT}")

    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    run_dir = REPO_ROOT / "imagery_alignment_checks" / f"{stamp}-{case['name']}"
    imagery_cache_dir = run_dir / "pydtmdl-cache"
    osm_dir = run_dir / "osm"
    run_dir.mkdir(parents=True, exist_ok=True)

    imagery_result = extract_project_imagery(
        center=case["center"],
        width_m=case["width_m"],
        height_m=case["height_m"],
        rotation_deg=case["rotation_deg"],
        provider_code=case["provider_code"],
        user_settings=build_user_settings(case),
        directory=str(imagery_cache_dir),
        max_edge=case["preview_max_edge"],
        jpeg_quality=90,
        cleanup_temp_files=False,
    )

    preview_copy = run_dir / "01-pydtmdl-imagery-preview.jpg"
    shutil.copy2(imagery_result.preview_output_path, preview_copy)

    osm_result_path, osm_result = run_osm_prepare(osm_dir, case)
    vector_path = Path(osm_result["derivedOutputPath"])

    overlay_stats = draw_osm_overlay(
        preview_path=preview_copy,
        vector_path=vector_path,
        output_path=run_dir / "02-pydtmdl-imagery-with-maptoplay-osm-overlay.png",
        transparent_overlay_path=run_dir / "03-maptoplay-osm-overlay-transparent.png",
        width_m=case["width_m"],
        height_m=case["height_m"],
    )

    source_files = list(imagery_result.source_files)
    if len(source_files) < case["min_source_files"]:
        raise RuntimeError(
            f"Validation imagery only rendered {len(source_files)} source file(s); "
            "choose a larger or different case."
        )

    summary = {
        "case": case,
        "imagery": imagery_result.model_dump(),
        "imageryPreviewCopy": str(preview_copy),
        "sourceFileCount": len(source_files),
        "sourceFiles": source_files,
        "cachePath": imagery_result.cache_path,
        "mergedTiff": str(Path(imagery_result.cache_path) / "merged.tif"),
        "reprojectedTiff": str(Path(imagery_result.cache_path) / "reprojected.tif"),
        "osmResultPath": str(osm_result_path),
        "osmDerivedVectorPath": str(vector_path),
        "osmStats": {
            "featureCount": osm_result["featureCount"],
            "pointCount": osm_result["pointCount"],
            "lineCount": osm_result["lineCount"],
            "polygonCount": osm_result["polygonCount"],
            "downloadedBbox": osm_result["downloadedBbox"],
            "localBounds": osm_result["localBounds"],
        },
        "overlayStats": overlay_stats,
        "outputs": {
            "preview": str(preview_copy),
            "overlay": str(run_dir / "02-pydtmdl-imagery-with-maptoplay-osm-overlay.png"),
            "transparentOverlay": str(run_dir / "03-maptoplay-osm-overlay-transparent.png"),
        },
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps({"runDir": str(run_dir), "summary": str(summary_path)}, indent=2))


if __name__ == "__main__":
    main()
