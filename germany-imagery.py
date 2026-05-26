"""Extract small preview images from German state orthophoto providers."""

import cv2

from pydtmdl import ImageryProvider


REQUESTS = [
    {
        "code": "nrw_dop",
        "center": (51.301, 7.602),
        "output": "output_germany_nrw_dop.png",
    },
    {
        "code": "bavaria_dop20",
        "center": (47.741, 11.431),
        "output": "output_germany_bavaria_dop20.png",
    },
    {
        "code": "hessen_dop20",
        "center": (50.511, 9.684),
        "output": "output_germany_hessen_dop20.png",
    },
    {
        "code": "niedersachsen_dop20",
        "center": (52.782, 9.122),
        "output": "output_germany_niedersachsen_dop20.png",
    },
    {
        "code": "thuringia_dop20",
        "center": (50.876, 11.588),
        "output": "output_germany_thuringia_dop20.png",
    },
]


def save_preview(provider_code: str, center: tuple[float, float], output_path: str) -> None:
    result = ImageryProvider.extract_area(
        center=center,
        width_m=512,
        height_m=512,
        provider_code=provider_code,
        min_valid_coverage=0.8,
    )
    preview = result.data.filled(0)
    preview = cv2.cvtColor(preview.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, preview)
    print(f"{provider_code}: saved {output_path}")


for request in REQUESTS:
    try:
        save_preview(
            provider_code=request["code"],
            center=request["center"],
            output_path=request["output"],
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"{request['code']}: failed: {exc}")
