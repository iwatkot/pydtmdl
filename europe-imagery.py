"""Extract small preview images from the newer European imagery providers."""

import cv2

from pydtmdl import ImageryProvider


REQUESTS = [
    {
        "code": "france_bdortho",
        "center": (48.8566, 2.3522),
        "output": "output_france_bdortho.png",
    },
    {
        "code": "spain_pnoa",
        "center": (40.4168, -3.7038),
        "output": "output_spain_pnoa.png",
    },
    {
        "code": "netherlands_luchtfoto_hr",
        "center": (52.3676, 4.9041),
        "output": "output_netherlands_luchtfoto_hr.png",
    },
    {
        "code": "luxembourg_orthophoto",
        "center": (49.6116, 6.1319),
        "output": "output_luxembourg_orthophoto.png",
    },
    {
        "code": "copernicus_vhr_2021",
        "center": (50.0755, 14.4378),
        "output": "output_copernicus_vhr_2021.png",
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