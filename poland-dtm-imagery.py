"""Extract small preview rasters from Polish DTM and orthophoto providers."""

import cv2

from pydtmdl import DTMProvider, ImageryProvider, postprocess_dtm_to_png


CENTER = (52.2297, 21.0122)  # Warsaw
WIDTH_M = 512
HEIGHT_M = 512


def save_imagery_preview() -> None:
    result = ImageryProvider.extract_area(
        center=CENTER,
        width_m=WIDTH_M,
        height_m=HEIGHT_M,
        provider_code="poland_orto_highres",
        min_valid_coverage=0.8,
    )
    preview = result.data.filled(0)
    preview = cv2.cvtColor(preview.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    output_path = "output_poland_orto_highres.png"
    cv2.imwrite(output_path, preview)
    print(f"poland_orto_highres: saved {output_path}")


def save_dtm_preview() -> None:
    result = DTMProvider.extract_area(
        center=CENTER,
        width_m=WIDTH_M,
        height_m=HEIGHT_M,
        provider_code="poland_dtm1m",
        fallback_provider_code="srtm30",
        min_valid_coverage=0.8,
    )
    output_path = "output_poland_dtm1m.png"
    postprocess_dtm_to_png(result.data, output_path, normalize_to_dtype=True)
    print(f"{result.metadata.actual_provider}: saved {output_path}")


for save_preview in (save_imagery_preview, save_dtm_preview):
    try:
        save_preview()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        print(f"{save_preview.__name__}: failed: {exc}")
