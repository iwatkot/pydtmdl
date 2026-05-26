"""Extract a small preview image from the Austria basemap.at orthophoto provider."""

import cv2

from pydtmdl import ImageryProvider


CENTER = (48.2082, 16.3738)
OUTPUT_PATH = "output_austria_basemap_orthofoto.png"


result = ImageryProvider.extract_area(
    center=CENTER,
    width_m=512,
    height_m=512,
    provider_code="austria_basemap_orthofoto",
    min_valid_coverage=0.8,
)

preview = result.data.filled(0)
preview = cv2.cvtColor(preview.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
cv2.imwrite(OUTPUT_PATH, preview)
print(f"austria_basemap_orthofoto: saved {OUTPUT_PATH}")