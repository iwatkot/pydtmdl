import cv2

from pydtmdl import DTMProvider, ImageryProvider
from pydtmdl.imagery_providers.sentinel2 import Sentinel2L2AImagerySettings

# 1️⃣ Create a virtual environment and install pydtmdl
# python -m venv .venv
# .venv\Scripts\activate # (Windows)
# source .venv/bin/activate # (Linux/Mac)
# pip install pydtmdl

# 2️⃣ Prepare coordinates of the center point and size (in meters).
coords = 45.285460396731374, 20.237491178279715  # Center point of the region of interest.
size = 6144  # Size of the region in meters (2048x2048 m).
rotation = 25


# 3️⃣ Option 1: Get the list of available providers for the given coordinates.
providers = DTMProvider.get_list(coords)
print(f"Found {len(providers)} providers:")
for provider in providers:
    print(f" - {provider.name()}")

# 4️⃣ Option 2: Get the best provider for the given coordinates.
best_provider = DTMProvider.get_best(coords)
print(f"Best provider: {best_provider.name()}")

# 5️⃣ Option 3: Get the best provider by code (if you know the code).
# provider_code = "srtm30"
# provider = DTMProvider.get_provider_by_code(provider_code)

# 6️⃣ Ensure that provider does not require additional settings.
if best_provider.settings_required():
    print(f"Provider {best_provider.name()} requires additional settings.")

    settings = best_provider.settings()
    print(f"Settings: {settings}")

# 7️⃣ Create an instance of the provider with the given coordinates and size.
# Optional: you can specify custom directory for temporary files (tile cache).
# Optional: you can provide a custom logger.
# provider = best_provider(coords, size=size, temp_dir="temp", logger=my_logger)
# If provider requires settings, you can pass them as well.
# provider = best_provider(coords, size=size, settings=settings)

provider = best_provider(coords, size=size, rotation_deg=rotation)

# 8️⃣ Get the DTM data as a numpy array.
# Remember to handle exceptions if the provider does not have data for the given coordinates.
# In this case, it's recommended to try another provider (from the list of available providers).
try:
    np_data = provider.get_numpy()
except ValueError as e:
    raise e

# 9️⃣ Implement additional processing if needed.
# For example, you can normalize the data, resize it, or save it to a file.

print(f"Data shape: {np_data.shape}, type: {np_data.dtype}")
print(f"Data min: {np_data.min()}, max: {np_data.max()}")

# Optional: use the high-level rectangular ROI API with structured metadata.
result = DTMProvider.extract_area(
    center=coords,
    width_m=4096,
    height_m=2048,
    rotation_deg=30,
    provider_code=best_provider.code(),
    fallback_provider_code="srtm30",
)
print(result.metadata.model_dump())

# Convert to 8-bit unsigned integer and resize for visualization.
np_data = cv2.normalize(np_data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
np_data = cv2.resize(np_data, (size, size), interpolation=cv2.INTER_LINEAR)
roi_preview = cv2.normalize(result.data.filled(0), None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

# Optional: extract Sentinel-2 RGB imagery for the same ROI.
imagery_result = ImageryProvider.extract_area(
    center=coords,
    width_m=4096,
    height_m=2048,
    rotation_deg=30,
    provider_code="sentinel2_l2a",
    # user_settings=Sentinel2L2AImagerySettings(
    #     date_from="2024-01-01",
    #     date_to="2024-12-31",
    #     max_cloud_cover=15,
    #     max_items=6,
    # ),
)
print(imagery_result.metadata.model_dump())

imagery_preview = imagery_result.data.filled(0)
imagery_preview = cv2.cvtColor(imagery_preview.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)

# Save the processed data to a file.
cv2.imwrite("output.png", np_data)
cv2.imwrite("output_roi.png", roi_preview)
cv2.imwrite("output_satellite.png", imagery_preview)
