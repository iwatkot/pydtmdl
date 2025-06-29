import cv2

from pydtmdl import DTMProvider

# 1️⃣ Create a virtual environment and install pydtmdl
# python -m venv .venv
# .venv\Scripts\activate # (Windows)
# source .venv/bin/activate # (Linux/Mac)
# pip install pydtmdl

# 2️⃣ Prepare coordinates of the center point and size (in meters).
coords = 45.285460396731374, 20.237491178279715  # Center point of the region of interest.
size = 2048  # Size of the region in meters (2048x2048 m).

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

provider = best_provider(coords, size=size)

# 8️⃣ Get the tile data as a numpy array.
# Remember to handle exceptions if the provider does not have data for the given coordinates.
# In this case, it's recommended to try another provider (from the list of available providers).
try:
    np_data = provider.image
except ValueError as e:
    raise e

# 9️⃣ Implement additional processing if needed.
# For example, you can normalize the data, resize it, or save it to a file.

print(f"Data shape: {np_data.shape}, type: {np_data.dtype}")
print(f"Data min: {np_data.min()}, max: {np_data.max()}")

# Convert to 8-bit unsigned integer and resize for visualization.
np_data = cv2.normalize(np_data, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
np_data = cv2.resize(np_data, (size, size), interpolation=cv2.INTER_LINEAR)

# Save the processed data to a file.
cv2.imwrite("output.png", np_data)
