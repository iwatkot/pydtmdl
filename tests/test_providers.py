import os
import shutil

import numpy as np

from pydtmdl import DTMProvider

non_base_providers = DTMProvider.get_non_base_providers()

# Remove the tiles directory before running tests.
tiles_directory = os.path.join(os.getcwd(), "tiles")
if os.path.isdir(tiles_directory):
    shutil.rmtree(tiles_directory)
os.makedirs(tiles_directory, exist_ok=True)


COORDINATE_CASES = {
    # "arctic": (70.0, -50.0),  # Northern Greenland # * Too big tiles for tests.
    "baden": (48.5, 9.0),  # Stuttgart area, Germany
    "bavaria": (48.0, 11.5),  # Munich area, Germany
    "canada": (50.0, -100.0),  # Central Canada
    "czech_dmr5g": (50.0, 14.5),  # Prague area, Czech Republic
    "czech": (50.0, 14.5),  # Prague area, Czech Republic
    "england1m": (52.5, -1.5),  # Birmingham area, England
    "flanders": (51.0, 4.5),  # Brussels area, Belgium
    "france": (48.8, 2.3),  # Paris area, France
    "hessen": (50.5, 9.0),  # Kassel area, Germany
    "italy": (42.0, 12.5),  # Rome area, Italy
    "lithuania": (55.0, 24.0),  # Vilnius area, Lithuania
    # "niedersachsen": (52.5, 9.5),  # Hannover area, Germany # ! Not working!
    "norway": (60.0, 10.0),  # Oslo area, Norway
    "NRW": (51.5, 7.5),  # Dortmund area, Germany
    # "rema": (-75.0, 0.0),  # Antarctica # * Too big tiles for tests.
    "sachsen-anhalt": (51.5, 12.0),  # Halle area, Germany
    "spain": (40.4, -3.7),  # Madrid area, Spain
    "srtm30": (45.5, 10.0),  # Northern Italy - global coverage
    "thuringia": (51.0, 11.0),  # Erfurt area, Germany
    "usgs_wcs": (40.0, -105.0),  # Colorado, USA
    "wales": (52.0, -3.5),  # Central Wales
}
SIZE_CASES = [1024]


def verify_provder_output(array: np.ndarray, provider_name: str):
    """Verify the output of a provider.

    Arguments:
        array (np.ndarray): The output array from the provider.
        provider_name (str): The name of the provider.

    Raises:
        AssertionError: If any of the checks fail.
    """
    # 1. Check that array is not empty (not all zeros).
    assert np.any(array != 0), f"{provider_name}: Array is empty (all zeros)"

    # 2. Check if the array at least has a 16-bit depth (signed or unsigned, int or float)
    assert array.dtype in [
        np.int16,
        np.uint16,
        np.float16,
        np.float32,
        np.int32,
        np.uint32,
        np.float64,
    ], f"{provider_name}: Expected 16-bit or higher depth, got {array.dtype}"

    print(f"{provider_name}: Passed all tests.")


def get_all_providers_without_settings() -> list[DTMProvider]:
    """Get all providers without settings.

    Returns:
        list[DTMProvider]: List of providers without settings.
    """
    providers_without_settings = []
    for provider in non_base_providers:
        settings_json = provider._settings().model_dump()
        if not settings_json:
            providers_without_settings.append(provider)
    return providers_without_settings


def test_all_providers():
    """Test all providers without settings."""
    providers_without_settings = get_all_providers_without_settings()
    for provider in providers_without_settings:
        coordinate_case = COORDINATE_CASES.get(provider._code)
        print(f"Testing provider: {provider._code}, coordinate case: {coordinate_case}")
        if not coordinate_case:
            print(f"Skipping provider {provider._code} as no coordinate case is defined.")
            continue

        for size in SIZE_CASES:
            testing_provider = provider(coordinate_case, size=size)

            array = testing_provider.get_numpy()
            verify_provder_output(array, provider._name)
            print(f"{provider._name}: Test passed for size {size}.")

    print("All providers without settings passed the tests.")


test_all_providers()
