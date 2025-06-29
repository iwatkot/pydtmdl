<p align="center">
<a href="https://github.com/iwatkot/maps4fs">maps4fs</a> •
<a href="https://github.com/iwatkot/maps4fsui">maps4fs UI</a> •
<a href="https://github.com/iwatkot/maps4fsapi">maps4fs API</a> •
<a href="https://github.com/iwatkot/maps4fsstats">maps4fs Stats</a> •
<a href="https://github.com/iwatkot/maps4fsbot">maps4fs Bot</a><br>
<a href="https://github.com/iwatkot/pygmdl">pygmdl</a> •
<a href="https://github.com/iwatkot/pydtmdl">pydtmdl</a>
</p>

<div align="center" markdown>
<img src="https://github.com/iwatkot/pydtmdl/releases/download/0.0.1/pydtmdl.png">
</a>

<p align="center">
    <a href="#Quick-Start">Quick Start</a> •
    <a href="#Overview">Overview</a> • 
    <a href="#What-is-a-DTM?">What is a DTM?</a> •
    <a href="#Supported-DTM-providers">Supported DTM providers</a> •
    <a href="#Contributing">Contributing</a>
</p>

[![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/iwatkot/pydtmdl)](https://github.com/iwatkot/pydtmdl/releases)
[![PyPI - Version](https://img.shields.io/pypi/v/pydtmdl)](https://pypi.org/project/pydtmdl)
[![GitHub issues](https://img.shields.io/github/issues/iwatkot/pydtmdl)](https://github.com/iwatkot/pydtmdl/issues)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/pydtmdl)](https://pypi.org/project/pydtmdl)
[![Checked with mypy](https://www.mypy-lang.org/static/mypy_badge.svg)](https://mypy-lang.org/)
[![Build Status](https://github.com/iwatkot/pydtmdl/actions/workflows/checks.yml/badge.svg)](https://github.com/iwatkot/pydtmdl/actions)
[![GitHub Repo stars](https://img.shields.io/github/stars/iwatkot/pydtmdl)](https://github.com/iwatkot/pydtmdl/stargazers)<br>

</div>

## Quick Start
Install the package using pip:

```bash
pip install pydtmdl
```

Then, you can use it in your Python scripts:

```python
from pydtmdl import DTMProvider

# Prepare coordinates of the center point and size (in meters).
coords = 45.285460396731374, 20.237491178279715  # Center point of the region of interest.
size = 2048  # Size of the region in meters (2048x2048 m).

# Get the best provider for the given coordinates.
best_provider = DTMProvider.get_best(coords)
print(f"Best provider: {best_provider.name()}")

# Create an instance of the provider with the given coordinates and size.
provider = best_provider(coords, size=size)

# Get the DTM data as a numpy array.
np_data = provider.image
```

## Overview
`pydtmdl` is a Python library designed to provide access to Digital Terrain Models (DTMs) from various providers. It supports multiple providers, each with its own resolution and data format. The library allows users to easily retrieve DTM data for specific geographic coordinates and sizes.  

Note, that some providers may require additional settings, such as API keys or selection of a specific dataset. More details can be found in the demo script and in the providers source code.  

The library will retrieve all the required tiles, merge them, window them and return the result as a numpy array. If additional processing is required, such as normalization or resizing, it can be done using OpenCV or other libraries (example code is provided in the demo script).

## What is a DTM?

First of all, it's important to understand what a DTM is.  
There are two main types of elevation models: Digital Terrain Model (DTM) and Digital Surface Model (DSM). The DTM represents the bare earth surface without any objects like buildings or vegetation. The DSM, on the other hand, represents the earth's surface including all objects.

![DTM vs DSM, example 1](https://github.com/user-attachments/assets/0bf691f3-6737-4663-86ca-c17a525ecda4)

![DTM vs DSM, example 2](https://github.com/user-attachments/assets/3ae1082c-1117-4073-ac98-a2bc1e22c1ba)

The library is focused on the DTM data and the DSM sources are not supported and will not be added in the future. The reason for this is that the DTM data is more suitable for terrain generation in games, as it provides a more accurate representation of the earth's surface without any objects.

## Supported DTM providers

![coverage map](https://github.com/user-attachments/assets/be5c5ce1-7318-4352-97eb-efba7ec587cd)

In addition to SRTM 30m, which provides global coverage, the map above highlights all countries and/or regions where higher resolution coverage is provided by one of the DTM providers.

| Provider Name                      | Resolution   | Developer                                   |
| ---------------------------------- | ------------ | ------------------------------------------- |
| 🌎 SRTM30                          | 30 meters    | [iwatkot](https://github.com/iwatkot)       |
| 🌎 ArcticDEM                       | 2 meters     | [kbrandwijk](https://github.com/kbrandwijk) |
| 🌎 REMA Antarctica                 | 2 meters     | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇺🇸 USGS                            | 1-90 meters  | [ZenJakey](https://github.com/ZenJakey)     |
| 🏴󠁧󠁢󠁥󠁮󠁧󠁿 England                         | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🏴󠁧󠁢󠁳󠁣󠁴󠁿 Scotland                        | 0.25-1 meter | [kbrandwijk](https://github.com/kbrandwijk) |
| 🏴󠁧󠁢󠁷󠁬󠁳󠁿󠁧󠁢󠁷󠁬󠁳󠁿 Wales                           | 1 meter      | [garnwenshared](https://github.com/garnshared) |
| 🇩🇪 Hessen, Germany                 | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇩🇪 Niedersachsen, Germany          | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇩🇪 Bayern, Germany                 | 1 meter      | [H4rdB4se](https://github.com/H4rdB4se)     |
| 🇩🇪 Nordrhein-Westfalen, Germany    | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇩🇪 Mecklenburg-Vorpommern, Germany | 1-25 meter   | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇩🇪 Baden-Württemberg, Germany      | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇩🇪 Sachsen-Anhalt, Germany         | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇩🇪 Thüringen, Germany              | 1 meter      | [H4rdB4se](https://github.com/H4rdB4se)     |
| 🇨🇦 Canada                          | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇧🇪 Flanders, Belgium               | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇫🇷 France                          | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇮🇹 Italy                           | 10 meter     | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇳🇴 Norway                          | 1 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇪🇸 Spain                           | 5 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇫🇮 Finland                         | 2 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇩🇰 Denmark                         | 0.4 meter    | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇨🇭 Switzerland                     | 0.5-2 meter  | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇨🇿 Czech Republic                  | 5 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇱🇹 Lithuania                       | 1 meter      | [Tox3](https://github.com/Tox3) |


## Contributing

Contributions are welcome! If you want to add your own DTM provider, please follow this guide.  
You can also contribute by reporting issues, suggesting improvements, or helping with documentation.
### What a DTM provider does?

A DTM provider is a service that provides elevation data for a given location. While there's plenty of DTM providers available, only the ones that provide a free and open access to their data can be used in this library.  

The base provider class, [DTMProvider](pydtmdl/base/dtm.py) that all DTM providers inherit from, is responsible for all processing of DEM data. Individual DTM providers are responsible only for downloading the DTM tile(s) for the area.

The process for generating the elevation data is:

- Download all DTM tiles for the desired map area (implemented by each DTM provider)
- If the DTM provider downloaded multiple tiles, merge these tiles into one
- If the tile uses a different projection, reproject it to EPSG:4326, which is used for all other data (like OSM)
- Extract the map area from the tile (some providers, like SRTM, return big tiles that are larger than just the desired area)

### How to implement a DTM provider?

So the DTM provider is a simple class, that receives coordinate of the center point, the size of the region of interest and should download all the needed DTM tiles and return a numpy array with the elevation data.

### Example of a DTM provider

➡️ Existing providers can be found in the [providers](pydtmdl/providers/) folder.

Let's take a look at an example of a DTM provider implementation.

**Step 1:** define description of the provider.

```python
class SRTM30Provider(DTMProvider):
    """Provider of Shuttle Radar Topography Mission (SRTM) 30m data."""

    _code = "srtm30"
    _name = "SRTM 30 m"
    _region = "Global"
    _icon = "🌎"
    _resolution = 30.0

    _url = "https://elevation-tiles-prod.s3.amazonaws.com/skadi/{latitude_band}/{tile_name}.hgt.gz"

    _instructions = "When working with SRTM provider..."
```

So, we inherit from the `DTMProvider` class, add some properties to identify the Provider (such as code and region). The most important part is the `_url` property, which is a template for the URL to download the elevation data. But if your provider uses some other approach, you can reimplement related methods.

If you want some additional information or guides you can set the `_instructions` property.

**Step 2 (optional):** use the `DTMProviderSetting` class to define your own settings (if needed).

```python
class SRTM30ProviderSettings(DTMProviderSettings):
    """Settings for the SRTM 30 m provider."""

    enable_something: bool = True
    input_something: int = 255
```

Also, you will need to add a new `_settings` property to the provider class.

```python
class SRTM30Provider(DTMProvider):
    ...
    _settings = SRTM30ProviderSettings
```

If those are provided you'll later be able to use the `user_settings` property to access the settings. In the example it would look like this:

```python
enable_something = self.user_settings.enable_something
input_something = self.user_settings.input_something
```

**Step 3:** implement the `download_tiles` method.

```python
    def download_tiles(self):
        """Download SRTM tiles."""
        north, south, east, west = self.get_bbox()

        tiles = []
        # Look at each corner of the bbox in case the bbox spans across multiple tiles
        for pair in [(north, east), (south, west), (south, east), (north, west)]:
            tile_parameters = self.get_tile_parameters(*pair)
            tile_name = tile_parameters["tile_name"]
            decompressed_tile_path = os.path.join(self.hgt_directory, f"{tile_name}.hgt")

            if not os.path.isfile(decompressed_tile_path):
                compressed_tile_path = os.path.join(self.gz_directory, f"{tile_name}.hgt.gz")
                if not self.get_or_download_tile(compressed_tile_path, **tile_parameters):
                    raise FileNotFoundError(f"Tile {tile_name} not found.")

                with gzip.open(compressed_tile_path, "rb") as f_in:
                    with open(decompressed_tile_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
            tiles.append(decompressed_tile_path)

        return tiles
```

This method uses the helper method `get_bbox` to get the coordinates of the bounding box of the map area. If your DTM provider requires you to provide the coordinates in a different projection, you need to make sure you convert. For an example of this, see the `transform_bbox` method in [nrw.py](pydtmdl/providers/nrw.py).
Then, it determines which tiles are needed, downloads them all to a temporary folder and extracts them. The base class provides a `_tile_directory` property for convenience that points to a temp folder for your provider.
Finally, it returns a list of file paths to the downloaded tiles.

As you can see, it's pretty simple to implement a DTM provider. You can use any source of elevation data, as long as it's free and open.
NOTE: If a DTM Provider requires an API key, paid subscription, or any other form of payment, you will be fully responsible for setting up your own access to the provider. The provider in the library will expose the settings needed to provide your authentication key or other required information.


