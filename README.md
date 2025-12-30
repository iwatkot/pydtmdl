<div align="center" markdown>

[![Maps4FS](https://img.shields.io/badge/maps4fs-gray?style=for-the-badge)](https://github.com/iwatkot/maps4fs)
[![Maps4FS](https://img.shields.io/badge/atlasfs-green?style=for-the-badge)](https://github.com/iwatkot/atlasfs)
[![Maps4FS](https://img.shields.io/badge/maps4fs-windows-purple?style=for-the-badge)](https://github.com/iwatkot/maps4fswindows)
[![PYDTMDL](https://img.shields.io/badge/pydtmdl-blue?style=for-the-badge)](https://github.com/iwatkot/pydtmdl)
[![PYGDMDL](https://img.shields.io/badge/pygmdl-teal?style=for-the-badge)](https://github.com/iwatkot/pygmdl)  
[![Maps4FS API](https://img.shields.io/badge/maps4fs-api-green?style=for-the-badge)](https://github.com/iwatkot/maps4fsapi)
[![Maps4FS UI](https://img.shields.io/badge/maps4fs-ui-blue?style=for-the-badge)](https://github.com/iwatkot/maps4fsui)
[![Maps4FS Data](https://img.shields.io/badge/maps4fs-data-orange?style=for-the-badge)](https://github.com/iwatkot/maps4fsdata)
[![Maps4FS ChromaDocs](https://img.shields.io/badge/maps4fs-chromadocs-orange?style=for-the-badge)](https://github.com/iwatkot/maps4fschromadocs)  
[![Maps4FS Upgrader](https://img.shields.io/badge/maps4fs-upgrader-yellow?style=for-the-badge)](https://github.com/iwatkot/maps4fsupgrader)
[![Maps4FS Stats](https://img.shields.io/badge/maps4fs-stats-red?style=for-the-badge)](https://github.com/iwatkot/maps4fsstats)
[![Maps4FS Bot](https://img.shields.io/badge/maps4fs-bot-teal?style=for-the-badge)](https://github.com/iwatkot/maps4fsbot)

</div>

<div align="center" markdown>
<img src="https://github.com/iwatkot/pydtmdl/releases/download/0.0.1/pydtmdl.png">
</a>

<p align="center">
    <a href="#quick-start">Quick Start</a> •
    <a href="#overview">Overview</a> • 
    <a href="#what-is-a-dtm">What is a DTM?</a> •
    <a href="#supported-dtm-providers">Supported DTM providers</a> •
    <a href="#licensing-and-data-usage">Licensing and Data Usage</a> •
    <a href="#contributing">Contributing</a>
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
| 🇸🇪 Sweden                          | 1 meter      | [GustavPersson](https://github.com/GustavPersson) |
| 🇨🇭 Switzerland                     | 0.5-2 meter  | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇨🇿 Czech Republic                  | 5 meter      | [kbrandwijk](https://github.com/kbrandwijk) |
| 🇨🇿 Czech Republic                  | 2 meter      | [VidhosticeSDK](https://github.com/VidhosticeSDK) |
| 🇱🇹 Lithuania                       | 1 meter      | [Tox3](https://github.com/Tox3) |

## Licensing and Data Usage

⚠️ **Important**: This library provides access to DTM data from various third-party providers. **PyDTMDL does not own, host, or distribute this data**. Each DTM provider has its own licensing terms and usage restrictions.

**It is your responsibility to:**
- Check the license and terms of use for each DTM provider you use
- Ensure compliance with the provider's licensing requirements
- Verify that your use case (commercial, research, personal, etc.) is permitted
- Provide proper attribution when required by the data provider
- Respect any usage limits or restrictions imposed by the provider

The library itself is licensed under the GNU Affero General Public License v3 (AGPL-3.0), but this **does not grant you any rights** to the DTM data accessed through the library. The data licenses are separate and must be obtained directly from the respective providers.

**By using this library, you acknowledge that you are solely responsible for ensuring compliance with all applicable data licenses and terms of use.**

For information about data licensing from specific providers, please refer to their official websites and documentation.

## Contributing

Contributions are welcome! If you want to add your own DTM provider, please follow this guide.  
You can also contribute by reporting issues, suggesting improvements, or helping with documentation.

### What a DTM provider does?

A DTM provider is a service that provides elevation data for a given location. While there's plenty of DTM providers available, only the ones that provide a free and open access to their data can be used in this library.  

The base provider class, [DTMProvider](pydtmdl/base/dtm.py), handles all the heavy lifting: merging tiles, reprojecting to EPSG:4326, and extracting the region of interest. Individual DTM providers only need to implement the `download_tiles()` method to fetch the raw data.

The process for generating elevation data is:

1. Download all DTM tiles for the desired map area (implemented by each DTM provider)
2. Merge multiple tiles if necessary (handled by base class)
3. Reproject to EPSG:4326 if needed (handled by base class)
4. Extract the map area from the tile (handled by base class)

### Provider Types

There are three main approaches to implementing a DTM provider:

1. **Custom implementation** - Inherit from `DTMProvider` directly for unique APIs
2. **WCS-based** - Inherit from both `WCSProvider` and `DTMProvider` for OGC WCS services
3. **WMS-based** - Inherit from both `WMSProvider` and `DTMProvider` for OGC WMS services

### Example 1: Custom Provider (SRTM)

➡️ Existing providers can be found in the [pydtmdl/providers/](pydtmdl/providers/) folder.

**Step 1:** Define the provider metadata.

```python
from pydtmdl.base.dtm import DTMProvider

class SRTM30Provider(DTMProvider):
    """Provider of Shuttle Radar Topography Mission (SRTM) 30m data."""

    _code = "srtm30"
    _name = "SRTM 30 m"
    _region = "Global"
    _icon = "🌎"
    _resolution = 30.0
    _url = "https://elevation-tiles-prod.s3.amazonaws.com/skadi/{latitude_band}/{tile_name}.hgt.gz"
```

**Step 2 (optional):** Define custom settings if your provider requires authentication or configuration.

```python
from pydtmdl.base.dtm import DTMProviderSettings

class SwedenProviderSettings(DTMProviderSettings):
    """Settings for the Sweden provider."""
    username: str = ""
    password: str = ""

class SwedenProvider(DTMProvider):
    _settings = SwedenProviderSettings
    _instructions = "ℹ️ This provider requires username and password..."
```

Access settings in your code:
```python
username = self.user_settings.username
password = self.user_settings.password
```

**Step 3:** Implement the `download_tiles()` method.

```python
def download_tiles(self) -> list[str]:
    """Download SRTM tiles."""
    north, south, east, west = self.get_bbox()
    
    tiles = []
    for pair in [(north, east), (south, west), (south, east), (north, west)]:
        tile_parameters = self.get_tile_parameters(*pair)
        tile_name = tile_parameters["tile_name"]
        tile_path = os.path.join(self.hgt_directory, f"{tile_name}.hgt")
        
        if not os.path.isfile(tile_path):
            # Download and decompress tile
            compressed_path = os.path.join(self.gz_directory, f"{tile_name}.hgt.gz")
            if not self.download_tile(compressed_path, **tile_parameters):
                raise FileNotFoundError(f"Tile {tile_name} not found.")
            # ... decompress logic ...
        
        tiles.append(tile_path)
    return list(set(tiles))
```

### Example 2: WCS Provider (England)

For WCS-based providers, inherit from both `WCSProvider` and `DTMProvider`. The base class handles coordinate transformation automatically using the `transform_bbox()` utility from [pydtmdl/utils.py](pydtmdl/utils.py).

```python
from pydtmdl.base.dtm import DTMProvider
from pydtmdl.base.wcs import WCSProvider

class England1MProvider(WCSProvider, DTMProvider):
    """Provider of England data."""
    
    _code = "england1m"
    _name = "England DGM1"
    _region = "UK"
    _icon = "🏴󠁧󠁢󠁥󠁮󠁧󠁿"
    _resolution = 1.0
    _extents = [(55.877, 49.851, 2.084, -7.105)]
    
    _url = "https://environment.data.gov.uk/geoservices/datasets/.../wcs"
    _wcs_version = "2.0.1"
    _source_crs = "EPSG:27700"  # British National Grid
    _tile_size = 1000
    
    def get_wcs_parameters(self, tile):
        return {
            "identifier": ["dataset_id"],
            "subsets": [("E", str(tile[1]), str(tile[3])), ("N", str(tile[0]), str(tile[2]))],
            "format": "tiff",
        }
```

The `WCSProvider` base class automatically:
- Transforms your bbox from EPSG:4326 to `_source_crs` 
- Tiles the area based on `_tile_size`
- Downloads each tile using your `get_wcs_parameters()` method
- Returns the list of downloaded files

### Example 3: Custom API with Authentication (Sweden)

For providers with custom APIs requiring authentication:

```python
class SwedenProvider(DTMProvider):
    _settings = SwedenProviderSettings  # Define custom settings
    
    def download_tiles(self):
        """Download tiles from STAC API."""
        download_urls = self.get_download_urls()
        return self.download_tif_files(download_urls, self.shared_tiff_path)
    
    def _get_auth_headers(self) -> dict[str, str]:
        """Generate auth headers from user settings."""
        credentials = f"{self.user_settings.username}:{self.user_settings.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded}"}
    
    def get_download_urls(self) -> list[str]:
        """Query STAC API for tile URLs within bbox."""
        bbox = self.get_bbox()
        # ... API logic using self._get_auth_headers() ...
        return urls
```

### Unified Download Methods

⚠️ **Important**: All DTM providers **must use** the unified download methods provided by the base `DTMProvider` class. Do not implement your own download logic.

The base class provides three unified download methods with built-in retry logic, error handling, and progress tracking:

#### 1. `download_tif_files(urls, output_path, headers=None, timeout=60)`
For downloading multiple GeoTIFF files from a list of URLs.

```python
def download_tiles(self) -> list[str]:
    download_urls = self.get_download_urls()
    return self.download_tif_files(download_urls, self.shared_tiff_path)
```

**Use for**: Simple URL-based downloads (SRTM, Scotland, Wales, etc.)

#### 2. `download_file(url, output_path, headers=None, method='GET', data=None, timeout=60)`
For downloading a single file with flexible HTTP methods (GET/POST).

```python
def download_tiles(self) -> list[str]:
    url = self.formatted_url(**tile_parameters)
    output_path = os.path.join(self._tile_directory, "tile.tif")
    self.download_file(url, output_path, method="POST", data=polygon_data)
    return [output_path]
```

**Use for**: Single file downloads or POST requests (Bavaria, custom APIs)

#### 3. `download_tiles_with_fetcher(tiles, output_path, data_fetcher, file_name_generator=None)`
For OGC Web Services (WCS/WMS) or any service requiring custom data fetching.

```python
def download_tiles(self) -> list[str]:
    bbox = self.get_bbox()
    bbox = transform_bbox(bbox, self._source_crs)
    tiles = tile_bbox(bbox, self._tile_size)
    
    wcs = WebCoverageService(self._url, version=self._wcs_version)
    
    def wcs_fetcher(tile):
        return wcs.getCoverage(**self.get_wcs_parameters(tile))
    
    return self.download_tiles_with_fetcher(tiles, self.shared_tiff_path, wcs_fetcher)
```

**Use for**: WCS/WMS providers (automatically handled by `WCSProvider`/`WMSProvider` base classes)

#### Why Use Unified Methods?

- ✅ **Built-in retry logic** - Automatic retries with configurable attempts and delays
- ✅ **Error handling** - Consistent error messages and logging
- ✅ **Progress tracking** - Visual progress bars with tqdm
- ✅ **File caching** - Skips already downloaded files
- ✅ **Timeout support** - Configurable timeouts for slow connections
- ✅ **Authentication** - Support for custom headers (API keys, Basic Auth, etc.)

**If you need functionality not provided by these methods, extend the base class methods rather than implementing your own. This ensures all providers benefit from improvements and bug fixes.**

### Other Helper Methods

The base `DTMProvider` class also provides:

- `get_bbox()` - Returns `(north, south, east, west)` in EPSG:4326
- `unzip_img_from_tif(file_name, output_path)` - Extracts .img or .tif from zip files
- `_tile_directory` - Temporary directory for your provider's tiles
- `_max_retries` - Number of retry attempts (default: 5)
- `_retry_pause` - Seconds between retries (default: 5)

For coordinate transformation, use the utility function from [pydtmdl/utils.py](pydtmdl/utils.py):
```python
from pydtmdl.utils import transform_bbox
bbox = self.get_bbox()
transformed_bbox = transform_bbox(bbox, "EPSG:25832")
```

### Requirements

- Providers must be free and openly accessible
- If authentication is required, users must provide their own credentials via settings
- The `download_tiles()` method must return a list of file paths to GeoTIFF files
- All tiles should contain valid elevation data readable by `rasterio`


