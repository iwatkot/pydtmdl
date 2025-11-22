"""This module contains the DTMProvider class and its subclasses. DTMProvider class is used to
define different providers of digital terrain models (DTM) data. Each provider has its own URL
and specific settings for downloading and processing the data."""

from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Type
from zipfile import ZipFile

import numpy as np
import osmnx as ox
import rasterio
import requests
from pydantic import BaseModel
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from requests.exceptions import RequestException
from tqdm import tqdm


class DTMProviderSettings(BaseModel):
    """Base class for DTM provider settings models."""


class DTMProvider(ABC):
    """Base class for DTM providers."""

    _code: str | None = None
    _name: str | None = None
    _region: str | None = None
    _icon: str | None = None
    _resolution: float | None = None

    _url: str | None = None

    _settings: Type[DTMProviderSettings] | None = DTMProviderSettings

    """Bounding box of the provider in the format (north, south, east, west)."""
    _extents: list[tuple[float, float, float, float]] | None = None

    _instructions: str | None = None

    _unreliable: bool = False

    _max_retries: int = 5
    _retry_pause: int = 5

    def __init__(
        self,
        coordinates: tuple[float, float],
        size: int,
        user_settings: DTMProviderSettings | None = None,
        directory: str = os.path.join(os.getcwd(), "tiles"),
        logger: Any = logging.getLogger(__name__),
    ):
        self._coordinates = coordinates
        self._user_settings = user_settings
        self._size = size

        if not self._code:
            raise ValueError("Provider code must be defined.")
        self._tile_directory = os.path.join(directory, self._code)
        os.makedirs(self._tile_directory, exist_ok=True)

        self.logger = logger

    @classmethod
    def resolution(cls) -> float | None:
        """Resolution of the provider in meters per pixel.

        Returns:
            float: Provider resolution.
        """
        return cls._resolution

    @classmethod
    def unreliable(cls) -> bool:
        """Check if the provider is unreliable.

        Returns:
            bool: True if the provider is unreliable, False otherwise.
        """
        return cls._unreliable

    @classmethod
    def name(cls) -> str | None:
        """Name of the provider.

        Returns:
            str: Provider name.
        """
        return cls._name

    @classmethod
    def code(cls) -> str | None:
        """Code of the provider.

        Returns:
            str: Provider code.
        """
        return cls._code

    @property
    def coordinates(self) -> tuple[float, float]:
        """Coordinates of the center point of the DTM data.

        Returns:
            tuple[float, float]: Coordinates of the center point of the DTM data.
        """
        return self._coordinates

    @property
    def size(self) -> int:
        """Size of the DTM data in meters.

        Returns:
            int: Size of the DTM data.
        """
        return self._size

    @property
    def url(self) -> str | None:
        """URL of the provider.

        Returns:
            str: URL of the provider or None if not defined.
        """
        return self._url

    def formatted_url(self, **kwargs) -> str:
        """Formatted URL of the provider.

        Arguments:
            **kwargs: Keyword arguments to format the URL.

        Returns:
            str: Formatted URL of the provider.
        """
        if not self.url:
            raise ValueError("URL must be defined.")
        return self.url.format(**kwargs)

    @classmethod
    def settings(cls) -> Type[DTMProviderSettings] | None:
        """Settings model of the provider.

        Returns:
            Type[DTMProviderSettings]: Settings model of the provider.
        """
        return cls._settings

    @classmethod
    def settings_required(cls) -> bool:
        """Check if the provider requires user settings.

        Returns:
            bool: True if the provider requires user settings, False otherwise.
        """
        return cls._settings is not None and cls._settings != DTMProviderSettings

    @classmethod
    def instructions(cls) -> str | None:
        """Instructions for using the provider.

        Returns:
            str: Instructions for using the provider.
        """
        return cls._instructions

    @property
    def user_settings(self) -> DTMProviderSettings | None:
        """User settings of the provider.

        Returns:
            DTMProviderSettings: User settings of the provider.
        """
        return self._user_settings

    @classmethod
    def description(cls) -> str:
        """Description of the provider.

        Returns:
            str: Provider description.
        """
        return f"{cls._icon} {cls._region} [{cls._resolution} m/px] {cls._name}"

    @classmethod
    def get_provider_by_code(cls, code: str) -> Type[DTMProvider] | None:
        """Get a provider by its code.

        Arguments:
            code (str): Provider code.

        Returns:
            DTMProvider: Provider class or None if not found.
        """
        for provider in cls.__subclasses__():
            if provider.code() == code:
                return provider
        return None

    @classmethod
    def get_provider_by_name(cls, name: str) -> Type[DTMProvider] | None:
        """Get a provider by its name.

        Arguments:
            name (str): Provider name.

        Returns:
            DTMProvider: Provider class or None if not found.
        """
        for provider in cls.__subclasses__():
            if provider.name() == name:
                return provider
        return None

    @classmethod
    def get_valid_provider_descriptions(
        cls, lat_lon: tuple[float, float], default_code: str = "srtm30"
    ) -> dict[str, str]:
        """Get descriptions of all providers, where keys are provider codes and
        values are provider descriptions.

        Arguments:
            lat_lon (tuple): Latitude and longitude of the center point.
            default_code (str): Default provider code.

        Returns:
            dict: Provider descriptions.
        """
        providers: dict[str, str] = {}
        for provider in cls.get_non_base_providers():
            if provider.inside_bounding_box(lat_lon):
                code = provider.code()
                if code is not None:
                    providers[code] = provider.description()

        # Sort the dictionary, to make sure that the default provider is the first one.
        providers = dict(sorted(providers.items(), key=lambda item: item[0] != default_code))

        return providers

    @classmethod
    def get_non_base_providers(cls) -> list[Type[DTMProvider]]:
        """Get all non-base providers.

        Returns:
            list: List of non-base provider classes.
        """
        from pydtmdl.base.wcs import WCSProvider
        from pydtmdl.base.wms import WMSProvider

        base_providers = [WCSProvider, WMSProvider]

        return [provider for provider in cls.__subclasses__() if provider not in base_providers]

    @classmethod
    def get_list(
        cls, lat_lon: tuple[float, float], include_unreliable: bool = False
    ) -> list[Type[DTMProvider]]:
        """Get all providers that can be used for the given coordinates.

        Arguments:
            lat_lon (tuple): Latitude and longitude of the center point.
            include_unreliable (bool): Whether to include unreliable providers.

        Returns:
            list: List of provider classes.
        """
        providers = []
        for provider in cls.get_non_base_providers():
            if provider.inside_bounding_box(lat_lon):
                if not include_unreliable and provider.unreliable():
                    continue
                providers.append(provider)
        return providers

    @classmethod
    def get_best(
        cls, lat_lon: tuple[float, float], default_code: str = "srtm30"
    ) -> Type[DTMProvider] | None:
        """Get the best provider for the given coordinates.

        Arguments:
            lat_lon (tuple): Latitude and longitude of the center point.
            default_code (str): Default provider code.

        Returns:
            DTMProvider: Best provider class or None if not found.
        """
        providers = cls.get_list(lat_lon)
        if not providers:
            return cls.get_provider_by_code(default_code)

        # Sort providers by priority and return the best one
        providers.sort(key=lambda p: p._resolution or float("inf"))
        return providers[0]

    @classmethod
    def inside_bounding_box(cls, lat_lon: tuple[float, float]) -> bool:
        """Check if the coordinates are inside the bounding box of the provider.

        Returns:
            bool: True if the coordinates are inside the bounding box, False otherwise.
        """
        lat, lon = lat_lon
        extents = cls._extents
        if extents is None:
            return True
        for extent in extents:
            if extent[0] >= lat >= extent[1] and extent[2] >= lon >= extent[3]:
                return True
        return False

    @abstractmethod
    def download_tiles(self) -> list[str]:
        """Download tiles from the provider.

        Returns:
            list: List of paths to the downloaded tiles.
        """
        raise NotImplementedError

    def get_numpy(self) -> np.ndarray:
        """Get numpy array of the tile.
        Resulting array must be 16 bit (signed or unsigned) integer, and it should be already
        windowed to the bounding box of ROI. It also must have only one channel.

        Raises:
            RuntimeError: If downloading tiles failed.
            ValueError: If no tiles were downloaded from the provider.

        Returns:
            np.ndarray: Numpy array of the tile.
        """
        # download tiles using DTM provider implementation

        try:
            tiles = self.download_tiles()
        except RequestException as e:
            error_message = (
                "Failed to download tiles from DTM provider servers. "
                "It's probably happening because the requested area is outside of the provider's "
                "coverage area or the provider's servers are currently unavailable. "
                "Please check the provider's coverage and ensure that the coordinates you specified "
                "are inside the coverage area. "
                "You can also try different providers."
            )
            self.logger.error(f"Error while downloading tiles: {e}")
            raise RuntimeError(error_message) from e
        self.logger.debug("Downloaded tiles: %s", tiles)

        if not tiles:
            error_message = (
                "No tiles were downloaded from the provider. "
                "The coordinates you provided are outside the coverage area of this provider, "
                "and the provider does not have data for this area. "
                "Try using a different provider."
            )
            self.logger.error(error_message)
            raise ValueError(error_message)

        # merge tiles if necessary
        if len(tiles) > 1:
            self.logger.debug("Multiple tiles downloaded. Merging tiles")
            tile, _ = self.merge_geotiff(tiles)
        else:
            tile = tiles[0]

        # determine CRS of the resulting tile and reproject if necessary
        with rasterio.open(tile) as src:
            crs = src.crs
        if crs != "EPSG:4326":
            self.logger.debug("Reprojecting GeoTIFF from %s to EPSG:4326...", crs)
            tile = self.reproject_geotiff(tile)

        # extract region of interest from the tile
        data = self.extract_roi(tile)

        return data

    @property
    def image(self) -> np.ndarray:
        """Get numpy array of the tile and check if it contains any data.

        Returns:
            np.ndarray: Numpy array of the tile.

        Raises:
            ValueError: If the tile does not contain any data.
        """
        data = self.get_numpy()
        if not np.any(data):
            raise ValueError("No data in the tile. Try different provider.")
        return data

    # region helpers
    def get_bbox(self) -> tuple[float, float, float, float]:
        """Get bounding box of the tile based on the center point and size.

        Returns:
            tuple: Bounding box of the tile (north, south, east, west).
        """
        west, south, east, north = ox.utils_geo.bbox_from_point(  # type: ignore
            self.coordinates, dist=self.size // 2, project_utm=False
        )
        bbox = float(north), float(south), float(east), float(west)
        return bbox

    def download_tif_files(
        self,
        urls: list[str],
        output_path: str,
        headers: dict[str, str] | None = None,
        timeout: int = 60,
    ) -> list[str]:
        """Download GeoTIFF files from the given URLs.

        Arguments:
            urls (list): List of URLs to download GeoTIFF files from.
            output_path (str): Path to save the downloaded GeoTIFF files.
            headers (dict): Optional HTTP headers for the request (e.g., for authentication).
            timeout (int): Request timeout in seconds. Default is 60.

        Returns:
            list: List of paths to the downloaded GeoTIFF files.
        """
        tif_files: list[str] = []

        existing_file_urls = [
            f for f in urls if os.path.exists(os.path.join(output_path, os.path.basename(f)))
        ]

        for url in existing_file_urls:
            self.logger.debug("File already exists: %s", os.path.basename(url))
            file_name = os.path.basename(url)
            file_path = os.path.join(output_path, file_name)
            if file_name.endswith(".zip"):
                file_path = self.unzip_img_from_tif(file_name, output_path)
            tif_files.append(file_path)

        for url in tqdm(
            (u for u in urls if u not in existing_file_urls),
            desc="Downloading tiles",
            unit="tile",
            initial=len(tif_files),
            total=len(urls),
        ):
            file_name = os.path.basename(url)
            file_path = os.path.join(output_path, file_name)

            # Retry logic
            for attempt in range(self._max_retries):
                try:
                    self.logger.debug(
                        "Retrieving TIFF: %s (attempt %d/%d)",
                        file_name,
                        attempt + 1,
                        self._max_retries,
                    )

                    # Send a GET request to the file URL
                    response = requests.get(url, stream=True, timeout=timeout, headers=headers)
                    response.raise_for_status()  # Raise an error for HTTP status codes 4xx/5xx

                    # Write the content of the response to the file
                    with open(file_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)

                    self.logger.debug("File downloaded successfully: %s", file_path)

                    if file_name.endswith(".zip"):
                        file_path = self.unzip_img_from_tif(file_name, output_path)

                    tif_files.append(file_path)
                    break  # Success, exit retry loop

                except requests.exceptions.RequestException as e:
                    if attempt < self._max_retries - 1:
                        self.logger.warning(
                            "Failed to download file from %s (attempt %d/%d): %s. Retrying in %d seconds...",
                            url,
                            attempt + 1,
                            self._max_retries,
                            e,
                            self._retry_pause,
                        )
                        time.sleep(self._retry_pause)
                    else:
                        self.logger.error(
                            "Failed to download file from %s after %d attempts: %s",
                            url,
                            self._max_retries,
                            e,
                        )
        return tif_files

    def download_file(
        self,
        url: str,
        output_path: str,
        headers: dict[str, str] | None = None,
        method: str = "GET",
        data: str | bytes | None = None,
        timeout: int = 60,
    ) -> bool:
        """Download a single file from a URL with flexible HTTP methods.

        Arguments:
            url (str): URL to download from.
            output_path (str): Path to save the downloaded file.
            headers (dict): Optional HTTP headers for the request.
            method (str): HTTP method to use ('GET' or 'POST'). Default is 'GET'.
            data (str | bytes): Optional data for POST requests.
            timeout (int): Request timeout in seconds. Default is 60.

        Returns:
            bool: True if download was successful, False otherwise.
        """
        # Retry logic
        for attempt in range(self._max_retries):
            try:
                self.logger.debug(
                    "Downloading file from %s to %s (attempt %d/%d)",
                    url,
                    output_path,
                    attempt + 1,
                    self._max_retries,
                )

                if method.upper() == "POST":
                    response = requests.post(
                        url, data=data, headers=headers, stream=True, timeout=timeout
                    )
                else:
                    response = requests.get(url, headers=headers, stream=True, timeout=timeout)

                if response.status_code == 200:
                    with open(output_path, "wb") as file:
                        for chunk in response.iter_content(chunk_size=8192):
                            file.write(chunk)
                    self.logger.debug("File downloaded successfully: %s", output_path)
                    return True

                self.logger.warning(
                    "Download failed. HTTP Status Code: %s for URL: %s",
                    response.status_code,
                    url,
                )

                if attempt < self._max_retries - 1:
                    self.logger.warning("Retrying in %d seconds...", self._retry_pause)
                    time.sleep(self._retry_pause)
                else:
                    self.logger.error(
                        "Failed to download file from %s after %d attempts", url, self._max_retries
                    )
                    return False

            except requests.exceptions.RequestException as e:
                if attempt < self._max_retries - 1:
                    self.logger.warning(
                        "Failed to download file from %s (attempt %d/%d): %s. Retrying in %d seconds...",
                        url,
                        attempt + 1,
                        self._max_retries,
                        e,
                        self._retry_pause,
                    )
                    time.sleep(self._retry_pause)
                else:
                    self.logger.error(
                        "Failed to download file from %s after %d attempts: %s",
                        url,
                        self._max_retries,
                        e,
                    )
                    return False

        return False

    def download_tiles_with_fetcher(
        self,
        tiles: list[tuple[float, float, float, float]],
        output_path: str,
        data_fetcher: Any,
        file_name_generator: Any = None,
    ) -> list[str]:
        """Download tiles using a custom data fetcher function.

        This unified method handles tile downloads for OGC Web Services (WCS/WMS)
        and any other service that requires a custom data fetching mechanism.

        Arguments:
            tiles (list): List of tile bounding boxes to download.
            output_path (str): Path to save the downloaded tiles.
            data_fetcher (callable): Function that takes a tile and returns the binary data.
                Should accept a tile tuple and return bytes-like object.
            file_name_generator (callable): Optional function to generate file names from tiles.
                If None, uses default naming: "{north}_{south}_{east}_{west}.tif"

        Returns:
            list: List of paths to the downloaded files.
        """
        all_tif_files = []

        def default_file_name(tile: tuple[float, float, float, float]) -> str:
            return "_".join(map(str, tile)) + ".tif"

        if file_name_generator is None:
            file_name_generator = default_file_name

        for tile in tqdm(tiles, desc="Downloading tiles with fetcher", unit="tile"):
            file_name = file_name_generator(tile)
            file_path = os.path.join(output_path, file_name)

            if not os.path.exists(file_path):
                # Retry logic
                success = False
                for attempt in range(self._max_retries):
                    try:
                        self.logger.debug(
                            "Fetching tile: %s (attempt %d/%d)",
                            tile,
                            attempt + 1,
                            self._max_retries,
                        )
                        output = data_fetcher(tile)
                        with open(file_path, "wb") as f:
                            f.write(output.read() if hasattr(output, "read") else output)
                        self.logger.debug("Tile downloaded successfully: %s", file_path)
                        success = True
                        break  # Success, exit retry loop
                    except Exception as e:
                        if attempt < self._max_retries - 1:
                            self.logger.warning(
                                "Failed to download tile %s (attempt %d/%d): %s. Retrying in %d seconds...",
                                tile,
                                attempt + 1,
                                self._max_retries,
                                e,
                                self._retry_pause,
                            )
                            time.sleep(self._retry_pause)
                        else:
                            self.logger.error(
                                "Failed to download tile %s after %d attempts: %s",
                                tile,
                                self._max_retries,
                                e,
                            )

                if not success:
                    continue  # Skip this tile if all retries failed
            else:
                self.logger.debug("File already exists: %s", file_name)

            all_tif_files.append(file_path)

        return all_tif_files

    def unzip_img_from_tif(self, file_name: str, output_path: str) -> str:
        """Unpacks the .img file from the zip file.

        Arguments:
            file_name (str): Name of the file to unzip.
            output_path (str): Path to the output directory.

        Returns:
            str: Path to the unzipped file.

        Raises:
            FileNotFoundError: If no .img or .tif file is found in the zip file
        """
        file_path = os.path.join(output_path, file_name)
        img_file_name = file_name.replace(".zip", ".img")
        tif_file_name = file_name.replace(".zip", ".tif")
        img_file_path = os.path.join(output_path, img_file_name)
        tif_file_path = os.path.join(output_path, tif_file_name)
        if os.path.exists(img_file_path):
            self.logger.debug("File already exists: %s", img_file_name)
            return img_file_path
        if os.path.exists(tif_file_path):
            self.logger.debug("File already exists: %s", tif_file_name)
            return tif_file_path
        with ZipFile(file_path, "r") as f_in:
            if img_file_name in f_in.namelist():
                f_in.extract(img_file_name, output_path)
                self.logger.debug("Unzipped file %s to %s", file_name, img_file_name)
                return img_file_path
            if tif_file_name in f_in.namelist():
                f_in.extract(tif_file_name, output_path)
                self.logger.debug("Unzipped file %s to %s", file_name, tif_file_name)
                return tif_file_path
        raise FileNotFoundError("No .img or .tif file found in the zip file.")

    def reproject_geotiff(self, input_tiff: str) -> str:
        """Reproject a GeoTIFF file to a new coordinate reference system (CRS).

        Arguments:
            input_tiff (str): Path to the input GeoTIFF file.

        Returns:
            str: Path to the reprojected GeoTIFF file.
        """
        output_tiff = os.path.join(self._tile_directory, "reprojected.tif")

        # Open the source GeoTIFF
        self.logger.debug("Reprojecting GeoTIFF to EPSG:4326 CRS...")
        with rasterio.open(input_tiff) as src:
            # Get the transform, width, and height of the target CRS
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )

            # Update the metadata for the target GeoTIFF
            kwargs = src.meta.copy()
            kwargs.update(
                {
                    "crs": "EPSG:4326",
                    "transform": transform,
                    "width": width,
                    "height": height,
                    "nodata": None,
                }
            )

            # Open the destination GeoTIFF file and reproject
            with rasterio.open(output_tiff, "w", **kwargs) as dst:
                for i in range(1, src.count + 1):  # Iterate over all raster bands
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs="EPSG:4326",
                        resampling=Resampling.average,  # Choose resampling method
                    )

        self.logger.debug("Reprojected GeoTIFF saved to %s", output_tiff)
        return output_tiff

    def merge_geotiff(self, input_files: list[str]) -> tuple[str, str]:
        """Merge multiple GeoTIFF files into a single GeoTIFF file.

        Arguments:
            input_files (list): List of input GeoTIFF files to merge.
        """
        output_file = os.path.join(self._tile_directory, "merged.tif")
        # Open all input GeoTIFF files as datasets
        self.logger.debug("Merging tiff files...")
        datasets = [rasterio.open(file) for file in input_files]

        # Merge datasets
        crs = datasets[0].crs
        mosaic, out_transform = merge(datasets, nodata=0)

        # Get metadata from the first file and update it for the output
        out_meta = datasets[0].meta.copy()
        out_meta.update(
            {
                "driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_transform,
                "count": mosaic.shape[0],  # Number of bands
            }
        )

        # Write merged GeoTIFF to the output file
        with rasterio.open(output_file, "w", **out_meta) as dest:
            dest.write(mosaic)

        self.logger.debug("GeoTIFF images merged successfully into %s", output_file)
        return output_file, crs

    def extract_roi(self, tile_path: str) -> np.ndarray:
        """Extract region of interest (ROI) from the GeoTIFF file.

        Arguments:
            tile_path (str): Path to the GeoTIFF file.

        Raises:
            ValueError: If the tile does not contain any data.

        Returns:
            np.ndarray: Numpy array of the ROI.
        """
        north, south, east, west = self.get_bbox()
        with rasterio.open(tile_path) as src:
            self.logger.debug("Opened tile, shape: %s, dtype: %s.", src.shape, src.dtypes[0])
            window = rasterio.windows.from_bounds(west, south, east, north, src.transform)
            self.logger.debug(
                "Window parameters. Column offset: %s, row offset: %s, width: %s, height: %s.",
                window.col_off,
                window.row_off,
                window.width,
                window.height,
            )
            data = src.read(1, window=window, masked=True)
        if not data.size > 0:
            raise ValueError("No data in the tile.")

        return data

    # endregion
