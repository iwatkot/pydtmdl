"""This module contains the DTMProvider class and its subclasses. DTMProvider class is used to
define different providers of digital terrain models (DTM) data. Each provider has its own URL
and specific settings for downloading and processing the data."""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Type, cast
from zipfile import ZipFile

import numpy as np
import rasterio
import requests
from affine import Affine
from pydantic import BaseModel
from pyproj import CRS, Transformer
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject
from rasterio.windows import from_bounds
from requests.exceptions import RequestException
from tqdm import tqdm


class DTMProviderSettings(BaseModel):
    """Base class for DTM provider settings models."""


class DTMErrorDetails(BaseModel):
    """Machine-readable error details for DTM provider failures."""

    error_type: str
    message: str
    provider_code: str | None = None
    provider_name: str | None = None


class DTMProviderError(Exception):
    """Base class for machine-readable DTM provider errors."""

    error_type = "provider_error"

    def __init__(
        self,
        message: str,
        provider_code: str | None = None,
        provider_name: str | None = None,
    ):
        super().__init__(message)
        self.provider_code = provider_code
        self.provider_name = provider_name

    def to_details(self) -> DTMErrorDetails:
        """Convert the error to a serializable details model."""
        return DTMErrorDetails(
            error_type=self.error_type,
            message=str(self),
            provider_code=self.provider_code,
            provider_name=self.provider_name,
        )


class DTMProviderRuntimeError(DTMProviderError, RuntimeError):
    """Base class for runtime DTM provider errors."""


class DTMProviderValueError(DTMProviderError, ValueError):
    """Base class for validation DTM provider errors."""


class ProviderUnavailableError(DTMProviderRuntimeError):
    """Raised when a requested provider is not available."""

    error_type = "provider_unavailable"


class OutsideCoverageError(DTMProviderValueError):
    """Raised when the requested geometry falls outside provider coverage."""

    error_type = "outside_coverage"


class AuthConfigMissingError(DTMProviderValueError):
    """Raised when a provider is missing required credentials or configuration."""

    error_type = "auth_config_missing"


class DownloadFailedError(DTMProviderRuntimeError):
    """Raised when source tiles cannot be downloaded."""

    error_type = "download_failed"


class ReprojectionFailedError(DTMProviderRuntimeError):
    """Raised when raster reprojection fails."""

    error_type = "reprojection_failed"


class CropExtractionError(DTMProviderRuntimeError):
    """Raised when the final ROI crop cannot be produced."""

    error_type = "crop_extraction_failed"


class DTMResultMetadata(BaseModel):
    """Structured metadata returned with DTM extraction results."""

    requested_provider: str
    requested_provider_name: str | None = None
    actual_provider: str
    actual_provider_name: str | None = None
    resolution: float | None = None
    output_path: str
    output_crs: str
    shape: tuple[int, int]
    dtype: str
    nodata: float | int | None = None
    cache_hit: bool = False
    cache_key: str
    cache_path: str
    fallback_used: bool = False
    primary_failure_reason: DTMErrorDetails | None = None
    source_files: list[str]
    center: tuple[float, float]
    width_m: int
    height_m: int
    rotation_deg: float = 0.0


@dataclass(slots=True)
class DTMExtractionResult:
    """Return type for structured DTM extraction calls."""

    data: np.ndarray
    metadata: DTMResultMetadata


class DTMProvider(ABC):
    """Base class for DTM providers."""

    _cache_version: int = 3

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
    _output_crs: str = "EPSG:4326"

    def __init__(
        self,
        coordinates: tuple[float, float],
        size: int | None = None,
        user_settings: DTMProviderSettings | None = None,
        directory: str = os.path.join(os.getcwd(), "tiles"),
        logger: Any = logging.getLogger(__name__),
        *,
        width_m: int | None = None,
        height_m: int | None = None,
        rotation_deg: float = 0.0,
    ):
        self._coordinates = coordinates
        self._user_settings = user_settings
        self._width_m, self._height_m = self._resolve_dimensions(size, width_m, height_m)
        self._size = int(size) if size is not None else max(self._width_m, self._height_m)
        self._rotation_deg = float(rotation_deg % 360)
        self._download_width_m, self._download_height_m = self._calculate_download_dimensions(
            self._width_m,
            self._height_m,
            self._rotation_deg,
        )
        self._directory = directory

        if not self._code:
            raise ValueError("Provider code must be defined.")
        self._cache_key = self.build_cache_key()
        self._provider_directory = os.path.join(directory, self._code)
        self._tile_directory = os.path.join(self._provider_directory, self._cache_key)
        os.makedirs(self._tile_directory, exist_ok=True)
        self._result_tiff_path = os.path.join(self._tile_directory, "result.tif")
        self._metadata_path = os.path.join(self._tile_directory, "result_metadata.json")

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
        """Legacy size of the DTM request in meters.

        Returns:
            int: Largest requested side length in meters.
        """
        return self._size

    @property
    def width_m(self) -> int:
        """Requested ROI width in meters."""
        return self._width_m

    @property
    def height_m(self) -> int:
        """Requested ROI height in meters."""
        return self._height_m

    @property
    def rotation_deg(self) -> float:
        """Rotation of the requested ROI around its center in degrees, clockwise-positive."""
        return self._rotation_deg

    @property
    def download_width_m(self) -> int:
        """Axis-aligned width required to fully cover the requested ROI."""
        return self._download_width_m

    @property
    def download_height_m(self) -> int:
        """Axis-aligned height required to fully cover the requested ROI."""
        return self._download_height_m

    @property
    def cache_key(self) -> str:
        """Stable cache identifier for the current request geometry."""
        return self._cache_key

    @property
    def cache_path(self) -> str:
        """Cache directory used for the current request geometry."""
        return self._tile_directory

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
        for provider in cls._all_provider_classes():
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
        for provider in cls._all_provider_classes():
            if provider.name() == name:
                return provider
        return None

    @classmethod
    def get_valid_provider_descriptions(
        cls,
        lat_lon: tuple[float, float],
        default_code: str = "srtm30",
        width_m: int | None = None,
        height_m: int | None = None,
        rotation_deg: float = 0.0,
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
            if cls._provider_matches_geometry(
                provider,
                lat_lon,
                width_m,
                height_m,
                rotation_deg,
            ):
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

        return [
            provider
            for provider in cls._all_provider_classes()
            if provider not in base_providers and not provider.__subclasses__()
        ]

    @classmethod
    def get_list(
        cls,
        lat_lon: tuple[float, float],
        include_unreliable: bool = False,
        width_m: int | None = None,
        height_m: int | None = None,
        rotation_deg: float = 0.0,
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
            if cls._provider_matches_geometry(
                provider,
                lat_lon,
                width_m,
                height_m,
                rotation_deg,
            ):
                if not include_unreliable and provider.unreliable():
                    continue
                providers.append(provider)
        return providers

    @classmethod
    def get_best(
        cls,
        lat_lon: tuple[float, float],
        default_code: str = "srtm30",
        width_m: int | None = None,
        height_m: int | None = None,
        rotation_deg: float = 0.0,
    ) -> Type[DTMProvider] | None:
        """Get the best provider for the given coordinates.

        Arguments:
            lat_lon (tuple): Latitude and longitude of the center point.
            default_code (str): Default provider code.

        Returns:
            DTMProvider: Best provider class or None if not found.
        """
        providers = cls.get_list(
            lat_lon,
            width_m=width_m,
            height_m=height_m,
            rotation_deg=rotation_deg,
        )
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

    @classmethod
    def covers_geometry(
        cls,
        lat_lon: tuple[float, float],
        width_m: int,
        height_m: int | None = None,
        rotation_deg: float = 0.0,
    ) -> bool:
        """Check whether the provider covers the full requested geometry."""
        resolved_height = height_m if height_m is not None else width_m
        bbox = cls.get_geometry_bbox(lat_lon, width_m, resolved_height, rotation_deg)
        extents = cls._extents
        if extents is None:
            return True
        north, south, east, west = bbox
        for extent in extents:
            if (
                extent[0] >= north
                and extent[1] <= south
                and extent[2] >= east
                and extent[3] <= west
            ):
                return True
        return False

    @classmethod
    def get_geometry_bbox(
        cls,
        coordinates: tuple[float, float],
        width_m: int,
        height_m: int,
        rotation_deg: float = 0.0,
    ) -> tuple[float, float, float, float]:
        """Get the EPSG:4326 bounding box for a rectangular ROI."""
        polygon = cls.build_roi_polygon(coordinates, width_m, height_m, rotation_deg)
        lons = [point[0] for point in polygon]
        lats = [point[1] for point in polygon]
        return max(lats), min(lats), max(lons), min(lons)

    @classmethod
    def build_roi_polygon(
        cls,
        coordinates: tuple[float, float],
        width_m: int,
        height_m: int,
        rotation_deg: float = 0.0,
    ) -> list[tuple[float, float]]:
        """Build a rectangular ROI polygon in EPSG:4326 as (lon, lat) points.

        Positive rotation values rotate the ROI clockwise around its center.
        """
        half_width = width_m / 2.0
        half_height = height_m / 2.0
        angle_rad = math.radians(-rotation_deg)
        cos_angle = math.cos(angle_rad)
        sin_angle = math.sin(angle_rad)

        _, to_wgs84 = cls._build_local_transformers(coordinates)
        local_corners = [
            (-half_width, -half_height),
            (half_width, -half_height),
            (half_width, half_height),
            (-half_width, half_height),
        ]
        polygon: list[tuple[float, float]] = []
        for x_offset, y_offset in local_corners:
            rotated_x = (x_offset * cos_angle) - (y_offset * sin_angle)
            rotated_y = (x_offset * sin_angle) + (y_offset * cos_angle)
            lon, lat = to_wgs84.transform(rotated_x, rotated_y)
            polygon.append((float(lon), float(lat)))
        return polygon

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
        return self.get_result().data

    def get_result(
        self,
        fallback_provider_code: str | None = None,
        fallback_user_settings: DTMProviderSettings | None = None,
    ) -> DTMExtractionResult:
        """Get the extracted DTM together with structured metadata."""
        requested_provider_code = self.code() or "unknown"
        requested_provider_name = self.name()

        try:
            return self._create_result(
                requested_provider_code=requested_provider_code,
                requested_provider_name=requested_provider_name,
            )
        except DTMProviderError as primary_error:
            if not fallback_provider_code:
                raise

            fallback_provider_class = self.get_provider_by_code(fallback_provider_code)
            if fallback_provider_class is None:
                raise ProviderUnavailableError(
                    f"Fallback provider '{fallback_provider_code}' is not available.",
                    provider_code=fallback_provider_code,
                ) from primary_error

            fallback_provider = fallback_provider_class(
                self.coordinates,
                size=self.size,
                user_settings=fallback_user_settings,
                directory=self._directory,
                logger=self.logger,
                width_m=self.width_m,
                height_m=self.height_m,
                rotation_deg=self.rotation_deg,
            )
            try:
                return fallback_provider._create_result(
                    requested_provider_code=requested_provider_code,
                    requested_provider_name=requested_provider_name,
                    fallback_used=True,
                    primary_failure=primary_error.to_details(),
                )
            except DTMProviderError as fallback_error:
                raise fallback_error from primary_error

    @classmethod
    def extract_area(
        cls,
        center: tuple[float, float],
        width_m: int,
        height_m: int | None = None,
        rotation_deg: float = 0.0,
        provider_code: str | None = None,
        fallback_provider_code: str | None = None,
        user_settings: DTMProviderSettings | None = None,
        fallback_user_settings: DTMProviderSettings | None = None,
        directory: str = os.path.join(os.getcwd(), "tiles"),
        logger: Any = logging.getLogger(__name__),
    ) -> DTMExtractionResult:
        """High-level extraction API for rectangular and rotated ROIs."""
        resolved_height = height_m if height_m is not None else width_m

        if cls is DTMProvider:
            provider_class = (
                cls.get_provider_by_code(provider_code)
                if provider_code
                else cls.get_best(
                    center,
                    width_m=width_m,
                    height_m=resolved_height,
                    rotation_deg=rotation_deg,
                )
            )
        else:
            provider_class = cls

        if provider_class is None:
            if provider_code:
                raise ProviderUnavailableError(
                    f"No provider is available for the requested provider_code: {provider_code!r}.",
                    provider_code=provider_code,
                )
            raise ProviderUnavailableError("No provider is available for the requested geometry.")

        provider = provider_class(
            center,
            size=width_m if resolved_height == width_m else max(width_m, resolved_height),
            user_settings=user_settings,
            directory=directory,
            logger=logger,
            width_m=width_m,
            height_m=resolved_height,
            rotation_deg=rotation_deg,
        )
        return provider.get_result(
            fallback_provider_code=fallback_provider_code,
            fallback_user_settings=fallback_user_settings,
        )

    @property
    def image(self) -> np.ndarray:
        """Get numpy array of the tile and check if it contains any data.

        Returns:
            np.ndarray: Numpy array of the tile.

        Raises:
            ValueError: If the tile does not contain any data.
        """
        data = self.get_numpy()
        if data.size == 0:
            raise OutsideCoverageError(
                "No data in the tile. Try different provider.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        if np.ma.isMaskedArray(data) and data.count() == 0:
            raise OutsideCoverageError(
                "No data in the tile. Try different provider.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        return data

    # region helpers
    def get_bbox(self) -> tuple[float, float, float, float]:
        """Get bounding box of the download area based on the request geometry.

        Returns:
            tuple: Bounding box of the download area (north, south, east, west).
        """
        return self.get_geometry_bbox(
            self.coordinates,
            self.width_m,
            self.height_m,
            self.rotation_deg,
        )

    def get_roi_polygon(self) -> list[tuple[float, float]]:
        """Get the requested ROI polygon as EPSG:4326 (lon, lat) coordinates."""
        return self.build_roi_polygon(
            self.coordinates,
            self.width_m,
            self.height_m,
            self.rotation_deg,
        )

    def get_roi_geometry(self) -> dict[str, Any]:
        """Get the requested ROI polygon as a GeoJSON geometry."""
        polygon = self.get_roi_polygon()
        return {
            "type": "Polygon",
            "coordinates": [[*polygon, polygon[0]]],
        }

    def build_cache_key(self) -> str:
        """Build a stable cache key for the current request geometry."""
        payload = {
            "cache_version": self._cache_version,
            "provider": self.code(),
            "center": [round(self.coordinates[0], 8), round(self.coordinates[1], 8)],
            "width_m": self.width_m,
            "height_m": self.height_m,
            "rotation_deg": round(self.rotation_deg, 6),
            "output_crs": self._output_crs,
            "resolution": self.resolution(),
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]

    @staticmethod
    def split_bbox(
        bbox: tuple[float, float, float, float],
        columns: int,
        rows: int,
    ) -> list[tuple[float, float, float, float]]:
        """Split a bounding box into an evenly spaced grid."""
        if columns <= 0 or rows <= 0:
            raise ValueError("Bounding box grid dimensions must be positive integers.")

        north, south, east, west = bbox
        lon_step = (east - west) / columns
        lat_step = (north - south) / rows
        tiles: list[tuple[float, float, float, float]] = []
        for column in range(columns):
            tile_west = west + (column * lon_step)
            tile_east = west + ((column + 1) * lon_step)
            for row in range(rows):
                tile_south = south + (row * lat_step)
                tile_north = south + ((row + 1) * lat_step)
                tiles.append((tile_north, tile_south, tile_east, tile_west))
        return tiles

    @staticmethod
    def get_tile_pixel_dimensions(
        tile_width_m: float,
        tile_height_m: float,
        max_pixels: int,
    ) -> tuple[int, int]:
        """Scale tile pixel dimensions while preserving aspect ratio."""
        max_pixels = max(1, int(max_pixels))
        largest_edge = max(tile_width_m, tile_height_m, 1.0)
        scale = max_pixels / largest_edge
        pixel_width = min(max_pixels, max(1, math.ceil(tile_width_m * scale)))
        pixel_height = min(max_pixels, max(1, math.ceil(tile_height_m * scale)))
        return pixel_width, pixel_height

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
        data, _ = self._extract_roi_raster(tile_path)
        return data[0]

    @staticmethod
    def _resolve_dimensions(
        size: int | None,
        width_m: int | None,
        height_m: int | None,
    ) -> tuple[int, int]:
        """Resolve the requested ROI dimensions while preserving legacy size support."""
        resolved_width = width_m if width_m is not None else size
        resolved_height = height_m if height_m is not None else size
        if resolved_width is None and resolved_height is not None:
            resolved_width = resolved_height
        elif resolved_height is None and resolved_width is not None:
            resolved_height = resolved_width
        if resolved_width is None or resolved_height is None:
            raise ValueError("Either size or width_m/height_m must be provided.")
        if resolved_width <= 0 or resolved_height <= 0:
            raise ValueError("Requested ROI dimensions must be positive integers.")
        return int(resolved_width), int(resolved_height)

    @staticmethod
    def _calculate_download_dimensions(
        width_m: int,
        height_m: int,
        rotation_deg: float,
    ) -> tuple[int, int]:
        """Calculate the axis-aligned extent required to cover a rotated rectangle."""
        angle_rad = math.radians(rotation_deg % 180)
        bbox_width = abs(width_m * math.cos(angle_rad)) + abs(height_m * math.sin(angle_rad))
        bbox_height = abs(width_m * math.sin(angle_rad)) + abs(height_m * math.cos(angle_rad))
        return math.ceil(bbox_width), math.ceil(bbox_height)

    @staticmethod
    def _build_local_crs(coordinates: tuple[float, float]) -> CRS:
        """Build a local metric CRS centered on the requested ROI."""
        lat, lon = coordinates
        return CRS.from_proj4(
            f"+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs"
        )

    @staticmethod
    def _build_local_transformers(
        coordinates: tuple[float, float],
    ) -> tuple[Transformer, Transformer]:
        """Build local metric transformers centered on the requested ROI."""
        local_crs = DTMProvider._build_local_crs(coordinates)
        to_local = Transformer.from_crs("EPSG:4326", local_crs, always_xy=True)
        to_wgs84 = Transformer.from_crs(local_crs, "EPSG:4326", always_xy=True)
        return to_local, to_wgs84

    @classmethod
    def _provider_matches_geometry(
        cls,
        provider: Type[DTMProvider],
        lat_lon: tuple[float, float],
        width_m: int | None,
        height_m: int | None,
        rotation_deg: float,
    ) -> bool:
        """Check whether a provider matches the requested point or full geometry."""
        if width_m is None and height_m is None:
            return provider.inside_bounding_box(lat_lon)
        if width_m is None:
            return provider.inside_bounding_box(lat_lon)
        resolved_height = height_m if height_m is not None else width_m
        return provider.covers_geometry(lat_lon, width_m, resolved_height, rotation_deg)

    @classmethod
    def _all_provider_classes(cls) -> list[Type[DTMProvider]]:
        """Collect all provider classes from the full inheritance tree."""
        providers: list[Type[DTMProvider]] = []
        seen: set[type[Any]] = set()
        stack: list[type[Any]] = [DTMProvider]

        while stack:
            provider_class = stack.pop()
            for child in provider_class.__subclasses__():
                if child in seen:
                    continue
                seen.add(child)
                providers.append(cast(Type[DTMProvider], child))
                stack.append(child)

        return providers

    def _create_result(
        self,
        requested_provider_code: str,
        requested_provider_name: str | None,
        fallback_used: bool = False,
        primary_failure: DTMErrorDetails | None = None,
    ) -> DTMExtractionResult:
        """Run the full extraction pipeline and return data plus metadata."""
        if self.settings_required() and self.user_settings is None:
            raise AuthConfigMissingError(
                "User settings are required for this provider.",
                provider_code=self.code(),
                provider_name=self.name(),
            )

        if os.path.exists(self._result_tiff_path):
            return self._build_result_from_output(
                self._result_tiff_path,
                requested_provider_code=requested_provider_code,
                requested_provider_name=requested_provider_name,
                fallback_used=fallback_used,
                primary_failure=primary_failure,
                cache_hit=True,
                source_files=self._get_cached_source_files(),
            )

        source_files = self._download_source_tiles()
        prepared_tile = self._prepare_source_tile(source_files)
        self._write_result_tiff(prepared_tile, self._result_tiff_path)
        result = self._build_result_from_output(
            self._result_tiff_path,
            requested_provider_code=requested_provider_code,
            requested_provider_name=requested_provider_name,
            fallback_used=fallback_used,
            primary_failure=primary_failure,
            cache_hit=False,
            source_files=source_files,
        )
        self._write_result_metadata(result.metadata)
        return result

    def _download_source_tiles(self) -> list[str]:
        """Download all source tiles for the current request geometry."""
        try:
            tiles = self.download_tiles()
        except DTMProviderError:
            raise
        except RequestException as e:
            self.logger.error("Error while downloading tiles: %s", e)
            raise DownloadFailedError(
                "Failed to download tiles from the provider.",
                provider_code=self.code(),
                provider_name=self.name(),
            ) from e
        except FileNotFoundError as e:
            self.logger.error("Requested geometry is outside provider coverage: %s", e)
            raise OutsideCoverageError(
                "The requested geometry is outside the provider coverage area.",
                provider_code=self.code(),
                provider_name=self.name(),
            ) from e
        except (RuntimeError, ValueError) as e:
            raise self._normalize_error(e) from e
        except Exception as e:
            self.logger.error("Unexpected download error: %s", e)
            raise DownloadFailedError(
                "Failed to download tiles from the provider.",
                provider_code=self.code(),
                provider_name=self.name(),
            ) from e

        self.logger.debug("Downloaded tiles: %s", tiles)
        if not tiles:
            raise OutsideCoverageError(
                "No tiles were downloaded from the provider.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        return tiles

    def _prepare_source_tile(self, tiles: list[str]) -> str:
        """Merge and reproject source tiles into a single crop-ready raster."""
        if len(tiles) > 1:
            self.logger.debug("Multiple tiles downloaded. Merging tiles")
            try:
                tile, _ = self.merge_geotiff(tiles)
            except Exception as e:
                raise CropExtractionError(
                    "Failed to merge the downloaded tiles.",
                    provider_code=self.code(),
                    provider_name=self.name(),
                ) from e
        else:
            tile = tiles[0]

        with rasterio.open(tile) as src:
            crs = src.crs.to_string() if src.crs else None

        if crs is None:
            raise ReprojectionFailedError(
                "Source tile does not define a CRS.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        if crs != self._output_crs:
            self.logger.debug("Reprojecting GeoTIFF from %s to %s...", crs, self._output_crs)
            try:
                tile = self.reproject_geotiff(tile)
            except Exception as e:
                raise ReprojectionFailedError(
                    f"Failed to reproject the source tile from {crs} to {self._output_crs}.",
                    provider_code=self.code(),
                    provider_name=self.name(),
                ) from e

        return tile

    def _write_result_tiff(self, tile_path: str, output_path: str) -> None:
        """Write the cropped ROI raster to the stable cache path."""
        data, metadata = self._extract_roi_raster(tile_path)
        filled_data = data.filled(metadata["nodata"]) if np.ma.isMaskedArray(data) else data
        with rasterio.open(output_path, "w", **metadata) as dst:
            dst.write(filled_data)

    def _extract_roi_raster(self, tile_path: str) -> tuple[np.ma.MaskedArray, dict[str, Any]]:
        """Extract the requested ROI using the appropriate rasterization path."""
        if math.isclose(self.rotation_deg % 360, 0.0):
            return self._crop_axis_aligned_roi(tile_path)
        return self._resample_rotated_roi(tile_path)

    def _crop_axis_aligned_roi(self, tile_path: str) -> tuple[np.ma.MaskedArray, dict[str, Any]]:
        """Crop an axis-aligned ROI by bounds to avoid masked border artifacts."""
        north, south, east, west = self.get_bbox()

        with rasterio.open(tile_path) as src:
            self.logger.debug("Opened tile, shape: %s, dtype: %s.", src.shape, src.dtypes[0])
            nodata = self._get_output_nodata(src)
            window = from_bounds(west, south, east, north, src.transform)
            window = window.round_offsets().round_lengths()
            data = src.read(window=window, masked=True)
            transform = src.window_transform(window)
            metadata = src.meta.copy()
            metadata.update(
                {
                    "driver": "GTiff",
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": transform,
                    "nodata": nodata,
                }
            )

        if data.size == 0:
            raise CropExtractionError(
                "The requested geometry does not contain any data.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        if np.ma.isMaskedArray(data) and data.count() == 0:
            raise CropExtractionError(
                "The cropped ROI does not contain any valid data.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        return data, metadata

    def _mask_roi(self, tile_path: str) -> tuple[np.ma.MaskedArray, dict[str, Any]]:
        """Mask a raster to the requested ROI polygon."""
        with rasterio.open(tile_path) as src:
            self.logger.debug("Opened tile, shape: %s, dtype: %s.", src.shape, src.dtypes[0])
            nodata = self._get_output_nodata(src)
            data, transform = mask(
                src,
                [self.get_roi_geometry()],
                crop=True,
                filled=False,
                nodata=nodata,
            )
            metadata = src.meta.copy()
            metadata.update(
                {
                    "driver": "GTiff",
                    "height": data.shape[1],
                    "width": data.shape[2],
                    "transform": transform,
                    "nodata": nodata,
                }
            )

        if data.size == 0:
            raise CropExtractionError(
                "The requested geometry does not contain any data.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        if np.ma.isMaskedArray(data) and data.count() == 0:
            raise CropExtractionError(
                "The cropped ROI does not contain any valid data.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        return data, metadata

    def _resample_rotated_roi(self, tile_path: str) -> tuple[np.ma.MaskedArray, dict[str, Any]]:
        """Resample a rotated ROI onto an aligned output grid without padded corners."""
        local_crs = self._build_local_crs(self.coordinates)
        to_local, _ = self._build_local_transformers(self.coordinates)

        with rasterio.open(tile_path) as src:
            self.logger.debug("Opened tile, shape: %s, dtype: %s.", src.shape, src.dtypes[0])
            nodata = self._get_output_nodata(src)

            center_lat, center_lon = self.coordinates
            center_x, center_y = to_local.transform(center_lon, center_lat)
            east_x, east_y = to_local.transform(center_lon + abs(src.transform.a), center_lat)
            south_x, south_y = to_local.transform(center_lon, center_lat + abs(src.transform.e))
            x_resolution_m = math.hypot(east_x - center_x, east_y - center_y)
            y_resolution_m = math.hypot(south_x - center_x, south_y - center_y)
            if x_resolution_m <= 0 or y_resolution_m <= 0:
                raise CropExtractionError(
                    "Failed to resolve the rotated ROI output resolution.",
                    provider_code=self.code(),
                    provider_name=self.name(),
                )

            width_px = max(1, math.ceil(self.width_m / x_resolution_m))
            height_px = max(1, math.ceil(self.height_m / y_resolution_m))
            pixel_width_m = self.width_m / width_px
            pixel_height_m = self.height_m / height_px

            angle_rad = math.radians(-self.rotation_deg)
            cos_angle = math.cos(angle_rad)
            sin_angle = math.sin(angle_rad)
            half_width = self.width_m / 2.0
            half_height = self.height_m / 2.0

            top_left_x = (-half_width * cos_angle) - (half_height * sin_angle)
            top_left_y = (-half_width * sin_angle) + (half_height * cos_angle)
            transform = Affine(
                pixel_width_m * cos_angle,
                pixel_height_m * sin_angle,
                top_left_x,
                pixel_width_m * sin_angle,
                -pixel_height_m * cos_angle,
                top_left_y,
            )

            destination = np.full((1, height_px, width_px), nodata, dtype=np.dtype(src.dtypes[0]))
            validity = np.zeros((height_px, width_px), dtype=np.uint8)

            reproject(
                source=rasterio.band(src, 1),
                destination=destination[0],
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=src.nodata,
                dst_transform=transform,
                dst_crs=local_crs,
                dst_nodata=nodata,
                resampling=Resampling.bilinear,
            )
            reproject(
                source=src.read_masks(1),
                destination=validity,
                src_transform=src.transform,
                src_crs=src.crs,
                src_nodata=0,
                dst_transform=transform,
                dst_crs=local_crs,
                dst_nodata=0,
                resampling=Resampling.nearest,
            )

            data = np.ma.array(destination, mask=validity[np.newaxis, ...] == 0)
            metadata = src.meta.copy()
            metadata.update(
                {
                    "driver": "GTiff",
                    "height": height_px,
                    "width": width_px,
                    "transform": transform,
                    "crs": local_crs,
                    "nodata": nodata,
                }
            )

        if data.size == 0:
            raise CropExtractionError(
                "The requested geometry does not contain any data.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        if np.ma.isMaskedArray(data) and data.count() == 0:
            raise CropExtractionError(
                "The cropped ROI does not contain any valid data.",
                provider_code=self.code(),
                provider_name=self.name(),
            )
        return data, metadata

    def _get_output_nodata(self, dataset: rasterio.DatasetReader) -> float | int:
        """Resolve a nodata value for cropped outputs."""
        if dataset.nodata is not None:
            return dataset.nodata
        dtype = np.dtype(dataset.dtypes[0])
        if np.issubdtype(dtype, np.floating):
            return float("nan")
        if np.issubdtype(dtype, np.signedinteger):
            return int(np.iinfo(dtype).min)
        return 0

    def _build_result_from_output(
        self,
        output_path: str,
        requested_provider_code: str,
        requested_provider_name: str | None,
        fallback_used: bool,
        primary_failure: DTMErrorDetails | None,
        cache_hit: bool,
        source_files: list[str] | None,
    ) -> DTMExtractionResult:
        """Create a structured result from a cached or freshly written raster."""
        with rasterio.open(output_path) as src:
            data = src.read(1, masked=True)
            if data.size == 0 or (np.ma.isMaskedArray(data) and data.count() == 0):
                raise CropExtractionError(
                    "The output raster does not contain any valid data.",
                    provider_code=self.code(),
                    provider_name=self.name(),
                )
            metadata = DTMResultMetadata(
                requested_provider=requested_provider_code,
                requested_provider_name=requested_provider_name,
                actual_provider=self.code() or "unknown",
                actual_provider_name=self.name(),
                resolution=self.resolution(),
                output_path=output_path,
                output_crs=src.crs.to_string() if src.crs else self._output_crs,
                shape=(src.height, src.width),
                dtype=src.dtypes[0],
                nodata=src.nodata,
                cache_hit=cache_hit,
                cache_key=self.cache_key,
                cache_path=self.cache_path,
                fallback_used=fallback_used,
                primary_failure_reason=primary_failure,
                source_files=source_files or [],
                center=self.coordinates,
                width_m=self.width_m,
                height_m=self.height_m,
                rotation_deg=self.rotation_deg,
            )
        return DTMExtractionResult(data=data, metadata=metadata)

    def _write_result_metadata(self, metadata: DTMResultMetadata) -> None:
        """Persist metadata next to the cached raster output."""
        with open(self._metadata_path, "w", encoding="utf-8") as metadata_file:
            metadata_file.write(metadata.model_dump_json(indent=2))

    def _get_cached_source_files(self) -> list[str]:
        """Read source file paths from cached metadata if available."""
        cached_metadata = self._load_cached_metadata()
        return cached_metadata.source_files if cached_metadata else []

    def _load_cached_metadata(self) -> DTMResultMetadata | None:
        """Load metadata from the cache directory if it exists and is valid."""
        if not os.path.exists(self._metadata_path):
            return None
        try:
            with open(self._metadata_path, "r", encoding="utf-8") as metadata_file:
                return DTMResultMetadata.model_validate_json(metadata_file.read())
        except (OSError, ValueError) as e:
            self.logger.warning(
                "Failed to read cached metadata from %s: %s", self._metadata_path, e
            )
            return None

    def _normalize_error(self, error: Exception) -> DTMProviderError:
        """Convert provider-specific exceptions into stable machine-readable errors."""
        if isinstance(error, DTMProviderError):
            return error

        message = str(error)
        normalized_message = message.lower()
        provider_code = self.code()
        provider_name = self.name()

        if any(
            token in normalized_message
            for token in [
                "user settings are required",
                "token is required",
                "api key is required",
                "username is required",
                "password is required",
                "dataset is required",
                "resolution is required",
            ]
        ):
            return AuthConfigMissingError(
                message, provider_code=provider_code, provider_name=provider_name
            )
        if (
            any(
                token in normalized_message
                for token in [
                    "outside the coverage",
                    "no tiles were downloaded",
                    "tile not found",
                    "not found",
                ]
            )
            and "crop" not in normalized_message
        ):
            return OutsideCoverageError(
                message, provider_code=provider_code, provider_name=provider_name
            )
        if "reproject" in normalized_message or "crs" in normalized_message:
            return ReprojectionFailedError(
                message,
                provider_code=provider_code,
                provider_name=provider_name,
            )
        if "crop" in normalized_message or "no data in the tile" in normalized_message:
            return CropExtractionError(
                message, provider_code=provider_code, provider_name=provider_name
            )
        return DownloadFailedError(
            message, provider_code=provider_code, provider_name=provider_name
        )

    # endregion
