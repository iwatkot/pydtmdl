"""This module contains provider of Sweden data."""

import base64
import os
from typing import Any
from urllib.parse import urljoin

import requests

from pydtmdl.base.dtm import DTMProvider, DTMProviderSettings


class SwedenProviderSettings(DTMProviderSettings):
    """Settings for the Sweden provider."""

    username: str = ""
    password: str = ""


class SwedenProvider(DTMProvider):
    """Provider of Sweden data, provided by Lantmäteriet under the CC0 1.0 Universal (CC0 1.0) license."""

    _code = "sweden"
    _name = "Sweden Lantmäteriet Markhöjdmodell"
    _region = "SE"
    _icon = "🇸🇪"
    _resolution = 1.0
    _settings = SwedenProviderSettings
    _extents = [(69.086555, 55.279995, 24.097910, 10.674677)]
    _source_crs = "EPSG:5845"  # SWEREF99 16 30 + RH2000 height (native CRS of downloaded files)

    _instructions = "ℹ️ This provider requires username and password. See [here](https://geotorget.lantmateriet.se/geodataprodukter/markhojdmodell-nedladdning-api) to request access free of charge, then enter your credentials below."

    _url = "https://api.lantmateriet.se/stac-hojd/v1"
    _collection_prefix = "mhm-"
    _geotiff_extensions = (".tif", ".tiff")

    def _get_auth_headers(self) -> dict[str, str]:
        """Get authentication headers for API requests.

        Returns:
            dict: Dictionary with Authorization header.
        """
        if not self.user_settings:
            raise ValueError("User settings are required for this provider.")
        if not self.user_settings.username:  # type: ignore
            raise ValueError("Username is required for this provider.")
        if not self.user_settings.password:  # type: ignore
            raise ValueError("Password is required for this provider.")

        credentials = f"{self.user_settings.username}:{self.user_settings.password}"  # type: ignore
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded_credentials}"}

    def download_tiles(self):
        """Download Sweden tiles from STAC API."""
        download_urls = self.get_download_urls()
        # Use base class method with authentication headers
        headers = self._get_auth_headers()
        return super().download_tif_files(download_urls, self.shared_tiff_path, headers=headers)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shared_tiff_path = os.path.join(self._source_tile_directory, "shared")
        os.makedirs(self.shared_tiff_path, exist_ok=True)

    @classmethod
    def _is_markhojdmodell_item(cls, item: dict[str, Any]) -> bool:
        """Return whether a STAC item belongs to a Markhojdmodell height-grid collection."""
        collection = item.get("collection")
        return isinstance(collection, str) and collection.startswith(cls._collection_prefix)

    @classmethod
    def _is_geotiff_asset(cls, asset: dict[str, Any]) -> bool:
        """Return whether a STAC asset points at a GeoTIFF/COG raster."""
        href = asset.get("href")
        media_type = str(asset.get("type") or "").lower()
        if isinstance(href, str) and href.lower().split("?", maxsplit=1)[0].endswith(cls._geotiff_extensions):
            return True
        return "geotiff" in media_type or media_type.startswith("image/tiff")

    @classmethod
    def _get_markhojdmodell_geotiff_href(cls, item: dict[str, Any]) -> str | None:
        """Extract the Markhojdmodell GeoTIFF data URL from a STAC item, if present."""
        if not cls._is_markhojdmodell_item(item):
            return None

        assets = item.get("assets")
        if not isinstance(assets, dict):
            return None

        data_asset = assets.get("data")
        if not isinstance(data_asset, dict) or not cls._is_geotiff_asset(data_asset):
            return None

        href = data_asset.get("href")
        return href if isinstance(href, str) and href else None

    def get_download_urls(self) -> list[str]:
        """Get download URLs of the GeoTIFF files from the STAC API.

        Returns:
            list: List of download URLs.
        """
        urls = []

        try:
            # Get authentication headers (validates user_settings)
            headers = self._get_auth_headers()

            # Get bounding box
            bbox = self.get_bbox()
            north, south, east, west = bbox
            # Format for STAC API (west,south,east,north)
            bbox_str = f"{west},{south},{east},{north}"

            # Make the GET request to /search endpoint
            search_url = f"{self._url}/search"
            request_params: dict[str, str] | None = {
                "bbox": bbox_str,
                "limit": "100",
            }
            seen_urls: set[str] = set()

            while search_url:
                response = requests.get(  # pylint: disable=W3101
                    search_url,
                    params=request_params,
                    headers=headers,
                    timeout=60,
                )

                # Check if the request was successful (HTTP status code 200)
                if response.status_code != 200:
                    self.logger.error("Failed to get data. HTTP Status Code: %s", response.status_code)
                    self.logger.error("  Response Body: %s", response.text)
                    break

                # Parse the JSON response
                json_data = response.json()
                items = json_data.get("features", [])
                for item in items:
                    if not isinstance(item, dict):
                        continue
                    href = self._get_markhojdmodell_geotiff_href(item)
                    if href and href not in seen_urls:
                        urls.append(href)
                        seen_urls.add(href)

                next_href = None
                for link in json_data.get("links", []):
                    if isinstance(link, dict) and link.get("rel") == "next" and isinstance(link.get("href"), str):
                        next_href = link["href"]
                        break
                search_url = urljoin(f"{self._url.rstrip('/')}/", next_href) if next_href else ""
                request_params = None
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to get data. Error: %s", e)
        return urls
