"""This module contains provider of Sweden data."""

import base64
import os

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
        self.shared_tiff_path = os.path.join(self._tile_directory, "shared")
        os.makedirs(self.shared_tiff_path, exist_ok=True)

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
            request_params = {
                "bbox": bbox_str,
                "limit": "100",
            }

            response = requests.get(  # pylint: disable=W3101
                search_url,
                params=request_params,
                headers=headers,
                timeout=60,
            )

            # Check if the request was successful (HTTP status code 200)
            if response.status_code == 200:
                # Parse the JSON response
                json_data = response.json()
                items = json_data["features"]
                for item in items:
                    urls.append(item["assets"]["data"]["href"])
            else:
                self.logger.error("Failed to get data. HTTP Status Code: %s", response.status_code)
                self.logger.error("  Response Body: %s", response.text)
        except requests.exceptions.RequestException as e:
            self.logger.error("Failed to get data. Error: %s", e)
        return urls
