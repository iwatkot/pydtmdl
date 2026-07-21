from pathlib import Path
from typing import Any

from pydtmdl.providers.sweden import SwedenProvider, SwedenProviderSettings


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200):
        self._payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self._payload


def _stac_item(collection: str, href: str, media_type: str) -> dict[str, Any]:
    return {
        "collection": collection,
        "assets": {
            "data": {
                "href": href,
                "type": media_type,
            },
        },
    }


def test_sweden_provider_filters_stac_search_to_markhojdmodell_geotiffs(
    tmp_path: Path,
    monkeypatch,
):
    first_tile = "https://dl1.lantmateriet.se/hojd/data/grid1m/63_4/55/63875_4975_25.tif"
    second_tile = "https://dl1.lantmateriet.se/hojd/data/grid1m/63_4/55/63850_4975_25.tif"
    point_cloud = (
        "https://dl1.lantmateriet.se/hojd/data/pointcloud/sls/22c001/"
        "m22c001-638_49.copc.laz"
    )
    duplicate_overview = "https://dl1.lantmateriet.se/hojd/data/grid/mhm/63_4/m638_49.tif"

    responses = [
        _FakeResponse(
            {
                "features": [
                    _stac_item(
                        "mhm-63_4",
                        first_tile,
                        "image/tiff; application=geotiff; profile=cloud-optimized",
                    ),
                    _stac_item("dsm-skoglig-copc", point_cloud, "application/vnd.laszip+copc"),
                    _stac_item(
                        "dtm-cog",
                        duplicate_overview,
                        "image/tiff; application=geotiff; profile=cloud-optimized",
                    ),
                    _stac_item("mhm-63_4", "https://example.com/metadata.json", "application/json"),
                ],
                "links": [
                    {
                        "rel": "next",
                        "href": "search?page=2",
                    },
                ],
            }
        ),
        _FakeResponse(
            {
                "features": [
                    _stac_item(
                        "mhm-63_4",
                        first_tile,
                        "image/tiff; application=geotiff; profile=cloud-optimized",
                    ),
                    _stac_item(
                        "mhm-63_4",
                        second_tile,
                        "image/tiff; application=geotiff; profile=cloud-optimized",
                    ),
                ],
                "links": [],
            }
        ),
    ]
    calls: list[dict[str, Any]] = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(
            {
                "url": url,
                "params": params,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return responses.pop(0)

    monkeypatch.setattr("pydtmdl.providers.sweden.requests.get", fake_get)

    provider = SwedenProvider(
        (57.619504756149034, 15.000275373458862),
        size=1024,
        user_settings=SwedenProviderSettings(username="user", password="pass"),
        directory=str(tmp_path),
    )

    assert provider.get_download_urls() == [first_tile, second_tile]
    assert calls[0]["params"]["limit"] == "100"
    assert calls[0]["headers"]["Authorization"].startswith("Basic ")
    assert calls[0]["timeout"] == 60
    assert calls[1]["url"] == "https://api.lantmateriet.se/stac-hojd/v1/search?page=2"
    assert calls[1]["params"] is None
