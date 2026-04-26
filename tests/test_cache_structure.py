"""Tests for the 0.2.3 optimized cache structure (global source tiles + per-geometry results)."""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

from pydtmdl.base.dtm import DTMProvider


def get_concrete_provider_class():
    """Get a concrete DTM provider class (not abstract)."""
    providers = DTMProvider.get_non_base_providers()
    # Filter out imagery providers to get only DTM providers
    dtm_providers = [p for p in providers if not hasattr(p, '_dataset')]
    return dtm_providers[0] if dtm_providers else providers[0]


class TestCacheStructure:
    """Test the new cache structure: global source tiles + per-geometry results."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Create temporary cache directory for tests."""
        self.cache_dir = tempfile.mkdtemp(prefix="pydtmdl_cache_test_")
        yield
        # Cleanup
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_source_tiles_directory_created(self):
        """Verify that _tiles/ source directory is created at provider level."""
        provider = get_concrete_provider_class()(
            coordinates=(40.0, 10.0),
            width_m=4096,
            directory=self.cache_dir,
        )

        expected_source_dir = os.path.join(self.cache_dir, provider.code(), "_tiles")
        assert os.path.exists(expected_source_dir), \
            f"Source tile directory not created: {expected_source_dir}"

    def test_result_directory_per_geometry(self):
        """Verify that each geometry gets its own result directory."""
        provider_class = get_concrete_provider_class()

        provider1 = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            directory=self.cache_dir,
        )

        provider2 = provider_class(
            coordinates=(40.0001, 10.0001),
            width_m=4096,
            directory=self.cache_dir,
        )

        # Different geometries should have different cache keys
        assert provider1.cache_key != provider2.cache_key, \
            "Different geometries should have different cache keys"

        # Result dirs should be different
        assert provider1._tile_directory != provider2._tile_directory, \
            "Different geometries should have different result directories"

        # Verify directories exist
        assert os.path.exists(provider1._tile_directory)
        assert os.path.exists(provider2._tile_directory)

    def test_shared_source_tile_directory(self):
        """Verify that different geometries share the same source tile directory."""
        provider_class = get_concrete_provider_class()

        provider1 = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            directory=self.cache_dir,
        )

        provider2 = provider_class(
            coordinates=(40.0001, 10.0001),
            width_m=4096,
            directory=self.cache_dir,
        )

        # Source tile directories should be identical
        assert provider1._source_tile_directory == provider2._source_tile_directory, \
            "Different geometries should share the same source tile directory"

    def test_cache_hierarchy(self):
        """Verify the complete cache directory hierarchy."""
        provider_class = get_concrete_provider_class()
        provider = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            directory=self.cache_dir,
        )

        provider_dir = os.path.join(self.cache_dir, provider.code())
        source_dir = os.path.join(provider_dir, "_tiles")
        result_dir = os.path.join(provider_dir, provider.cache_key)

        # All directories should exist
        assert os.path.exists(provider_dir), f"Provider directory missing: {provider_dir}"
        assert os.path.exists(source_dir), f"Source tile directory missing: {source_dir}"
        assert os.path.exists(result_dir), f"Result directory missing: {result_dir}"

        # Verify structure matches paths
        assert provider._provider_directory == provider_dir
        assert provider._source_tile_directory == source_dir
        assert provider._tile_directory == result_dir

    def test_result_tiff_path_in_result_dir(self):
        """Verify result.tif is stored in the result directory, not source directory."""
        provider_class = get_concrete_provider_class()
        provider = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            directory=self.cache_dir,
        )

        # Result TIFF should be in result directory
        assert provider._result_tiff_path.startswith(provider._tile_directory), \
            "Result TIFF should be in result directory"

        # Result TIFF should NOT be in source directory
        assert not provider._result_tiff_path.startswith(provider._source_tile_directory), \
            "Result TIFF should NOT be in source directory"

    def test_metadata_path_in_result_dir(self):
        """Verify metadata.json is stored in the result directory."""
        provider_class = get_concrete_provider_class()
        provider = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            directory=self.cache_dir,
        )

        # Metadata should be in result directory
        assert provider._metadata_path.startswith(provider._tile_directory), \
            "Metadata should be in result directory"

    def test_multiple_geometries_same_source_tiles(self):
        """Integration test: verify multiple geometries can share source tiles."""
        provider_class = get_concrete_provider_class()

        geometries = [
            (40.0, 10.0),
            (40.0001, 10.0),
            (40.0, 10.0001),
            (40.0001, 10.0001),
        ]

        providers = []
        source_dirs = []

        for center in geometries:
            provider = provider_class(
                coordinates=center,
                width_m=4096,
                directory=self.cache_dir,
            )
            providers.append(provider)
            source_dirs.append(provider._source_tile_directory)

        # All should share the same source directory
        assert all(d == source_dirs[0] for d in source_dirs), \
            "All providers should share the same source tile directory"

        # All result directories should be unique
        result_dirs = [p._tile_directory for p in providers]
        assert len(result_dirs) == len(set(result_dirs)), \
            "Each geometry should have a unique result directory"

    def test_cache_key_stability(self):
        """Verify cache keys are stable for the same geometry."""
        provider_class = get_concrete_provider_class()

        # Create two providers with identical parameters
        provider1 = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            height_m=4096,
            rotation_deg=0.0,
            directory=self.cache_dir,
        )

        # Create a second instance with same parameters
        provider2 = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            height_m=4096,
            rotation_deg=0.0,
            directory=self.cache_dir,
        )

        # Cache keys should be identical
        assert provider1.cache_key == provider2.cache_key, \
            "Same geometry should produce same cache key"

    def test_rotation_affects_cache_key(self):
        """Verify that rotation changes the cache key."""
        provider_class = get_concrete_provider_class()

        provider1 = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            rotation_deg=0.0,
            directory=self.cache_dir,
        )

        provider2 = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            rotation_deg=1.0,
            directory=self.cache_dir,
        )

        # Different rotation should produce different cache key
        assert provider1.cache_key != provider2.cache_key, \
            "Different rotation should produce different cache keys"

        # But they should share source tiles
        assert provider1._source_tile_directory == provider2._source_tile_directory, \
            "Different rotation should share source tiles"

    def test_coordinate_rounding_tolerance(self):
        """Verify coordinate rounding tolerance (8 decimals)."""
        provider_class = get_concrete_provider_class()

        # These coordinates differ in the 9th decimal place (beyond rounding tolerance)
        provider1 = provider_class(
            coordinates=(40.00000000, 10.00000000),
            width_m=4096,
            directory=self.cache_dir,
        )

        provider2 = provider_class(
            coordinates=(40.000000001, 10.000000001),  # Beyond 8-decimal rounding
            width_m=4096,
            directory=self.cache_dir,
        )

        # Should have same cache key (within rounding tolerance)
        assert provider1.cache_key == provider2.cache_key, \
            "Coordinates differing beyond 8-decimal precision should be treated as identical"

    def test_coordinate_beyond_tolerance(self):
        """Verify coordinates beyond rounding tolerance create new cache keys."""
        provider_class = get_concrete_provider_class()

        provider1 = provider_class(
            coordinates=(40.0, 10.0),
            width_m=4096,
            directory=self.cache_dir,
        )

        provider2 = provider_class(
            coordinates=(40.00001, 10.0),  # 0.00001° = 5th decimal place
            width_m=4096,
            directory=self.cache_dir,
        )

        # Should have different cache keys
        assert provider1.cache_key != provider2.cache_key, \
            "Coordinates differing at 5th decimal place should produce different cache keys"

        # But they should share source tiles
        assert provider1._source_tile_directory == provider2._source_tile_directory, \
            "Different cache keys should still share source tiles"


class TestImageryProviderCache:
    """Test cache structure for imagery providers."""

    @pytest.fixture(autouse=True)
    def setup_and_cleanup(self):
        """Create temporary cache directory for tests."""
        self.cache_dir = tempfile.mkdtemp(prefix="pydtmdl_imagery_cache_test_")
        yield
        if os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir)

    def test_imagery_provider_source_directory(self):
        """Verify imagery providers also use _source_tile_directory."""
        try:
            from pydtmdl.base.imagery import ImageryProvider
        except ImportError:
            pytest.skip("ImageryProvider not available")

        providers = ImageryProvider.get_non_base_providers()
        if not providers:
            pytest.skip("No concrete imagery providers available")

        provider = providers[0](
            coordinates=(40.0, 10.0),
            width_m=4096,
            directory=self.cache_dir,
        )

        expected_source_dir = os.path.join(self.cache_dir, provider.code(), "_tiles")
        assert os.path.exists(expected_source_dir), \
            f"Imagery provider source directory not created: {expected_source_dir}"

        assert provider._source_tile_directory == expected_source_dir
