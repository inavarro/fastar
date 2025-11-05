import pytest
import os
from fastar.tools.assets import get_asset_path


@pytest.mark.unit
class TestAssetsFunctions:
    """Test suite for assets utility functions."""

    def test_get_asset_path_returns_absolute_path(self):
        """Test that get_asset_path returns an absolute path."""
        path = get_asset_path('filters_default.res')
        assert os.path.isabs(path), 'Path should be absolute'

    def test_get_asset_path_points_to_assets_dir(self):
        """Test that get_asset_path points to the assets directory."""
        path = get_asset_path('filters_default.res')
        assert 'assets' in path, "Path should contain 'assets' directory"

    def test_get_asset_path_returns_string(self):
        """Test that get_asset_path returns a string."""
        path = get_asset_path('test_file.txt')
        assert isinstance(path, str), 'Path should be a string'

    def test_get_asset_path_consistent_results(self):
        """Test that get_asset_path returns consistent results."""
        path1 = get_asset_path('filters_default.res')
        path2 = get_asset_path('filters_default.res')
        assert path1 == path2, 'Path should be consistent across calls'

    def test_get_asset_path_ends_with_filename(self):
        """Test that get_asset_path ends with the requested filename."""
        filename = 'filters_default.res'
        path = get_asset_path(filename)
        assert path.endswith(filename), f'Path should end with {filename}'
