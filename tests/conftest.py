"""Shared test fixtures for chuk-mcp-dem."""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def sample_elevation():
    """100x100 elevation array with values 100-500m."""
    np.random.seed(42)
    return np.random.uniform(100, 500, (100, 100)).astype(np.float32)


@pytest.fixture
def sample_elevation_with_voids(sample_elevation):
    """Elevation array with NaN voids."""
    arr = sample_elevation.copy()
    arr[10:15, 10:15] = np.nan
    return arr


@pytest.fixture
def sample_transform():
    """Affine transform for a 1-degree tile at N46 E007."""
    from rasterio.transform import Affine

    return Affine(0.01, 0.0, 7.0, 0.0, -0.01, 47.0)


@pytest.fixture
def sample_crs():
    """EPSG:4326 CRS."""
    from rasterio.crs import CRS

    return CRS.from_epsg(4326)


@pytest.fixture
def mock_artifact_store():
    """Mock artifact store."""
    store = AsyncMock()
    store.store = AsyncMock(return_value=None)
    store.retrieve = AsyncMock(return_value=b"fake-geotiff-bytes")
    return store


@pytest.fixture
def mock_manager(mock_artifact_store):
    """DEMManager with mocked store."""
    from chuk_mcp_dem.core.dem_manager import DEMManager

    manager = DEMManager()
    manager._get_store = MagicMock(return_value=mock_artifact_store)
    return manager


@pytest.fixture
def mock_mcp():
    """Mock ChukMCPServer."""
    mcp = MagicMock()
    mcp.tool = MagicMock(return_value=lambda fn: fn)
    return mcp
