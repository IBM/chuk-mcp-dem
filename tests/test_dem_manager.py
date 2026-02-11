"""
Comprehensive tests for DEMManager.

Covers discovery (sync), tile URL construction, validation,
async download methods (mocked), and LRU tile cache.
"""

from unittest.mock import patch

import numpy as np
import pytest

from chuk_mcp_dem.constants import (
    DEM_SOURCES,
)
from chuk_mcp_dem.core.dem_manager import (
    DEMManager,
    FeatureResult,
    FetchResult,
    MultiPointResult,
    PointResult,
    ProfileResult,
    TerrainResult,
    ViewshedResult,
    TemporalChangeResult,
    LandformResult,
    AnomalyResult,
)


# ===================================================================
# Discovery methods (sync)
# ===================================================================


class TestListSources:
    """Tests for DEMManager.list_sources()."""

    def test_returns_list_of_dicts(self, mock_manager):
        result = mock_manager.list_sources()
        assert isinstance(result, list)
        assert all(isinstance(item, dict) for item in result)

    def test_correct_count(self, mock_manager):
        result = mock_manager.list_sources()
        assert len(result) == 6

    def test_correct_keys(self, mock_manager):
        result = mock_manager.list_sources()
        expected_keys = {"id", "name", "resolution_m", "coverage", "vertical_datum", "void_filled"}
        for item in result:
            assert set(item.keys()) == expected_keys

    def test_cop30_present(self, mock_manager):
        result = mock_manager.list_sources()
        ids = [item["id"] for item in result]
        assert "cop30" in ids

    def test_all_source_ids_present(self, mock_manager):
        result = mock_manager.list_sources()
        ids = {item["id"] for item in result}
        assert ids == set(DEM_SOURCES.keys())


class TestDescribeSource:
    """Tests for DEMManager.describe_source()."""

    def test_valid_source_returns_full_metadata(self, mock_manager):
        result = mock_manager.describe_source("cop30")
        assert isinstance(result, dict)
        assert result["id"] == "cop30"
        assert result["name"] == "Copernicus GLO-30"
        assert result["resolution_m"] == 30
        assert "coverage_bounds" in result
        assert "llm_guidance" in result

    def test_returns_copy(self, mock_manager):
        """Ensure returned dict is a copy, not the original."""
        result = mock_manager.describe_source("cop30")
        result["id"] = "modified"
        original = mock_manager.describe_source("cop30")
        assert original["id"] == "cop30"

    def test_unknown_source_raises_value_error(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            mock_manager.describe_source("nonexistent")


class TestCheckCoverage:
    """Tests for DEMManager.check_coverage()."""

    def test_fully_covered_bbox(self, mock_manager):
        # cop30 has global coverage [-180, -90, 180, 90]
        bbox = [7.0, 46.0, 8.0, 47.0]
        result = mock_manager.check_coverage(bbox, "cop30")
        assert result["fully_covered"] is True
        assert result["coverage_percentage"] == 100.0

    def test_partially_covered_bbox(self, mock_manager):
        # SRTM covers [-180, -56, 180, 60], query spanning beyond 60N
        bbox = [7.0, 55.0, 8.0, 65.0]
        result = mock_manager.check_coverage(bbox, "srtm")
        assert result["fully_covered"] is False
        assert 0.0 < result["coverage_percentage"] < 100.0

    def test_no_overlap_returns_zero(self, mock_manager):
        # SRTM covers [-180, -56, 180, 60], query entirely above 60N
        bbox = [7.0, 70.0, 8.0, 80.0]
        result = mock_manager.check_coverage(bbox, "srtm")
        assert result["coverage_percentage"] == 0.0
        assert result["tiles_required"] == 0
        assert result["tile_ids"] == []

    def test_correct_tile_count(self, mock_manager):
        # 2x2 degree bbox -> 4 tiles for cop30 (tile_size_degrees=1.0)
        bbox = [7.0, 46.0, 9.0, 48.0]
        result = mock_manager.check_coverage(bbox, "cop30")
        assert result["tiles_required"] == 4
        assert len(result["tile_ids"]) == 4

    def test_tile_ids_populated(self, mock_manager):
        bbox = [7.0, 46.0, 8.0, 47.0]
        result = mock_manager.check_coverage(bbox, "cop30")
        assert result["tiles_required"] >= 1
        assert len(result["tile_ids"]) == result["tiles_required"]
        # tile IDs should be non-empty strings
        for tid in result["tile_ids"]:
            assert isinstance(tid, str)
            assert len(tid) > 0

    def test_estimated_size_positive(self, mock_manager):
        bbox = [7.0, 46.0, 8.0, 47.0]
        result = mock_manager.check_coverage(bbox, "cop30")
        assert result["estimated_size_mb"] > 0


class TestEstimateSize:
    """Tests for DEMManager.estimate_size()."""

    def test_correct_dimensions(self, mock_manager):
        bbox = [7.0, 46.0, 8.0, 47.0]
        result = mock_manager.estimate_size(bbox, "cop30")
        assert "dimensions" in result
        assert len(result["dimensions"]) == 2
        n_rows, n_cols = result["dimensions"]
        assert n_rows > 0
        assert n_cols > 0

    def test_estimated_mb_positive(self, mock_manager):
        bbox = [7.0, 46.0, 8.0, 47.0]
        result = mock_manager.estimate_size(bbox, "cop30")
        assert result["estimated_mb"] > 0
        assert result["estimated_bytes"] > 0

    def test_warning_for_large_area(self, mock_manager):
        # Very large area: 90 degrees x 90 degrees -> should trigger warning
        bbox = [-45.0, -45.0, 45.0, 45.0]
        result = mock_manager.estimate_size(bbox, "cop30")
        assert result["warning"] is not None

    def test_no_warning_for_small_area(self, mock_manager):
        bbox = [7.0, 46.0, 7.1, 46.1]
        result = mock_manager.estimate_size(bbox, "cop30")
        assert result["warning"] is None

    def test_unknown_source_raises_value_error(self, mock_manager):
        bbox = [7.0, 46.0, 8.0, 47.0]
        with pytest.raises(ValueError, match="Unknown DEM source"):
            mock_manager.estimate_size(bbox, "nonexistent")

    def test_custom_resolution(self, mock_manager):
        bbox = [7.0, 46.0, 8.0, 47.0]
        native = mock_manager.estimate_size(bbox, "cop30")
        coarser = mock_manager.estimate_size(bbox, "cop30", resolution_m=90.0)
        # Coarser resolution -> fewer pixels
        assert coarser["pixels"] < native["pixels"]

    def test_result_keys(self, mock_manager):
        bbox = [7.0, 46.0, 8.0, 47.0]
        result = mock_manager.estimate_size(bbox, "cop30")
        expected_keys = {
            "native_resolution_m",
            "target_resolution_m",
            "dimensions",
            "pixels",
            "dtype",
            "estimated_bytes",
            "estimated_mb",
            "warning",
        }
        assert set(result.keys()) == expected_keys


# ===================================================================
# Tile URL construction
# ===================================================================


class TestMakeTileUrl:
    """Tests for DEMManager._make_tile_url()."""

    def test_cop30_positive_lat_lon(self, mock_manager):
        url = mock_manager._make_tile_url("cop30", 46, 7)
        assert url is not None
        assert "copernicus-dem-30m" in url
        assert "N46" in url
        assert "E007" in url
        assert url.endswith(".tif")

    def test_cop30_negative_lat_lon(self, mock_manager):
        url = mock_manager._make_tile_url("cop30", -33, -70)
        assert url is not None
        assert "S33" in url
        assert "W070" in url

    def test_cop90_different_url_pattern(self, mock_manager):
        url = mock_manager._make_tile_url("cop90", 46, 7)
        assert url is not None
        assert "copernicus-dem-90m" in url
        assert "COG_30" in url  # cop90 uses COG_30 (30 arcsecond)

    def test_cop30_vs_cop90_different_prefix(self, mock_manager):
        url30 = mock_manager._make_tile_url("cop30", 46, 7)
        url90 = mock_manager._make_tile_url("cop90", 46, 7)
        assert "COG_10" in url30  # cop30 uses COG_10 (10 metre / 1 arcsecond)
        assert "COG_30" in url90

    def test_unknown_source_returns_none(self, mock_manager):
        url = mock_manager._make_tile_url("nonexistent", 46, 7)
        assert url is None

    def test_srtm_positive_lat_lon(self, mock_manager):
        url = mock_manager._make_tile_url("srtm", 46, 7)
        assert url is not None
        assert "elevation-tiles-prod" in url
        assert "/skadi/N46/" in url
        assert "N46E007.hgt.gz" in url

    def test_srtm_negative_lat_lon(self, mock_manager):
        url = mock_manager._make_tile_url("srtm", -33, -70)
        assert url is not None
        assert "S33W070.hgt.gz" in url

    def test_3dep_positive_lat_lon(self, mock_manager):
        url = mock_manager._make_tile_url("3dep", 46, -121)
        assert url is not None
        assert "prd-tnm" in url
        assert "USGS_13_" in url
        assert "n46w121" in url
        assert url.endswith(".tif")

    def test_3dep_negative_lat(self, mock_manager):
        """3DEP uses lowercase n/s e/w."""
        url = mock_manager._make_tile_url("3dep", 35, -106)
        assert url is not None
        assert "n35w106" in url

    def test_fabdem_positive_lat_lon(self, mock_manager):
        url = mock_manager._make_tile_url("fabdem", 46, 7)
        assert url is not None
        assert "data.bris.ac.uk" in url
        assert "N46E007_FABDEM_V1-2.tif" in url

    def test_fabdem_negative_lat_lon(self, mock_manager):
        url = mock_manager._make_tile_url("fabdem", -33, -70)
        assert url is not None
        assert "S33W070_FABDEM_V1-2.tif" in url

    def test_aster_returns_none(self, mock_manager):
        """ASTER requires NASA Earthdata auth, returns None."""
        url = mock_manager._make_tile_url("aster", 46, 7)
        assert url is None


class TestMakeTileId:
    """Tests for DEMManager._make_tile_id()."""

    def test_cop30_tile_id(self, mock_manager):
        tile_id = mock_manager._make_tile_id("cop30", 46, 7)
        assert tile_id == "Copernicus_DSM_COG_10_N46_00_E007_00"

    def test_cop90_tile_id(self, mock_manager):
        tile_id = mock_manager._make_tile_id("cop90", 46, 7)
        assert tile_id == "Copernicus_DSM_COG_10_N46_00_E007_00"

    def test_other_source_tile_id(self, mock_manager):
        tile_id = mock_manager._make_tile_id("srtm", 46, 7)
        assert tile_id == "N46E007"

    def test_negative_coords(self, mock_manager):
        tile_id = mock_manager._make_tile_id("srtm", -33, -70)
        assert tile_id == "S33W070"


class TestGetTileUrls:
    """Tests for DEMManager._get_tile_urls()."""

    def test_single_tile_bbox(self, mock_manager):
        bbox = [7.0, 46.0, 7.5, 46.5]
        urls = mock_manager._get_tile_urls("cop30", bbox)
        assert len(urls) == 1
        assert "N46" in urls[0]
        assert "E007" in urls[0]

    def test_multi_tile_bbox(self, mock_manager):
        bbox = [7.0, 46.0, 9.0, 48.0]
        urls = mock_manager._get_tile_urls("cop30", bbox)
        assert len(urls) == 4  # 2x2 grid


class TestGetTileIds:
    """Tests for DEMManager._get_tile_ids()."""

    def test_matching_tile_count(self, mock_manager):
        bbox = [7.0, 46.0, 9.0, 48.0]
        ids = mock_manager._get_tile_ids("cop30", bbox)
        urls = mock_manager._get_tile_urls("cop30", bbox)
        assert len(ids) == len(urls)

    def test_single_tile(self, mock_manager):
        bbox = [7.0, 46.0, 7.5, 46.5]
        ids = mock_manager._get_tile_ids("cop30", bbox)
        assert len(ids) == 1


# ===================================================================
# Validation
# ===================================================================


class TestValidateBbox:
    """Tests for DEMManager._validate_bbox()."""

    def test_valid_bbox_passes(self, mock_manager):
        # Should not raise
        mock_manager._validate_bbox([7.0, 46.0, 8.0, 47.0])

    def test_wrong_length_raises(self, mock_manager):
        with pytest.raises(ValueError, match="Invalid bounding box"):
            mock_manager._validate_bbox([7.0, 46.0, 8.0])

    def test_five_elements_raises(self, mock_manager):
        with pytest.raises(ValueError, match="Invalid bounding box"):
            mock_manager._validate_bbox([7.0, 46.0, 8.0, 47.0, 0.0])

    def test_west_gte_east_raises(self, mock_manager):
        with pytest.raises(ValueError, match="west.*east"):
            mock_manager._validate_bbox([8.0, 46.0, 7.0, 47.0])

    def test_west_equals_east_raises(self, mock_manager):
        with pytest.raises(ValueError, match="west.*east"):
            mock_manager._validate_bbox([7.0, 46.0, 7.0, 47.0])

    def test_south_gte_north_raises(self, mock_manager):
        with pytest.raises(ValueError, match="south.*north"):
            mock_manager._validate_bbox([7.0, 47.0, 8.0, 46.0])

    def test_south_equals_north_raises(self, mock_manager):
        with pytest.raises(ValueError, match="south.*north"):
            mock_manager._validate_bbox([7.0, 46.0, 8.0, 46.0])


# ===================================================================
# Async download methods (mocked)
# ===================================================================


class TestFetchElevation:
    """Tests for DEMManager.fetch_elevation()."""

    async def test_fetch_elevation_stores_artifact(
        self, mock_manager, mock_artifact_store, sample_elevation, sample_crs, sample_transform
    ):
        geotiff_bytes = b"fake-geotiff-bytes"
        png_bytes = b"fake-png-bytes"

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=geotiff_bytes,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.elevation_to_hillshade_png",
                return_value=png_bytes,
            ),
        ):
            result = await mock_manager.fetch_elevation(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert isinstance(result, FetchResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.artifact_ref.endswith(".tif")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30
        assert result.shape == list(sample_elevation.shape)
        assert result.dtype == "float32"
        # Verify store was called (geotiff + hillshade)
        assert mock_artifact_store.store.call_count == 2

    async def test_fetch_elevation_elevation_range(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"bytes",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.elevation_to_hillshade_png",
                return_value=b"png",
            ),
        ):
            result = await mock_manager.fetch_elevation(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert len(result.elevation_range) == 2
        assert result.elevation_range[0] <= result.elevation_range[1]
        assert result.elevation_range[0] >= 100.0
        assert result.elevation_range[1] <= 500.0

    async def test_fetch_elevation_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_elevation(bbox=[7.0, 46.0, 8.0, 47.0], source="bad_source")

    async def test_fetch_elevation_empty_tile_urls(self, mock_manager):
        """Source exists but _make_tile_url returns None (ASTER requires auth)."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_elevation(bbox=[7.0, 46.0, 8.0, 47.0], source="aster")

    async def test_fetch_elevation_nodata_counted(self, mock_manager, sample_crs, sample_transform):
        elev = np.full((10, 10), 100.0, dtype=np.float32)
        elev[0, 0] = np.nan
        elev[0, 1] = np.nan

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(elev, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=np.full((10, 10), 100.0, dtype=np.float32),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"bytes",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.elevation_to_hillshade_png",
                return_value=b"png",
            ),
        ):
            result = await mock_manager.fetch_elevation(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", fill_voids=True
            )

        # After fill_voids the elevation is fully filled, but nodata_count is
        # computed from the filled array returned by fill_voids (all 100.0).
        # The code computes nodata from the filled elevation, so should be 0.
        assert result.nodata_pixels == 0

    async def test_fetch_elevation_preview_ref(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"bytes",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.elevation_to_hillshade_png",
                return_value=b"preview-png",
            ),
        ):
            result = await mock_manager.fetch_elevation(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert result.preview_ref is not None
        assert "_hillshade.png" in result.preview_ref


class TestFetchPoint:
    """Tests for DEMManager.fetch_point()."""

    async def test_fetch_point_returns_elevation(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_dem_tile",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.sample_elevation",
                return_value=250.5,
            ),
        ):
            result = await mock_manager.fetch_point(lon=7.5, lat=46.5, source="cop30")

        assert isinstance(result, PointResult)
        assert result.elevation_m == 250.5
        assert result.uncertainty_m == 4.0  # cop30 accuracy

    async def test_fetch_point_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_point(lon=7.5, lat=46.5, source="bad")

    async def test_fetch_point_no_coverage(self, mock_manager):
        """Source with no tile URL constructor (aster) has empty tile_urls."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_point(lon=7.5, lat=46.5, source="aster")


class TestFetchPoints:
    """Tests for DEMManager.fetch_points()."""

    async def test_fetch_points_multiple_elevations(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.sample_elevations",
                return_value=[100.0, 200.0, 300.0],
            ),
        ):
            result = await mock_manager.fetch_points(
                points=[[7.1, 46.1], [7.5, 46.5], [7.9, 46.9]],
                source="cop30",
            )

        assert isinstance(result, MultiPointResult)
        assert len(result.elevations) == 3
        assert result.elevations == [100.0, 200.0, 300.0]
        assert result.elevation_range == [100.0, 300.0]

    async def test_fetch_points_with_nan(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.sample_elevations",
                return_value=[float("nan"), float("nan")],
            ),
        ):
            result = await mock_manager.fetch_points(
                points=[[7.1, 46.1], [7.5, 46.5]],
                source="cop30",
            )

        # All NaN -> range defaults to [0.0, 0.0]
        assert result.elevation_range == [0.0, 0.0]

    async def test_fetch_points_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_points(points=[[7.1, 46.1]], source="nonexistent")


# ===================================================================
# Cache management
# ===================================================================


class TestCacheTile:
    """Tests for DEMManager._cache_tile() and _get_cached_tile()."""

    def test_stores_data_and_updates_total(self, mock_manager):
        data = b"x" * 100
        mock_manager._cache_tile("tile_a", data)
        assert "tile_a" in mock_manager._tile_cache
        assert mock_manager._tile_cache_total == 100

    def test_get_cached_tile_returns_data(self, mock_manager):
        data = b"elevation-data"
        mock_manager._cache_tile("tile_b", data)
        result = mock_manager._get_cached_tile("tile_b")
        assert result == data

    def test_get_cached_tile_moves_to_lru_end(self, mock_manager):
        mock_manager._cache_tile("tile_1", b"a")
        mock_manager._cache_tile("tile_2", b"b")
        mock_manager._cache_tile("tile_3", b"c")

        # Access tile_1 to move it to end
        mock_manager._get_cached_tile("tile_1")

        keys = list(mock_manager._tile_cache.keys())
        assert keys[-1] == "tile_1"

    def test_get_cached_tile_returns_none_for_missing(self, mock_manager):
        result = mock_manager._get_cached_tile("nonexistent_tile")
        assert result is None

    def test_evicts_oldest_when_full(self):
        """Set a low max bytes and verify oldest tile is evicted."""
        manager = DEMManager()

        # Temporarily patch the max bytes constant
        with patch("chuk_mcp_dem.core.dem_manager.TILE_CACHE_MAX_BYTES", 200):
            manager._cache_tile("old_tile", b"x" * 100)
            manager._cache_tile("new_tile", b"y" * 150)

            # old_tile should have been evicted to make room
            assert "old_tile" not in manager._tile_cache
            assert "new_tile" in manager._tile_cache
            assert manager._tile_cache_total == 150

    def test_rejects_items_larger_than_max_item(self):
        """Items exceeding TILE_CACHE_MAX_ITEM are silently rejected."""
        manager = DEMManager()

        with patch("chuk_mcp_dem.core.dem_manager.TILE_CACHE_MAX_ITEM", 50):
            manager._cache_tile("big_tile", b"x" * 100)
            assert "big_tile" not in manager._tile_cache
            assert manager._tile_cache_total == 0

    def test_multiple_evictions(self):
        """Multiple old tiles evicted when needed."""
        manager = DEMManager()

        with patch("chuk_mcp_dem.core.dem_manager.TILE_CACHE_MAX_BYTES", 300):
            manager._cache_tile("a", b"x" * 100)
            manager._cache_tile("b", b"y" * 100)
            manager._cache_tile("c", b"z" * 100)
            # Cache is now 300/300; adding a 150-byte tile requires evicting a+b
            manager._cache_tile("d", b"w" * 150)

            assert "a" not in manager._tile_cache
            assert "b" not in manager._tile_cache
            assert "c" in manager._tile_cache
            assert "d" in manager._tile_cache
            assert manager._tile_cache_total == 250

    def test_cache_tile_overwrite(self, mock_manager):
        """Caching the same key twice updates the data."""
        mock_manager._cache_tile("key", b"old")
        mock_manager._cache_tile("key", b"new_data")
        # The second write is simply appended (dict allows duplicate insert
        # without explicit delete). Total increases because the code doesn't
        # check for duplicates. This documents the current behavior.
        assert mock_manager._tile_cache["key"] == b"new_data"


# ===================================================================
# Edge cases and internal helpers
# ===================================================================


class TestGetSource:
    """Tests for DEMManager._get_source()."""

    def test_valid_source(self, mock_manager):
        src = mock_manager._get_source("cop30")
        assert src["id"] == "cop30"
        assert src["resolution_m"] == 30

    def test_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            mock_manager._get_source("missing")


class TestDefaultSource:
    """Tests for default source initialization."""

    def test_default_source_is_cop30(self):
        manager = DEMManager()
        assert manager.default_source == "cop30"

    def test_custom_default_source(self):
        manager = DEMManager(default_source="cop90")
        assert manager.default_source == "cop90"


class TestDataclasses:
    """Tests for result dataclasses."""

    def test_fetch_result_fields(self):
        result = FetchResult(
            artifact_ref="dem/abc.tif",
            preview_ref="dem/abc_hillshade.png",
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            elevation_range=[100.0, 500.0],
            dtype="float32",
            nodata_pixels=5,
        )
        assert result.artifact_ref == "dem/abc.tif"
        assert result.nodata_pixels == 5

    def test_point_result_fields(self):
        result = PointResult(elevation_m=1234.5, uncertainty_m=4.0)
        assert result.elevation_m == 1234.5

    def test_multi_point_result_fields(self):
        result = MultiPointResult(
            elevations=[100.0, 200.0],
            elevation_range=[100.0, 200.0],
        )
        assert len(result.elevations) == 2

    def test_terrain_result_fields(self):
        result = TerrainResult(
            artifact_ref="dem/hs_abc.tif",
            preview_ref="dem/hs_abc_preview.png",
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            value_range=[0.0, 254.0],
            dtype="float32",
        )
        assert result.artifact_ref == "dem/hs_abc.tif"
        assert result.preview_ref == "dem/hs_abc_preview.png"
        assert result.dtype == "float32"

    def test_terrain_result_no_preview(self):
        result = TerrainResult(
            artifact_ref="dem/slope.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[50, 50],
            value_range=[0.0, 89.0],
            dtype="float32",
        )
        assert result.preview_ref is None

    def test_profile_result_fields(self):
        result = ProfileResult(
            longitudes=[7.0, 7.5, 8.0],
            latitudes=[46.0, 46.5, 47.0],
            distances_m=[0.0, 5000.0, 10000.0],
            elevations=[1000.0, 1500.0, 1200.0],
            total_distance_m=10000.0,
            elevation_range=[1000.0, 1500.0],
            elevation_gain_m=500.0,
            elevation_loss_m=300.0,
        )
        assert len(result.longitudes) == 3
        assert result.total_distance_m == 10000.0
        assert result.elevation_gain_m == 500.0
        assert result.elevation_loss_m == 300.0

    def test_viewshed_result_fields(self):
        result = ViewshedResult(
            artifact_ref="dem/vs_abc.tif",
            preview_ref="dem/vs_preview.png",
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[200, 200],
            visible_percentage=45.2,
            observer_elevation_m=1500.0,
            radius_m=5000.0,
        )
        assert result.artifact_ref == "dem/vs_abc.tif"
        assert result.visible_percentage == 45.2
        assert result.observer_elevation_m == 1500.0
        assert result.radius_m == 5000.0

    def test_viewshed_result_no_preview(self):
        result = ViewshedResult(
            artifact_ref="dem/vs.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            visible_percentage=0.0,
            observer_elevation_m=0.0,
            radius_m=1000.0,
        )
        assert result.preview_ref is None


# ===================================================================
# Terrain analysis: fetch_hillshade
# ===================================================================


class TestFetchHillshade:
    """Tests for DEMManager.fetch_hillshade()."""

    async def test_fetch_hillshade_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        hillshade_arr = np.ones((100, 100), dtype=np.float32) * 127.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_hillshade",
                return_value=hillshade_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.elevation_to_hillshade_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_hillshade(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert isinstance(result, TerrainResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30
        assert result.shape == [100, 100]
        assert result.dtype == "float32"

    async def test_fetch_hillshade_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_hillshade(bbox=[7.0, 46.0, 8.0, 47.0], source="bad_source")

    async def test_fetch_hillshade_custom_params(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        hillshade_arr = np.ones((100, 100), dtype=np.float32) * 200.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_hillshade",
                return_value=hillshade_arr,
            ) as mock_hs,
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.elevation_to_hillshade_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_hillshade(
                bbox=[7.0, 46.0, 8.0, 47.0],
                source="cop30",
                azimuth=180.0,
                altitude=30.0,
                z_factor=2.0,
            )

        # Verify custom params were passed to compute_hillshade
        mock_hs.assert_called_once()
        call_args = mock_hs.call_args
        assert call_args[0][2] == 180.0  # azimuth
        assert call_args[0][3] == 30.0  # altitude
        assert call_args[0][4] == 2.0  # z_factor
        assert result.value_range == [200.0, 200.0]

    async def test_fetch_hillshade_value_range(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        hillshade_arr = np.array([[10.0, 250.0], [50.0, 200.0]], dtype=np.float32)

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_hillshade",
                return_value=hillshade_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.elevation_to_hillshade_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_hillshade(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert result.value_range[0] == pytest.approx(10.0)
        assert result.value_range[1] == pytest.approx(250.0)

    async def test_fetch_hillshade_no_coverage(self, mock_manager):
        """Source with no tile URL constructor returns empty tile_urls."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_hillshade(bbox=[7.0, 46.0, 8.0, 47.0], source="aster")

    async def test_fetch_hillshade_png_output(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        hillshade_arr = np.ones((100, 100), dtype=np.float32) * 127.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_hillshade",
                return_value=hillshade_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.elevation_to_hillshade_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_hillshade(
                bbox=[7.0, 46.0, 8.0, 47.0],
                source="cop30",
                output_format="png",
            )

        assert result.artifact_ref.endswith(".png")
        # No separate preview for png output
        assert result.preview_ref is None


# ===================================================================
# Terrain analysis: fetch_slope
# ===================================================================


class TestFetchSlope:
    """Tests for DEMManager.fetch_slope()."""

    async def test_fetch_slope_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        slope_arr = np.ones((100, 100), dtype=np.float32) * 15.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_slope",
                return_value=slope_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.slope_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_slope(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert isinstance(result, TerrainResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30
        assert result.value_range == [15.0, 15.0]

    async def test_fetch_slope_default_units(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        slope_arr = np.ones((100, 100), dtype=np.float32) * 20.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_slope",
                return_value=slope_arr,
            ) as mock_slope,
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.slope_to_png",
                return_value=b"fake-png",
            ),
        ):
            await mock_manager.fetch_slope(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        # Default units is "degrees"
        mock_slope.assert_called_once()
        assert mock_slope.call_args[0][2] == "degrees"

    async def test_fetch_slope_custom_units(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        slope_arr = np.ones((100, 100), dtype=np.float32) * 50.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_slope",
                return_value=slope_arr,
            ) as mock_slope,
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.slope_to_png",
                return_value=b"fake-png",
            ),
        ):
            await mock_manager.fetch_slope(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", units="percent"
            )

        mock_slope.assert_called_once()
        assert mock_slope.call_args[0][2] == "percent"

    async def test_fetch_slope_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_slope(bbox=[7.0, 46.0, 8.0, 47.0], source="nonexistent")


# ===================================================================
# Terrain analysis: fetch_aspect
# ===================================================================


class TestFetchAspect:
    """Tests for DEMManager.fetch_aspect()."""

    async def test_fetch_aspect_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        aspect_arr = np.ones((100, 100), dtype=np.float32) * 180.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_aspect",
                return_value=aspect_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.aspect_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_aspect(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert isinstance(result, TerrainResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.value_range == [180.0, 180.0]

    async def test_fetch_aspect_custom_flat_value(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        aspect_arr = np.ones((100, 100), dtype=np.float32) * 90.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_aspect",
                return_value=aspect_arr,
            ) as mock_asp,
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.aspect_to_png",
                return_value=b"fake-png",
            ),
        ):
            await mock_manager.fetch_aspect(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", flat_value=-9999.0
            )

        mock_asp.assert_called_once()
        assert mock_asp.call_args[0][2] == -9999.0

    async def test_fetch_aspect_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_aspect(bbox=[7.0, 46.0, 8.0, 47.0], source="bad")


# ===================================================================
# Terrain analysis: fetch_curvature
# ===================================================================


class TestFetchCurvature:
    """Tests for DEMManager.fetch_curvature()."""

    async def test_fetch_curvature_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        curv_arr = np.ones((100, 100), dtype=np.float32) * 0.005

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_curvature",
                return_value=curv_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.curvature_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_curvature(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert isinstance(result, TerrainResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30
        assert result.value_range[0] == pytest.approx(0.005, rel=1e-5)
        assert result.value_range[1] == pytest.approx(0.005, rel=1e-5)

    async def test_fetch_curvature_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_curvature(bbox=[7.0, 46.0, 8.0, 47.0], source="nonexistent")

    async def test_fetch_curvature_png_format(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        curv_arr = np.ones((100, 100), dtype=np.float32) * 0.01

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_curvature",
                return_value=curv_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.curvature_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_curvature(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", output_format="png"
            )

        assert isinstance(result, TerrainResult)
        assert result.artifact_ref.startswith("dem/")


# ===================================================================
# Terrain analysis: fetch_tri
# ===================================================================


class TestFetchTRI:
    """Tests for DEMManager.fetch_tri()."""

    async def test_fetch_tri_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        tri_arr = np.ones((100, 100), dtype=np.float32) * 50.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_tri",
                return_value=tri_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.tri_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_tri(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

        assert isinstance(result, TerrainResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30
        assert result.value_range == [50.0, 50.0]

    async def test_fetch_tri_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_tri(bbox=[7.0, 46.0, 8.0, 47.0], source="nonexistent")

    async def test_fetch_tri_png_format(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        tri_arr = np.ones((100, 100), dtype=np.float32) * 75.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_tri",
                return_value=tri_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.tri_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_tri(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", output_format="png"
            )

        assert isinstance(result, TerrainResult)
        assert result.artifact_ref.startswith("dem/")


# ===================================================================
# Contour: fetch_contours
# ===================================================================


class TestFetchContours:
    """Tests for DEMManager.fetch_contours()."""

    async def test_fetch_contours_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        from unittest.mock import patch, MagicMock

        contour_array = np.full_like(sample_elevation, np.nan, dtype=np.float32)
        contour_array[10, :] = 200.0
        contour_array[50, :] = 300.0
        levels = [200.0, 300.0]

        with (
            patch("chuk_mcp_dem.core.raster_io.read_and_merge_tiles") as mock_read,
            patch("chuk_mcp_dem.core.raster_io.fill_voids") as mock_fill,
            patch("chuk_mcp_dem.core.raster_io.compute_contours") as mock_contour,
            patch("chuk_mcp_dem.core.raster_io.arrays_to_geotiff") as mock_tiff,
            patch("chuk_mcp_dem.core.raster_io.contours_to_png") as mock_png,
        ):
            mock_read.return_value = (sample_elevation, sample_crs, sample_transform)
            mock_fill.return_value = sample_elevation
            mock_contour.return_value = (contour_array, levels)
            mock_tiff.return_value = b"fake-tiff"
            mock_png.return_value = b"\x89PNG-fake"

            mock_manager._get_tile_urls = MagicMock(return_value=["https://example.com/tile.tif"])

            result = await mock_manager.fetch_contours(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", interval_m=100.0
            )

            assert result.artifact_ref is not None
            assert result.interval_m == 100.0
            assert result.contour_count == 2
            assert result.crs == str(sample_crs)
            assert result.shape == list(contour_array.shape)

    async def test_fetch_contours_png_output(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        from unittest.mock import patch, MagicMock

        contour_array = np.full_like(sample_elevation, np.nan, dtype=np.float32)
        levels = [200.0]

        with (
            patch("chuk_mcp_dem.core.raster_io.read_and_merge_tiles") as mock_read,
            patch("chuk_mcp_dem.core.raster_io.fill_voids") as mock_fill,
            patch("chuk_mcp_dem.core.raster_io.compute_contours") as mock_contour,
            patch("chuk_mcp_dem.core.raster_io.contours_to_png") as mock_png,
        ):
            mock_read.return_value = (sample_elevation, sample_crs, sample_transform)
            mock_fill.return_value = sample_elevation
            mock_contour.return_value = (contour_array, levels)
            mock_png.return_value = b"\x89PNG-fake"

            mock_manager._get_tile_urls = MagicMock(return_value=["https://example.com/tile.tif"])

            result = await mock_manager.fetch_contours(
                bbox=[7.0, 46.0, 8.0, 47.0],
                source="cop30",
                interval_m=100.0,
                output_format="png",
            )

            assert result.artifact_ref is not None
            mock_png.assert_called_once()

    async def test_fetch_contours_empty_tiles(self, mock_manager):
        from unittest.mock import MagicMock

        mock_manager._get_tile_urls = MagicMock(return_value=[])
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_contours(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", interval_m=100.0
            )


# ===================================================================
# Watershed: fetch_watershed
# ===================================================================


class TestFetchWatershed:
    """Tests for DEMManager.fetch_watershed()."""

    async def test_fetch_watershed_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        from unittest.mock import patch, MagicMock

        accum_arr = np.ones((100, 100), dtype=np.float32) * 10.0
        accum_arr[50, 50] = 5000.0

        with (
            patch("chuk_mcp_dem.core.raster_io.read_and_merge_tiles") as mock_read,
            patch("chuk_mcp_dem.core.raster_io.fill_voids") as mock_fill,
            patch("chuk_mcp_dem.core.raster_io.compute_flow_accumulation") as mock_accum,
            patch("chuk_mcp_dem.core.raster_io.arrays_to_geotiff") as mock_tiff,
            patch("chuk_mcp_dem.core.raster_io.watershed_to_png") as mock_png,
        ):
            mock_read.return_value = (sample_elevation, sample_crs, sample_transform)
            mock_fill.return_value = sample_elevation
            mock_accum.return_value = accum_arr
            mock_tiff.return_value = b"fake-tiff"
            mock_png.return_value = b"\x89PNG-fake"

            mock_manager._get_tile_urls = MagicMock(return_value=["https://example.com/tile.tif"])

            result = await mock_manager.fetch_watershed(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

            assert result.artifact_ref is not None
            assert result.crs == str(sample_crs)
            assert result.shape == [100, 100]
            assert result.value_range == [pytest.approx(10.0), pytest.approx(5000.0)]
            assert result.dtype == "float32"

    async def test_fetch_watershed_png_output(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        from unittest.mock import patch, MagicMock

        accum_arr = np.ones((100, 100), dtype=np.float32)

        with (
            patch("chuk_mcp_dem.core.raster_io.read_and_merge_tiles") as mock_read,
            patch("chuk_mcp_dem.core.raster_io.fill_voids") as mock_fill,
            patch("chuk_mcp_dem.core.raster_io.compute_flow_accumulation") as mock_accum,
            patch("chuk_mcp_dem.core.raster_io.watershed_to_png") as mock_png,
        ):
            mock_read.return_value = (sample_elevation, sample_crs, sample_transform)
            mock_fill.return_value = sample_elevation
            mock_accum.return_value = accum_arr
            mock_png.return_value = b"\x89PNG-fake"

            mock_manager._get_tile_urls = MagicMock(return_value=["https://example.com/tile.tif"])

            result = await mock_manager.fetch_watershed(
                bbox=[7.0, 46.0, 8.0, 47.0],
                source="cop30",
                output_format="png",
            )

            assert result.artifact_ref is not None
            mock_png.assert_called_once()

    async def test_fetch_watershed_empty_tiles(self, mock_manager):
        from unittest.mock import MagicMock

        mock_manager._get_tile_urls = MagicMock(return_value=[])
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_watershed(bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")

    async def test_fetch_watershed_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_watershed(bbox=[7.0, 46.0, 8.0, 47.0], source="invalid_source")


# ===================================================================
# Profile: fetch_profile
# ===================================================================


class TestFetchProfile:
    """Tests for DEMManager.fetch_profile()."""

    async def test_fetch_profile_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        mock_lons = [7.0, 7.5, 8.0]
        mock_lats = [46.0, 46.5, 47.0]
        mock_distances = [0.0, 5000.0, 10000.0]
        mock_elevations = [1000.0, 1500.0, 1200.0]

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_profile_points",
                return_value=(mock_lons, mock_lats, mock_distances, mock_elevations),
            ),
        ):
            result = await mock_manager.fetch_profile(
                start=[7.0, 46.0], end=[8.0, 47.0], source="cop30", num_points=3
            )

        assert isinstance(result, ProfileResult)
        assert result.longitudes == mock_lons
        assert result.latitudes == mock_lats
        assert result.distances_m == mock_distances
        assert result.elevations == mock_elevations
        assert result.total_distance_m == 10000.0
        assert result.elevation_range == [1000.0, 1500.0]

    async def test_fetch_profile_gain_loss(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        """Verify gain and loss are computed correctly."""
        # Elevation goes: 100 -> 300 -> 200 -> 400
        # gain = (300-100) + (400-200) = 200 + 200 = 400
        # loss = (200-300) = 100
        mock_elevations = [100.0, 300.0, 200.0, 400.0]

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_profile_points",
                return_value=(
                    [7.0, 7.33, 7.66, 8.0],
                    [46.0, 46.33, 46.66, 47.0],
                    [0.0, 3000.0, 6000.0, 10000.0],
                    mock_elevations,
                ),
            ),
        ):
            result = await mock_manager.fetch_profile(
                start=[7.0, 46.0], end=[8.0, 47.0], source="cop30", num_points=4
            )

        assert result.elevation_gain_m == pytest.approx(400.0)
        assert result.elevation_loss_m == pytest.approx(100.0)

    async def test_fetch_profile_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_profile(
                start=[7.0, 46.0], end=[8.0, 47.0], source="nonexistent"
            )

    async def test_fetch_profile_no_coverage(self, mock_manager):
        """Source with no tile URL constructor raises."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_profile(start=[7.0, 46.0], end=[8.0, 47.0], source="aster")

    async def test_fetch_profile_all_nan(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        """When all elevations are NaN, range defaults to [0.0, 0.0]."""
        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_profile_points",
                return_value=(
                    [7.0, 8.0],
                    [46.0, 47.0],
                    [0.0, 10000.0],
                    [float("nan"), float("nan")],
                ),
            ),
        ):
            result = await mock_manager.fetch_profile(
                start=[7.0, 46.0], end=[8.0, 47.0], source="cop30", num_points=2
            )

        assert result.elevation_range == [0.0, 0.0]
        assert result.elevation_gain_m == 0.0
        assert result.elevation_loss_m == 0.0


# ===================================================================
# Viewshed: fetch_viewshed
# ===================================================================


class TestFetchViewshed:
    """Tests for DEMManager.fetch_viewshed()."""

    async def test_fetch_viewshed_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        # Create a viewshed array: 60% visible, 40% hidden within radius
        vs_arr = np.full((100, 100), np.nan, dtype=np.float32)
        vs_arr[40:60, 40:60] = 1.0  # 400 visible
        vs_arr[60:70, 40:60] = 0.0  # 200 hidden
        # rest is NaN (outside radius)

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_viewshed",
                return_value=vs_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
        ):
            result = await mock_manager.fetch_viewshed(
                observer=[7.5, 46.5], radius_m=5000.0, source="cop30"
            )

        assert isinstance(result, ViewshedResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30
        assert result.radius_m == 5000.0
        # 400 visible out of 600 analyzed = 66.7%
        assert result.visible_percentage == pytest.approx(66.7, abs=0.1)

    async def test_fetch_viewshed_radius_too_large(self, mock_manager):
        with pytest.raises(ValueError, match="exceeds maximum"):
            await mock_manager.fetch_viewshed(
                observer=[7.5, 46.5], radius_m=100000.0, source="cop30"
            )

    async def test_fetch_viewshed_observer_elevation(
        self, mock_manager, sample_crs, sample_transform
    ):
        """Observer elevation is read from the elevation array at observer position."""
        elev = np.full((100, 100), 500.0, dtype=np.float32)
        vs_arr = np.full((100, 100), np.nan, dtype=np.float32)
        vs_arr[50, 50] = 1.0

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(elev, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_viewshed",
                return_value=vs_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
        ):
            result = await mock_manager.fetch_viewshed(
                observer=[7.5, 46.5], radius_m=1000.0, source="cop30"
            )

        # The observer elevation should be read from the elevation grid
        # Since we filled with 500.0, we expect it to be 500.0
        assert result.observer_elevation_m == pytest.approx(500.0)

    async def test_fetch_viewshed_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_viewshed(
                observer=[7.5, 46.5], radius_m=5000.0, source="bad_source"
            )

    async def test_fetch_viewshed_no_coverage(self, mock_manager):
        """Source with no tile URL constructor raises."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_viewshed(observer=[7.5, 46.5], radius_m=1000.0, source="aster")

    async def test_fetch_viewshed_all_visible(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        """100% visibility case."""
        vs_arr = np.ones((100, 100), dtype=np.float32)  # all visible

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_viewshed",
                return_value=vs_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
        ):
            result = await mock_manager.fetch_viewshed(
                observer=[7.5, 46.5], radius_m=1000.0, source="cop30"
            )

        assert result.visible_percentage == pytest.approx(100.0)


# ===================================================================
# Phase 3: Temporal Change, Landforms, Anomalies, Features
# ===================================================================


class TestFetchTemporalChange:
    """Tests for DEMManager.fetch_temporal_change()."""

    async def test_fetch_temporal_change_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        change_arr = np.ones((100, 100), dtype=np.float32) * 2.0
        regions = [
            {
                "bbox": [7.0, 46.0, 8.0, 47.0],
                "area_m2": 1000.0,
                "mean_change_m": 2.0,
                "max_change_m": 2.0,
                "change_type": "gain",
            }
        ]

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=sample_elevation,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_elevation_change",
                return_value=(change_arr, regions),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.change_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_temporal_change(
                bbox=[7.0, 46.0, 8.0, 47.0],
                before_source="srtm",
                after_source="cop30",
            )

        assert isinstance(result, TemporalChangeResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30  # SRTM resolution
        assert result.dtype == "float32"
        assert result.significance_threshold_m == 1.0
        assert len(result.significant_regions) == 1

    async def test_fetch_temporal_change_invalid_before_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_temporal_change(
                bbox=[7.0, 46.0, 8.0, 47.0],
                before_source="bad",
                after_source="cop30",
            )

    async def test_fetch_temporal_change_invalid_after_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_temporal_change(
                bbox=[7.0, 46.0, 8.0, 47.0],
                before_source="srtm",
                after_source="bad",
            )

    async def test_fetch_temporal_change_no_coverage_before(self, mock_manager):
        """Before source with no tile URL constructor raises."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_temporal_change(
                bbox=[7.0, 46.0, 8.0, 47.0],
                before_source="aster",
                after_source="cop30",
            )

    async def test_fetch_temporal_change_png_output(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        change_arr = np.ones((100, 100), dtype=np.float32) * 0.5
        regions = []

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=sample_elevation,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_elevation_change",
                return_value=(change_arr, regions),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.change_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_temporal_change(
                bbox=[7.0, 46.0, 8.0, 47.0],
                before_source="srtm",
                after_source="cop30",
                output_format="png",
            )

        assert isinstance(result, TemporalChangeResult)
        assert result.artifact_ref.startswith("dem/")


class TestFetchLandforms:
    """Tests for DEMManager.fetch_landforms()."""

    async def test_fetch_landforms_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        landform_arr = np.zeros((100, 100), dtype=np.uint8)
        landform_arr[0:50, :] = 3  # plateau

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=sample_elevation,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_landforms",
                return_value=landform_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.landform_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_landforms(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30"
            )

        assert isinstance(result, LandformResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30
        assert result.dtype == "uint8"
        assert "plain" in result.class_distribution
        assert "plateau" in result.class_distribution
        assert result.dominant_landform in ("plain", "plateau")

    async def test_fetch_landforms_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_landforms(
                bbox=[7.0, 46.0, 8.0, 47.0], source="bad"
            )

    async def test_fetch_landforms_no_coverage(self, mock_manager):
        """Source with no tile URL constructor raises."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_landforms(
                bbox=[7.0, 46.0, 8.0, 47.0], source="aster"
            )

    async def test_fetch_landforms_png_output(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        landform_arr = np.zeros((100, 100), dtype=np.uint8)

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=sample_elevation,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_landforms",
                return_value=landform_arr,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.landform_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_landforms(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", output_format="png"
            )

        assert isinstance(result, LandformResult)
        assert result.artifact_ref.startswith("dem/")


class TestFetchAnomalies:
    """Tests for DEMManager.fetch_anomalies()."""

    async def test_fetch_anomalies_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        scores_arr = np.random.uniform(0, 1, (100, 100)).astype(np.float32)
        anomalies = [
            {
                "bbox": [7.0, 46.0, 7.5, 46.5],
                "area_m2": 500.0,
                "confidence": 0.8,
                "mean_anomaly_score": 0.75,
            }
        ]

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=sample_elevation,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_anomaly_scores",
                return_value=(scores_arr, anomalies),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.anomaly_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_anomalies(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30"
            )

        assert isinstance(result, AnomalyResult)
        assert result.artifact_ref.startswith("dem/")
        assert result.crs == str(sample_crs)
        assert result.resolution_m == 30
        assert result.anomaly_count == 1
        assert len(result.anomalies) == 1
        assert result.dtype == "float32"

    async def test_fetch_anomalies_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_anomalies(
                bbox=[7.0, 46.0, 8.0, 47.0], source="bad"
            )

    async def test_fetch_anomalies_no_coverage(self, mock_manager):
        """Source with no tile URL constructor raises."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_anomalies(
                bbox=[7.0, 46.0, 8.0, 47.0], source="aster"
            )

    async def test_fetch_anomalies_png_output(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        scores_arr = np.random.uniform(0, 1, (100, 100)).astype(np.float32)
        anomalies = []

        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=sample_elevation,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_anomaly_scores",
                return_value=(scores_arr, anomalies),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.anomaly_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_anomalies(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", output_format="png"
            )

        assert isinstance(result, AnomalyResult)
        assert result.artifact_ref.startswith("dem/")


class TestFetchFeatures:
    """Tests for DEMManager.fetch_features()."""

    async def test_fetch_features_success(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=sample_elevation,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_feature_detection",
                return_value=(
                    np.zeros(sample_elevation.shape, dtype=np.float32),
                    [{"bbox": [7, 46, 8, 47], "area_m2": 100.0, "feature_type": "ridge", "confidence": 0.5}],
                ),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.arrays_to_geotiff",
                return_value=b"fake-tiff",
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.feature_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_features(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30"
            )
            assert result.artifact_ref
            assert result.crs == str(sample_crs)
            assert result.resolution_m == 30
            assert result.feature_count == 1
            assert result.feature_summary == {"ridge": 1}
            assert len(result.features) == 1
            assert result.dtype == "float32"

    async def test_fetch_features_png_output(
        self, mock_manager, sample_elevation, sample_crs, sample_transform
    ):
        with (
            patch(
                "chuk_mcp_dem.core.raster_io.read_and_merge_tiles",
                return_value=(sample_elevation, sample_crs, sample_transform),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.fill_voids",
                return_value=sample_elevation,
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.compute_feature_detection",
                return_value=(
                    np.zeros(sample_elevation.shape, dtype=np.float32),
                    [],
                ),
            ),
            patch(
                "chuk_mcp_dem.core.raster_io.feature_to_png",
                return_value=b"fake-png",
            ),
        ):
            result = await mock_manager.fetch_features(
                bbox=[7.0, 46.0, 8.0, 47.0], source="cop30", output_format="png"
            )
            assert result.artifact_ref
            assert result.preview_ref is None  # No separate preview for png output

    async def test_fetch_features_invalid_source(self, mock_manager):
        with pytest.raises(ValueError, match="Unknown DEM source"):
            await mock_manager.fetch_features(
                bbox=[7.0, 46.0, 8.0, 47.0], source="bad"
            )

    async def test_fetch_features_no_coverage(self, mock_manager):
        """Source with no tile URL constructor raises."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_features(
                bbox=[7.0, 46.0, 8.0, 47.0], source="aster"
            )


# ===================================================================
# Phase 3: Dataclass tests
# ===================================================================


class TestPhase3Dataclasses:
    """Tests for Phase 3 result dataclasses."""

    def test_temporal_change_result_fields(self):
        result = TemporalChangeResult(
            artifact_ref="dem/change.tif",
            preview_ref="dem/change.png",
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            significance_threshold_m=1.0,
            volume_gained_m3=5000.0,
            volume_lost_m3=3000.0,
            significant_regions=[],
            dtype="float32",
        )
        assert result.artifact_ref == "dem/change.tif"
        assert result.volume_gained_m3 == 5000.0
        assert result.volume_lost_m3 == 3000.0

    def test_landform_result_fields(self):
        result = LandformResult(
            artifact_ref="dem/landforms.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            class_distribution={"plain": 80.0, "ridge": 20.0},
            dominant_landform="plain",
            dtype="uint8",
        )
        assert result.artifact_ref == "dem/landforms.tif"
        assert result.dominant_landform == "plain"
        assert result.preview_ref is None

    def test_anomaly_result_fields(self):
        result = AnomalyResult(
            artifact_ref="dem/anomalies.tif",
            preview_ref="dem/anomalies.png",
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            anomaly_count=5,
            anomalies=[],
            dtype="float32",
        )
        assert result.artifact_ref == "dem/anomalies.tif"
        assert result.anomaly_count == 5
        assert result.preview_ref == "dem/anomalies.png"

    def test_feature_result(self):
        r = FeatureResult(
            artifact_ref="abc",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            feature_count=5,
            feature_summary={"ridge": 3, "peak": 2},
            features=[],
            dtype="float32",
        )
        assert r.feature_count == 5
        assert r.feature_summary == {"ridge": 3, "peak": 2}
