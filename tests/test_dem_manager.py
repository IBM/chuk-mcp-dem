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
from chuk_mcp_dem.core.dem_manager import DEMManager, FetchResult, MultiPointResult, PointResult


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

    def test_srtm_returns_none(self, mock_manager):
        """Sources other than cop30/cop90 return None (no URL constructor)."""
        url = mock_manager._make_tile_url("srtm", 46, 7)
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
        """Source exists but _make_tile_url returns None for non-cop sources."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_elevation(bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")

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
        """Source with no tile URL constructor (srtm) has empty tile_urls."""
        with pytest.raises(ValueError, match="does not cover"):
            await mock_manager.fetch_point(lon=7.5, lat=46.5, source="srtm")


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
