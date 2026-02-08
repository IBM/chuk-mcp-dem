"""
Comprehensive tests for download tools (dem_fetch, dem_fetch_point,
dem_fetch_points, dem_check_coverage, dem_estimate_size).

Tests cover:
- Success paths (JSON and text output modes)
- Parameter forwarding to manager methods
- Input validation (interpolation methods)
- Error handling (exception -> ErrorResponse)
- Edge cases (empty points, large areas, no coverage)
- Default parameter values
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from chuk_mcp_dem.core.dem_manager import FetchResult, MultiPointResult, PointResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def download_tools(mock_manager):
    """Register download tools and return (tools_dict, manager)."""
    tools = {}
    mcp = MagicMock()

    def capture_tool(**kwargs):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp.tool = capture_tool

    from chuk_mcp_dem.tools.download.api import register_download_tools

    register_download_tools(mcp, mock_manager)
    return tools, mock_manager


@pytest.fixture
def standard_fetch_result():
    """Reusable FetchResult for dem_fetch tests."""
    return FetchResult(
        artifact_ref="dem/abc123.tif",
        preview_ref="dem/abc123_hillshade.png",
        crs="EPSG:4326",
        resolution_m=30.0,
        shape=[100, 100],
        elevation_range=[200.0, 1500.0],
        dtype="float32",
        nodata_pixels=0,
    )


@pytest.fixture
def standard_point_result():
    """Reusable PointResult for dem_fetch_point tests."""
    return PointResult(elevation_m=1625.3, uncertainty_m=4.0)


@pytest.fixture
def standard_multi_point_result():
    """Reusable MultiPointResult for dem_fetch_points tests."""
    return MultiPointResult(
        elevations=[1625.3, 2100.5, 800.0],
        elevation_range=[800.0, 2100.5],
    )


# ===========================================================================
# dem_fetch
# ===========================================================================


class TestDemFetch:
    """Tests for the dem_fetch tool."""

    @pytest.mark.asyncio
    async def test_success_json(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/abc123.tif"
        assert data["preview_ref"] == "dem/abc123_hillshade.png"
        assert data["crs"] == "EPSG:4326"
        assert data["resolution_m"] == 30
        assert data["shape"] == [100, 100]
        assert data["elevation_range"] == [200.0, 1500.0]
        assert data["dtype"] == "float32"
        assert data["nodata_pixels"] == 0

    @pytest.mark.asyncio
    async def test_success_text(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Fetched DEM:" in result
        assert "dem/abc123.tif" in result
        assert "100x100" in result
        assert "EPSG:4326" in result

    @pytest.mark.asyncio
    async def test_bbox_forwarded(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        await tools["dem_fetch"](bbox=bbox)

        manager.fetch_elevation.assert_awaited_once()
        call_kwargs = manager.fetch_elevation.call_args.kwargs
        assert call_kwargs["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_source_forwarded(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], source="cop90")

        call_kwargs = manager.fetch_elevation.call_args.kwargs
        assert call_kwargs["source"] == "cop90"

    @pytest.mark.asyncio
    async def test_default_source(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_elevation.call_args.kwargs
        assert call_kwargs["source"] == "cop30"

    @pytest.mark.asyncio
    async def test_fill_voids_default_true(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_elevation.call_args.kwargs
        assert call_kwargs["fill_voids"] is True

    @pytest.mark.asyncio
    async def test_fill_voids_false(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], fill_voids=False)

        call_kwargs = manager.fetch_elevation.call_args.kwargs
        assert call_kwargs["fill_voids"] is False

    @pytest.mark.asyncio
    async def test_resolution_forwarded(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], resolution_m=90.0)

        call_kwargs = manager.fetch_elevation.call_args.kwargs
        assert call_kwargs["resolution_m"] == 90.0

    @pytest.mark.asyncio
    async def test_output_crs_forwarded(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], output_crs="EPSG:32632")

        call_kwargs = manager.fetch_elevation.call_args.kwargs
        assert call_kwargs["output_crs"] == "EPSG:32632"

    @pytest.mark.asyncio
    async def test_message_contains_estimated_mb(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        # 100*100*4 / (1024*1024) ~ 0.04 MB
        assert "message" in data
        assert "Downloaded" in data["message"]
        assert "MB" in data["message"]

    @pytest.mark.asyncio
    async def test_source_in_response(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)

        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_bbox_in_response(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        result = await tools["dem_fetch"](bbox=bbox)
        data = json.loads(result)

        assert data["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_error_returns_error_response(self, download_tools):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(side_effect=ValueError("Invalid bounding box"))

        result = await tools["dem_fetch"](bbox=[10.0, 10.0, 5.0, 5.0])
        data = json.loads(result)

        assert "error" in data
        assert "Invalid bounding box" in data["error"]

    @pytest.mark.asyncio
    async def test_error_text_mode(self, download_tools):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Network timeout" in result

    @pytest.mark.asyncio
    async def test_no_preview_ref(self, download_tools):
        tools, manager = download_tools
        result_no_preview = FetchResult(
            artifact_ref="dem/xyz.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[50, 50],
            elevation_range=[0.0, 100.0],
            dtype="float32",
            nodata_pixels=5,
        )
        manager.fetch_elevation = AsyncMock(return_value=result_no_preview)

        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["preview_ref"] is None
        assert data["nodata_pixels"] == 5

    @pytest.mark.asyncio
    async def test_large_shape_estimated_mb(self, download_tools):
        tools, manager = download_tools
        large_result = FetchResult(
            artifact_ref="dem/large.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[10000, 10000],
            elevation_range=[0.0, 8848.0],
            dtype="float32",
            nodata_pixels=0,
        )
        manager.fetch_elevation = AsyncMock(return_value=large_result)

        result = await tools["dem_fetch"](bbox=[-180.0, -90.0, 180.0, 90.0])
        data = json.loads(result)

        # 10000*10000*4 / (1024*1024) ~ 381 MB
        assert "381" in data["message"]


# ===========================================================================
# dem_fetch_point
# ===========================================================================


class TestDemFetchPoint:
    """Tests for the dem_fetch_point tool."""

    @pytest.mark.asyncio
    async def test_success_json(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        result = await tools["dem_fetch_point"](lon=-105.27, lat=40.015)
        data = json.loads(result)

        assert data["lon"] == -105.27
        assert data["lat"] == 40.015
        assert data["elevation_m"] == pytest.approx(1625.3)
        assert data["uncertainty_m"] == pytest.approx(4.0)
        assert data["source"] == "cop30"
        assert data["interpolation"] == "bilinear"

    @pytest.mark.asyncio
    async def test_success_text(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        result = await tools["dem_fetch_point"](lon=-105.27, lat=40.015, output_mode="text")

        assert "Elevation at" in result
        assert "1625.3m" in result
        assert "+/-4.0m" in result

    @pytest.mark.asyncio
    async def test_default_interpolation(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        await tools["dem_fetch_point"](lon=0.0, lat=0.0)

        call_kwargs = manager.fetch_point.call_args.kwargs
        assert call_kwargs["interpolation"] == "bilinear"

    @pytest.mark.asyncio
    async def test_nearest_interpolation(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        result = await tools["dem_fetch_point"](lon=0.0, lat=0.0, interpolation="nearest")
        data = json.loads(result)

        assert data["interpolation"] == "nearest"
        call_kwargs = manager.fetch_point.call_args.kwargs
        assert call_kwargs["interpolation"] == "nearest"

    @pytest.mark.asyncio
    async def test_cubic_interpolation(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        result = await tools["dem_fetch_point"](lon=10.0, lat=50.0, interpolation="cubic")
        data = json.loads(result)

        assert data["interpolation"] == "cubic"

    @pytest.mark.asyncio
    async def test_invalid_interpolation(self, download_tools):
        tools, manager = download_tools

        result = await tools["dem_fetch_point"](lon=0.0, lat=0.0, interpolation="lanczos")
        data = json.loads(result)

        assert "error" in data
        assert "Invalid interpolation" in data["error"]
        assert "lanczos" in data["error"]
        assert "nearest" in data["error"]
        assert "bilinear" in data["error"]
        assert "cubic" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_interpolation_text_mode(self, download_tools):
        tools, manager = download_tools

        result = await tools["dem_fetch_point"](
            lon=0.0, lat=0.0, interpolation="hermite", output_mode="text"
        )

        assert "Error:" in result
        assert "Invalid interpolation" in result
        assert "hermite" in result

    @pytest.mark.asyncio
    async def test_source_forwarded(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        result = await tools["dem_fetch_point"](lon=0.0, lat=0.0, source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"
        call_kwargs = manager.fetch_point.call_args.kwargs
        assert call_kwargs["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_source_3dep(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        result = await tools["dem_fetch_point"](lon=-105.0, lat=40.0, source="3dep")
        data = json.loads(result)

        assert data["source"] == "3dep"

    @pytest.mark.asyncio
    async def test_lon_lat_forwarded(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        await tools["dem_fetch_point"](lon=13.405, lat=52.52)

        call_kwargs = manager.fetch_point.call_args.kwargs
        assert call_kwargs["lon"] == pytest.approx(13.405)
        assert call_kwargs["lat"] == pytest.approx(52.52)

    @pytest.mark.asyncio
    async def test_message_contains_elevation(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)

        result = await tools["dem_fetch_point"](lon=0.0, lat=0.0)
        data = json.loads(result)

        assert "1625.3" in data["message"]
        assert "4.0" in data["message"]

    @pytest.mark.asyncio
    async def test_error_returns_error_response(self, download_tools):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(side_effect=ValueError("Coverage error"))

        result = await tools["dem_fetch_point"](lon=0.0, lat=0.0)
        data = json.loads(result)

        assert "error" in data
        assert "Coverage error" in data["error"]

    @pytest.mark.asyncio
    async def test_negative_elevation(self, download_tools):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(
            return_value=PointResult(elevation_m=-422.0, uncertainty_m=4.0)
        )

        result = await tools["dem_fetch_point"](lon=35.47, lat=31.52, source="cop30")
        data = json.loads(result)

        assert data["elevation_m"] == pytest.approx(-422.0)


# ===========================================================================
# dem_fetch_points
# ===========================================================================


class TestDemFetchPoints:
    """Tests for the dem_fetch_points tool."""

    @pytest.mark.asyncio
    async def test_success_json(self, download_tools, standard_multi_point_result):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(return_value=standard_multi_point_result)
        points = [[-105.0, 40.0], [-104.0, 39.0], [-106.0, 41.0]]

        result = await tools["dem_fetch_points"](points=points)
        data = json.loads(result)

        assert data["source"] == "cop30"
        assert data["point_count"] == 3
        assert len(data["points"]) == 3
        assert data["points"][0]["lon"] == -105.0
        assert data["points"][0]["lat"] == 40.0
        assert data["points"][0]["elevation_m"] == pytest.approx(1625.3)
        assert data["elevation_range"] == [800.0, 2100.5]
        assert data["interpolation"] == "bilinear"

    @pytest.mark.asyncio
    async def test_success_text(self, download_tools, standard_multi_point_result):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(return_value=standard_multi_point_result)
        points = [[-105.0, 40.0], [-104.0, 39.0], [-106.0, 41.0]]

        result = await tools["dem_fetch_points"](points=points, output_mode="text")

        assert "3 point(s)" in result
        assert "800.0m to 2100.5m" in result
        assert "-105.000000" in result

    @pytest.mark.asyncio
    async def test_points_forwarded(self, download_tools, standard_multi_point_result):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(return_value=standard_multi_point_result)
        points = [[10.0, 50.0], [11.0, 51.0], [12.0, 52.0]]

        await tools["dem_fetch_points"](points=points)

        call_kwargs = manager.fetch_points.call_args.kwargs
        assert call_kwargs["points"] == points

    @pytest.mark.asyncio
    async def test_interpolation_forwarded(self, download_tools, standard_multi_point_result):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(return_value=standard_multi_point_result)
        points = [[10.0, 50.0]]

        await tools["dem_fetch_points"](points=points, interpolation="cubic")

        call_kwargs = manager.fetch_points.call_args.kwargs
        assert call_kwargs["interpolation"] == "cubic"

    @pytest.mark.asyncio
    async def test_invalid_interpolation(self, download_tools):
        tools, manager = download_tools
        points = [[10.0, 50.0]]

        result = await tools["dem_fetch_points"](points=points, interpolation="spline")
        data = json.loads(result)

        assert "error" in data
        assert "Invalid interpolation" in data["error"]
        assert "spline" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_interpolation_text(self, download_tools):
        tools, manager = download_tools

        result = await tools["dem_fetch_points"](
            points=[[1.0, 2.0]], interpolation="quadratic", output_mode="text"
        )

        assert "Error:" in result
        assert "quadratic" in result

    @pytest.mark.asyncio
    async def test_source_forwarded(self, download_tools, standard_multi_point_result):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(return_value=standard_multi_point_result)

        result = await tools["dem_fetch_points"](
            points=[[10.0, 50.0], [11.0, 51.0], [12.0, 52.0]],
            source="fabdem",
        )
        data = json.loads(result)

        assert data["source"] == "fabdem"
        call_kwargs = manager.fetch_points.call_args.kwargs
        assert call_kwargs["source"] == "fabdem"

    @pytest.mark.asyncio
    async def test_single_point(self, download_tools):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(
            return_value=MultiPointResult(
                elevations=[500.0],
                elevation_range=[500.0, 500.0],
            )
        )

        result = await tools["dem_fetch_points"](points=[[10.0, 50.0]])
        data = json.loads(result)

        assert data["point_count"] == 1
        assert len(data["points"]) == 1
        assert data["elevation_range"] == [500.0, 500.0]

    @pytest.mark.asyncio
    async def test_message_contains_point_count(self, download_tools, standard_multi_point_result):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(return_value=standard_multi_point_result)

        result = await tools["dem_fetch_points"](points=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        data = json.loads(result)

        assert "3" in data["message"]

    @pytest.mark.asyncio
    async def test_error_returns_error_response(self, download_tools):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(side_effect=RuntimeError("Tile download failed"))

        result = await tools["dem_fetch_points"](points=[[10.0, 50.0], [11.0, 51.0]])
        data = json.loads(result)

        assert "error" in data
        assert "Tile download failed" in data["error"]

    @pytest.mark.asyncio
    async def test_two_points_correct_mapping(self, download_tools):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(
            return_value=MultiPointResult(
                elevations=[100.0, 200.0],
                elevation_range=[100.0, 200.0],
            )
        )

        result = await tools["dem_fetch_points"](points=[[1.0, 2.0], [3.0, 4.0]])
        data = json.loads(result)

        assert data["points"][0]["lon"] == 1.0
        assert data["points"][0]["lat"] == 2.0
        assert data["points"][0]["elevation_m"] == pytest.approx(100.0)
        assert data["points"][1]["lon"] == 3.0
        assert data["points"][1]["lat"] == 4.0
        assert data["points"][1]["elevation_m"] == pytest.approx(200.0)


# ===========================================================================
# dem_check_coverage
# ===========================================================================


class TestDemCheckCoverage:
    """Tests for the dem_check_coverage tool."""

    @pytest.fixture
    def full_coverage_result(self):
        return {
            "fully_covered": True,
            "coverage_percentage": 100.0,
            "tiles_required": 4,
            "tile_ids": ["N46E007", "N46E008", "N47E007", "N47E008"],
            "estimated_size_mb": 52.0,
        }

    @pytest.fixture
    def partial_coverage_result(self):
        return {
            "fully_covered": False,
            "coverage_percentage": 65.3,
            "tiles_required": 2,
            "tile_ids": ["N46E007", "N46E008"],
            "estimated_size_mb": 26.0,
        }

    @pytest.fixture
    def no_coverage_result(self):
        return {
            "fully_covered": False,
            "coverage_percentage": 0.0,
            "tiles_required": 0,
            "tile_ids": [],
            "estimated_size_mb": 0.0,
        }

    @pytest.mark.asyncio
    async def test_full_coverage_json(self, download_tools, full_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=full_coverage_result)

        result = await tools["dem_check_coverage"](bbox=[7.0, 46.0, 9.0, 48.0])
        data = json.loads(result)

        assert data["fully_covered"] is True
        assert data["coverage_percentage"] == 100.0
        assert data["tiles_required"] == 4
        assert len(data["tile_ids"]) == 4
        assert data["estimated_size_mb"] == 52.0
        assert data["message"] == "Full coverage available"

    @pytest.mark.asyncio
    async def test_full_coverage_text(self, download_tools, full_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=full_coverage_result)

        result = await tools["dem_check_coverage"](bbox=[7.0, 46.0, 9.0, 48.0], output_mode="text")

        assert "fully covered" in result
        assert "cop30" in result
        assert "Tiles required: 4" in result
        assert "52.0 MB" in result

    @pytest.mark.asyncio
    async def test_partial_coverage_json(self, download_tools, partial_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=partial_coverage_result)

        result = await tools["dem_check_coverage"](bbox=[7.0, 46.0, 9.0, 48.0])
        data = json.loads(result)

        assert data["fully_covered"] is False
        assert data["coverage_percentage"] == 65.3
        assert "65.3" in data["message"]

    @pytest.mark.asyncio
    async def test_partial_coverage_text(self, download_tools, partial_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=partial_coverage_result)

        result = await tools["dem_check_coverage"](bbox=[7.0, 46.0, 9.0, 48.0], output_mode="text")

        assert "65.3% covered" in result
        assert "fully covered" not in result

    @pytest.mark.asyncio
    async def test_no_coverage_json(self, download_tools, no_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=no_coverage_result)

        result = await tools["dem_check_coverage"](bbox=[7.0, 46.0, 9.0, 48.0])
        data = json.loads(result)

        assert data["fully_covered"] is False
        assert data["coverage_percentage"] == 0.0
        assert data["tiles_required"] == 0
        assert data["tile_ids"] == []
        assert data["estimated_size_mb"] == 0.0

    @pytest.mark.asyncio
    async def test_bbox_and_source_forwarded(self, download_tools, full_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=full_coverage_result)
        bbox = [10.0, 50.0, 12.0, 52.0]

        await tools["dem_check_coverage"](bbox=bbox, source="srtm")

        manager.check_coverage.assert_called_once_with(bbox, "srtm")

    @pytest.mark.asyncio
    async def test_default_source(self, download_tools, full_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=full_coverage_result)

        await tools["dem_check_coverage"](bbox=[7.0, 46.0, 8.0, 47.0])

        manager.check_coverage.assert_called_once_with([7.0, 46.0, 8.0, 47.0], "cop30")

    @pytest.mark.asyncio
    async def test_source_in_response(self, download_tools, full_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=full_coverage_result)

        result = await tools["dem_check_coverage"](bbox=[7.0, 46.0, 8.0, 47.0], source="aster")
        data = json.loads(result)

        assert data["source"] == "aster"

    @pytest.mark.asyncio
    async def test_bbox_in_response(self, download_tools, full_coverage_result):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(return_value=full_coverage_result)
        bbox = [10.0, 50.0, 12.0, 52.0]

        result = await tools["dem_check_coverage"](bbox=bbox)
        data = json.loads(result)

        assert data["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_error_returns_error_response(self, download_tools):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(side_effect=ValueError("Invalid bounding box"))

        result = await tools["dem_check_coverage"](bbox=[10.0, 10.0, 5.0, 5.0])
        data = json.loads(result)

        assert "error" in data
        assert "Invalid bounding box" in data["error"]

    @pytest.mark.asyncio
    async def test_error_text_mode(self, download_tools):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(side_effect=RuntimeError("Source unavailable"))

        result = await tools["dem_check_coverage"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Source unavailable" in result


# ===========================================================================
# dem_estimate_size
# ===========================================================================


class TestDemEstimateSize:
    """Tests for the dem_estimate_size tool."""

    @pytest.fixture
    def small_estimate_result(self):
        return {
            "native_resolution_m": 30,
            "target_resolution_m": 30,
            "dimensions": [3601, 3601],
            "pixels": 12967201,
            "dtype": "float32",
            "estimated_bytes": 51868804,
            "estimated_mb": 49.5,
            "warning": None,
        }

    @pytest.fixture
    def large_estimate_result(self):
        return {
            "native_resolution_m": 30,
            "target_resolution_m": 30,
            "dimensions": [36010, 36010],
            "pixels": 1296720100,
            "dtype": "float32",
            "estimated_bytes": 5186880400,
            "estimated_mb": 4946.1,
            "warning": "Large download: 4946 MB. Consider using a smaller bbox or coarser resolution.",
        }

    @pytest.fixture
    def custom_resolution_result(self):
        return {
            "native_resolution_m": 30,
            "target_resolution_m": 90,
            "dimensions": [1201, 1201],
            "pixels": 1442401,
            "dtype": "float32",
            "estimated_bytes": 5769604,
            "estimated_mb": 5.5,
            "warning": None,
        }

    @pytest.mark.asyncio
    async def test_small_area_json(self, download_tools, small_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=small_estimate_result)

        result = await tools["dem_estimate_size"](bbox=[-105.0, 39.0, -104.0, 40.0])
        data = json.loads(result)

        assert data["source"] == "cop30"
        assert data["native_resolution_m"] == 30
        assert data["target_resolution_m"] == 30
        assert data["dimensions"] == [3601, 3601]
        assert data["pixels"] == 12967201
        assert data["dtype"] == "float32"
        assert data["estimated_bytes"] == 51868804
        assert data["estimated_mb"] == 49.5
        assert data["warning"] is None

    @pytest.mark.asyncio
    async def test_small_area_text(self, download_tools, small_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=small_estimate_result)

        result = await tools["dem_estimate_size"](
            bbox=[-105.0, 39.0, -104.0, 40.0], output_mode="text"
        )

        assert "Size estimate:" in result
        assert "3601x3601" in result
        assert "49.5 MB" in result
        assert "WARNING" not in result

    @pytest.mark.asyncio
    async def test_large_area_with_warning_json(self, download_tools, large_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=large_estimate_result)

        result = await tools["dem_estimate_size"](bbox=[-115.0, 30.0, -105.0, 40.0])
        data = json.loads(result)

        assert data["warning"] is not None
        assert "Large download" in data["warning"]
        assert data["estimated_mb"] == 4946.1

    @pytest.mark.asyncio
    async def test_large_area_text_with_warning(self, download_tools, large_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=large_estimate_result)

        result = await tools["dem_estimate_size"](
            bbox=[-115.0, 30.0, -105.0, 40.0], output_mode="text"
        )

        assert "WARNING:" in result
        assert "Large download" in result

    @pytest.mark.asyncio
    async def test_custom_resolution_json(self, download_tools, custom_resolution_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=custom_resolution_result)

        result = await tools["dem_estimate_size"](
            bbox=[-105.0, 39.0, -104.0, 40.0], resolution_m=90.0
        )
        data = json.loads(result)

        assert data["native_resolution_m"] == 30
        assert data["target_resolution_m"] == 90
        assert data["estimated_mb"] == 5.5

    @pytest.mark.asyncio
    async def test_bbox_source_resolution_forwarded(self, download_tools, small_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=small_estimate_result)
        bbox = [-105.0, 39.0, -104.0, 40.0]

        await tools["dem_estimate_size"](bbox=bbox, source="cop90", resolution_m=90.0)

        manager.estimate_size.assert_called_once_with(bbox, "cop90", 90.0)

    @pytest.mark.asyncio
    async def test_default_source_and_resolution(self, download_tools, small_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=small_estimate_result)
        bbox = [-105.0, 39.0, -104.0, 40.0]

        await tools["dem_estimate_size"](bbox=bbox)

        manager.estimate_size.assert_called_once_with(bbox, "cop30", None)

    @pytest.mark.asyncio
    async def test_source_in_response(self, download_tools, small_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=small_estimate_result)

        result = await tools["dem_estimate_size"](bbox=[-105.0, 39.0, -104.0, 40.0], source="3dep")
        data = json.loads(result)

        assert data["source"] == "3dep"

    @pytest.mark.asyncio
    async def test_bbox_in_response(self, download_tools, small_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=small_estimate_result)
        bbox = [10.0, 50.0, 12.0, 52.0]

        result = await tools["dem_estimate_size"](bbox=bbox)
        data = json.loads(result)

        assert data["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_message_contains_megapixels(self, download_tools, small_estimate_result):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(return_value=small_estimate_result)

        result = await tools["dem_estimate_size"](bbox=[-105.0, 39.0, -104.0, 40.0])
        data = json.loads(result)

        # 12967201 / 1_000_000 ~ 13.0 megapixels
        assert "megapixels" in data["message"]
        assert "49.5" in data["message"]

    @pytest.mark.asyncio
    async def test_error_returns_error_response(self, download_tools):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(side_effect=ValueError("Invalid bounding box"))

        result = await tools["dem_estimate_size"](bbox=[10.0, 10.0, 5.0, 5.0])
        data = json.loads(result)

        assert "error" in data
        assert "Invalid bounding box" in data["error"]

    @pytest.mark.asyncio
    async def test_error_text_mode(self, download_tools):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(side_effect=RuntimeError("Unknown source"))

        result = await tools["dem_estimate_size"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Unknown source" in result


# ===========================================================================
# Cross-cutting concerns
# ===========================================================================


class TestToolRegistration:
    """Verify that all 5 download tools are registered."""

    def test_all_tools_registered(self, download_tools):
        tools, _ = download_tools
        expected = {
            "dem_fetch",
            "dem_fetch_point",
            "dem_fetch_points",
            "dem_check_coverage",
            "dem_estimate_size",
        }
        assert set(tools.keys()) == expected

    def test_tools_are_callable(self, download_tools):
        tools, _ = download_tools
        for name, fn in tools.items():
            assert callable(fn), f"{name} should be callable"


class TestOutputModeConsistency:
    """Verify that every tool supports both json and text output_mode."""

    @pytest.mark.asyncio
    async def test_dem_fetch_json_parseable(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)
        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_dem_fetch_text_not_json(self, download_tools, standard_fetch_result):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(return_value=standard_fetch_result)
        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        # Text output should not be valid JSON (it's human-readable)
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_dem_fetch_point_json_parseable(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)
        result = await tools["dem_fetch_point"](lon=0.0, lat=0.0)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_dem_fetch_point_text_not_json(self, download_tools, standard_point_result):
        tools, manager = download_tools
        manager.fetch_point = AsyncMock(return_value=standard_point_result)
        result = await tools["dem_fetch_point"](lon=0.0, lat=0.0, output_mode="text")
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_dem_fetch_points_json_parseable(
        self, download_tools, standard_multi_point_result
    ):
        tools, manager = download_tools
        manager.fetch_points = AsyncMock(return_value=standard_multi_point_result)
        result = await tools["dem_fetch_points"](points=[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_dem_check_coverage_json_parseable(self, download_tools):
        tools, manager = download_tools
        manager.check_coverage = MagicMock(
            return_value={
                "fully_covered": True,
                "coverage_percentage": 100.0,
                "tiles_required": 1,
                "tile_ids": ["N46E007"],
                "estimated_size_mb": 13.0,
            }
        )
        result = await tools["dem_check_coverage"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_dem_estimate_size_json_parseable(self, download_tools):
        tools, manager = download_tools
        manager.estimate_size = MagicMock(
            return_value={
                "native_resolution_m": 30,
                "target_resolution_m": 30,
                "dimensions": [100, 100],
                "pixels": 10000,
                "dtype": "float32",
                "estimated_bytes": 40000,
                "estimated_mb": 0.04,
                "warning": None,
            }
        )
        result = await tools["dem_estimate_size"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_error_json_parseable(self, download_tools):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(side_effect=Exception("fail"))
        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_error_text_has_prefix(self, download_tools):
        tools, manager = download_tools
        manager.fetch_elevation = AsyncMock(side_effect=Exception("boom"))
        result = await tools["dem_fetch"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        assert result.startswith("Error:")
