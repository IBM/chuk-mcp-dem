"""
Comprehensive tests for analysis tools (dem_hillshade, dem_slope,
dem_aspect, dem_profile, dem_viewshed).

Tests cover:
- Success paths (JSON and text output modes)
- Parameter forwarding to manager methods
- Input validation (output formats, slope units, interpolation, radius)
- Error handling (exception -> ErrorResponse)
- Default parameter values
- Response field verification
"""

import json

import pytest
from unittest.mock import AsyncMock, MagicMock

from chuk_mcp_dem.core.dem_manager import (
    TerrainResult,
    ProfileResult,
    ViewshedResult,
    ContourResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def analysis_tools(mock_manager):
    """Register analysis tools and return (tools_dict, manager)."""
    tools = {}
    mcp = MagicMock()

    def capture_tool(**kwargs):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp.tool = capture_tool

    from chuk_mcp_dem.tools.analysis.api import register_analysis_tools

    register_analysis_tools(mcp, mock_manager)
    return tools, mock_manager


@pytest.fixture
def standard_terrain_result():
    """Reusable TerrainResult for hillshade/slope/aspect tests."""
    return TerrainResult(
        artifact_ref="dem/abc123.tif",
        preview_ref="dem/abc123_hs.png",
        crs="EPSG:4326",
        resolution_m=30.0,
        shape=[100, 100],
        value_range=[0.0, 255.0],
        dtype="float32",
    )


@pytest.fixture
def standard_profile_result():
    """Reusable ProfileResult for dem_profile tests."""
    return ProfileResult(
        longitudes=[7.0, 7.5, 8.0],
        latitudes=[46.0, 46.5, 47.0],
        distances_m=[0.0, 50000.0, 100000.0],
        elevations=[500.0, 1200.0, 800.0],
        total_distance_m=100000.0,
        elevation_range=[500.0, 1200.0],
        elevation_gain_m=700.0,
        elevation_loss_m=400.0,
    )


@pytest.fixture
def standard_viewshed_result():
    """Reusable ViewshedResult for dem_viewshed tests."""
    return ViewshedResult(
        artifact_ref="dem/vs123.tif",
        preview_ref="dem/vs123.png",
        crs="EPSG:4326",
        resolution_m=30.0,
        shape=[200, 200],
        visible_percentage=65.3,
        observer_elevation_m=500.0,
        radius_m=5000.0,
    )


@pytest.fixture
def standard_contour_result():
    """Reusable ContourResult for dem_contour tests."""
    return ContourResult(
        artifact_ref="dem/contour123.tif",
        preview_ref="dem/contour123_preview.png",
        crs="EPSG:4326",
        resolution_m=30.0,
        shape=[100, 100],
        interval_m=100.0,
        contour_count=5,
        elevation_range=[200.0, 700.0],
        dtype="float32",
    )


# ===========================================================================
# dem_hillshade
# ===========================================================================


class TestDemHillshade:
    """Tests for the dem_hillshade tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/abc123.tif"
        assert data["preview_ref"] == "dem/abc123_hs.png"
        assert data["crs"] == "EPSG:4326"
        assert data["resolution_m"] == 30.0
        assert data["shape"] == [100, 100]
        assert data["value_range"] == [0.0, 255.0]
        assert data["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Hillshade:" in result
        assert "dem/abc123.tif" in result
        assert "100x100" in result
        assert "EPSG:4326" in result

    @pytest.mark.asyncio
    async def test_default_params(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_hillshade.call_args.kwargs
        assert call_kwargs["azimuth"] == 315.0
        assert call_kwargs["altitude"] == 45.0
        assert call_kwargs["z_factor"] == 1.0
        assert call_kwargs["source"] == "cop30"
        assert call_kwargs["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_custom_params(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_hillshade"](
            bbox=[7.0, 46.0, 8.0, 47.0],
            azimuth=45.0,
            altitude=60.0,
            z_factor=2.0,
        )

        call_kwargs = manager.fetch_hillshade.call_args.kwargs
        assert call_kwargs["azimuth"] == 45.0
        assert call_kwargs["altitude"] == 60.0
        assert call_kwargs["z_factor"] == 2.0

    @pytest.mark.asyncio
    async def test_png_format(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="png")
        data = json.loads(result)

        assert data["output_format"] == "png"
        call_kwargs = manager.fetch_hillshade.call_args.kwargs
        assert call_kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_invalid_format(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp")
        data = json.loads(result)

        assert "error" in data
        assert "bmp" in data["error"]
        assert "geotiff" in data["error"]
        assert "png" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_format_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_hillshade"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp", output_mode="text"
        )

        assert "Error:" in result
        assert "bmp" in result

    @pytest.mark.asyncio
    async def test_invalid_source(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(side_effect=ValueError("Unknown DEM source 'badname'"))

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], source="badname")
        data = json.loads(result)

        assert "error" in data
        assert "badname" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "error" in data
        assert "Network timeout" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(side_effect=RuntimeError("Tile download failed"))

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Tile download failed" in result

    @pytest.mark.asyncio
    async def test_bbox_forwarded(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        await tools["dem_hillshade"](bbox=bbox)

        call_kwargs = manager.fetch_hillshade.call_args.kwargs
        assert call_kwargs["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_bbox_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        result = await tools["dem_hillshade"](bbox=bbox)
        data = json.loads(result)

        assert data["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_source_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_message_contains_shape(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "message" in data
        assert "Hillshade" in data["message"]
        assert "100x100" in data["message"]

    @pytest.mark.asyncio
    async def test_azimuth_altitude_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](
            bbox=[7.0, 46.0, 8.0, 47.0], azimuth=90.0, altitude=30.0
        )
        data = json.loads(result)

        assert data["azimuth"] == 90.0
        assert data["altitude"] == 30.0

    @pytest.mark.asyncio
    async def test_z_factor_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], z_factor=3.0)
        data = json.loads(result)

        assert data["z_factor"] == 3.0

    @pytest.mark.asyncio
    async def test_no_preview_ref(self, analysis_tools):
        tools, manager = analysis_tools
        result_no_preview = TerrainResult(
            artifact_ref="dem/xyz.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[50, 50],
            value_range=[0.0, 200.0],
            dtype="float32",
        )
        manager.fetch_hillshade = AsyncMock(return_value=result_no_preview)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["preview_ref"] is None
        assert data["artifact_ref"] == "dem/xyz.tif"


# ===========================================================================
# dem_slope
# ===========================================================================


class TestDemSlope:
    """Tests for the dem_slope tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/abc123.tif"
        assert data["preview_ref"] == "dem/abc123_hs.png"
        assert data["crs"] == "EPSG:4326"
        assert data["resolution_m"] == 30.0
        assert data["shape"] == [100, 100]
        assert data["value_range"] == [0.0, 255.0]
        assert data["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Slope:" in result
        assert "dem/abc123.tif" in result
        assert "100x100" in result

    @pytest.mark.asyncio
    async def test_degrees_units(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["units"] == "degrees"
        call_kwargs = manager.fetch_slope.call_args.kwargs
        assert call_kwargs["units"] == "degrees"

    @pytest.mark.asyncio
    async def test_percent_units(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], units="percent")
        data = json.loads(result)

        assert data["units"] == "percent"
        call_kwargs = manager.fetch_slope.call_args.kwargs
        assert call_kwargs["units"] == "percent"

    @pytest.mark.asyncio
    async def test_invalid_units(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], units="radians")
        data = json.loads(result)

        assert "error" in data
        assert "radians" in data["error"]
        assert "degrees" in data["error"]
        assert "percent" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_units_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_slope"](
            bbox=[7.0, 46.0, 8.0, 47.0], units="radians", output_mode="text"
        )

        assert "Error:" in result
        assert "radians" in result

    @pytest.mark.asyncio
    async def test_invalid_format(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="jpeg")
        data = json.loads(result)

        assert "error" in data
        assert "jpeg" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(side_effect=RuntimeError("Tile download failed"))

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "error" in data
        assert "Tile download failed" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Network timeout" in result

    @pytest.mark.asyncio
    async def test_bbox_forwarded(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)
        bbox = [10.0, 50.0, 12.0, 52.0]

        await tools["dem_slope"](bbox=bbox)

        call_kwargs = manager.fetch_slope.call_args.kwargs
        assert call_kwargs["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_source_forwarded(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], source="cop90")
        data = json.loads(result)

        assert data["source"] == "cop90"
        call_kwargs = manager.fetch_slope.call_args.kwargs
        assert call_kwargs["source"] == "cop90"

    @pytest.mark.asyncio
    async def test_message_contains_shape(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "message" in data
        assert "Slope" in data["message"]
        assert "100x100" in data["message"]

    @pytest.mark.asyncio
    async def test_png_format(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="png")
        data = json.loads(result)

        assert data["output_format"] == "png"
        call_kwargs = manager.fetch_slope.call_args.kwargs
        assert call_kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_default_source(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_slope.call_args.kwargs
        assert call_kwargs["source"] == "cop30"


# ===========================================================================
# dem_aspect
# ===========================================================================


class TestDemAspect:
    """Tests for the dem_aspect tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/abc123.tif"
        assert data["preview_ref"] == "dem/abc123_hs.png"
        assert data["crs"] == "EPSG:4326"
        assert data["resolution_m"] == 30.0
        assert data["shape"] == [100, 100]
        assert data["value_range"] == [0.0, 255.0]
        assert data["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Aspect:" in result
        assert "dem/abc123.tif" in result
        assert "100x100" in result

    @pytest.mark.asyncio
    async def test_default_flat_value(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["flat_value"] == -1.0
        call_kwargs = manager.fetch_aspect.call_args.kwargs
        assert call_kwargs["flat_value"] == -1.0

    @pytest.mark.asyncio
    async def test_custom_flat_value(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0], flat_value=0.0)
        data = json.loads(result)

        assert data["flat_value"] == 0.0
        call_kwargs = manager.fetch_aspect.call_args.kwargs
        assert call_kwargs["flat_value"] == 0.0

    @pytest.mark.asyncio
    async def test_invalid_format(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="tiff")
        data = json.loads(result)

        assert "error" in data
        assert "tiff" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_format_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_aspect"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_format="svg", output_mode="text"
        )

        assert "Error:" in result
        assert "svg" in result

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(side_effect=RuntimeError("Coverage error"))

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "error" in data
        assert "Coverage error" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(side_effect=RuntimeError("Tile fetch failed"))

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Tile fetch failed" in result

    @pytest.mark.asyncio
    async def test_bbox_forwarded(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)
        bbox = [10.0, 50.0, 12.0, 52.0]

        await tools["dem_aspect"](bbox=bbox)

        call_kwargs = manager.fetch_aspect.call_args.kwargs
        assert call_kwargs["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_source_forwarded(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0], source="fabdem")
        data = json.loads(result)

        assert data["source"] == "fabdem"
        call_kwargs = manager.fetch_aspect.call_args.kwargs
        assert call_kwargs["source"] == "fabdem"

    @pytest.mark.asyncio
    async def test_message_contains_shape(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "message" in data
        assert "Aspect" in data["message"]
        assert "100x100" in data["message"]

    @pytest.mark.asyncio
    async def test_png_format(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="png")
        data = json.loads(result)

        assert data["output_format"] == "png"
        call_kwargs = manager.fetch_aspect.call_args.kwargs
        assert call_kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_default_source(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_aspect.call_args.kwargs
        assert call_kwargs["source"] == "cop30"


# ===========================================================================
# dem_curvature
# ===========================================================================


class TestDemCurvature:
    """Tests for the dem_curvature tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/abc123.tif"
        assert data["preview_ref"] == "dem/abc123_hs.png"
        assert data["crs"] == "EPSG:4326"
        assert data["resolution_m"] == 30.0
        assert data["shape"] == [100, 100]
        assert data["value_range"] == [0.0, 255.0]
        assert data["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Curvature:" in result
        assert "dem/abc123.tif" in result
        assert "100x100" in result

    @pytest.mark.asyncio
    async def test_default_params(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_curvature.call_args.kwargs
        assert call_kwargs["source"] == "cop30"
        assert call_kwargs["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_png_format(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="png")
        data = json.loads(result)

        assert data["output_format"] == "png"
        call_kwargs = manager.fetch_curvature.call_args.kwargs
        assert call_kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_invalid_format(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp")
        data = json.loads(result)

        assert "error" in data
        assert "bmp" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_format_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_curvature"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp", output_mode="text"
        )

        assert "Error:" in result
        assert "bmp" in result

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "error" in data
        assert "Network timeout" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(side_effect=RuntimeError("Tile download failed"))

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Tile download failed" in result

    @pytest.mark.asyncio
    async def test_bbox_forwarded(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        await tools["dem_curvature"](bbox=bbox)

        call_kwargs = manager.fetch_curvature.call_args.kwargs
        assert call_kwargs["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_bbox_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        result = await tools["dem_curvature"](bbox=bbox)
        data = json.loads(result)

        assert data["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_source_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_message_contains_shape(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "message" in data
        assert "Curvature" in data["message"]
        assert "100x100" in data["message"]

    @pytest.mark.asyncio
    async def test_no_preview_ref(self, analysis_tools):
        tools, manager = analysis_tools
        result_no_preview = TerrainResult(
            artifact_ref="dem/xyz.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[50, 50],
            value_range=[-0.01, 0.01],
            dtype="float32",
        )
        manager.fetch_curvature = AsyncMock(return_value=result_no_preview)

        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["preview_ref"] is None
        assert data["artifact_ref"] == "dem/xyz.tif"


# ===========================================================================
# dem_terrain_ruggedness
# ===========================================================================


class TestDemTerrainRuggedness:
    """Tests for the dem_terrain_ruggedness tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_terrain_ruggedness"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/abc123.tif"
        assert data["preview_ref"] == "dem/abc123_hs.png"
        assert data["crs"] == "EPSG:4326"
        assert data["resolution_m"] == 30.0
        assert data["shape"] == [100, 100]
        assert data["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_terrain_ruggedness"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text"
        )

        assert "Terrain Ruggedness:" in result
        assert "dem/abc123.tif" in result
        assert "100x100" in result

    @pytest.mark.asyncio
    async def test_default_params(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_terrain_ruggedness"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_tri.call_args.kwargs
        assert call_kwargs["source"] == "cop30"
        assert call_kwargs["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_png_format(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_terrain_ruggedness"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_format="png"
        )
        data = json.loads(result)

        assert data["output_format"] == "png"
        call_kwargs = manager.fetch_tri.call_args.kwargs
        assert call_kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_invalid_format(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_terrain_ruggedness"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp"
        )
        data = json.loads(result)

        assert "error" in data
        assert "bmp" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_format_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_terrain_ruggedness"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp", output_mode="text"
        )

        assert "Error:" in result
        assert "bmp" in result

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_terrain_ruggedness"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "error" in data
        assert "Network timeout" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(side_effect=RuntimeError("Tile download failed"))

        result = await tools["dem_terrain_ruggedness"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text"
        )

        assert "Error:" in result
        assert "Tile download failed" in result

    @pytest.mark.asyncio
    async def test_bbox_forwarded(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        await tools["dem_terrain_ruggedness"](bbox=bbox)

        call_kwargs = manager.fetch_tri.call_args.kwargs
        assert call_kwargs["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_bbox_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        result = await tools["dem_terrain_ruggedness"](bbox=bbox)
        data = json.loads(result)

        assert data["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_source_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_terrain_ruggedness"](bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_message_contains_shape(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_terrain_ruggedness"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "message" in data
        assert "Terrain Ruggedness" in data["message"]
        assert "100x100" in data["message"]

    @pytest.mark.asyncio
    async def test_no_preview_ref(self, analysis_tools):
        tools, manager = analysis_tools
        result_no_preview = TerrainResult(
            artifact_ref="dem/xyz.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[50, 50],
            value_range=[0.0, 150.0],
            dtype="float32",
        )
        manager.fetch_tri = AsyncMock(return_value=result_no_preview)

        result = await tools["dem_terrain_ruggedness"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["preview_ref"] is None
        assert data["artifact_ref"] == "dem/xyz.tif"


# ===========================================================================
# dem_contour
# ===========================================================================


class TestDemContour:
    """Tests for the dem_contour tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/contour123.tif"
        assert data["preview_ref"] == "dem/contour123_preview.png"
        assert data["interval_m"] == 100.0
        assert data["contour_count"] == 5
        assert data["elevation_range"] == [200.0, 700.0]
        assert data["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Contours:" in result
        assert "dem/contour123.tif" in result
        assert "100x100" in result

    @pytest.mark.asyncio
    async def test_default_params(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)

        await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_contours.call_args.kwargs
        assert call_kwargs["source"] == "cop30"
        assert call_kwargs["interval_m"] == 100.0
        assert call_kwargs["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_custom_interval(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)

        await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], interval_m=50.0)

        call_kwargs = manager.fetch_contours.call_args.kwargs
        assert call_kwargs["interval_m"] == 50.0

    @pytest.mark.asyncio
    async def test_png_format(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="png")
        data = json.loads(result)

        assert data["output_format"] == "png"
        call_kwargs = manager.fetch_contours.call_args.kwargs
        assert call_kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_invalid_format(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp")
        data = json.loads(result)

        assert "error" in data
        assert "bmp" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_interval_zero(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], interval_m=0)
        data = json.loads(result)

        assert "error" in data
        assert "interval_m" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_interval_negative(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], interval_m=-50.0)
        data = json.loads(result)

        assert "error" in data

    @pytest.mark.asyncio
    async def test_invalid_format_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_contour"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp", output_mode="text"
        )

        assert "Error:" in result
        assert "bmp" in result

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "error" in data
        assert "Network timeout" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(side_effect=RuntimeError("Tile download failed"))

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Tile download failed" in result

    @pytest.mark.asyncio
    async def test_bbox_forwarded(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        await tools["dem_contour"](bbox=bbox)

        call_kwargs = manager.fetch_contours.call_args.kwargs
        assert call_kwargs["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_bbox_in_response(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)
        bbox = [7.0, 46.0, 8.0, 47.0]

        result = await tools["dem_contour"](bbox=bbox)
        data = json.loads(result)

        assert data["bbox"] == bbox

    @pytest.mark.asyncio
    async def test_source_in_response(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_message_contains_interval(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "message" in data
        assert "Contour" in data["message"]
        assert "100" in data["message"]


# ===========================================================================
# dem_profile
# ===========================================================================


class TestDemProfile:
    """Tests for the dem_profile tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0])
        data = json.loads(result)

        assert data["source"] == "cop30"
        assert data["start"] == [7.0, 46.0]
        assert data["end"] == [8.0, 47.0]
        assert data["num_points"] == 100
        assert len(data["points"]) == 3
        assert data["total_distance_m"] == pytest.approx(100000.0)
        assert data["elevation_range"] == [500.0, 1200.0]
        assert data["elevation_gain_m"] == pytest.approx(700.0)
        assert data["elevation_loss_m"] == pytest.approx(400.0)
        assert data["interpolation"] == "bilinear"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0], output_mode="text")

        assert "Profile:" in result
        assert "100000.0m" in result
        assert "500.0m" in result
        assert "1200.0m" in result

    @pytest.mark.asyncio
    async def test_default_params(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0])

        call_kwargs = manager.fetch_profile.call_args.kwargs
        assert call_kwargs["source"] == "cop30"
        assert call_kwargs["num_points"] == 100
        assert call_kwargs["interpolation"] == "bilinear"

    @pytest.mark.asyncio
    async def test_custom_num_points(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0], num_points=50)
        data = json.loads(result)

        assert data["num_points"] == 50
        call_kwargs = manager.fetch_profile.call_args.kwargs
        assert call_kwargs["num_points"] == 50

    @pytest.mark.asyncio
    async def test_invalid_num_points(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0], num_points=1)
        data = json.loads(result)

        assert "error" in data
        assert "num_points" in data["error"].lower() or "1" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_num_points_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_profile"](
            start=[7.0, 46.0], end=[8.0, 47.0], num_points=0, output_mode="text"
        )

        assert "Error:" in result

    @pytest.mark.asyncio
    async def test_invalid_interpolation(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_profile"](
            start=[7.0, 46.0], end=[8.0, 47.0], interpolation="spline"
        )
        data = json.loads(result)

        assert "error" in data
        assert "spline" in data["error"]
        assert "nearest" in data["error"]
        assert "bilinear" in data["error"]
        assert "cubic" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_interpolation_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_profile"](
            start=[7.0, 46.0],
            end=[8.0, 47.0],
            interpolation="hermite",
            output_mode="text",
        )

        assert "Error:" in result
        assert "hermite" in result

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(side_effect=RuntimeError("Coverage error"))

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0])
        data = json.loads(result)

        assert "error" in data
        assert "Coverage error" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0], output_mode="text")

        assert "Error:" in result
        assert "Network timeout" in result

    @pytest.mark.asyncio
    async def test_point_count_matches(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0])
        data = json.loads(result)

        # ProfileResult has 3 items, so points should have 3 entries
        assert len(data["points"]) == 3
        assert data["points"][0]["lon"] == pytest.approx(7.0)
        assert data["points"][0]["lat"] == pytest.approx(46.0)
        assert data["points"][0]["distance_m"] == pytest.approx(0.0)
        assert data["points"][0]["elevation_m"] == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_source_forwarded(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0], source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"
        call_kwargs = manager.fetch_profile.call_args.kwargs
        assert call_kwargs["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_start_end_forwarded(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)
        start = [7.0, 46.0]
        end = [8.0, 47.0]

        await tools["dem_profile"](start=start, end=end)

        call_kwargs = manager.fetch_profile.call_args.kwargs
        assert call_kwargs["start"] == start
        assert call_kwargs["end"] == end

    @pytest.mark.asyncio
    async def test_message_contains_distance(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0])
        data = json.loads(result)

        assert "message" in data
        assert "Profile" in data["message"]
        assert "100000.0" in data["message"]

    @pytest.mark.asyncio
    async def test_cubic_interpolation(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](
            start=[7.0, 46.0], end=[8.0, 47.0], interpolation="cubic"
        )
        data = json.loads(result)

        assert data["interpolation"] == "cubic"
        call_kwargs = manager.fetch_profile.call_args.kwargs
        assert call_kwargs["interpolation"] == "cubic"

    @pytest.mark.asyncio
    async def test_nearest_interpolation(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](
            start=[7.0, 46.0], end=[8.0, 47.0], interpolation="nearest"
        )
        data = json.loads(result)

        assert data["interpolation"] == "nearest"

    @pytest.mark.asyncio
    async def test_elevation_gain_loss_in_response(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0])
        data = json.loads(result)

        assert data["elevation_gain_m"] == pytest.approx(700.0)
        assert data["elevation_loss_m"] == pytest.approx(400.0)

    @pytest.mark.asyncio
    async def test_start_end_in_response(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)

        result = await tools["dem_profile"](start=[13.0, 52.0], end=[14.0, 53.0])
        data = json.loads(result)

        assert data["start"] == [13.0, 52.0]
        assert data["end"] == [14.0, 53.0]


# ===========================================================================
# dem_viewshed
# ===========================================================================


class TestDemViewshed:
    """Tests for the dem_viewshed tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=5000.0)
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/vs123.tif"
        assert data["preview_ref"] == "dem/vs123.png"
        assert data["crs"] == "EPSG:4326"
        assert data["resolution_m"] == 30.0
        assert data["shape"] == [200, 200]
        assert data["visible_percentage"] == pytest.approx(65.3)
        assert data["observer_elevation_m"] == pytest.approx(500.0)
        assert data["radius_m"] == pytest.approx(5000.0)
        assert data["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=5000.0, output_mode="text"
        )

        assert "Viewshed:" in result
        assert "dem/vs123.tif" in result
        assert "200x200" in result
        assert "65.3" in result

    @pytest.mark.asyncio
    async def test_default_params(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=5000.0)

        call_kwargs = manager.fetch_viewshed.call_args.kwargs
        assert call_kwargs["source"] == "cop30"
        assert call_kwargs["observer_height_m"] == 1.8
        assert call_kwargs["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_custom_params(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        await tools["dem_viewshed"](
            observer=[7.5, 46.5],
            radius_m=10000.0,
            observer_height_m=10.0,
            source="cop90",
        )

        call_kwargs = manager.fetch_viewshed.call_args.kwargs
        assert call_kwargs["observer_height_m"] == 10.0
        assert call_kwargs["source"] == "cop90"
        assert call_kwargs["radius_m"] == 10000.0

    @pytest.mark.asyncio
    async def test_invalid_radius_zero(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=0.0)
        data = json.loads(result)

        assert "error" in data
        assert "radius_m" in data["error"].lower() or "0" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_radius_negative(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=-100.0)
        data = json.loads(result)

        assert "error" in data
        assert "-100" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_radius_negative_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=-50.0, output_mode="text"
        )

        assert "Error:" in result
        assert "-50" in result

    @pytest.mark.asyncio
    async def test_radius_too_large(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=100000.0)
        data = json.loads(result)

        assert "error" in data
        assert "100000" in data["error"]
        assert "50000" in data["error"]

    @pytest.mark.asyncio
    async def test_radius_too_large_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=60000.0, output_mode="text"
        )

        assert "Error:" in result
        assert "60000" in result

    @pytest.mark.asyncio
    async def test_invalid_format(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=5000.0, output_format="bmp"
        )
        data = json.loads(result)

        assert "error" in data
        assert "bmp" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_format_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5],
            radius_m=5000.0,
            output_format="jpeg",
            output_mode="text",
        )

        assert "Error:" in result
        assert "jpeg" in result

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(side_effect=RuntimeError("Tile download failed"))

        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=5000.0)
        data = json.loads(result)

        assert "error" in data
        assert "Tile download failed" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=5000.0, output_mode="text"
        )

        assert "Error:" in result
        assert "Network timeout" in result

    @pytest.mark.asyncio
    async def test_observer_forwarded(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)
        observer = [13.405, 52.52]

        await tools["dem_viewshed"](observer=observer, radius_m=5000.0)

        call_kwargs = manager.fetch_viewshed.call_args.kwargs
        assert call_kwargs["observer"] == observer

    @pytest.mark.asyncio
    async def test_observer_in_response(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        result = await tools["dem_viewshed"](observer=[13.405, 52.52], radius_m=5000.0)
        data = json.loads(result)

        assert data["observer"] == [13.405, 52.52]

    @pytest.mark.asyncio
    async def test_source_in_response(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=5000.0, source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_message_contains_percentage(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=5000.0)
        data = json.loads(result)

        assert "message" in data
        assert "Viewshed" in data["message"]
        assert "65.3" in data["message"]
        assert "5000" in data["message"]

    @pytest.mark.asyncio
    async def test_png_format(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=5000.0, output_format="png"
        )
        data = json.loads(result)

        assert data["output_format"] == "png"
        call_kwargs = manager.fetch_viewshed.call_args.kwargs
        assert call_kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_observer_height_in_response(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=5000.0, observer_height_m=10.0
        )
        data = json.loads(result)

        assert data["observer_height_m"] == 10.0

    @pytest.mark.asyncio
    async def test_radius_at_max_boundary(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        # Exactly 50000 should succeed (the max)
        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=50000.0)
        data = json.loads(result)

        assert "error" not in data
        assert data["radius_m"] == pytest.approx(5000.0)  # from fixture

    @pytest.mark.asyncio
    async def test_manager_value_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(side_effect=ValueError("Unknown DEM source 'badname'"))

        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=5000.0, source="badname"
        )
        data = json.loads(result)

        assert "error" in data
        assert "badname" in data["error"]


# ===========================================================================
# Cross-cutting concerns
# ===========================================================================


class TestAnalysisToolRegistration:
    """Verify that all 8 analysis tools are registered."""

    def test_all_tools_registered(self, analysis_tools):
        tools, _ = analysis_tools
        expected = {
            "dem_hillshade",
            "dem_slope",
            "dem_aspect",
            "dem_curvature",
            "dem_terrain_ruggedness",
            "dem_contour",
            "dem_watershed",
            "dem_profile",
            "dem_viewshed",
        }
        assert set(tools.keys()) == expected

    def test_tools_are_callable(self, analysis_tools):
        tools, _ = analysis_tools
        for name, fn in tools.items():
            assert callable(fn), f"{name} should be callable"


class TestAnalysisOutputModeConsistency:
    """Verify that every analysis tool supports both json and text output_mode."""

    @pytest.mark.asyncio
    async def test_hillshade_json_parseable(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_hillshade_text_not_json(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_slope_json_parseable(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_slope_text_not_json(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_aspect_json_parseable(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_aspect_text_not_json(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_aspect = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_aspect"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_curvature_json_parseable(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_curvature_text_not_json(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_curvature = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_curvature"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_tri_json_parseable(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_terrain_ruggedness"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_tri_text_not_json(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_tri = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_terrain_ruggedness"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text"
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_contour_json_parseable(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)
        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_contour_text_not_json(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)
        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_profile_json_parseable(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)
        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_profile_text_not_json(self, analysis_tools, standard_profile_result):
        tools, manager = analysis_tools
        manager.fetch_profile = AsyncMock(return_value=standard_profile_result)
        result = await tools["dem_profile"](start=[7.0, 46.0], end=[8.0, 47.0], output_mode="text")
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_viewshed_json_parseable(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)
        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=5000.0)
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_viewshed_text_not_json(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)
        result = await tools["dem_viewshed"](
            observer=[7.5, 46.5], radius_m=5000.0, output_mode="text"
        )
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_watershed_json_parseable(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert isinstance(parsed, dict)

    @pytest.mark.asyncio
    async def test_watershed_text_not_json(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)
        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        with pytest.raises(json.JSONDecodeError):
            json.loads(result)

    @pytest.mark.asyncio
    async def test_error_json_parseable(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(side_effect=Exception("fail"))
        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0])
        parsed = json.loads(result)
        assert "error" in parsed

    @pytest.mark.asyncio
    async def test_error_text_has_prefix(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(side_effect=Exception("boom"))
        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")
        assert result.startswith("Error:")


# ===========================================================================
# dem_watershed
# ===========================================================================


class TestDemWatershed:
    """Tests for the dem_watershed tool."""

    @pytest.mark.asyncio
    async def test_json_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert data["artifact_ref"] == "dem/abc123.tif"
        assert data["preview_ref"] == "dem/abc123_hs.png"
        assert data["crs"] == "EPSG:4326"
        assert data["resolution_m"] == 30.0
        assert data["shape"] == [100, 100]
        assert data["value_range"] == [0.0, 255.0]
        assert data["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_text_success(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert "Watershed:" in result
        assert "dem/abc123.tif" in result
        assert "100x100" in result

    @pytest.mark.asyncio
    async def test_default_params(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0])

        call_kwargs = manager.fetch_watershed.call_args.kwargs
        assert call_kwargs["source"] == "cop30"
        assert call_kwargs["output_format"] == "geotiff"

    @pytest.mark.asyncio
    async def test_custom_source(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")

        call_kwargs = manager.fetch_watershed.call_args.kwargs
        assert call_kwargs["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_png_format(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="png")
        data = json.loads(result)

        assert data["output_format"] == "png"
        call_kwargs = manager.fetch_watershed.call_args.kwargs
        assert call_kwargs["output_format"] == "png"

    @pytest.mark.asyncio
    async def test_invalid_format(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp")
        data = json.loads(result)

        assert "error" in data
        assert "bmp" in data["error"]

    @pytest.mark.asyncio
    async def test_invalid_format_text(self, analysis_tools):
        tools, manager = analysis_tools

        result = await tools["dem_watershed"](
            bbox=[7.0, 46.0, 8.0, 47.0], output_format="bmp", output_mode="text"
        )

        assert "Error:" in result
        assert "bmp" in result

    @pytest.mark.asyncio
    async def test_manager_error(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(side_effect=RuntimeError("Network timeout"))

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "error" in data
        assert "Network timeout" in data["error"]

    @pytest.mark.asyncio
    async def test_manager_error_text(self, analysis_tools):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(side_effect=RuntimeError("Tile download failed"))

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], output_mode="text")

        assert result.startswith("Error:")
        assert "Tile download failed" in result

    @pytest.mark.asyncio
    async def test_bbox_forwarded(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        await tools["dem_watershed"](bbox=[10.0, 20.0, 11.0, 21.0])

        call_kwargs = manager.fetch_watershed.call_args.kwargs
        assert call_kwargs["bbox"] == [10.0, 20.0, 11.0, 21.0]

    @pytest.mark.asyncio
    async def test_bbox_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_watershed"](bbox=[10.0, 20.0, 11.0, 21.0])
        data = json.loads(result)

        assert data["bbox"] == [10.0, 20.0, 11.0, 21.0]

    @pytest.mark.asyncio
    async def test_source_in_response(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")
        data = json.loads(result)

        assert data["source"] == "srtm"

    @pytest.mark.asyncio
    async def test_message_contains_shape(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0])
        data = json.loads(result)

        assert "100x100" in data["message"]


# ===========================================================================
# FABDEM license warning integration
# ===========================================================================


class TestFabdemLicenseWarning:
    """Tests that FABDEM source triggers license warnings in tool responses."""

    @pytest.mark.asyncio
    async def test_hillshade_fabdem_warning(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], source="fabdem")
        data = json.loads(result)

        assert data["license_warning"] is not None
        assert "non-commercial" in data["license_warning"]
        assert "CC-BY-NC-SA-4.0" in data["license_warning"]

    @pytest.mark.asyncio
    async def test_hillshade_cop30_no_warning(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](bbox=[7.0, 46.0, 8.0, 47.0], source="cop30")
        data = json.loads(result)

        assert data["license_warning"] is None

    @pytest.mark.asyncio
    async def test_slope_fabdem_warning(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_slope = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_slope"](bbox=[7.0, 46.0, 8.0, 47.0], source="fabdem")
        data = json.loads(result)

        assert data["license_warning"] is not None
        assert "non-commercial" in data["license_warning"]

    @pytest.mark.asyncio
    async def test_watershed_fabdem_warning(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], source="fabdem")
        data = json.loads(result)

        assert data["license_warning"] is not None
        assert "non-commercial" in data["license_warning"]

    @pytest.mark.asyncio
    async def test_watershed_srtm_no_warning(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_watershed = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_watershed"](bbox=[7.0, 46.0, 8.0, 47.0], source="srtm")
        data = json.loads(result)

        assert data["license_warning"] is None

    @pytest.mark.asyncio
    async def test_fabdem_warning_in_text_mode(self, analysis_tools, standard_terrain_result):
        tools, manager = analysis_tools
        manager.fetch_hillshade = AsyncMock(return_value=standard_terrain_result)

        result = await tools["dem_hillshade"](
            bbox=[7.0, 46.0, 8.0, 47.0], source="fabdem", output_mode="text"
        )

        assert "WARNING:" in result
        assert "non-commercial" in result

    @pytest.mark.asyncio
    async def test_contour_fabdem_warning(self, analysis_tools, standard_contour_result):
        tools, manager = analysis_tools
        manager.fetch_contours = AsyncMock(return_value=standard_contour_result)

        result = await tools["dem_contour"](bbox=[7.0, 46.0, 8.0, 47.0], source="fabdem")
        data = json.loads(result)

        assert data["license_warning"] is not None
        assert "CC-BY-NC-SA-4.0" in data["license_warning"]

    @pytest.mark.asyncio
    async def test_viewshed_fabdem_warning(self, analysis_tools, standard_viewshed_result):
        tools, manager = analysis_tools
        manager.fetch_viewshed = AsyncMock(return_value=standard_viewshed_result)

        result = await tools["dem_viewshed"](observer=[7.5, 46.5], radius_m=5000.0, source="fabdem")
        data = json.loads(result)

        assert data["license_warning"] is not None
        assert "non-commercial" in data["license_warning"]
