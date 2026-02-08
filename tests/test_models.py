"""
Comprehensive tests for chuk-mcp-dem response models.

Tests all Pydantic models in chuk_mcp_dem.models.responses for:
- Valid creation
- extra="forbid" rejects unknown fields
- to_text() output contains expected strings
- format_response() in json/text modes
"""

import json

import pytest
from pydantic import ValidationError

from chuk_mcp_dem.models.responses import (
    AspectResponse,
    CapabilitiesResponse,
    CoverageCheckResponse,
    ErrorResponse,
    FetchResponse,
    HillshadeResponse,
    MultiPointResponse,
    PointElevationResponse,
    PointInfo,
    ProfilePointInfo,
    ProfileResponse,
    SizeEstimateResponse,
    SlopeResponse,
    SourceDetailResponse,
    SourceInfo,
    SourcesResponse,
    StatusResponse,
    ViewshedResponse,
    format_response,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_source_info(**overrides) -> SourceInfo:
    defaults = dict(
        id="cop30",
        name="Copernicus DEM 30m",
        resolution_m=30,
        coverage="global",
        vertical_datum="EGM2008",
        void_filled=True,
    )
    defaults.update(overrides)
    return SourceInfo(**defaults)


def _make_source_detail(**overrides) -> SourceDetailResponse:
    defaults = dict(
        id="cop30",
        name="Copernicus DEM 30m",
        resolution_m=30,
        coverage="global",
        coverage_bounds=[-180.0, -90.0, 180.0, 90.0],
        vertical_datum="EGM2008",
        vertical_unit="metres",
        horizontal_crs="EPSG:4326",
        tile_size_degrees=1.0,
        dtype="float32",
        nodata_value=-9999.0,
        void_filled=True,
        acquisition_period="2010-2015",
        source_sensors=["TanDEM-X", "SRTM"],
        accuracy_vertical_m=4.0,
        access_url="https://example.com/cop30",
        license="CC-BY-4.0",
        llm_guidance="Use cop30 for global queries.",
        message="OK",
    )
    defaults.update(overrides)
    return SourceDetailResponse(**defaults)


def _make_point_info(**overrides) -> PointInfo:
    defaults = dict(lon=-105.0, lat=40.0, elevation_m=1625.3)
    defaults.update(overrides)
    return PointInfo(**defaults)


# ===========================================================================
# ErrorResponse
# ===========================================================================


class TestErrorResponse:
    def test_creation(self):
        resp = ErrorResponse(error="something broke")
        assert resp.error == "something broke"

    def test_to_text(self):
        resp = ErrorResponse(error="tile not found")
        assert resp.to_text() == "Error: tile not found"

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ErrorResponse(error="x", extra_field="bad")

    def test_missing_required_field(self):
        with pytest.raises(ValidationError):
            ErrorResponse()


# ===========================================================================
# SourceInfo
# ===========================================================================


class TestSourceInfo:
    def test_creation(self):
        si = _make_source_info()
        assert si.id == "cop30"
        assert si.resolution_m == 30
        assert si.void_filled is True

    def test_to_text_void_filled(self):
        text = _make_source_info(void_filled=True).to_text()
        assert "void-filled" in text
        assert "cop30" in text
        assert "30m" in text

    def test_to_text_may_have_voids(self):
        text = _make_source_info(void_filled=False).to_text()
        assert "may have voids" in text
        assert "void-filled" not in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            _make_source_info(bogus=42)

    def test_to_text_format(self):
        text = _make_source_info().to_text()
        assert text == "cop30: Copernicus DEM 30m (30m, global, void-filled)"


# ===========================================================================
# SourcesResponse
# ===========================================================================


class TestSourcesResponse:
    def test_creation(self):
        si = _make_source_info()
        resp = SourcesResponse(sources=[si], default="cop30", message="Found 1 source")
        assert len(resp.sources) == 1
        assert resp.default == "cop30"

    def test_to_text(self):
        si = _make_source_info()
        resp = SourcesResponse(sources=[si], default="cop30", message="Available sources")
        text = resp.to_text()
        assert "Available sources" in text
        assert "Default: cop30" in text
        assert "cop30:" in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            SourcesResponse(
                sources=[_make_source_info()],
                default="cop30",
                message="ok",
                unknown="nope",
            )

    def test_multiple_sources(self):
        s1 = _make_source_info(id="cop30")
        s2 = _make_source_info(id="cop90", resolution_m=90)
        resp = SourcesResponse(sources=[s1, s2], default="cop30", message="2 sources")
        text = resp.to_text()
        assert "cop30:" in text
        assert "cop90:" in text


# ===========================================================================
# SourceDetailResponse
# ===========================================================================


class TestSourceDetailResponse:
    def test_creation(self):
        resp = _make_source_detail()
        assert resp.id == "cop30"
        assert resp.coverage_bounds == [-180.0, -90.0, 180.0, 90.0]
        assert resp.source_sensors == ["TanDEM-X", "SRTM"]

    def test_to_text(self):
        resp = _make_source_detail()
        text = resp.to_text()
        assert "Copernicus DEM 30m (cop30)" in text
        assert "Resolution: 30m" in text
        assert "-180.0" in text
        assert "EGM2008" in text
        assert "float32" in text
        assert "void-filled" in text
        assert "TanDEM-X, SRTM" in text
        assert "2010-2015" in text
        assert "CC-BY-4.0" in text
        assert "Use cop30 for global queries." in text

    def test_to_text_may_have_voids(self):
        text = _make_source_detail(void_filled=False).to_text()
        assert "may have voids" in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            _make_source_detail(nope=True)


# ===========================================================================
# CoverageCheckResponse
# ===========================================================================


class TestCoverageCheckResponse:
    def _make(self, **overrides):
        defaults = dict(
            source="cop30",
            bbox=[-105.5, 39.5, -104.5, 40.5],
            fully_covered=True,
            coverage_percentage=100.0,
            tiles_required=4,
            tile_ids=["N39W106", "N39W105", "N40W106", "N40W105"],
            estimated_size_mb=52.0,
            message="Fully covered",
        )
        defaults.update(overrides)
        return CoverageCheckResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.fully_covered is True
        assert resp.tiles_required == 4
        assert len(resp.tile_ids) == 4

    def test_to_text_fully_covered(self):
        text = self._make().to_text()
        assert "fully covered" in text
        assert "cop30" in text
        assert "Tiles required: 4" in text
        assert "52.0 MB" in text
        assert "N39W106" in text

    def test_to_text_partial_coverage(self):
        text = self._make(fully_covered=False, coverage_percentage=73.5).to_text()
        assert "73.5% covered" in text
        assert "fully covered" not in text

    def test_tile_ids_truncation(self):
        tiles = [f"tile_{i}" for i in range(15)]
        text = self._make(tile_ids=tiles, tiles_required=15).to_text()
        # Only first 10 shown
        assert "tile_0" in text
        assert "tile_9" in text
        assert "tile_10" not in text
        assert "and 5 more" in text

    def test_tile_ids_no_truncation_at_10(self):
        tiles = [f"tile_{i}" for i in range(10)]
        text = self._make(tile_ids=tiles, tiles_required=10).to_text()
        assert "tile_9" in text
        assert "more" not in text

    def test_empty_tile_ids(self):
        text = self._make(
            tile_ids=[], tiles_required=0, fully_covered=False, coverage_percentage=0.0
        ).to_text()
        assert "Tile IDs" not in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(extra_stuff="no")

    def test_coverage_percentage_bounds(self):
        with pytest.raises(ValidationError):
            self._make(coverage_percentage=101.0)
        with pytest.raises(ValidationError):
            self._make(coverage_percentage=-1.0)


# ===========================================================================
# SizeEstimateResponse
# ===========================================================================


class TestSizeEstimateResponse:
    def _make(self, **overrides):
        defaults = dict(
            source="cop30",
            bbox=[-105.0, 39.0, -104.0, 40.0],
            native_resolution_m=30,
            target_resolution_m=30,
            dimensions=[3601, 3601],
            pixels=12967201,
            dtype="float32",
            estimated_bytes=51868804,
            estimated_mb=49.5,
            warning=None,
            message="Size estimated",
        )
        defaults.update(overrides)
        return SizeEstimateResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.pixels == 12967201
        assert resp.estimated_mb == 49.5
        assert resp.warning is None

    def test_to_text_no_warning(self):
        text = self._make().to_text()
        assert "Size estimate: cop30" in text
        assert "3601x3601" in text
        assert "megapixels" in text
        assert "30m" in text
        assert "float32" in text
        assert "49.5 MB" in text
        assert "WARNING" not in text

    def test_to_text_with_warning(self):
        text = self._make(warning="Large area: consider reducing resolution").to_text()
        assert "WARNING: Large area: consider reducing resolution" in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(nope="x")

    def test_megapixels_calculation(self):
        text = self._make(pixels=5_000_000).to_text()
        assert "5.0 megapixels" in text


# ===========================================================================
# FetchResponse
# ===========================================================================


class TestFetchResponse:
    def _make(self, **overrides):
        defaults = dict(
            source="cop30",
            bbox=[-105.0, 39.0, -104.0, 40.0],
            artifact_ref="artifacts/dem/cop30_abc123.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30,
            shape=[3601, 3601],
            elevation_range=[1500.0, 4300.0],
            dtype="float32",
            nodata_pixels=0,
            message="Fetch complete",
        )
        defaults.update(overrides)
        return FetchResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.artifact_ref == "artifacts/dem/cop30_abc123.tif"
        assert resp.preview_ref is None
        assert resp.nodata_pixels == 0

    def test_to_text_no_preview(self):
        text = self._make().to_text()
        assert "Fetched DEM: cop30" in text
        assert "artifacts/dem/cop30_abc123.tif" in text
        assert "3601x3601" in text
        assert "EPSG:4326" in text
        assert "30m" in text
        assert "1500.0m to 4300.0m" in text
        assert "float32" in text
        assert "NoData pixels: 0" in text
        assert "Preview" not in text

    def test_to_text_with_preview(self):
        text = self._make(preview_ref="artifacts/dem/preview_abc.png").to_text()
        assert "Preview: artifacts/dem/preview_abc.png" in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(oops=1)

    def test_nodata_pixels_non_negative(self):
        with pytest.raises(ValidationError):
            self._make(nodata_pixels=-1)


# ===========================================================================
# PointElevationResponse
# ===========================================================================


class TestPointElevationResponse:
    def _make(self, **overrides):
        defaults = dict(
            lon=-105.2705,
            lat=40.0150,
            source="cop30",
            elevation_m=1625.3,
            interpolation="bilinear",
            uncertainty_m=4.0,
            message="Elevation retrieved",
        )
        defaults.update(overrides)
        return PointElevationResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.lon == pytest.approx(-105.2705)
        assert resp.elevation_m == pytest.approx(1625.3)

    def test_to_text(self):
        text = self._make().to_text()
        assert "-105.270500" in text
        assert "40.015000" in text
        assert "1625.3m" in text
        assert "cop30" in text
        assert "bilinear" in text
        assert "+/-4.0m" in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(foo="bar")


# ===========================================================================
# PointInfo
# ===========================================================================


class TestPointInfo:
    def test_creation(self):
        pi = _make_point_info()
        assert pi.lon == -105.0
        assert pi.lat == 40.0
        assert pi.elevation_m == pytest.approx(1625.3)

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            PointInfo(lon=0, lat=0, elevation_m=0, extra="bad")

    def test_no_to_text(self):
        """PointInfo has no to_text method."""
        pi = _make_point_info()
        assert not hasattr(pi, "to_text")


# ===========================================================================
# MultiPointResponse
# ===========================================================================


class TestMultiPointResponse:
    def _make(self, **overrides):
        points = [
            _make_point_info(lon=-105.0, lat=40.0, elevation_m=1600.0),
            _make_point_info(lon=-104.0, lat=39.0, elevation_m=2100.5),
        ]
        defaults = dict(
            source="cop30",
            point_count=2,
            points=points,
            elevation_range=[1600.0, 2100.5],
            interpolation="bilinear",
            message="Multi-point query complete",
        )
        defaults.update(overrides)
        return MultiPointResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.point_count == 2
        assert len(resp.points) == 2

    def test_to_text(self):
        text = self._make().to_text()
        assert "2 point(s)" in text
        assert "cop30" in text
        assert "bilinear" in text
        assert "1600.0m to 2100.5m" in text
        assert "-105.000000" in text
        assert "2100.5m" in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(surplus="yes")

    def test_point_count_minimum(self):
        with pytest.raises(ValidationError):
            self._make(point_count=0)


# ===========================================================================
# StatusResponse
# ===========================================================================


class TestStatusResponse:
    def _make(self, **overrides):
        defaults = dict(
            default_source="cop30",
            available_sources=["cop30", "cop90"],
            storage_provider="memory",
        )
        defaults.update(overrides)
        return StatusResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.server == "chuk-mcp-dem"
        assert resp.version == "0.1.0"
        assert resp.artifact_store_available is False
        assert resp.cache_size_mb == 0.0

    def test_default_values(self):
        """server, version, artifact_store_available, cache_size_mb all have defaults."""
        resp = self._make()
        assert resp.server == "chuk-mcp-dem"
        assert resp.version == "0.1.0"
        assert resp.artifact_store_available is False
        assert resp.cache_size_mb == pytest.approx(0.0)

    def test_to_text(self):
        text = self._make(artifact_store_available=True, cache_size_mb=12.5).to_text()
        assert "chuk-mcp-dem v0.1.0" in text
        assert "Default source: cop30" in text
        assert "cop30, cop90" in text
        assert "memory" in text
        assert "available" in text
        assert "12.5 MB" in text

    def test_to_text_store_not_available(self):
        text = self._make(artifact_store_available=False).to_text()
        assert "not available" in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(extra_field="no")

    def test_custom_server_and_version(self):
        resp = self._make(server="custom-server", version="2.0.0")
        assert resp.server == "custom-server"
        text = resp.to_text()
        assert "custom-server v2.0.0" in text


# ===========================================================================
# CapabilitiesResponse
# ===========================================================================


class TestCapabilitiesResponse:
    def _make(self, **overrides):
        defaults = dict(
            server="chuk-mcp-dem",
            version="0.1.0",
            sources=[_make_source_info()],
            default_source="cop30",
            terrain_derivatives=["slope", "aspect", "hillshade"],
            analysis_tools=["viewshed", "profile"],
            output_formats=["geotiff", "png", "numpy"],
            tool_count=8,
            llm_guidance="Use dem_capabilities first.",
            message="Capabilities listed",
        )
        defaults.update(overrides)
        return CapabilitiesResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.tool_count == 8
        assert len(resp.sources) == 1
        assert resp.terrain_derivatives == ["slope", "aspect", "hillshade"]

    def test_to_text(self):
        text = self._make().to_text()
        assert "chuk-mcp-dem v0.1.0" in text
        assert "Tools: 8" in text
        assert "Default source: cop30" in text
        assert "cop30" in text
        assert "slope, aspect, hillshade" in text
        assert "viewshed, profile" in text
        assert "geotiff, png, numpy" in text
        assert "Use dem_capabilities first." in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(unknown=True)

    def test_tool_count_non_negative(self):
        with pytest.raises(ValidationError):
            self._make(tool_count=-1)


# ===========================================================================
# format_response
# ===========================================================================


class TestFormatResponse:
    def test_json_mode_returns_json(self):
        resp = ErrorResponse(error="bad request")
        result = format_response(resp, output_mode="json")
        parsed = json.loads(result)
        assert parsed["error"] == "bad request"

    def test_text_mode_calls_to_text(self):
        resp = ErrorResponse(error="not found")
        result = format_response(resp, output_mode="text")
        assert result == "Error: not found"

    def test_default_mode_is_json(self):
        resp = ErrorResponse(error="oops")
        result = format_response(resp)
        parsed = json.loads(result)
        assert parsed["error"] == "oops"

    def test_text_mode_fallback_to_json_when_no_to_text(self):
        """If a model lacks to_text(), text mode should fall back to JSON."""
        # PointInfo has no to_text method, so format_response should fall back to JSON
        pi = PointInfo(lon=1.0, lat=2.0, elevation_m=100.0)
        result = format_response(pi, output_mode="text")
        parsed = json.loads(result)
        assert parsed["lon"] == 1.0
        assert parsed["lat"] == 2.0

    def test_json_mode_complex_model(self):
        si = _make_source_info()
        resp = SourcesResponse(sources=[si], default="cop30", message="ok")
        result = format_response(resp, output_mode="json")
        parsed = json.loads(result)
        assert parsed["default"] == "cop30"
        assert len(parsed["sources"]) == 1
        assert parsed["sources"][0]["id"] == "cop30"

    def test_text_mode_complex_model(self):
        si = _make_source_info()
        resp = SourcesResponse(sources=[si], default="cop30", message="ok")
        result = format_response(resp, output_mode="text")
        assert "Default: cop30" in result
        assert "cop30:" in result


# ===========================================================================
# HillshadeResponse
# ===========================================================================


class TestHillshadeResponse:
    def _make(self, **overrides):
        defaults = dict(
            source="cop30",
            bbox=[7.0, 46.0, 8.0, 47.0],
            artifact_ref="dem/abc123.tif",
            preview_ref=None,
            azimuth=315.0,
            altitude=45.0,
            z_factor=1.0,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            value_range=[0.0, 254.0],
            output_format="geotiff",
            message="Hillshade computed",
        )
        defaults.update(overrides)
        return HillshadeResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.source == "cop30"
        assert resp.azimuth == 315.0
        assert resp.altitude == 45.0
        assert resp.z_factor == 1.0
        assert resp.shape == [100, 100]
        assert resp.preview_ref is None

    def test_to_text_returns_string(self):
        text = self._make().to_text()
        assert isinstance(text, str)
        assert "Hillshade: cop30" in text
        assert "dem/abc123.tif" in text
        assert "100x100" in text
        assert "315" in text
        assert "45" in text

    def test_to_text_with_preview(self):
        text = self._make(preview_ref="dem/preview_hs.png").to_text()
        assert "Preview: dem/preview_hs.png" in text

    def test_to_text_no_preview(self):
        text = self._make(preview_ref=None).to_text()
        assert "Preview" not in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(unknown_field="bad")

    def test_optional_preview_ref_none(self):
        resp = self._make(preview_ref=None)
        assert resp.preview_ref is None

    def test_value_range_in_text(self):
        text = self._make(value_range=[10.5, 250.3]).to_text()
        assert "10.5" in text
        assert "250.3" in text


# ===========================================================================
# SlopeResponse
# ===========================================================================


class TestSlopeResponse:
    def _make(self, **overrides):
        defaults = dict(
            source="cop30",
            bbox=[7.0, 46.0, 8.0, 47.0],
            artifact_ref="dem/slope_abc.tif",
            preview_ref=None,
            units="degrees",
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            value_range=[0.0, 85.0],
            output_format="geotiff",
            message="Slope computed",
        )
        defaults.update(overrides)
        return SlopeResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.source == "cop30"
        assert resp.units == "degrees"
        assert resp.resolution_m == 30.0
        assert resp.preview_ref is None

    def test_to_text_returns_string(self):
        text = self._make().to_text()
        assert isinstance(text, str)
        assert "Slope: cop30 (degrees)" in text
        assert "dem/slope_abc.tif" in text
        assert "100x100" in text

    def test_to_text_percent_units(self):
        text = self._make(units="percent").to_text()
        assert "Slope: cop30 (percent)" in text

    def test_to_text_with_preview(self):
        text = self._make(preview_ref="dem/slope_preview.png").to_text()
        assert "Preview: dem/slope_preview.png" in text

    def test_to_text_no_preview(self):
        text = self._make(preview_ref=None).to_text()
        assert "Preview" not in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(bogus=42)

    def test_optional_preview_ref_none(self):
        resp = self._make(preview_ref=None)
        assert resp.preview_ref is None


# ===========================================================================
# AspectResponse
# ===========================================================================


class TestAspectResponse:
    def _make(self, **overrides):
        defaults = dict(
            source="cop30",
            bbox=[7.0, 46.0, 8.0, 47.0],
            artifact_ref="dem/aspect_abc.tif",
            preview_ref=None,
            flat_value=-1.0,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[100, 100],
            value_range=[-1.0, 359.9],
            output_format="geotiff",
            message="Aspect computed",
        )
        defaults.update(overrides)
        return AspectResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.source == "cop30"
        assert resp.flat_value == -1.0
        assert resp.shape == [100, 100]
        assert resp.preview_ref is None

    def test_to_text_returns_string(self):
        text = self._make().to_text()
        assert isinstance(text, str)
        assert "Aspect: cop30" in text
        assert "dem/aspect_abc.tif" in text
        assert "100x100" in text
        assert "Flat value: -1.0" in text

    def test_to_text_with_preview(self):
        text = self._make(preview_ref="dem/aspect_preview.png").to_text()
        assert "Preview: dem/aspect_preview.png" in text

    def test_to_text_no_preview(self):
        text = self._make(preview_ref=None).to_text()
        assert "Preview" not in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(extra=True)

    def test_optional_preview_ref_none(self):
        resp = self._make(preview_ref=None)
        assert resp.preview_ref is None

    def test_custom_flat_value(self):
        resp = self._make(flat_value=-9999.0)
        assert resp.flat_value == -9999.0
        text = resp.to_text()
        assert "Flat value: -9999.0" in text


# ===========================================================================
# ProfilePointInfo
# ===========================================================================


class TestProfilePointInfo:
    def test_creation(self):
        p = ProfilePointInfo(lon=7.5, lat=46.5, distance_m=0.0, elevation_m=1500.0)
        assert p.lon == 7.5
        assert p.lat == 46.5
        assert p.distance_m == 0.0
        assert p.elevation_m == 1500.0

    def test_no_to_text(self):
        """ProfilePointInfo has no to_text method."""
        p = ProfilePointInfo(lon=0.0, lat=0.0, distance_m=0.0, elevation_m=0.0)
        assert not hasattr(p, "to_text")

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            ProfilePointInfo(lon=0.0, lat=0.0, distance_m=0.0, elevation_m=0.0, extra="bad")

    def test_negative_coords(self):
        p = ProfilePointInfo(lon=-105.27, lat=-33.5, distance_m=5000.0, elevation_m=200.0)
        assert p.lon == pytest.approx(-105.27)
        assert p.lat == pytest.approx(-33.5)


# ===========================================================================
# ProfileResponse
# ===========================================================================


class TestProfileResponse:
    def _make(self, **overrides):
        points = [
            ProfilePointInfo(lon=7.0, lat=46.0, distance_m=0.0, elevation_m=1000.0),
            ProfilePointInfo(lon=7.5, lat=46.5, distance_m=5000.0, elevation_m=1500.0),
            ProfilePointInfo(lon=8.0, lat=47.0, distance_m=10000.0, elevation_m=1200.0),
        ]
        defaults = dict(
            source="cop30",
            start=[7.0, 46.0],
            end=[8.0, 47.0],
            num_points=3,
            points=points,
            total_distance_m=10000.0,
            elevation_range=[1000.0, 1500.0],
            elevation_gain_m=500.0,
            elevation_loss_m=300.0,
            interpolation="bilinear",
            message="Profile extracted",
        )
        defaults.update(overrides)
        return ProfileResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.source == "cop30"
        assert resp.num_points == 3
        assert len(resp.points) == 3
        assert resp.total_distance_m == 10000.0
        assert resp.elevation_gain_m == 500.0
        assert resp.elevation_loss_m == 300.0

    def test_to_text_returns_string(self):
        text = self._make().to_text()
        assert isinstance(text, str)
        assert "Profile: cop30" in text
        assert "7.000000" in text
        assert "46.000000" in text
        assert "10000.0m" in text
        assert "3 points" in text
        assert "Gain: 500.0m" in text
        assert "Loss: 300.0m" in text
        assert "bilinear" in text

    def test_to_text_elevation_range(self):
        text = self._make().to_text()
        assert "1000.0m to 1500.0m" in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(nope="x")

    def test_start_end_in_text(self):
        text = self._make().to_text()
        assert "Start:" in text
        assert "End:" in text


# ===========================================================================
# ViewshedResponse
# ===========================================================================


class TestViewshedResponse:
    def _make(self, **overrides):
        defaults = dict(
            source="cop30",
            observer=[7.5, 46.5],
            radius_m=5000.0,
            observer_height_m=1.8,
            observer_elevation_m=1500.0,
            artifact_ref="dem/viewshed_abc.tif",
            preview_ref=None,
            crs="EPSG:4326",
            resolution_m=30.0,
            shape=[200, 200],
            visible_percentage=45.2,
            output_format="geotiff",
            message="Viewshed computed",
        )
        defaults.update(overrides)
        return ViewshedResponse(**defaults)

    def test_creation(self):
        resp = self._make()
        assert resp.source == "cop30"
        assert resp.observer == [7.5, 46.5]
        assert resp.radius_m == 5000.0
        assert resp.observer_height_m == 1.8
        assert resp.observer_elevation_m == 1500.0
        assert resp.visible_percentage == 45.2
        assert resp.preview_ref is None

    def test_to_text_returns_string(self):
        text = self._make().to_text()
        assert isinstance(text, str)
        assert "Viewshed: cop30" in text
        assert "7.500000" in text
        assert "46.500000" in text
        assert "5000m" in text
        assert "1.8m" in text
        assert "1500.0m" in text
        assert "45.2%" in text
        assert "dem/viewshed_abc.tif" in text

    def test_to_text_with_preview(self):
        text = self._make(preview_ref="dem/viewshed_preview.png").to_text()
        assert "Preview: dem/viewshed_preview.png" in text

    def test_to_text_no_preview(self):
        text = self._make(preview_ref=None).to_text()
        assert "Preview" not in text

    def test_extra_fields_rejected(self):
        with pytest.raises(ValidationError):
            self._make(unknown="bad")

    def test_optional_preview_ref_none(self):
        resp = self._make(preview_ref=None)
        assert resp.preview_ref is None

    def test_visible_percentage_bounds(self):
        """visible_percentage has ge=0, le=100 constraint."""
        with pytest.raises(ValidationError):
            self._make(visible_percentage=101.0)
        with pytest.raises(ValidationError):
            self._make(visible_percentage=-1.0)

    def test_visible_percentage_zero(self):
        resp = self._make(visible_percentage=0.0)
        assert resp.visible_percentage == 0.0

    def test_visible_percentage_hundred(self):
        resp = self._make(visible_percentage=100.0)
        assert resp.visible_percentage == 100.0
