"""Comprehensive tests for chuk_mcp_dem.tools.discovery.api module.

Tests all four discovery tools: dem_list_sources, dem_describe_source,
dem_status, and dem_capabilities. Covers JSON/text output modes, error
handling, response structure, and edge cases.
"""

import json
import os

import pytest
from unittest.mock import MagicMock, patch

from chuk_mcp_dem.constants import (
    ALL_SOURCE_IDS,
    ANALYSIS_TOOLS,
    DEFAULT_SOURCE,
    OUTPUT_FORMATS,
    TERRAIN_DERIVATIVES,
    ServerConfig,
)
from chuk_mcp_dem.tools.discovery.api import register_discovery_tools


# ── Fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def discovery_tools(mock_manager):
    """Register discovery tools and return a dict mapping name -> coroutine function."""
    tools = {}
    mcp = MagicMock()

    def capture_tool(**kwargs):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp.tool = capture_tool
    register_discovery_tools(mcp, mock_manager)
    return tools


@pytest.fixture
def discovery_tools_with_manager(mock_manager):
    """Return both the tools dict and the mock manager for tests that need to modify it."""
    tools = {}
    mcp = MagicMock()

    def capture_tool(**kwargs):
        def decorator(fn):
            tools[fn.__name__] = fn
            return fn

        return decorator

    mcp.tool = capture_tool
    register_discovery_tools(mcp, mock_manager)
    return tools, mock_manager


# ── Registration tests ─────────────────────────────────────────────


class TestRegistration:
    """Verify that register_discovery_tools registers exactly 4 tools."""

    def test_registers_four_tools(self, discovery_tools):
        assert len(discovery_tools) == 4

    def test_registers_dem_list_sources(self, discovery_tools):
        assert "dem_list_sources" in discovery_tools

    def test_registers_dem_describe_source(self, discovery_tools):
        assert "dem_describe_source" in discovery_tools

    def test_registers_dem_status(self, discovery_tools):
        assert "dem_status" in discovery_tools

    def test_registers_dem_capabilities(self, discovery_tools):
        assert "dem_capabilities" in discovery_tools

    def test_all_tools_are_coroutines(self, discovery_tools):
        import asyncio

        for name, fn in discovery_tools.items():
            assert asyncio.iscoroutinefunction(fn), f"{name} is not a coroutine"


# ── dem_list_sources ───────────────────────────────────────────────


class TestDemListSources:
    """Tests for the dem_list_sources tool."""

    async def test_json_output_returns_valid_json(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        assert isinstance(data, dict)

    async def test_json_has_sources_key(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        assert "sources" in data

    async def test_json_has_default_key(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        assert "default" in data

    async def test_json_has_message_key(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        assert "message" in data

    async def test_returns_six_sources(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        assert len(data["sources"]) == 6

    async def test_default_source_is_cop30(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        assert data["default"] == "cop30"

    async def test_each_source_has_required_keys(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        required = {"id", "name", "resolution_m", "coverage", "vertical_datum", "void_filled"}
        for src in data["sources"]:
            assert required.issubset(set(src.keys())), f"Source {src.get('id')} missing keys"

    async def test_source_ids_match_constants(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        ids = {s["id"] for s in data["sources"]}
        assert ids == set(ALL_SOURCE_IDS)

    async def test_message_contains_count(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        assert "6" in data["message"]

    async def test_text_output_is_string(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="text")
        assert isinstance(result, str)

    async def test_text_output_contains_default(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="text")
        assert "Default:" in result

    async def test_text_output_contains_source_ids(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="text")
        for source_id in ALL_SOURCE_IDS:
            assert source_id in result, f"Source '{source_id}' not in text output"

    async def test_text_output_contains_cop30(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="text")
        assert "cop30" in result

    async def test_default_output_mode_is_json(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"]()
        data = json.loads(result)
        assert "sources" in data

    async def test_void_filled_values_are_booleans(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        for src in data["sources"]:
            assert isinstance(src["void_filled"], bool)

    async def test_resolution_m_values_are_positive(self, discovery_tools):
        result = await discovery_tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        for src in data["sources"]:
            assert src["resolution_m"] > 0


# ── dem_list_sources error handling ────────────────────────────────


class TestDemListSourcesErrors:
    """Test error handling in dem_list_sources."""

    async def test_error_returns_error_response(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        manager.list_sources = MagicMock(side_effect=RuntimeError("boom"))
        result = await tools["dem_list_sources"](output_mode="json")
        data = json.loads(result)
        assert "error" in data
        assert "boom" in data["error"]

    async def test_error_text_mode(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        manager.list_sources = MagicMock(side_effect=RuntimeError("text boom"))
        result = await tools["dem_list_sources"](output_mode="text")
        assert "Error:" in result
        assert "text boom" in result


# ── dem_describe_source ────────────────────────────────────────────


class TestDemDescribeSource:
    """Tests for the dem_describe_source tool."""

    async def test_json_output_returns_valid_json(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="json")
        data = json.loads(result)
        assert isinstance(data, dict)

    async def test_describes_cop30(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="json")
        data = json.loads(result)
        assert data["id"] == "cop30"
        assert data["name"] == "Copernicus GLO-30"

    async def test_describes_cop90(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop90", output_mode="json")
        data = json.loads(result)
        assert data["id"] == "cop90"
        assert data["resolution_m"] == 90

    async def test_describes_srtm(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="srtm", output_mode="json")
        data = json.loads(result)
        assert data["id"] == "srtm"
        assert data["void_filled"] is False

    async def test_describes_aster(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="aster", output_mode="json")
        data = json.loads(result)
        assert data["id"] == "aster"
        assert data["coverage"] == "83N-83S"

    async def test_describes_3dep(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="3dep", output_mode="json")
        data = json.loads(result)
        assert data["id"] == "3dep"
        assert data["coverage"] == "USA"
        assert data["resolution_m"] == 10

    async def test_describes_fabdem(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="fabdem", output_mode="json")
        data = json.loads(result)
        assert data["id"] == "fabdem"
        assert "NC" in data["license"]

    async def test_has_all_metadata_fields(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="json")
        data = json.loads(result)
        expected_keys = {
            "id",
            "name",
            "resolution_m",
            "coverage",
            "coverage_bounds",
            "vertical_datum",
            "vertical_unit",
            "horizontal_crs",
            "tile_size_degrees",
            "dtype",
            "nodata_value",
            "void_filled",
            "acquisition_period",
            "source_sensors",
            "accuracy_vertical_m",
            "access_url",
            "license",
            "llm_guidance",
            "message",
        }
        assert expected_keys.issubset(set(data.keys()))

    async def test_coverage_bounds_is_list_of_four(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="json")
        data = json.loads(result)
        assert len(data["coverage_bounds"]) == 4

    async def test_source_sensors_is_nonempty_list(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="json")
        data = json.loads(result)
        assert isinstance(data["source_sensors"], list)
        assert len(data["source_sensors"]) >= 1

    async def test_message_contains_source_name(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="json")
        data = json.loads(result)
        assert "Copernicus GLO-30" in data["message"]

    async def test_message_contains_resolution(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="json")
        data = json.loads(result)
        assert "30" in data["message"]

    async def test_text_output_contains_source_name(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="text")
        assert "Copernicus GLO-30" in result

    async def test_text_output_contains_resolution(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="text")
        assert "30m" in result

    async def test_text_output_contains_coverage(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="text")
        assert "global" in result

    async def test_text_output_contains_vertical_datum(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="text")
        assert "EGM2008" in result

    async def test_text_output_contains_accuracy(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="cop30", output_mode="text")
        assert "4.0" in result

    async def test_default_source_is_cop30(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](output_mode="json")
        data = json.loads(result)
        assert data["id"] == "cop30"

    async def test_unknown_source_returns_error(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](
            source="nonexistent", output_mode="json"
        )
        data = json.loads(result)
        assert "error" in data
        assert "nonexistent" in data["error"]

    async def test_unknown_source_error_lists_available(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="bogus", output_mode="json")
        data = json.loads(result)
        assert "cop30" in data["error"]

    async def test_unknown_source_text_mode(self, discovery_tools):
        result = await discovery_tools["dem_describe_source"](source="bogus", output_mode="text")
        assert "Error:" in result


# ── dem_describe_source error handling ─────────────────────────────


class TestDemDescribeSourceErrors:
    """Test error handling in dem_describe_source."""

    async def test_generic_exception_returns_error(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        manager.describe_source = MagicMock(side_effect=Exception("internal error"))
        result = await tools["dem_describe_source"](source="cop30", output_mode="json")
        data = json.loads(result)
        assert "error" in data
        assert "internal error" in data["error"]


# ── dem_status ─────────────────────────────────────────────────────


class TestDemStatus:
    """Tests for the dem_status tool."""

    async def test_json_output_returns_valid_json(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert isinstance(data, dict)

    async def test_server_name(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert data["server"] == ServerConfig.NAME

    async def test_version(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert data["version"] == ServerConfig.VERSION

    async def test_default_source(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert data["default_source"] == DEFAULT_SOURCE

    async def test_available_sources_has_six(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert len(data["available_sources"]) == 6

    async def test_available_sources_match_constants(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert set(data["available_sources"]) == set(ALL_SOURCE_IDS)

    async def test_artifact_store_available_true(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        # mock_manager has _get_store mocked to return successfully
        assert data["artifact_store_available"] is True

    async def test_artifact_store_available_false(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        manager._get_store = MagicMock(side_effect=RuntimeError("no store"))
        result = await tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert data["artifact_store_available"] is False

    async def test_cache_size_mb_is_float(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert isinstance(data["cache_size_mb"], (int, float))

    async def test_cache_size_zero_initially(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert data["cache_size_mb"] == 0.0

    async def test_cache_size_reflects_manager_state(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        manager._tile_cache_total = 5 * 1024 * 1024  # 5 MB
        result = await tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert data["cache_size_mb"] == 5.0

    async def test_storage_provider_from_env(self, discovery_tools):
        with patch.dict(os.environ, {"CHUK_ARTIFACTS_PROVIDER": "s3"}):
            result = await discovery_tools["dem_status"](output_mode="json")
            data = json.loads(result)
            assert data["storage_provider"] == "s3"

    async def test_storage_provider_default_memory(self, discovery_tools):
        with patch.dict(os.environ, {}, clear=True):
            # Ensure the env var is not set; we need to remove it specifically
            env = os.environ.copy()
            env.pop("CHUK_ARTIFACTS_PROVIDER", None)
            with patch.dict(os.environ, env, clear=True):
                result = await discovery_tools["dem_status"](output_mode="json")
                data = json.loads(result)
                assert data["storage_provider"] == "memory"

    async def test_text_output_contains_server_name(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="text")
        assert ServerConfig.NAME in result

    async def test_text_output_contains_version(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="text")
        assert ServerConfig.VERSION in result

    async def test_text_output_contains_default_source(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="text")
        assert DEFAULT_SOURCE in result

    async def test_text_artifact_store_available(self, discovery_tools):
        result = await discovery_tools["dem_status"](output_mode="text")
        assert "available" in result

    async def test_text_artifact_store_not_available(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        manager._get_store = MagicMock(side_effect=RuntimeError("no store"))
        result = await tools["dem_status"](output_mode="text")
        assert "not available" in result


# ── dem_status error handling ──────────────────────────────────────


class TestDemStatusErrors:
    """Test error handling in dem_status."""

    async def test_generic_exception_returns_error(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        # Make _tile_cache_total a property that raises
        type(manager)._tile_cache_total = property(
            lambda self: (_ for _ in ()).throw(RuntimeError("cache broken"))
        )
        result = await tools["dem_status"](output_mode="json")
        data = json.loads(result)
        assert "error" in data
        # Clean up to not affect other tests
        if hasattr(type(manager), "_tile_cache_total"):
            del type(manager)._tile_cache_total
        manager._tile_cache_total = 0


# ── dem_capabilities ───────────────────────────────────────────────


class TestDemCapabilities:
    """Tests for the dem_capabilities tool."""

    async def test_json_output_returns_valid_json(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert isinstance(data, dict)

    async def test_server_name(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert data["server"] == ServerConfig.NAME

    async def test_version(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert data["version"] == ServerConfig.VERSION

    async def test_sources_has_six(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert len(data["sources"]) == 6

    async def test_source_ids_match(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        ids = {s["id"] for s in data["sources"]}
        assert ids == set(ALL_SOURCE_IDS)

    async def test_default_source(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert data["default_source"] == DEFAULT_SOURCE

    async def test_terrain_derivatives(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert data["terrain_derivatives"] == TERRAIN_DERIVATIVES

    async def test_terrain_derivatives_count(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert len(data["terrain_derivatives"]) == 7

    async def test_analysis_tools(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert data["analysis_tools"] == ANALYSIS_TOOLS

    async def test_analysis_tools_count(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert len(data["analysis_tools"]) == 2

    async def test_output_formats(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert data["output_formats"] == OUTPUT_FORMATS

    async def test_output_formats_count(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert len(data["output_formats"]) == 2

    async def test_tool_count_is_eighteen(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert data["tool_count"] == 18

    async def test_llm_guidance_present(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert "llm_guidance" in data
        assert len(data["llm_guidance"]) > 10

    async def test_llm_guidance_mentions_dem_list_sources(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert "dem_list_sources" in data["llm_guidance"]

    async def test_llm_guidance_mentions_dem_fetch(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert "dem_fetch" in data["llm_guidance"]

    async def test_llm_guidance_mentions_cop30(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert "cop30" in data["llm_guidance"]

    async def test_message_contains_server_name(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert ServerConfig.NAME in data["message"]

    async def test_message_contains_version(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert ServerConfig.VERSION in data["message"]

    async def test_text_output_contains_server_name(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="text")
        assert ServerConfig.NAME in result

    async def test_text_output_contains_tool_count(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="text")
        assert "9" in result

    async def test_text_output_contains_terrain_derivatives(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="text")
        for deriv in TERRAIN_DERIVATIVES:
            assert deriv in result, f"Derivative '{deriv}' not in text output"

    async def test_text_output_contains_analysis_tools(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="text")
        for tool in ANALYSIS_TOOLS:
            assert tool in result, f"Analysis tool '{tool}' not in text output"

    async def test_text_output_contains_output_formats(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="text")
        for fmt in OUTPUT_FORMATS:
            assert fmt in result, f"Output format '{fmt}' not in text output"

    async def test_text_output_contains_guidance(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="text")
        assert "Guidance:" in result

    async def test_default_output_mode_is_json(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"]()
        data = json.loads(result)
        assert "sources" in data

    async def test_each_source_has_summary_keys(self, discovery_tools):
        result = await discovery_tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        required = {"id", "name", "resolution_m", "coverage", "vertical_datum", "void_filled"}
        for src in data["sources"]:
            assert required.issubset(set(src.keys())), f"Source {src.get('id')} missing keys"


# ── dem_capabilities error handling ────────────────────────────────


class TestDemCapabilitiesErrors:
    """Test error handling in dem_capabilities."""

    async def test_list_sources_error_returns_error_response(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        manager.list_sources = MagicMock(side_effect=RuntimeError("sources broken"))
        result = await tools["dem_capabilities"](output_mode="json")
        data = json.loads(result)
        assert "error" in data
        assert "sources broken" in data["error"]

    async def test_list_sources_error_text_mode(self, discovery_tools_with_manager):
        tools, manager = discovery_tools_with_manager
        manager.list_sources = MagicMock(side_effect=RuntimeError("sources broken"))
        result = await tools["dem_capabilities"](output_mode="text")
        assert "Error:" in result
        assert "sources broken" in result


# ── Cross-tool consistency ─────────────────────────────────────────


class TestCrossToolConsistency:
    """Verify consistency across discovery tool responses."""

    async def test_list_and_capabilities_same_source_count(self, discovery_tools):
        list_result = await discovery_tools["dem_list_sources"](output_mode="json")
        caps_result = await discovery_tools["dem_capabilities"](output_mode="json")
        list_data = json.loads(list_result)
        caps_data = json.loads(caps_result)
        assert len(list_data["sources"]) == len(caps_data["sources"])

    async def test_list_and_capabilities_same_source_ids(self, discovery_tools):
        list_result = await discovery_tools["dem_list_sources"](output_mode="json")
        caps_result = await discovery_tools["dem_capabilities"](output_mode="json")
        list_ids = {s["id"] for s in json.loads(list_result)["sources"]}
        caps_ids = {s["id"] for s in json.loads(caps_result)["sources"]}
        assert list_ids == caps_ids

    async def test_status_and_capabilities_same_server_name(self, discovery_tools):
        status_result = await discovery_tools["dem_status"](output_mode="json")
        caps_result = await discovery_tools["dem_capabilities"](output_mode="json")
        assert json.loads(status_result)["server"] == json.loads(caps_result)["server"]

    async def test_status_and_capabilities_same_version(self, discovery_tools):
        status_result = await discovery_tools["dem_status"](output_mode="json")
        caps_result = await discovery_tools["dem_capabilities"](output_mode="json")
        assert json.loads(status_result)["version"] == json.loads(caps_result)["version"]

    async def test_describe_source_matches_list_entry(self, discovery_tools):
        list_result = await discovery_tools["dem_list_sources"](output_mode="json")
        desc_result = await discovery_tools["dem_describe_source"](
            source="cop30", output_mode="json"
        )
        list_data = json.loads(list_result)
        desc_data = json.loads(desc_result)
        cop30_entry = next(s for s in list_data["sources"] if s["id"] == "cop30")
        assert cop30_entry["name"] == desc_data["name"]
        assert cop30_entry["resolution_m"] == desc_data["resolution_m"]
        assert cop30_entry["coverage"] == desc_data["coverage"]
        assert cop30_entry["vertical_datum"] == desc_data["vertical_datum"]
        assert cop30_entry["void_filled"] == desc_data["void_filled"]

    async def test_status_sources_match_list_sources(self, discovery_tools):
        list_result = await discovery_tools["dem_list_sources"](output_mode="json")
        status_result = await discovery_tools["dem_status"](output_mode="json")
        list_ids = {s["id"] for s in json.loads(list_result)["sources"]}
        status_ids = set(json.loads(status_result)["available_sources"])
        assert list_ids == status_ids
