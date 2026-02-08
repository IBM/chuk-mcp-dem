"""
Discovery tools — DEM source listing, description, status, capabilities.

These tools require no network I/O and return information about
available DEM sources and server configuration.
"""

import logging
import os

from ...constants import (
    ALL_SOURCE_IDS,
    ANALYSIS_TOOLS,
    DEFAULT_SOURCE,
    OUTPUT_FORMATS,
    TERRAIN_DERIVATIVES,
    ServerConfig,
    StorageProvider,
    EnvVar,
)
from ...models.responses import (
    CapabilitiesResponse,
    ErrorResponse,
    SourceDetailResponse,
    SourceInfo,
    SourcesResponse,
    StatusResponse,
    format_response,
)

logger = logging.getLogger(__name__)


def register_discovery_tools(mcp, manager):
    """Register discovery tools with the MCP server."""

    @mcp.tool()
    async def dem_list_sources(output_mode: str = "json") -> str:
        """List all available DEM data sources with resolution, coverage, and vertical datum.

        Use this to discover which elevation datasets are available before fetching data.

        Args:
            output_mode: "json" for structured data, "text" for human-readable summary

        Returns:
            List of available DEM sources with metadata
        """
        try:
            sources_data = manager.list_sources()
            sources = [SourceInfo(**s) for s in sources_data]
            from ...constants import SuccessMessages

            response = SourcesResponse(
                sources=sources,
                default=manager.default_source,
                message=SuccessMessages.SOURCES_LIST.format(len(sources)),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_list_sources failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_describe_source(source: str = DEFAULT_SOURCE, output_mode: str = "json") -> str:
        """Get detailed metadata for a specific DEM source including resolution, coverage bounds,
        vertical datum, accuracy, and LLM usage guidance.

        Args:
            source: DEM source ID (cop30, cop90, srtm, aster, 3dep, fabdem)
            output_mode: "json" for structured data, "text" for human-readable summary

        Returns:
            Detailed source metadata including LLM guidance
        """
        try:
            data = manager.describe_source(source)
            from ...constants import SuccessMessages

            response = SourceDetailResponse(
                **data,
                message=SuccessMessages.SOURCE_DESCRIBE.format(
                    data["name"], data["resolution_m"], data["coverage"]
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_describe_source failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_status(output_mode: str = "json") -> str:
        """Get server status including version, available sources, and storage configuration.

        Args:
            output_mode: "json" for structured data, "text" for human-readable summary

        Returns:
            Server status information
        """
        try:
            provider = os.environ.get(EnvVar.ARTIFACTS_PROVIDER, StorageProvider.MEMORY)

            store_available = False
            try:
                manager._get_store()
                store_available = True
            except Exception:
                pass

            cache_mb = manager._tile_cache_total / (1024 * 1024)

            response = StatusResponse(
                server=ServerConfig.NAME,
                version=ServerConfig.VERSION,
                default_source=manager.default_source,
                available_sources=ALL_SOURCE_IDS,
                storage_provider=provider,
                artifact_store_available=store_available,
                cache_size_mb=round(cache_mb, 1),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_status failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_capabilities(output_mode: str = "json") -> str:
        """Get full server capabilities including sources, terrain derivatives,
        analysis tools, and output formats.

        Args:
            output_mode: "json" for structured data, "text" for human-readable summary

        Returns:
            Complete server capabilities
        """
        try:
            sources_data = manager.list_sources()
            sources = [SourceInfo(**s) for s in sources_data]

            response = CapabilitiesResponse(
                server=ServerConfig.NAME,
                version=ServerConfig.VERSION,
                sources=sources,
                default_source=manager.default_source,
                terrain_derivatives=TERRAIN_DERIVATIVES,
                analysis_tools=ANALYSIS_TOOLS,
                output_formats=OUTPUT_FORMATS,
                tool_count=9,
                llm_guidance=(
                    "Use dem_list_sources to discover available DEMs. "
                    "Use dem_describe_source for detailed source info. "
                    "Use dem_check_coverage before fetching to verify data availability. "
                    "Use dem_fetch to download elevation data for a bounding box. "
                    "Use dem_fetch_point for single-point elevation queries. "
                    "Default source is cop30 (Copernicus GLO-30, 30m global)."
                ),
                message=f"{ServerConfig.NAME} v{ServerConfig.VERSION} capabilities",
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_capabilities failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)
