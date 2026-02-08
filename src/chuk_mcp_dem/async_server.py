#!/usr/bin/env python3
"""
Async DEM MCP Server using chuk-mcp-server

Digital elevation model discovery, retrieval, and terrain analysis.
Fetches DEM tiles from Copernicus, SRTM, and other sources, and stores
them in chuk-artifacts for downstream analysis.

Storage is managed through chuk-mcp-server's built-in artifact store context.
"""

import logging

from chuk_mcp_server import ChukMCPServer

from .core.dem_manager import DEMManager
from .tools.analysis import register_analysis_tools
from .tools.discovery import register_discovery_tools
from .tools.download import register_download_tools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the MCP server instance
mcp = ChukMCPServer("chuk-mcp-dem")

# Create DEM manager instance
manager = DEMManager()

# Register all tool modules
register_discovery_tools(mcp, manager)
register_download_tools(mcp, manager)
register_analysis_tools(mcp, manager)

# Run the server
if __name__ == "__main__":
    logger.info("Starting DEM MCP Server...")
    logger.info("Storage: Using chuk-mcp-server artifact store context")
    mcp.run(stdio=True)
