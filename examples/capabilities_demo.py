#!/usr/bin/env python3
"""
Capabilities Demo -- chuk-mcp-dem

Quick-start script showing what the server can do, without any network
access. Lists DEM sources, server status, full capabilities, and
demonstrates the dual output mode (JSON vs text).

Usage:
    python examples/capabilities_demo.py
"""

import asyncio

from tool_runner import ToolRunner


async def main() -> None:
    runner = ToolRunner()

    print("=" * 60)
    print("chuk-mcp-dem -- Server Capabilities")
    print("=" * 60)

    # List all registered tools
    print(f"\nRegistered tools ({len(runner.tool_names)}):")
    for name in sorted(runner.tool_names):
        print(f"  - {name}")

    # List DEM sources
    sources = await runner.run("dem_list_sources")
    print(f"\nDEM Sources ({len(sources['sources'])}):")
    print(f"  Default: {sources['default']}")
    for s in sources["sources"]:
        voids = "void-filled" if s["void_filled"] else "may have voids"
        print(
            f"  {s['id']:8s}  {s['name']:20s}  {s['resolution_m']:4d}m  {s['coverage']:10s}  {voids}"
        )

    # Describe the default source in detail
    detail = await runner.run("dem_describe_source", source="cop30")
    print(f"\nDefault Source Detail: {detail['name']}")
    print(f"  Resolution: {detail['resolution_m']}m")
    print(f"  Coverage: {detail['coverage']} {detail['coverage_bounds']}")
    print(f"  Vertical datum: {detail['vertical_datum']} ({detail['vertical_unit']})")
    print(f"  Accuracy: +/-{detail['accuracy_vertical_m']}m vertical")
    print(f"  Sensors: {', '.join(detail['source_sensors'])}")
    print(f"  Period: {detail['acquisition_period']}")
    print(f"  License: {detail['license']}")
    print(f"  Guidance: {detail['llm_guidance']}")

    # Server status
    status = await runner.run("dem_status")
    print("\nServer Status:")
    print(f"  {status['server']} v{status['version']}")
    print(f"  Storage: {status['storage_provider']}")
    print(
        f"  Artifact store: {'available' if status['artifact_store_available'] else 'not available'}"
    )
    print(f"  Cache: {status['cache_size_mb']:.1f} MB")
    print(f"  Sources: {', '.join(status['available_sources'])}")

    # Full capabilities
    caps = await runner.run("dem_capabilities")
    print("\nCapabilities:")
    print(f"  Tools: {caps['tool_count']}")
    print(f"  Terrain derivatives: {', '.join(caps['terrain_derivatives'])}")
    print(f"  Analysis tools: {', '.join(caps['analysis_tools'])}")
    print(f"  Output formats: {', '.join(caps['output_formats'])}")
    print(f"  Guidance: {caps['llm_guidance']}")

    # ---------------------------------------------------------------
    # Dual output mode: text vs JSON
    # ---------------------------------------------------------------
    print("\n" + "-" * 60)
    print("Dual Output Mode Demo")
    print("-" * 60)

    # Status in text mode
    print("\ndem_status (output_mode='text'):")
    status_text = await runner.run_text("dem_status")
    print(status_text)

    # Capabilities in text mode
    print("\ndem_capabilities (output_mode='text'):")
    caps_text = await runner.run_text("dem_capabilities")
    print(caps_text)

    # List sources in text mode
    print("\ndem_list_sources (output_mode='text'):")
    sources_text = await runner.run_text("dem_list_sources")
    print(sources_text)

    print("\n" + "=" * 60)
    print("All capabilities shown above require no network access.")
    print("Run other demos (mount_rainier, grand_canyon_profile,")
    print("mountain_peaks, viewshed, source_comparison) to see the")
    print("full pipeline in action with live DEM data.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
