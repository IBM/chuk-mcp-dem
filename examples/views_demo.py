#!/usr/bin/env python3
"""
Interactive View Tools Demo -- chuk-mcp-dem

Demonstrates dem_profile_chart and dem_map using the Grand Canyon
as a showcase area. These tools return structuredContent for MCP
clients that support rich UI rendering (e.g. Claude Desktop with
mcp-views). The demo prints the structured content to show what
would be rendered.

Usage:
    python examples/views_demo.py

Requirements:
    pip install chuk-mcp-dem
    (Requires network access to Copernicus DEM S3 buckets)
"""

import asyncio
import sys

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

# Grand Canyon rim-to-rim profile (South Rim → North Rim)
SOUTH_RIM = [-112.1125, 36.0580]
NORTH_RIM = [-112.0550, 36.2165]

# Grand Canyon bounding box
GRAND_CANYON_BBOX = [-112.2, 36.0, -111.9, 36.3]

SOURCE = "cop30"


# -- Helpers ------------------------------------------------------------------


def print_structured_content(result: dict, label: str) -> None:
    """Pretty-print structuredContent from a view tool result."""
    content = result.get("structuredContent", result)
    print(f"\n  {label} content type: {content.get('type', 'unknown')}")
    print(f"  Version: {content.get('version', 'unknown')}")
    # Print all fields except 'points'/'layers' (too verbose)
    for key, value in content.items():
        if key not in ("type", "version", "points", "layers"):
            print(f"  {key}: {value}")


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    runner = ToolRunner()

    print("=" * 60)
    print("chuk-mcp-dem -- Interactive View Tools Demo")
    print("Grand Canyon, Arizona")
    print("=" * 60)

    # -------------------------------------------------------------------------
    # Tool 1: dem_profile_chart
    # -------------------------------------------------------------------------
    print("\n--- dem_profile_chart ---")
    print(f"  Route: South Rim {SOUTH_RIM} → North Rim {NORTH_RIM}")

    result = await runner.run(
        "dem_profile_chart",
        start=SOUTH_RIM,
        end=NORTH_RIM,
        source=SOURCE,
        num_points=100,
    )

    if "error" in result:
        print(f"  ERROR: {result['error']}")
        sys.exit(1)

    content = result.get("structuredContent", result)
    print_structured_content(result, "Profile chart")

    points = content.get("points", [])
    if points:
        valid = [p for p in points if p.get("y") is not None]
        if valid:
            min_elev = min(p["y"] for p in valid)
            max_elev = max(p["y"] for p in valid)
            total_dist = valid[-1]["x"]
            print("\n  Data summary:")
            print(f"    {len(valid)} valid sample points")
            print(f"    Distance: {total_dist:.0f} m ({total_dist / 1000:.1f} km)")
            print(f"    Elevation range: {min_elev:.0f} m – {max_elev:.0f} m")
            print(f"    Canyon depth: ~{max_elev - min_elev:.0f} m")

    print("\n  In a supporting MCP client this would render as an")
    print("  interactive elevation profile chart with filled area.")

    # -------------------------------------------------------------------------
    # Tool 2: dem_map
    # -------------------------------------------------------------------------
    print("\n--- dem_map ---")
    print(f"  Bbox: {GRAND_CANYON_BBOX}")

    for basemap in ("terrain", "satellite"):
        result = await runner.run(
            "dem_map",
            bbox=GRAND_CANYON_BBOX,
            source=SOURCE,
            basemap=basemap,
        )

        if "error" in result:
            print(f"  ERROR ({basemap}): {result['error']}")
            continue

        content = result.get("structuredContent", result)
        center = content.get("center", {})
        zoom = content.get("zoom")
        layers = content.get("layers", [])

        print(f"\n  Basemap: {basemap}")
        print(f"  Center: lat={center.get('lat', '?'):.4f}, lon={center.get('lon', '?'):.4f}")
        print(f"  Zoom: {zoom}")
        print(f"  Layers: {len(layers)}")

        if layers:
            layer = layers[0]
            geojson = layer.get("features", {})
            features = geojson.get("features", [])
            print(f"  Layer label: {layer.get('label', '?')}")
            print(f"  GeoJSON features: {len(features)}")
            if features:
                geometry = features[0].get("geometry", {})
                print(f"  Geometry type: {geometry.get('type', '?')}")

    print("\n  In a supporting MCP client this would render as an")
    print("  interactive slippy map with a bbox polygon overlay.")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Demo complete!")
    print()
    print("View tools return structuredContent for MCP clients that")
    print("support rich UI rendering. To see these rendered:")
    print("  1. Use Claude Desktop with mcp-views enabled")
    print("  2. Connect to the chuk-mcp-dem server")
    print("  3. Ask: 'Show an elevation profile chart from the South")
    print("     Rim to the North Rim of the Grand Canyon'")
    print("  4. Ask: 'Show the Grand Canyon on a map'")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
