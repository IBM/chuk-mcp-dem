#!/usr/bin/env python3
"""
Viewshed Analysis Demo -- chuk-mcp-dem

Computes visibility from Cadillac Mountain summit in Acadia National Park.
Demonstrates point elevation query and viewshed analysis.

Usage:
    python examples/viewshed_demo.py

Output:
    examples/output/cadillac_viewshed.png

Requirements:
    pip install chuk-mcp-dem matplotlib
    (Requires network access to Copernicus DEM S3 buckets)
"""

import asyncio
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

OBSERVER = [-68.2265, 44.3525]  # Cadillac Mountain summit, Acadia NP, ME
RADIUS_M = 5000.0  # 5 km radius
OBSERVER_HEIGHT_M = 1.8  # Standing height
SOURCE = "cop30"
OUTPUT_DIR = Path(__file__).parent / "output"


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()

    print("=" * 60)
    print("Cadillac Mountain -- Viewshed Analysis")
    print("=" * 60)

    # Step 1: Query summit elevation
    print("\nStep 1: Querying summit elevation...")
    print(f"  Location: ({OBSERVER[0]}, {OBSERVER[1]})")

    point = await runner.run("dem_fetch_point", lon=OBSERVER[0], lat=OBSERVER[1], source=SOURCE)
    if "error" in point:
        print(f"  ERROR: {point['error']}")
        sys.exit(1)
    summit_elev = point["elevation_m"]
    print(f"  Summit elevation: {summit_elev:.1f}m")
    print(f"  Uncertainty: +/-{point['uncertainty_m']}m")

    # Step 2: Compute viewshed
    print(f"\nStep 2: Computing viewshed (radius={RADIUS_M:.0f}m)...")
    print(f"  Observer height: {OBSERVER_HEIGHT_M}m above ground")

    viewshed = await runner.run(
        "dem_viewshed",
        observer=OBSERVER,
        radius_m=RADIUS_M,
        source=SOURCE,
        observer_height_m=OBSERVER_HEIGHT_M,
    )
    if "error" in viewshed:
        print(f"  ERROR: {viewshed['error']}")
        sys.exit(1)

    print(f"  Artifact: {viewshed['artifact_ref']}")
    print(f"  Shape: {viewshed['shape']}")
    print(f"  Observer elevation: {viewshed['observer_elevation_m']:.1f}m")
    print(f"  Visible: {viewshed['visible_percentage']:.1f}%")

    # Retrieve viewshed raster
    store = runner.manager._get_store()
    vs_data = await store.retrieve(viewshed["artifact_ref"])
    with rasterio.open(io.BytesIO(vs_data)) as src:
        vis_array = src.read(1)

    # Step 3: Render viewshed map
    print("\nStep 3: Rendering viewshed map...")

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Create RGB visualisation: green=visible, red=hidden, gray=outside radius
    h, w = vis_array.shape
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)  # gray for outside
    rgb[vis_array == 1.0] = [50, 180, 50]  # green = visible
    rgb[vis_array == 0.0] = [200, 60, 60]  # red = hidden

    ax.imshow(rgb)

    # Mark observer position
    obs_row = h // 2
    obs_col = w // 2
    ax.plot(obs_col, obs_row, "w*", markersize=15, markeredgecolor="black")
    ax.annotate(
        f"Observer\n{summit_elev:.0f}m",
        xy=(obs_col, obs_row),
        xytext=(obs_col + w * 0.08, obs_row - h * 0.08),
        fontsize=10,
        fontweight="bold",
        color="white",
        arrowprops=dict(arrowstyle="->", color="white", lw=2),
    )

    # Legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor=(50 / 255, 180 / 255, 50 / 255), label="Visible"),
        Patch(facecolor=(200 / 255, 60 / 255, 60 / 255), label="Hidden"),
        Patch(facecolor=(180 / 255, 180 / 255, 180 / 255), label="Outside radius"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=11)

    ax.set_title(
        f"Cadillac Mountain Viewshed\n"
        f"Radius: {RADIUS_M / 1000:.0f} km | "
        f"Visible: {viewshed['visible_percentage']:.1f}% | "
        f"Source: {SOURCE}",
        fontsize=14,
        fontweight="bold",
    )
    ax.axis("off")

    fig.tight_layout()
    output_path = OUTPUT_DIR / "cadillac_viewshed.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("  Location: Cadillac Mountain, Acadia National Park")
    print(f"  Summit: {summit_elev:.0f}m")
    print(f"  Radius: {RADIUS_M / 1000:.0f} km")
    print(f"  Visible: {viewshed['visible_percentage']:.1f}%")
    print(f"  Grid: {viewshed['shape'][0]}x{viewshed['shape'][1]} pixels")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
