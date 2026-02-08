#!/usr/bin/env python3
"""
DEM Source Comparison Demo -- chuk-mcp-dem

Compares Copernicus GLO-30 (30m) and GLO-90 (90m) side-by-side
for Yosemite Valley. Shows the resolution difference in hillshade
visualisations.

Usage:
    python examples/source_comparison_demo.py

Output:
    examples/output/yosemite_comparison.png

Requirements:
    pip install chuk-mcp-dem matplotlib
    (Requires network access to Copernicus DEM S3 buckets)
"""

import asyncio
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import rasterio

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

BBOX = [-119.65, 37.71, -119.55, 37.76]  # Yosemite Valley, CA
SOURCES = ["cop30", "cop90"]
SOURCE_NAMES = {"cop30": "Copernicus GLO-30 (30m)", "cop90": "Copernicus GLO-90 (90m)"}
OUTPUT_DIR = Path(__file__).parent / "output"


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()
    store = runner.manager._get_store()

    print("=" * 60)
    print("Yosemite Valley -- DEM Source Comparison")
    print("=" * 60)
    print(f"\nbbox: {BBOX}")

    # Step 1: Compare size estimates
    print("\nStep 1: Comparing size estimates...")
    for source in SOURCES:
        size = await runner.run("dem_estimate_size", bbox=BBOX, source=source)
        dims = size["dimensions"]
        print(
            f"  {SOURCE_NAMES[source]}: {dims[0]}x{dims[1]} pixels, {size['estimated_mb']:.1f} MB"
        )

    # Step 2: Download elevation data from both sources
    print("\nStep 2: Downloading elevation data...")
    elevation_data = {}
    fetch_results = {}

    for source in SOURCES:
        print(f"\n  Fetching {SOURCE_NAMES[source]}...")
        result = await runner.run("dem_fetch", bbox=BBOX, source=source)
        if "error" in result:
            print(f"    ERROR: {result['error']}")
            sys.exit(1)

        fetch_results[source] = result
        print(f"    Artifact: {result['artifact_ref']}")
        print(f"    Shape: {result['shape']}")
        elev_min, elev_max = result["elevation_range"]
        print(f"    Elevation: {elev_min:.0f}m to {elev_max:.0f}m")

        data = await store.retrieve(result["artifact_ref"])
        with rasterio.open(io.BytesIO(data)) as src:
            elevation_data[source] = src.read(1)

    # Step 3: Compute hillshades
    print("\nStep 3: Computing hillshades...")
    hillshade_data = {}

    for source in SOURCES:
        print(f"  Computing hillshade for {SOURCE_NAMES[source]}...")
        hs = await runner.run("dem_hillshade", bbox=BBOX, source=source)
        if "error" in hs:
            print(f"    ERROR: {hs['error']}")
            sys.exit(1)

        data = await store.retrieve(hs["artifact_ref"])
        with rasterio.open(io.BytesIO(data)) as src:
            hillshade_data[source] = src.read(1)
        print(f"    Shape: {hs['shape']}")

    # Step 4: Render side-by-side comparison
    print("\nStep 4: Rendering comparison...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    for col, source in enumerate(SOURCES):
        elev = elevation_data[source]
        hs = hillshade_data[source]
        fr = fetch_results[source]
        elev_min, elev_max = fr["elevation_range"]

        # Elevation (top row)
        ax = axes[0, col]
        im = ax.imshow(elev, cmap="terrain", vmin=elev_min, vmax=elev_max)
        ax.set_title(f"Elevation -- {SOURCE_NAMES[source]}", fontsize=12)
        ax.axis("off")
        fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)

        # Hillshade (bottom row)
        ax = axes[1, col]
        ax.imshow(hs, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"Hillshade -- {SOURCE_NAMES[source]}", fontsize=12)
        ax.axis("off")

        # Add pixel count annotation
        shape = fr["shape"]
        ax.text(
            0.02,
            0.02,
            f"{shape[0]}x{shape[1]} px",
            transform=ax.transAxes,
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round", facecolor="black", alpha=0.6),
        )

    fig.suptitle(
        "Yosemite Valley -- DEM Source Comparison\nCopernicus GLO-30 (30m) vs GLO-90 (90m)",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "yosemite_comparison.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    for source in SOURCES:
        fr = fetch_results[source]
        shape = fr["shape"]
        elev_min, elev_max = fr["elevation_range"]
        print(f"  {SOURCE_NAMES[source]}: {shape[0]}x{shape[1]} px, {elev_min:.0f}-{elev_max:.0f}m")
    ratio = (
        fetch_results["cop30"]["shape"][0]
        * fetch_results["cop30"]["shape"][1]
        / max(
            1,
            fetch_results["cop90"]["shape"][0] * fetch_results["cop90"]["shape"][1],
        )
    )
    print(f"  Resolution ratio: {ratio:.1f}x more pixels in GLO-30")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
