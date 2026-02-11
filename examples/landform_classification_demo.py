#!/usr/bin/env python3
"""
Landform Classification Demo -- chuk-mcp-dem

Classifies terrain around the Matterhorn (Swiss Alps) into landform types
using rule-based analysis of slope, curvature, and terrain ruggedness.

Landform classes: plain, ridge, valley, plateau, escarpment, depression,
saddle, terrace, alluvial_fan.

Usage:
    python examples/landform_classification_demo.py

Output:
    examples/output/matterhorn_landforms.png

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
from matplotlib.colors import ListedColormap

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

BBOX = [7.62, 45.96, 7.70, 46.02]  # Matterhorn, Switzerland
SOURCE = "cop30"
OUTPUT_DIR = Path(__file__).parent / "output"

LANDFORM_CLASSES = [
    "plain",
    "ridge",
    "valley",
    "plateau",
    "escarpment",
    "depression",
    "saddle",
    "terrace",
    "alluvial_fan",
]

LANDFORM_COLORS = [
    "#a6d96a",  # plain - green
    "#d73027",  # ridge - red
    "#4575b4",  # valley - blue
    "#fee08b",  # plateau - yellow
    "#8c510a",  # escarpment - brown
    "#542788",  # depression - purple
    "#f46d43",  # saddle - orange
    "#66c2a5",  # terrace - teal
    "#e0e0e0",  # alluvial_fan - light gray
]


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()
    store = runner.manager._get_store()

    print("=" * 60)
    print("Matterhorn -- Landform Classification")
    print("=" * 60)

    # Step 1: Check coverage
    print("\nStep 1: Checking coverage...")
    print(f"  bbox: {BBOX}")
    print(f"  source: {SOURCE}")

    coverage = await runner.run("dem_check_coverage", bbox=BBOX, source=SOURCE)
    print(f"  Coverage: {coverage['coverage_percentage']}%")
    print(f"  Tiles required: {coverage['tiles_required']}")

    if not coverage["fully_covered"]:
        print("  WARNING: Area not fully covered!")

    # Step 2: Fetch elevation data
    print("\nStep 2: Fetching elevation data...")
    fetch = await runner.run("dem_fetch", bbox=BBOX, source=SOURCE)
    if "error" in fetch:
        print(f"  ERROR: {fetch['error']}")
        sys.exit(1)
    print(f"  Shape: {fetch['shape']}")
    elev_min, elev_max = fetch["elevation_range"]
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")

    elev_data = await store.retrieve(fetch["artifact_ref"])
    with rasterio.open(io.BytesIO(elev_data)) as src:
        elevation = src.read(1)

    # Step 3: Compute hillshade
    print("\nStep 3: Computing hillshade...")
    hs_result = await runner.run(
        "dem_hillshade", bbox=BBOX, source=SOURCE, azimuth=315.0, altitude=45.0
    )
    if "error" in hs_result:
        print(f"  ERROR: {hs_result['error']}")
        sys.exit(1)

    hs_data = await store.retrieve(hs_result["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_data)) as src:
        hillshade = src.read(1)

    # Step 4: Classify landforms
    print("\nStep 4: Classifying landforms...")
    landform_result = await runner.run(
        "dem_classify_landforms", bbox=BBOX, source=SOURCE, method="rule_based"
    )
    if "error" in landform_result:
        print(f"  ERROR: {landform_result['error']}")
        sys.exit(1)
    print(f"  Artifact: {landform_result['artifact_ref']}")
    print(f"  Dominant landform: {landform_result['dominant_landform']}")
    print("  Class distribution:")
    for cls, pct in sorted(
        landform_result["class_distribution"].items(), key=lambda x: -x[1]
    ):
        if pct > 0:
            print(f"    {cls}: {pct:.1f}%")

    lf_data = await store.retrieve(landform_result["artifact_ref"])
    with rasterio.open(io.BytesIO(lf_data)) as src:
        landforms = src.read(1)

    # Step 5: Render 2x2 panel
    print("\nStep 5: Rendering landform analysis panel...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Elevation (top-left)
    ax = axes[0, 0]
    im = ax.imshow(elevation, cmap="terrain", vmin=elev_min, vmax=elev_max)
    ax.set_title("Elevation", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)

    # Hillshade (top-right)
    ax = axes[0, 1]
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Hillshade (az=315, alt=45)", fontsize=13)
    ax.axis("off")

    # Landform classification (bottom-left)
    ax = axes[1, 0]
    cmap = ListedColormap(LANDFORM_COLORS)
    im = ax.imshow(landforms, cmap=cmap, vmin=-0.5, vmax=8.5, interpolation="nearest")
    ax.set_title("Landform Classification", fontsize=13)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, ticks=range(9), shrink=0.7)
    cbar.ax.set_yticklabels(LANDFORM_CLASSES, fontsize=8)

    # Class distribution bar chart (bottom-right)
    ax = axes[1, 1]
    dist = landform_result["class_distribution"]
    classes = [c for c in LANDFORM_CLASSES if dist.get(c, 0) > 0]
    percentages = [dist[c] for c in classes]
    colors = [LANDFORM_COLORS[LANDFORM_CLASSES.index(c)] for c in classes]
    bars = ax.barh(classes, percentages, color=colors, edgecolor="gray", linewidth=0.5)
    ax.set_xlabel("Coverage (%)", fontsize=11)
    ax.set_title("Class Distribution", fontsize=13)
    for bar, pct in zip(bars, percentages):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(0, max(percentages) * 1.2 if percentages else 100)

    fig.suptitle(
        f"Matterhorn -- Landform Classification\nCopernicus GLO-30 | bbox: {BBOX}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "matterhorn_landforms.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Source: {SOURCE} (Copernicus GLO-30)")
    print(f"  Shape:  {elevation.shape[0]}x{elevation.shape[1]} pixels")
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print(f"  Dominant landform: {landform_result['dominant_landform']}")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
