#!/usr/bin/env python3
"""
CNN-Inspired Feature Detection Demo -- chuk-mcp-dem

Uses multi-angle hillshade with Sobel edge filters to detect geomorphological
features in the Grand Canyon: peaks, ridges, valleys, cliffs, saddles, channels.

Usage:
    python examples/feature_detection_demo.py

Output:
    examples/output/grand_canyon_features.png

Requirements:
    pip install chuk-mcp-dem matplotlib
    (Requires scipy and network access to Copernicus DEM S3 buckets)
"""

import asyncio
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

BBOX = [-112.18, 36.04, -112.08, 36.12]  # Grand Canyon, AZ
SOURCE = "cop30"
OUTPUT_DIR = Path(__file__).parent / "output"

FEATURE_CLASSES = ["none", "peak", "ridge", "valley", "cliff", "saddle", "channel"]
FEATURE_COLORS = [
    "#e0e0e0",  # none - light gray
    "#d73027",  # peak - red
    "#8c510a",  # ridge - brown
    "#1a9850",  # valley - green
    "#f46d43",  # cliff - orange
    "#fee08b",  # saddle - gold
    "#4575b4",  # channel - blue
]


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()
    store = runner.manager._get_store()

    print("=" * 60)
    print("Grand Canyon -- CNN-Inspired Feature Detection")
    print("=" * 60)

    # Step 1: Fetch elevation data
    print("\nStep 1: Fetching elevation data...")
    print(f"  bbox: {BBOX}")
    print(f"  source: {SOURCE}")

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

    # Step 2: Compute hillshade
    print("\nStep 2: Computing hillshade...")
    hs_result = await runner.run("dem_hillshade", bbox=BBOX, source=SOURCE)
    if "error" in hs_result:
        print(f"  ERROR: {hs_result['error']}")
        sys.exit(1)

    hs_data = await store.retrieve(hs_result["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_data)) as src:
        hillshade = src.read(1)

    # Step 3: Compute slope
    print("\nStep 3: Computing slope...")
    slope_result = await runner.run("dem_slope", bbox=BBOX, source=SOURCE, units="degrees")
    if "error" in slope_result:
        print(f"  ERROR: {slope_result['error']}")
        sys.exit(1)

    sl_data = await store.retrieve(slope_result["artifact_ref"])
    with rasterio.open(io.BytesIO(sl_data)) as src:
        slope = src.read(1)

    # Step 4: Detect features
    print("\nStep 4: Detecting features (CNN-inspired multi-angle hillshade)...")
    feature_result = await runner.run(
        "dem_detect_features", bbox=BBOX, source=SOURCE, method="cnn_hillshade"
    )
    if "error" in feature_result:
        print(f"  ERROR: {feature_result['error']}")
        sys.exit(1)

    print(f"  Artifact: {feature_result['artifact_ref']}")
    print(f"  Features detected: {feature_result['feature_count']}")
    if feature_result.get("feature_summary"):
        print("  Feature summary:")
        for ftype, count in sorted(feature_result["feature_summary"].items()):
            print(f"    {ftype}: {count}")

    if feature_result.get("features"):
        print("  Top feature regions:")
        for i, f in enumerate(feature_result["features"][:5]):
            print(
                f"    #{i + 1}: type={f['feature_type']}, "
                f"confidence={f['confidence']:.2f}, area={f['area_m2']:.0f}m2"
            )

    feat_data = await store.retrieve(feature_result["artifact_ref"])
    with rasterio.open(io.BytesIO(feat_data)) as src:
        feature_map = src.read(1)

    # Step 5: Render 2x2 panel
    print("\nStep 5: Rendering feature detection panel...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Hillshade (top-left)
    ax = axes[0, 0]
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Hillshade", fontsize=13)
    ax.axis("off")

    # Slope (top-right)
    ax = axes[0, 1]
    slope_display = np.clip(slope, 0, 60)
    im = ax.imshow(slope_display, cmap="RdYlGn_r", vmin=0, vmax=60)
    ax.set_title("Slope (degrees)", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Slope (deg)", shrink=0.7)

    # Feature map (bottom-left)
    ax = axes[1, 0]
    cmap = ListedColormap(FEATURE_COLORS)
    im = ax.imshow(feature_map, cmap=cmap, vmin=-0.5, vmax=6.5, interpolation="nearest")
    ax.set_title(
        f"Feature Map ({feature_result['feature_count']} regions)", fontsize=13
    )
    ax.axis("off")
    legend_patches = [
        Patch(facecolor=FEATURE_COLORS[i], label=FEATURE_CLASSES[i])
        for i in range(len(FEATURE_CLASSES))
        if i == 0 or FEATURE_CLASSES[i] in feature_result.get("feature_summary", {})
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8, framealpha=0.8)

    # Hillshade + feature overlay (bottom-right)
    ax = axes[1, 1]
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=255, alpha=0.7)
    feature_masked = np.ma.masked_where(feature_map < 0.5, feature_map)
    ax.imshow(
        feature_masked, cmap=cmap, vmin=-0.5, vmax=6.5,
        interpolation="nearest", alpha=0.6,
    )
    n_classified = int(np.sum(feature_map > 0.5))
    ax.set_title(
        f"Feature Overlay ({n_classified} classified pixels)", fontsize=13
    )
    ax.axis("off")

    fig.suptitle(
        f"Grand Canyon -- CNN-Inspired Feature Detection\n"
        f"Multi-Angle Hillshade + Sobel Filters | Copernicus GLO-30 | bbox: {BBOX}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "grand_canyon_features.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Source: {SOURCE} (Copernicus GLO-30)")
    print(f"  Shape:  {elevation.shape[0]}x{elevation.shape[1]} pixels")
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print(f"  Features detected: {feature_result['feature_count']}")
    if feature_result.get("feature_summary"):
        for ftype, count in sorted(feature_result["feature_summary"].items()):
            print(f"    {ftype}: {count}")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
