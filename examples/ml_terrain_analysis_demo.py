#!/usr/bin/env python3
"""
ML-Enhanced Terrain Analysis Demo -- chuk-mcp-dem

Showcases all Phase 3.0 ML-enhanced terrain analysis tools on the
Grand Canyon: landform classification, anomaly detection, temporal
change, and CNN-inspired feature detection.

Gracefully handles missing scikit-learn.

Usage:
    python examples/ml_terrain_analysis_demo.py

Output:
    examples/output/grand_canyon_ml_analysis.png

Requirements:
    pip install chuk-mcp-dem[ml] matplotlib
    (Requires network access to Copernicus DEM S3 buckets)
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
    "#a6d96a",  # plain
    "#d73027",  # ridge
    "#4575b4",  # valley
    "#fee08b",  # plateau
    "#8c510a",  # escarpment
    "#542788",  # depression
    "#f46d43",  # saddle
    "#66c2a5",  # terrace
    "#e0e0e0",  # alluvial_fan
]


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()
    store = runner.manager._get_store()

    print("=" * 60)
    print("Grand Canyon -- ML-Enhanced Terrain Analysis")
    print("=" * 60)
    print(f"\nbbox: {BBOX}")
    print(f"source: {SOURCE}")

    # Step 1: Fetch elevation + hillshade
    print("\nStep 1: Fetching elevation data...")
    fetch = await runner.run("dem_fetch", bbox=BBOX, source=SOURCE)
    if "error" in fetch:
        print(f"  ERROR: {fetch['error']}")
        sys.exit(1)
    elev_min, elev_max = fetch["elevation_range"]
    print(f"  Shape: {fetch['shape']}, Elevation: {elev_min:.0f}m to {elev_max:.0f}m")

    elev_data = await store.retrieve(fetch["artifact_ref"])
    with rasterio.open(io.BytesIO(elev_data)) as src:
        elevation = src.read(1)

    print("\n  Computing hillshade...")
    hs_result = await runner.run("dem_hillshade", bbox=BBOX, source=SOURCE)
    if "error" in hs_result:
        print(f"  ERROR: {hs_result['error']}")
        sys.exit(1)
    hs_data = await store.retrieve(hs_result["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_data)) as src:
        hillshade = src.read(1)

    # Step 2: Classify landforms
    print("\nStep 2: Classifying landforms...")
    landform_result = await runner.run(
        "dem_classify_landforms", bbox=BBOX, source=SOURCE, method="rule_based"
    )
    landforms = None
    if "error" in landform_result:
        print(f"  ERROR: {landform_result['error']}")
    else:
        print(f"  Dominant landform: {landform_result['dominant_landform']}")
        lf_data = await store.retrieve(landform_result["artifact_ref"])
        with rasterio.open(io.BytesIO(lf_data)) as src:
            landforms = src.read(1)

    # Step 3: Detect anomalies
    print("\nStep 3: Detecting anomalies (Isolation Forest)...")
    anomaly_scores = None
    anomaly_result = await runner.run(
        "dem_detect_anomalies", bbox=BBOX, source=SOURCE, sensitivity=0.1
    )
    if "error" in anomaly_result:
        print(f"  Skipped: {anomaly_result['error']}")
    else:
        print(f"  Anomalies detected: {anomaly_result['anomaly_count']}")
        anom_data = await store.retrieve(anomaly_result["artifact_ref"])
        with rasterio.open(io.BytesIO(anom_data)) as src:
            anomaly_scores = src.read(1)

    # Step 4: Temporal change
    print("\nStep 4: Computing temporal change (GLO-90 vs GLO-30)...")
    change_map = None
    change_result = await runner.run(
        "dem_compare_temporal",
        bbox=BBOX,
        before_source="cop90",
        after_source="cop30",
        significance_threshold_m=1.0,
    )
    if "error" in change_result:
        print(f"  ERROR: {change_result['error']}")
    else:
        print(f"  Volume gained: {change_result['volume_gained_m3']:,.0f} m3")
        print(f"  Volume lost:   {change_result['volume_lost_m3']:,.0f} m3")
        ch_data = await store.retrieve(change_result["artifact_ref"])
        with rasterio.open(io.BytesIO(ch_data)) as src:
            change_map = src.read(1)

    # Step 5: Feature detection (CNN-inspired multi-angle hillshade)
    print("\nStep 5: Feature detection (CNN-inspired)...")
    feature_map = None
    feature_result = await runner.run(
        "dem_detect_features", bbox=BBOX, source=SOURCE, method="cnn_hillshade"
    )
    if "error" in feature_result:
        print(f"  ERROR: {feature_result['error']}")
    else:
        print(f"  Features detected: {feature_result['feature_count']}")
        if feature_result.get("feature_summary"):
            for ftype, count in sorted(feature_result["feature_summary"].items()):
                print(f"    {ftype}: {count}")
        feat_data = await store.retrieve(feature_result["artifact_ref"])
        with rasterio.open(io.BytesIO(feat_data)) as src:
            feature_map = src.read(1)

    # Step 6: Render 2x3 panel
    print("\nStep 6: Rendering ML analysis panel...")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Elevation + hillshade blend (top-left)
    ax = axes[0, 0]
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=255, alpha=0.5)
    im = ax.imshow(elevation, cmap="terrain", vmin=elev_min, vmax=elev_max, alpha=0.6)
    ax.set_title("Elevation + Hillshade", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)

    # Landform classification (top-center)
    ax = axes[0, 1]
    if landforms is not None:
        cmap = ListedColormap(LANDFORM_COLORS)
        im = ax.imshow(landforms, cmap=cmap, vmin=-0.5, vmax=8.5, interpolation="nearest")
        ax.set_title(
            f"Landforms (dominant: {landform_result['dominant_landform']})", fontsize=13
        )
        cbar = fig.colorbar(im, ax=ax, ticks=range(9), shrink=0.7)
        cbar.ax.set_yticklabels(LANDFORM_CLASSES, fontsize=7)
    else:
        ax.text(0.5, 0.5, "Landform classification\nfailed", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red")
        ax.set_title("Landforms (error)", fontsize=13)
    ax.axis("off")

    # Anomaly scores (top-right)
    ax = axes[0, 2]
    if anomaly_scores is not None:
        im = ax.imshow(anomaly_scores, cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_title(
            f"Anomaly Scores ({anomaly_result['anomaly_count']} regions)", fontsize=13
        )
        fig.colorbar(im, ax=ax, label="Score", shrink=0.7)
    else:
        ax.text(0.5, 0.5, "Anomaly detection\nrequires scikit-learn\n\n"
                "pip install chuk-mcp-dem[ml]",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=12, color="gray",
                bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
        ax.set_title("Anomaly Detection (unavailable)", fontsize=13)
    ax.axis("off")

    # Temporal change (bottom-left)
    ax = axes[1, 0]
    if change_map is not None:
        abs_max = max(
            abs(float(np.nanmin(change_map))),
            abs(float(np.nanmax(change_map))),
            1.0,
        )
        im = ax.imshow(change_map, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
        ax.set_title("Temporal Change (GLO-90 vs GLO-30)", fontsize=13)
        fig.colorbar(im, ax=ax, label="Change (m)", shrink=0.7)
    else:
        ax.text(0.5, 0.5, "Temporal change\nfailed", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red")
        ax.set_title("Temporal Change (error)", fontsize=13)
    ax.axis("off")

    # Feature detection (bottom-center)
    feat_classes = ["none", "peak", "ridge", "valley", "cliff", "saddle", "channel"]
    feat_colors = ["#e0e0e0", "#d73027", "#8c510a", "#1a9850", "#f46d43", "#fee08b", "#4575b4"]
    ax = axes[1, 1]
    if feature_map is not None:
        feat_cmap = ListedColormap(feat_colors)
        ax.imshow(feature_map, cmap=feat_cmap, vmin=-0.5, vmax=6.5, interpolation="nearest")
        ax.set_title(
            f"Feature Detection ({feature_result['feature_count']} regions)", fontsize=13
        )
        legend_patches = [
            Patch(facecolor=feat_colors[i], label=feat_classes[i])
            for i in range(len(feat_classes))
            if i == 0 or feat_classes[i] in feature_result.get("feature_summary", {})
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)
    else:
        ax.text(0.5, 0.5, "Feature detection\nfailed", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="red")
        ax.set_title("Feature Detection (error)", fontsize=13)
    ax.axis("off")

    # Class distribution summary (bottom-right)
    ax = axes[1, 2]
    if landforms is not None and "class_distribution" in landform_result:
        dist = landform_result["class_distribution"]
        classes = [c for c in LANDFORM_CLASSES if dist.get(c, 0) > 0]
        percentages = [dist[c] for c in classes]
        colors = [LANDFORM_COLORS[LANDFORM_CLASSES.index(c)] for c in classes]
        bars = ax.barh(classes, percentages, color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_xlabel("Coverage (%)", fontsize=10)
        ax.set_title("Landform Distribution", fontsize=13)
        for bar, pct in zip(bars, percentages):
            ax.text(
                bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=8,
            )
        ax.set_xlim(0, max(percentages) * 1.2 if percentages else 100)
    else:
        ax.text(0.5, 0.5, "No landform data", transform=ax.transAxes,
                ha="center", va="center", fontsize=14, color="gray")
        ax.set_title("Landform Distribution", fontsize=13)

    fig.suptitle(
        f"Grand Canyon -- ML-Enhanced Terrain Analysis\n"
        f"Phase 3.0 Tools | Copernicus GLO-30 | bbox: {BBOX}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "grand_canyon_ml_analysis.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Source: {SOURCE} (Copernicus GLO-30)")
    print(f"  Shape:  {elevation.shape[0]}x{elevation.shape[1]} pixels")
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print("\n  Phase 3.0 Tool Results:")
    if landforms is not None:
        print(f"    Landforms: dominant={landform_result['dominant_landform']}")
    else:
        print("    Landforms: failed")
    if anomaly_scores is not None:
        print(f"    Anomalies: {anomaly_result['anomaly_count']} regions detected")
    else:
        print("    Anomalies: scikit-learn not available")
    if change_map is not None:
        net = change_result["volume_gained_m3"] - change_result["volume_lost_m3"]
        print(f"    Temporal change: net={net:+,.0f} m3")
    else:
        print("    Temporal change: failed")
    if feature_map is not None:
        print(f"    Features: {feature_result['feature_count']} regions detected")
    else:
        print("    Features: failed")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
