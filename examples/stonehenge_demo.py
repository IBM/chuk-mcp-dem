#!/usr/bin/env python3
"""
Stonehenge Archaeological Anomaly Detection Demo -- chuk-mcp-dem

Uses Isolation Forest and CNN-inspired feature detection on flat chalk
downland around Stonehenge. Total relief is ~40m. The algorithm has to
find barrow mounds and henge ditches that modify terrain by just a few
metres -- proving the approach works on subtle, low-relief archaeology.

Key technique: low sun angle hillshade (altitude=15 degrees) is how
archaeologists actually use hillshade -- long shadows reveal subtle
earthworks invisible at standard illumination angles.

Usage:
    python examples/stonehenge_demo.py

Output:
    examples/output/stonehenge_anomalies.png

Requirements:
    pip install chuk-mcp-dem[ml] matplotlib
    (Requires scikit-learn and network access to Copernicus DEM S3 buckets)
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

# Covers Stonehenge monument, The Avenue, Cursus, and surrounding barrow
# cemeteries on Salisbury Plain chalk downland.
BBOX = [-1.87, 51.16, -1.77, 51.20]
SOURCE = "cop30"
SENSITIVITY = 0.05  # Low -- we want fewer, higher-confidence anomalies
HILLSHADE_ALTITUDE = 15.0  # Archaeological low sun angle (degrees)
OUTPUT_DIR = Path(__file__).parent / "output"

FEATURE_CLASSES = ["none", "peak", "ridge", "valley", "cliff", "saddle", "channel"]
FEATURE_COLORS = [
    "#e0e0e0",  # none
    "#d73027",  # peak
    "#8c510a",  # ridge
    "#1a9850",  # valley
    "#f46d43",  # cliff
    "#fee08b",  # saddle
    "#4575b4",  # channel
]


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()
    store = runner.manager._get_store()

    print("=" * 70)
    print("Stonehenge Landscape -- Archaeological Anomaly Detection")
    print("=" * 70)
    print()
    print("Site: Salisbury Plain, Wiltshire, UK")
    print("Terrain: Chalk downland, ~40m total relief")
    print("Challenge: Find subtle earthworks on nearly flat terrain at 30m")
    print(f"  bbox: {BBOX}")
    print(f"  source: {SOURCE} (Copernicus GLO-30)")

    # Step 1: Fetch elevation data
    print("\n--- Step 1: Fetching elevation data ---")

    fetch = await runner.run("dem_fetch", bbox=BBOX, source=SOURCE)
    if "error" in fetch:
        print(f"  ERROR: {fetch['error']}")
        sys.exit(1)
    print(f"  Shape: {fetch['shape']}")
    elev_min, elev_max = fetch["elevation_range"]
    total_relief = elev_max - elev_min
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print(f"  Total relief: {total_relief:.0f}m -- this is nearly flat terrain")

    elev_data = await store.retrieve(fetch["artifact_ref"])
    with rasterio.open(io.BytesIO(elev_data)) as src:
        elevation = src.read(1)

    # Step 2: Low-angle archaeological hillshade
    print(f"\n--- Step 2: Computing low-angle hillshade (altitude={HILLSHADE_ALTITUDE}) ---")
    print("  Low sun angle is how archaeologists reveal subtle earthworks.")
    print("  Long shadows make features visible that disappear at 45 degrees.")

    hs_low = await runner.run(
        "dem_hillshade", bbox=BBOX, source=SOURCE, altitude=HILLSHADE_ALTITUDE
    )
    if "error" in hs_low:
        print(f"  ERROR: {hs_low['error']}")
        sys.exit(1)
    hs_low_data = await store.retrieve(hs_low["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_low_data)) as src:
        hillshade_low = src.read(1)

    # Standard hillshade for comparison
    hs_std = await runner.run("dem_hillshade", bbox=BBOX, source=SOURCE)
    hs_std_data = await store.retrieve(hs_std["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_std_data)) as src:
        hillshade_std = src.read(1)

    # Step 3: Compute slope
    print("\n--- Step 3: Computing slope ---")
    slope_result = await runner.run("dem_slope", bbox=BBOX, source=SOURCE, units="degrees")
    if "error" in slope_result:
        print(f"  ERROR: {slope_result['error']}")
        sys.exit(1)
    sl_data = await store.retrieve(slope_result["artifact_ref"])
    with rasterio.open(io.BytesIO(sl_data)) as src:
        slope = src.read(1)
    print(f"  Max slope: {float(np.nanmax(slope)):.1f} degrees")
    print(f"  Mean slope: {float(np.nanmean(slope)):.2f} degrees -- confirming flat terrain")

    # Step 4: Detect anomalies
    print(f"\n--- Step 4: Detecting anomalies (Isolation Forest, sensitivity={SENSITIVITY}) ---")
    print("  We did not tell the algorithm what to look for.")
    print("  It receives only raw terrain data -- slope, curvature, TRI, roughness.")
    print("  No labels, no training data, no archaeological priors.")

    anomaly_result = await runner.run(
        "dem_detect_anomalies", bbox=BBOX, source=SOURCE, sensitivity=SENSITIVITY
    )
    if "error" in anomaly_result:
        print(f"  ERROR: {anomaly_result['error']}")
        if "scikit-learn" in anomaly_result["error"].lower():
            print("\n  Install scikit-learn: pip install chuk-mcp-dem[ml]")
        sys.exit(1)

    print(f"  Anomalies detected: {anomaly_result['anomaly_count']}")
    if anomaly_result.get("anomalies"):
        print("  Top anomaly regions:")
        for i, a in enumerate(anomaly_result["anomalies"][:5]):
            print(
                f"    #{i + 1}: confidence={a['confidence']:.2f}, "
                f"area={a['area_m2']:.0f}m2, score={a['mean_anomaly_score']:.3f}"
            )

    anom_data = await store.retrieve(anomaly_result["artifact_ref"])
    with rasterio.open(io.BytesIO(anom_data)) as src:
        anomaly_scores = src.read(1)

    # Step 5: Feature detection
    print("\n--- Step 5: CNN-inspired feature detection ---")
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
    print("\n--- Step 6: Rendering archaeological analysis panel ---")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # Low-angle hillshade (top-left)
    ax = axes[0, 0]
    ax.imshow(hillshade_low, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Archaeological Hillshade (altitude={HILLSHADE_ALTITUDE}\u00b0)", fontsize=12)
    ax.axis("off")

    # Standard hillshade for comparison (top-center)
    ax = axes[0, 1]
    ax.imshow(hillshade_std, cmap="gray", vmin=0, vmax=255)
    ax.set_title("Standard Hillshade (altitude=45\u00b0)", fontsize=12)
    ax.axis("off")

    # Slope (top-right)
    ax = axes[0, 2]
    slope_display = np.clip(slope, 0, 10)  # Narrow range for flat terrain
    im = ax.imshow(slope_display, cmap="RdYlGn_r", vmin=0, vmax=10)
    ax.set_title("Slope (degrees)", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Slope (deg)", shrink=0.7)

    # Anomaly scores (bottom-left)
    ax = axes[1, 0]
    im = ax.imshow(anomaly_scores, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(
        f"Anomaly Scores ({anomaly_result['anomaly_count']} regions, sensitivity={SENSITIVITY})",
        fontsize=12,
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Anomaly score", shrink=0.7)

    # Hillshade + anomaly overlay (bottom-center)
    ax = axes[1, 1]
    ax.imshow(hillshade_low, cmap="gray", vmin=0, vmax=255, alpha=0.7)
    anomaly_mask = anomaly_scores > 0.5
    overlay = np.ma.masked_where(~anomaly_mask, anomaly_scores)
    im = ax.imshow(overlay, cmap="Reds", vmin=0.5, vmax=1.0, alpha=0.8)
    n_flagged = int(anomaly_mask.sum())
    ax.set_title(f"Anomaly Overlay on Low-Angle Hillshade ({n_flagged} px)", fontsize=12)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Anomaly score", shrink=0.7)

    # Feature detection (bottom-right)
    ax = axes[1, 2]
    if feature_map is not None:
        feat_cmap = ListedColormap(FEATURE_COLORS)
        ax.imshow(feature_map, cmap=feat_cmap, vmin=-0.5, vmax=6.5, interpolation="nearest")
        ax.set_title(f"Feature Detection ({feature_result['feature_count']} regions)", fontsize=12)
        legend_patches = [
            Patch(facecolor=FEATURE_COLORS[i], label=FEATURE_CLASSES[i])
            for i in range(len(FEATURE_CLASSES))
            if i == 0 or FEATURE_CLASSES[i] in feature_result.get("feature_summary", {})
        ]
        ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)
    else:
        ax.text(
            0.5,
            0.5,
            "Feature detection\nfailed",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="red",
        )
        ax.set_title("Feature Detection (error)", fontsize=12)
    ax.axis("off")

    fig.suptitle(
        "Stonehenge Landscape -- Archaeological Anomaly Detection\n"
        f"Isolation Forest + CNN-Inspired Features | Copernicus GLO-30 | "
        f"bbox: {BBOX}",
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "stonehenge_anomalies.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary with narrative
    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("  Site:    Stonehenge landscape, Salisbury Plain")
    print(f"  Terrain: Chalk downland, {total_relief:.0f}m total relief")
    print(f"  Grid:    {elevation.shape[0]}x{elevation.shape[1]} pixels at 30m")
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print(f"  Mean slope: {float(np.nanmean(slope)):.2f} degrees")
    print()
    print("  We didn't tell the AI what to look for.")
    print("  It received only raw elevation data -- no labels, no training,")
    print(f"  no archaeological context. On terrain with just {total_relief:.0f}m of relief,")
    print(f"  Isolation Forest found {anomaly_result['anomaly_count']} anomaly regions where the")
    print("  terrain deviates from the surrounding chalk downland pattern.")
    if feature_map is not None:
        print(
            f"  Feature detection identified {feature_result['feature_count']} "
            "geomorphological features."
        )
    print()
    print("  These anomalies coincide with one of the densest archaeological")
    print("  landscapes in Europe -- barrow cemeteries, henge monuments, and")
    print("  ancient earthworks that modify the terrain by just a few metres.")
    print()
    print(f"Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
