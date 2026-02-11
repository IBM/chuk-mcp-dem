#!/usr/bin/env python3
"""
Colchester (Camulodunum) -- Archaeological Anomaly Detection

Britain's oldest recorded Roman town. The Romans chose this hilltop
above the River Colne because the terrain gave them a natural defensive
position -- a ridge with river valleys on two sides. Every modification
they made (walls, roads, the circus, temples, cemeteries) changed the
terrain in ways that should still be detectable.

At 30m this captures landscape-scale features: the hilltop the Romans
fortified, the river valleys they used as defences, the road alignments
radiating outward. The anomaly detector should flag the town centre
(most modified terrain) and the river crossings (engineered approaches).

Known features in the bbox:
  - Roman wall circuit (oldest in Britain, largely intact)
  - Colchester Castle (Norman, on Roman temple platform)
  - Roman circus (chariot racing, only one in Britain)
  - Balkerne Gate (largest surviving Roman gateway in Britain)
  - River Colne crossing points (Roman bridge sites)
  - Lexden tumulus (Iron Age/early Roman burial mound)
  - Roman roads: Stane Street to London, road to Mersea
  - Gosbecks archaeological park (Roman theatre + temple)
  - Stanway elite burial complex
  - Sheepen industrial site (pre-Roman oppidum)

Environment Agency LiDAR at 1m resolution is available for this area
and would reveal individual wall lines, road surfaces, and earthworks.

Usage:
    python examples/colchester_demo.py

Output:
    examples/output/colchester_anomalies.png

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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

# Centred on the Roman town. West to Stanway burial complex, south to
# Gosbecks archaeological park (Roman theatre + temple), north across
# the Colne valley, east towards the Hythe port area.
BBOX = [0.83, 51.85, 0.95, 51.92]
SOURCE = "cop30"
SENSITIVITY = 0.1
HILLSHADE_ALTITUDE_LOW = 15.0  # Archaeological low sun angle
HILLSHADE_ALTITUDE_STD = 45.0  # Standard for comparison
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
    print("Colchester (Camulodunum) -- Archaeological Anomaly Detection")
    print("=" * 70)
    print()
    print("Site: Colchester, Essex, UK")
    print("Terrain: River valley and hilltop, gently undulating")
    print("Challenge: Detect Roman town modifications on natural terrain at 30m")
    print()
    print(f"  bbox: {BBOX}")
    print(f"  source: {SOURCE} (Copernicus GLO-30)")
    print()

    # ----------------------------------------------------------
    # Step 1: Fetch elevation data
    # ----------------------------------------------------------
    print("--- Step 1: Fetching elevation data ---")

    fetch = await runner.run("dem_fetch", bbox=BBOX, source=SOURCE)
    if "error" in fetch:
        print(f"  ERROR: {fetch['error']}")
        sys.exit(1)
    print(f"  Shape: {fetch['shape']}")
    elev_min, elev_max = fetch["elevation_range"]
    total_relief = elev_max - elev_min
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print(f"  Total relief: {total_relief:.0f}m")
    print("  -> Low-lying terrain with gentle hills and river valleys")
    print()

    elev_data = await store.retrieve(fetch["artifact_ref"])
    with rasterio.open(io.BytesIO(elev_data)) as src:
        elevation = src.read(1)

    # ----------------------------------------------------------
    # Step 2: Archaeological hillshade (low sun angle)
    # ----------------------------------------------------------
    print(f"--- Step 2: Computing low-angle hillshade (altitude={HILLSHADE_ALTITUDE_LOW}) ---")
    print("  Low sun angle reveals subtle earthworks and terrain breaks.")

    hs_low = await runner.run(
        "dem_hillshade", bbox=BBOX, source=SOURCE, altitude=HILLSHADE_ALTITUDE_LOW
    )
    if "error" in hs_low:
        print(f"  ERROR: {hs_low['error']}")
        sys.exit(1)
    hs_low_data = await store.retrieve(hs_low["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_low_data)) as src:
        hillshade_low = src.read(1)
    print()

    # ----------------------------------------------------------
    # Step 3: Standard hillshade for comparison
    # ----------------------------------------------------------
    print(f"--- Step 3: Computing standard hillshade (altitude={HILLSHADE_ALTITUDE_STD}) ---")

    hs_std = await runner.run(
        "dem_hillshade", bbox=BBOX, source=SOURCE, altitude=HILLSHADE_ALTITUDE_STD
    )
    if "error" in hs_std:
        print(f"  ERROR: {hs_std['error']}")
        sys.exit(1)
    hs_std_data = await store.retrieve(hs_std["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_std_data)) as src:
        hillshade_std = src.read(1)
    print()

    # ----------------------------------------------------------
    # Step 4: Slope analysis
    # ----------------------------------------------------------
    print("--- Step 4: Computing slope ---")

    slope_result = await runner.run(
        "dem_slope", bbox=BBOX, source=SOURCE, units="degrees"
    )
    if "error" in slope_result:
        print(f"  ERROR: {slope_result['error']}")
        sys.exit(1)

    sl_data = await store.retrieve(slope_result["artifact_ref"])
    with rasterio.open(io.BytesIO(sl_data)) as src:
        slope = src.read(1)

    print(f"  Max slope: {float(np.nanmax(slope)):.1f} degrees")
    print(f"  Mean slope: {float(np.nanmean(slope)):.2f} degrees")
    print()

    # ----------------------------------------------------------
    # Step 5: Anomaly detection (Isolation Forest)
    # ----------------------------------------------------------
    print(f"--- Step 5: Detecting anomalies (Isolation Forest, sensitivity={SENSITIVITY}) ---")
    print("  The algorithm receives only raw terrain data.")
    print("  No labels, no training data, no archaeological priors.")
    print("  It knows nothing about Roman towns, walls, or roads.")

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
    print()

    anom_data = await store.retrieve(anomaly_result["artifact_ref"])
    with rasterio.open(io.BytesIO(anom_data)) as src:
        anomaly_scores = src.read(1)

    # ----------------------------------------------------------
    # Step 6: Feature detection (CNN-inspired)
    # ----------------------------------------------------------
    print("--- Step 6: CNN-inspired feature detection ---")

    feature_map = None
    feature_result = await runner.run(
        "dem_detect_features", bbox=BBOX, source=SOURCE, method="cnn_hillshade"
    )
    if "error" in feature_result:
        print(f"  ERROR: {feature_result['error']}")
    else:
        print(f"  Features detected: {feature_result['feature_count']}")
        if feature_result.get("feature_summary"):
            print("  Feature summary:")
            for ftype, count in sorted(feature_result["feature_summary"].items()):
                print(f"    {ftype}: {count}")
        feat_data = await store.retrieve(feature_result["artifact_ref"])
        with rasterio.open(io.BytesIO(feat_data)) as src:
            feature_map = src.read(1)
    print()

    # ----------------------------------------------------------
    # Step 7: Landform classification
    # ----------------------------------------------------------
    print("--- Step 7: Classifying landforms ---")

    landform_result = await runner.run(
        "dem_classify_landforms", bbox=BBOX, source=SOURCE, method="rule_based"
    )
    if "error" in landform_result:
        print(f"  ERROR: {landform_result['error']}")
    else:
        print(f"  Dominant landform: {landform_result['dominant_landform']}")
        print("  Class distribution:")
        dist = landform_result.get("class_distribution", {})
        for cls, pct in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            if pct > 0.1:
                print(f"    {cls}: {pct:.1f}%")
    print()

    # ----------------------------------------------------------
    # Step 8: Render combined panel
    # ----------------------------------------------------------
    print("--- Step 8: Rendering archaeological analysis panel ---")

    fig = plt.figure(figsize=(20, 13))
    fig.suptitle(
        "Colchester (Camulodunum) -- Archaeological Anomaly Detection\n"
        f"Isolation Forest + CNN-Inspired Features | Copernicus GLO-30 | bbox: {BBOX}",
        fontsize=14, fontweight="bold", y=0.98,
    )

    gs = GridSpec(
        2, 3, figure=fig, hspace=0.25, wspace=0.25,
        top=0.91, bottom=0.04, left=0.03, right=0.97,
    )

    # --- Row 1: Hillshade (low), Hillshade (standard), Slope ---

    # Archaeological hillshade
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(hillshade_low, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Archaeological Hillshade (altitude={HILLSHADE_ALTITUDE_LOW}\u00b0)", fontsize=10)
    ax.axis("off")

    # Standard hillshade
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(hillshade_std, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"Standard Hillshade (altitude={HILLSHADE_ALTITUDE_STD}\u00b0)", fontsize=10)
    ax.axis("off")

    # Slope
    ax = fig.add_subplot(gs[0, 2])
    slope_display = np.clip(slope, 0, 10)
    im = ax.imshow(slope_display, cmap="YlGn_r", vmin=0, vmax=10)
    ax.set_title("Slope (degrees)", fontsize=10)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Slope (deg)")

    # --- Row 2: Anomaly heatmap, Anomaly overlay, Feature detection ---

    # Anomaly scores
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(anomaly_scores, cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_title(
        f"Anomaly Scores ({anomaly_result['anomaly_count']} regions, "
        f"sensitivity={SENSITIVITY})",
        fontsize=10,
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Anomaly score")

    # Anomaly overlay on low-angle hillshade
    ax = fig.add_subplot(gs[1, 1])
    ax.imshow(hillshade_low, cmap="gray", vmin=0, vmax=255)
    anomaly_mask = anomaly_scores > 0.5
    overlay = np.ma.masked_where(~anomaly_mask, anomaly_scores)
    n_flagged = int(anomaly_mask.sum())
    im = ax.imshow(overlay, cmap="YlOrBr", vmin=0.5, vmax=1, alpha=0.7)
    ax.set_title(
        f"Anomaly Overlay on Low-Angle Hillshade ({n_flagged} px)", fontsize=10
    )
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Anomaly score")

    # Feature detection
    ax = fig.add_subplot(gs[1, 2])
    if feature_map is not None:
        feat_cmap = ListedColormap(FEATURE_COLORS)
        ax.imshow(
            feature_map, cmap=feat_cmap, vmin=-0.5, vmax=6.5,
            interpolation="nearest",
        )
        ax.set_title(
            f"Feature Detection ({feature_result['feature_count']} regions)", fontsize=10
        )
        unique_classes = sorted(
            set(int(v) for v in np.unique(feature_map) if v > 0)
        )
        legend_patches = [
            Patch(facecolor=FEATURE_COLORS[c], label=FEATURE_CLASSES[c])
            for c in unique_classes
            if c < len(FEATURE_CLASSES)
        ]
        if legend_patches:
            ax.legend(handles=legend_patches, loc="lower right", fontsize=7, framealpha=0.8)
    else:
        ax.text(
            0.5, 0.5, "Feature detection\nfailed",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=14, color="red",
        )
        ax.set_title("Feature Detection (error)", fontsize=10)
    ax.axis("off")

    output_path = OUTPUT_DIR / "colchester_anomalies.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
    print()

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)
    print()
    print("  Site:    Colchester (Camulodunum), Essex")
    print(f"  Terrain: River valley and hilltop, {total_relief:.0f}m total relief")
    print(f"  Grid:    {elevation.shape[0]}x{elevation.shape[1]} pixels at 30m")
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print(f"  Mean slope: {float(np.nanmean(slope)):.2f} degrees")
    print()
    print("  The AI received only raw elevation data -- no labels, no training,")
    print("  no archaeological context. It doesn't know this is a Roman town.")
    print(f"  Isolation Forest found {anomaly_result['anomaly_count']} anomaly regions")
    print("  where the terrain deviates from the surrounding natural pattern.")
    if feature_map is not None:
        print(f"  Feature detection identified {feature_result['feature_count']}"
              " geomorphological features.")
    print()
    print("  On this terrain, anomalies should correlate with:")
    print("    - The hilltop the Romans fortified (highest anomaly density)")
    print("    - River Colne crossings (engineered approaches)")
    print("    - Roman road alignments radiating from the town")
    print("    - Gosbecks archaeological park (theatre + temple complex)")
    print("    - Lexden tumulus (large Iron Age burial mound)")
    print()
    print("  AT 30m RESOLUTION:")
    print("    We see landscape-scale patterns -- the hill, the valleys,")
    print("    the terrain breaks. Individual walls and roads are subpixel.")
    print()
    print("  AT 1m LIDAR (Environment Agency, available for this area):")
    print("    Individual wall lines, road surfaces, building platforms,")
    print("    burial mounds, field boundaries, and ditch systems would")
    print("    all become visible. That's the next upgrade path.")
    print()
    print("  PERSONAL NOTE:")
    print("    This is my town. I live 2 miles from the Roman wall.")
    print("    I walk past Balkerne Gate most weeks. Running this pipeline")
    print("    on data from my own backyard -- that's the point. This")
    print("    isn't theoretical. It's real terrain I know by foot.")
    print()
    print(f"Output: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
