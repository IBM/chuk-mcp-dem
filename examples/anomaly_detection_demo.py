#!/usr/bin/env python3
"""
Terrain Anomaly Detection Demo -- chuk-mcp-dem

Uses Isolation Forest (scikit-learn) to detect terrain anomalies near
Hoover Dam. The algorithm identifies outlier terrain based on slope,
curvature, and terrain ruggedness index feature vectors.

Usage:
    python examples/anomaly_detection_demo.py

Output:
    examples/output/hoover_dam_anomalies.png

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

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

BBOX = [-114.76, 36.00, -114.70, 36.04]  # Hoover Dam, NV/AZ
SOURCE = "cop30"
SENSITIVITY = 0.1  # Isolation Forest contamination parameter
OUTPUT_DIR = Path(__file__).parent / "output"


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()
    store = runner.manager._get_store()

    print("=" * 60)
    print("Hoover Dam -- Terrain Anomaly Detection")
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

    # Step 4: Detect anomalies
    print(f"\nStep 4: Detecting anomalies (sensitivity={SENSITIVITY})...")
    anomaly_result = await runner.run(
        "dem_detect_anomalies", bbox=BBOX, source=SOURCE, sensitivity=SENSITIVITY
    )
    if "error" in anomaly_result:
        print(f"  ERROR: {anomaly_result['error']}")
        if "scikit-learn" in anomaly_result["error"].lower():
            print("\n  Install scikit-learn: pip install chuk-mcp-dem[ml]")
        sys.exit(1)

    print(f"  Artifact: {anomaly_result['artifact_ref']}")
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

    # Step 5: Render 2x2 panel
    print("\nStep 5: Rendering anomaly analysis panel...")

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

    # Anomaly scores (bottom-left)
    ax = axes[1, 0]
    im = ax.imshow(anomaly_scores, cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_title(f"Anomaly Scores (sensitivity={SENSITIVITY})", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Anomaly score", shrink=0.7)

    # Hillshade + anomaly overlay (bottom-right)
    ax = axes[1, 1]
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=255, alpha=0.7)
    anomaly_mask = anomaly_scores > 0.5
    overlay = np.ma.masked_where(~anomaly_mask, anomaly_scores)
    im = ax.imshow(overlay, cmap="Reds", vmin=0.5, vmax=1.0, alpha=0.8)
    ax.set_title(f"Anomaly Overlay (score > 0.5, n={int(anomaly_mask.sum())} px)", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Anomaly score", shrink=0.7)

    fig.suptitle(
        f"Hoover Dam -- Terrain Anomaly Detection\n"
        f"Isolation Forest | Copernicus GLO-30 | bbox: {BBOX}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "hoover_dam_anomalies.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Source: {SOURCE} (Copernicus GLO-30)")
    print(f"  Shape:  {elevation.shape[0]}x{elevation.shape[1]} pixels")
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print(f"  Anomalies detected: {anomaly_result['anomaly_count']}")
    print(f"  Sensitivity: {SENSITIVITY}")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
