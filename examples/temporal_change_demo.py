#!/usr/bin/env python3
"""
Temporal Elevation Change Demo -- chuk-mcp-dem

Compares Copernicus GLO-90 and GLO-30 elevation data for Mount St. Helens
to demonstrate the elevation change detection pipeline. Shows volume
gained/lost and significant change regions arising from resolution and
processing differences between the two datasets.

For a true temporal comparison, substitute before_source="srtm" (year 2000)
once SRTM tiles are available in your environment.

Usage:
    python examples/temporal_change_demo.py

Output:
    examples/output/st_helens_change.png

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

BBOX = [-122.22, 46.17, -122.14, 46.23]  # Mount St. Helens, WA
BEFORE_SOURCE = "cop90"  # Copernicus GLO-90 (90m)
AFTER_SOURCE = "cop30"  # Copernicus GLO-30 (30m)
SIGNIFICANCE_M = 1.0
OUTPUT_DIR = Path(__file__).parent / "output"

SOURCE_LABELS = {
    "cop90": "Copernicus GLO-90 (90m)",
    "cop30": "Copernicus GLO-30 (30m)",
    "srtm": "SRTM (~2000)",
}


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()
    store = runner.manager._get_store()

    before_label = SOURCE_LABELS.get(BEFORE_SOURCE, BEFORE_SOURCE)
    after_label = SOURCE_LABELS.get(AFTER_SOURCE, AFTER_SOURCE)

    print("=" * 60)
    print("Mount St. Helens -- Elevation Change Detection")
    print("=" * 60)
    print(f"\nbbox: {BBOX}")
    print(f"Before: {before_label}")
    print(f"After:  {after_label}")

    # Step 1: Fetch before-epoch elevation
    print(f"\nStep 1: Fetching {before_label} elevation (before)...")
    before_fetch = await runner.run("dem_fetch", bbox=BBOX, source=BEFORE_SOURCE)
    if "error" in before_fetch:
        print(f"  ERROR: {before_fetch['error']}")
        sys.exit(1)
    print(f"  Shape: {before_fetch['shape']}")
    before_min, before_max = before_fetch["elevation_range"]
    print(f"  Elevation: {before_min:.0f}m to {before_max:.0f}m")

    before_data = await store.retrieve(before_fetch["artifact_ref"])
    with rasterio.open(io.BytesIO(before_data)) as src:
        before_elev = src.read(1)

    # Step 2: Fetch after-epoch elevation
    print(f"\nStep 2: Fetching {after_label} elevation (after)...")
    after_fetch = await runner.run("dem_fetch", bbox=BBOX, source=AFTER_SOURCE)
    if "error" in after_fetch:
        print(f"  ERROR: {after_fetch['error']}")
        sys.exit(1)
    print(f"  Shape: {after_fetch['shape']}")
    after_min, after_max = after_fetch["elevation_range"]
    print(f"  Elevation: {after_min:.0f}m to {after_max:.0f}m")

    after_data = await store.retrieve(after_fetch["artifact_ref"])
    with rasterio.open(io.BytesIO(after_data)) as src:
        after_elev = src.read(1)

    # Step 3: Compute hillshade of after-epoch for context
    print(f"\nStep 3: Computing hillshade ({after_label})...")
    hs_result = await runner.run("dem_hillshade", bbox=BBOX, source=AFTER_SOURCE)
    if "error" in hs_result:
        print(f"  ERROR: {hs_result['error']}")
        sys.exit(1)

    hs_data = await store.retrieve(hs_result["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_data)) as src:
        hillshade = src.read(1)

    # Step 4: Compare temporal change
    print(f"\nStep 4: Computing elevation change (threshold={SIGNIFICANCE_M}m)...")
    change_result = await runner.run(
        "dem_compare_temporal",
        bbox=BBOX,
        before_source=BEFORE_SOURCE,
        after_source=AFTER_SOURCE,
        significance_threshold_m=SIGNIFICANCE_M,
    )
    if "error" in change_result:
        print(f"  ERROR: {change_result['error']}")
        sys.exit(1)

    print(f"  Artifact: {change_result['artifact_ref']}")
    print(f"  Volume gained: {change_result['volume_gained_m3']:,.0f} m3")
    print(f"  Volume lost:   {change_result['volume_lost_m3']:,.0f} m3")
    n_regions = len(change_result.get("significant_regions", []))
    print(f"  Significant regions: {n_regions}")
    if change_result.get("significant_regions"):
        print("  Top change regions:")
        for i, r in enumerate(change_result["significant_regions"][:5]):
            print(
                f"    #{i + 1}: {r['change_type']}, "
                f"mean={r['mean_change_m']:+.1f}m, "
                f"max={r['max_change_m']:+.1f}m, "
                f"area={r['area_m2']:,.0f}m2"
            )

    change_data = await store.retrieve(change_result["artifact_ref"])
    with rasterio.open(io.BytesIO(change_data)) as src:
        change_map = src.read(1)

    # Step 5: Render 2x2 panel
    print("\nStep 5: Rendering elevation change panel...")

    # Shared elevation range for consistent coloring
    shared_min = min(before_min, after_min)
    shared_max = max(before_max, after_max)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # Before elevation (top-left)
    ax = axes[0, 0]
    im = ax.imshow(before_elev, cmap="terrain", vmin=shared_min, vmax=shared_max)
    ax.set_title(f"{before_label}", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)

    # After elevation (top-right)
    ax = axes[0, 1]
    im = ax.imshow(after_elev, cmap="terrain", vmin=shared_min, vmax=shared_max)
    ax.set_title(f"{after_label}", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Elevation (m)", shrink=0.7)

    # Change map (bottom-left)
    ax = axes[1, 0]
    abs_max = max(abs(float(np.nanmin(change_map))), abs(float(np.nanmax(change_map))), 1.0)
    im = ax.imshow(change_map, cmap="RdBu", vmin=-abs_max, vmax=abs_max)
    ax.set_title(
        f"Elevation Change (blue=loss, red=gain)\nthreshold={SIGNIFICANCE_M}m",
        fontsize=13,
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Change (m)", shrink=0.7)

    # Hillshade + significant change overlay (bottom-right)
    ax = axes[1, 1]
    ax.imshow(hillshade, cmap="gray", vmin=0, vmax=255, alpha=0.7)
    sig_mask = np.abs(change_map) > SIGNIFICANCE_M
    sig_overlay = np.ma.masked_where(~sig_mask, change_map)
    im = ax.imshow(sig_overlay, cmap="RdBu", vmin=-abs_max, vmax=abs_max, alpha=0.8)
    n_sig_pixels = int(sig_mask.sum())
    ax.set_title(
        f"Significant Changes on Hillshade\n({n_sig_pixels:,} pixels > {SIGNIFICANCE_M}m)",
        fontsize=13,
    )
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Change (m)", shrink=0.7)

    fig.suptitle(
        f"Mount St. Helens -- Elevation Change Detection\n"
        f"{before_label} vs {after_label} | bbox: {BBOX}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "st_helens_change.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Before: {before_label} ({before_elev.shape[0]}x{before_elev.shape[1]} px)")
    print(f"  After:  {after_label} ({after_elev.shape[0]}x{after_elev.shape[1]} px)")
    print(f"  Volume gained: {change_result['volume_gained_m3']:,.0f} m3")
    print(f"  Volume lost:   {change_result['volume_lost_m3']:,.0f} m3")
    net = change_result["volume_gained_m3"] - change_result["volume_lost_m3"]
    print(f"  Net change:    {net:+,.0f} m3")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
