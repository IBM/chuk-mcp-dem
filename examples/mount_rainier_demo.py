#!/usr/bin/env python3
"""
Mount Rainier Terrain Analysis -- chuk-mcp-dem Demo

Demonstrates the full DEM terrain analysis pipeline for Mount Rainier:
    dem_check_coverage -> dem_estimate_size -> dem_fetch ->
    dem_hillshade -> dem_slope -> dem_aspect -> dem_curvature ->
    dem_terrain_ruggedness -> dem_contour -> dem_watershed

Each tool call returns JSON with artifact references and metadata.
Artifacts are retrieved from the store and rendered with matplotlib.

Usage:
    python examples/mount_rainier_demo.py

Output:
    examples/output/rainier_terrain.png

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

BBOX = [-121.82, 46.82, -121.70, 46.90]  # Mount Rainier, WA
SOURCE = "cop30"
OUTPUT_DIR = Path(__file__).parent / "output"


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()

    print("=" * 60)
    print("Mount Rainier -- Terrain Analysis Pipeline")
    print("=" * 60)

    # Step 1: Check coverage
    print("\nStep 1: Checking coverage...")
    print(f"  bbox: {BBOX}")
    print(f"  source: {SOURCE}")

    coverage = await runner.run("dem_check_coverage", bbox=BBOX, source=SOURCE)
    print(f"  Coverage: {coverage['coverage_percentage']}%")
    print(f"  Tiles required: {coverage['tiles_required']}")
    print(f"  Estimated size: {coverage['estimated_size_mb']:.1f} MB")

    if not coverage["fully_covered"]:
        print("  WARNING: Area not fully covered!")

    # Step 2: Estimate size
    print("\nStep 2: Estimating download size...")
    size = await runner.run("dem_estimate_size", bbox=BBOX, source=SOURCE)
    dims = size["dimensions"]
    print(f"  Dimensions: {dims[0]}x{dims[1]} pixels")
    print(f"  Resolution: {size['target_resolution_m']}m")
    print(f"  Estimated size: {size['estimated_mb']:.1f} MB")
    if size.get("warning"):
        print(f"  WARNING: {size['warning']}")

    # Step 3: Fetch elevation data
    print("\nStep 3: Fetching elevation data...")
    fetch = await runner.run("dem_fetch", bbox=BBOX, source=SOURCE)
    if "error" in fetch:
        print(f"  ERROR: {fetch['error']}")
        sys.exit(1)
    print(f"  Artifact: {fetch['artifact_ref']}")
    print(f"  Shape: {fetch['shape']}")
    elev_min, elev_max = fetch["elevation_range"]
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    if fetch.get("preview_ref"):
        print(f"  Preview: {fetch['preview_ref']}")

    # Retrieve elevation raster from artifact store
    store = runner.manager._get_store()
    elev_data = await store.retrieve(fetch["artifact_ref"])
    with rasterio.open(io.BytesIO(elev_data)) as src:
        elevation = src.read(1)
    print(f"  Array shape: {elevation.shape}")

    # Step 4: Compute hillshade
    print("\nStep 4: Computing hillshade...")
    hillshade_result = await runner.run(
        "dem_hillshade", bbox=BBOX, source=SOURCE, azimuth=315.0, altitude=45.0
    )
    if "error" in hillshade_result:
        print(f"  ERROR: {hillshade_result['error']}")
        sys.exit(1)
    print(f"  Artifact: {hillshade_result['artifact_ref']}")
    print(f"  Value range: {hillshade_result['value_range']}")

    hs_data = await store.retrieve(hillshade_result["artifact_ref"])
    with rasterio.open(io.BytesIO(hs_data)) as src:
        hillshade = src.read(1)

    # Step 5: Compute slope
    print("\nStep 5: Computing slope...")
    slope_result = await runner.run("dem_slope", bbox=BBOX, source=SOURCE, units="degrees")
    if "error" in slope_result:
        print(f"  ERROR: {slope_result['error']}")
        sys.exit(1)
    print(f"  Artifact: {slope_result['artifact_ref']}")
    print(f"  Value range: {slope_result['value_range']}")

    sl_data = await store.retrieve(slope_result["artifact_ref"])
    with rasterio.open(io.BytesIO(sl_data)) as src:
        slope = src.read(1)

    # Step 6: Compute aspect
    print("\nStep 6: Computing aspect...")
    aspect_result = await runner.run("dem_aspect", bbox=BBOX, source=SOURCE)
    if "error" in aspect_result:
        print(f"  ERROR: {aspect_result['error']}")
        sys.exit(1)
    print(f"  Artifact: {aspect_result['artifact_ref']}")
    print(f"  Value range: {aspect_result['value_range']}")

    asp_data = await store.retrieve(aspect_result["artifact_ref"])
    with rasterio.open(io.BytesIO(asp_data)) as src:
        aspect = src.read(1)

    # Step 7: Compute curvature
    print("\nStep 7: Computing curvature...")
    curvature_result = await runner.run("dem_curvature", bbox=BBOX, source=SOURCE)
    if "error" in curvature_result:
        print(f"  ERROR: {curvature_result['error']}")
        sys.exit(1)
    print(f"  Artifact: {curvature_result['artifact_ref']}")
    print(f"  Value range: {curvature_result['value_range']}")

    curv_data = await store.retrieve(curvature_result["artifact_ref"])
    with rasterio.open(io.BytesIO(curv_data)) as src:
        curvature = src.read(1)

    # Step 8: Compute Terrain Ruggedness Index
    print("\nStep 8: Computing Terrain Ruggedness Index...")
    tri_result = await runner.run("dem_terrain_ruggedness", bbox=BBOX, source=SOURCE)
    if "error" in tri_result:
        print(f"  ERROR: {tri_result['error']}")
        sys.exit(1)
    print(f"  Artifact: {tri_result['artifact_ref']}")
    print(f"  Value range: {tri_result['value_range']}")

    tri_data = await store.retrieve(tri_result["artifact_ref"])
    with rasterio.open(io.BytesIO(tri_data)) as src:
        tri = src.read(1)

    # Step 9: Compute contours
    print("\nStep 9: Computing contours (100m interval)...")
    contour_result = await runner.run("dem_contour", bbox=BBOX, source=SOURCE, interval_m=100.0)
    if "error" in contour_result:
        print(f"  ERROR: {contour_result['error']}")
        sys.exit(1)
    print(f"  Artifact: {contour_result['artifact_ref']}")
    print(f"  Contour levels: {contour_result['contour_count']}")
    print(f"  Elevation range: {contour_result['elevation_range']}")

    contour_data = await store.retrieve(contour_result["artifact_ref"])
    with rasterio.open(io.BytesIO(contour_data)) as src:
        contours = src.read(1)

    # Step 10: Compute watershed (flow accumulation)
    print("\nStep 10: Computing watershed flow accumulation...")
    watershed_result = await runner.run("dem_watershed", bbox=BBOX, source=SOURCE)
    if "error" in watershed_result:
        print(f"  ERROR: {watershed_result['error']}")
        sys.exit(1)
    print(f"  Artifact: {watershed_result['artifact_ref']}")
    print(f"  Value range: {watershed_result['value_range']}")

    ws_data = await store.retrieve(watershed_result["artifact_ref"])
    with rasterio.open(io.BytesIO(ws_data)) as src:
        watershed = src.read(1)

    # Step 11: Render 4x2 terrain analysis panel
    print("\nStep 11: Rendering terrain analysis panel...")

    fig, axes = plt.subplots(4, 2, figsize=(16, 26))

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

    # Slope (middle-left)
    ax = axes[1, 0]
    slope_display = np.clip(slope, 0, 60)
    im = ax.imshow(slope_display, cmap="RdYlGn_r", vmin=0, vmax=60)
    ax.set_title("Slope (degrees)", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Slope (deg)", shrink=0.7)

    # Aspect (middle-right)
    ax = axes[1, 1]
    aspect_display = np.where(aspect < 0, np.nan, aspect)
    im = ax.imshow(aspect_display, cmap="hsv", vmin=0, vmax=360)
    ax.set_title("Aspect (degrees from N)", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Aspect (deg)", shrink=0.7)

    # Curvature (bottom-left)
    ax = axes[2, 0]
    abs_max = max(abs(np.nanmin(curvature)), abs(np.nanmax(curvature)), 1e-10)
    im = ax.imshow(curvature, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max)
    ax.set_title("Curvature (ridges=red, valleys=blue)", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="Curvature (1/m)", shrink=0.7)

    # TRI (row 2, right)
    ax = axes[2, 1]
    tri_display = np.clip(tri, 0, 500)
    im = ax.imshow(tri_display, cmap="YlOrRd", vmin=0, vmax=500)
    ax.set_title("Terrain Ruggedness Index", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="TRI (m)", shrink=0.7)

    # Contours (row 3, left)
    ax = axes[3, 0]
    ax.imshow(elevation, cmap="terrain", vmin=elev_min, vmax=elev_max, alpha=0.7)
    contour_mask = ~np.isnan(contours)
    contour_overlay = np.ma.masked_where(~contour_mask, contours)
    im = ax.imshow(contour_overlay, cmap="copper_r", vmin=elev_min, vmax=elev_max)
    ax.set_title(f"Contours (100m interval, {contour_result['contour_count']} levels)", fontsize=13)
    ax.axis("off")

    # Watershed (row 3, right)
    ax = axes[3, 1]
    log_ws = np.log1p(watershed)
    im = ax.imshow(log_ws, cmap="Blues", vmin=0, vmax=max(float(np.max(log_ws)), 1.0))
    ax.set_title("Watershed (flow accumulation)", fontsize=13)
    ax.axis("off")
    fig.colorbar(im, ax=ax, label="log(cells)", shrink=0.7)

    fig.suptitle(
        f"Mount Rainier -- Terrain Analysis\nCopernicus GLO-30 | bbox: {BBOX}",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = OUTPUT_DIR / "rainier_terrain.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Source: {SOURCE} (Copernicus GLO-30)")
    print(f"  Shape:  {elevation.shape[0]}x{elevation.shape[1]} pixels")
    print(f"  Elevation: {elev_min:.0f}m to {elev_max:.0f}m")
    print(
        f"  Slope range: {slope_result['value_range'][0]:.1f} to {slope_result['value_range'][1]:.1f} degrees"
    )
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
