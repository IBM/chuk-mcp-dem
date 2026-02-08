#!/usr/bin/env python3
"""
Mountain Peaks Demo -- chuk-mcp-dem

Queries elevation for famous mountain peaks using single-point and
multi-point elevation tools. Compares interpolation methods.

Usage:
    python examples/mountain_peaks_demo.py

Requirements:
    pip install chuk-mcp-dem
    (Requires network access to Copernicus DEM S3 buckets)
"""

import asyncio
import sys

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

PEAKS = [
    {"name": "Denali", "lon": -151.0063, "lat": 63.0695, "known_m": 6190},
    {"name": "Mount Rainier", "lon": -121.7603, "lat": 46.8523, "known_m": 4392},
    {"name": "Mount Whitney", "lon": -118.2923, "lat": 36.5785, "known_m": 4421},
    {"name": "Pikes Peak", "lon": -105.0423, "lat": 38.8409, "known_m": 4302},
    {"name": "Mount Washington", "lon": -71.3033, "lat": 44.2706, "known_m": 1917},
]

SOURCE = "cop30"


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    runner = ToolRunner()

    print("=" * 60)
    print("Mountain Peaks -- Elevation Queries")
    print("=" * 60)

    # Step 1: Single-point query (detailed output)
    peak = PEAKS[1]  # Mount Rainier
    print(f"\nStep 1: Single-point query -- {peak['name']}")
    print(f"  Location: ({peak['lon']}, {peak['lat']})")

    result = await runner.run("dem_fetch_point", lon=peak["lon"], lat=peak["lat"], source=SOURCE)
    if "error" in result:
        print(f"  ERROR: {result['error']}")
        sys.exit(1)

    print(f"  DEM elevation: {result['elevation_m']:.1f}m")
    print(f"  Known elevation: {peak['known_m']}m")
    print(f"  Difference: {result['elevation_m'] - peak['known_m']:.1f}m")
    print(f"  Uncertainty: +/-{result['uncertainty_m']}m")
    print(f"  Interpolation: {result['interpolation']}")

    # Also show text output mode
    text = await runner.run_text("dem_fetch_point", lon=peak["lon"], lat=peak["lat"], source=SOURCE)
    print("\n  Text mode output:")
    for line in text.strip().split("\n"):
        print(f"    {line}")

    # Step 2: Query all peaks individually
    print(f"\nStep 2: Querying all {len(PEAKS)} peaks...")

    peak_data = []
    for p in PEAKS:
        r = await runner.run("dem_fetch_point", lon=p["lon"], lat=p["lat"], source=SOURCE)
        if "error" in r:
            print(f"  {p['name']}: ERROR - {r['error']}")
            continue
        peak_data.append(
            {
                "name": p["name"],
                "dem_m": r["elevation_m"],
                "known_m": p["known_m"],
                "diff": r["elevation_m"] - p["known_m"],
            }
        )
        print(f"  {p['name']}: {r['elevation_m']:.1f}m")

    # Print ranked table
    peak_data.sort(key=lambda x: x["dem_m"], reverse=True)
    print(f"\n  {'Rank':<6}{'Peak':<20}{'DEM (m)':<10}{'Known (m)':<12}{'Diff (m)':<10}")
    print(f"  {'-' * 56}")
    for i, p in enumerate(peak_data, 1):
        print(f"  {i:<6}{p['name']:<20}{p['dem_m']:<10.1f}{p['known_m']:<12}{p['diff']:<+10.1f}")

    # Step 2b: Multi-point batch query (nearby peaks sharing tiles)
    # Use 3 peaks in the western US that share a manageable tile footprint
    batch_peaks = [
        p for p in PEAKS if p["name"] in ("Mount Whitney", "Pikes Peak", "Mount Rainier")
    ]
    batch_points = [[p["lon"], p["lat"]] for p in batch_peaks]
    print(f"\n  Batch query ({len(batch_peaks)} nearby peaks via dem_fetch_points):")
    multi_result = await runner.run("dem_fetch_points", points=batch_points, source=SOURCE)
    if "error" in multi_result:
        print(f"  ERROR: {multi_result['error']}")
    else:
        elev_range = multi_result["elevation_range"]
        print(f"  Range: {elev_range[0]:.0f}m to {elev_range[1]:.0f}m")
        for p_info, p_meta in zip(multi_result["points"], batch_peaks):
            print(f"    {p_meta['name']}: {p_info['elevation_m']:.1f}m")

    # Step 3: Interpolation comparison for one peak
    print(f"\nStep 3: Interpolation comparison -- {PEAKS[1]['name']}")
    methods = ["nearest", "bilinear", "cubic"]

    print(f"\n  {'Method':<12}{'Elevation (m)':<16}{'Diff from known':<16}")
    print(f"  {'-' * 42}")

    for method in methods:
        r = await runner.run(
            "dem_fetch_point",
            lon=PEAKS[1]["lon"],
            lat=PEAKS[1]["lat"],
            source=SOURCE,
            interpolation=method,
        )
        if "error" in r:
            print(f"  {method:<12}ERROR: {r['error']}")
            continue
        diff = r["elevation_m"] - PEAKS[1]["known_m"]
        print(f"  {method:<12}{r['elevation_m']:<16.1f}{diff:<+16.1f}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Source: {SOURCE} (Copernicus GLO-30, 30m resolution)")
    print(f"  Peaks queried: {len(PEAKS)}")
    print(f"  Highest: {peak_data[0]['name']} ({peak_data[0]['dem_m']:.0f}m)")
    print("  Note: DEM elevations may differ from surveyed peaks due to")
    print("        30m pixel averaging and different vertical datums.")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
