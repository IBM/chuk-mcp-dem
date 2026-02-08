#!/usr/bin/env python3
"""
Grand Canyon Elevation Profile -- chuk-mcp-dem Demo

Extracts an elevation cross-section from the South Rim to the North Rim
of the Grand Canyon, demonstrating point elevation queries and the
elevation profile tool.

Usage:
    python examples/grand_canyon_profile_demo.py

Output:
    examples/output/grand_canyon_profile.png

Requirements:
    pip install chuk-mcp-dem matplotlib
    (Requires network access to Copernicus DEM S3 buckets)
"""

import asyncio
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from tool_runner import ToolRunner

# -- Configuration -----------------------------------------------------------

SOUTH_RIM = [-112.1125, 36.0580]  # South Rim viewpoint
NORTH_RIM = [-112.0550, 36.2165]  # North Rim viewpoint
NUM_POINTS = 200
SOURCE = "cop30"
OUTPUT_DIR = Path(__file__).parent / "output"


# -- Main pipeline -----------------------------------------------------------


async def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner()

    print("=" * 60)
    print("Grand Canyon -- Rim-to-Rim Elevation Profile")
    print("=" * 60)

    # Step 1: Query South Rim elevation
    print("\nStep 1: Querying South Rim elevation...")
    print(f"  Location: ({SOUTH_RIM[0]}, {SOUTH_RIM[1]})")

    south_result = await runner.run(
        "dem_fetch_point", lon=SOUTH_RIM[0], lat=SOUTH_RIM[1], source=SOURCE
    )
    if "error" in south_result:
        print(f"  ERROR: {south_result['error']}")
        sys.exit(1)
    south_elev = south_result["elevation_m"]
    print(f"  Elevation: {south_elev:.1f}m (+/-{south_result['uncertainty_m']}m)")

    # Step 2: Query North Rim elevation
    print("\nStep 2: Querying North Rim elevation...")
    print(f"  Location: ({NORTH_RIM[0]}, {NORTH_RIM[1]})")

    north_result = await runner.run(
        "dem_fetch_point", lon=NORTH_RIM[0], lat=NORTH_RIM[1], source=SOURCE
    )
    if "error" in north_result:
        print(f"  ERROR: {north_result['error']}")
        sys.exit(1)
    north_elev = north_result["elevation_m"]
    print(f"  Elevation: {north_elev:.1f}m (+/-{north_result['uncertainty_m']}m)")

    print(f"\n  Rim difference: {abs(north_elev - south_elev):.0f}m")

    # Step 3: Extract elevation profile
    print(f"\nStep 3: Extracting {NUM_POINTS}-point elevation profile...")

    profile = await runner.run(
        "dem_profile",
        start=SOUTH_RIM,
        end=NORTH_RIM,
        source=SOURCE,
        num_points=NUM_POINTS,
    )
    if "error" in profile:
        print(f"  ERROR: {profile['error']}")
        sys.exit(1)

    total_dist = profile["total_distance_m"]
    elev_range = profile["elevation_range"]
    gain = profile["elevation_gain_m"]
    loss = profile["elevation_loss_m"]

    print(f"  Total distance: {total_dist:.0f}m ({total_dist / 1000:.1f} km)")
    print(f"  Elevation range: {elev_range[0]:.0f}m to {elev_range[1]:.0f}m")
    print(f"  Elevation gain: {gain:.0f}m")
    print(f"  Elevation loss: {loss:.0f}m")
    print(f"  Canyon depth: ~{elev_range[1] - elev_range[0]:.0f}m")

    # Extract arrays for plotting
    distances = [p["distance_m"] for p in profile["points"]]
    elevations = [p["elevation_m"] for p in profile["points"]]

    # Step 4: Render profile chart
    print("\nStep 4: Rendering profile chart...")

    fig, ax = plt.subplots(1, 1, figsize=(14, 6))

    # Fill under the profile
    ax.fill_between(
        [d / 1000 for d in distances],
        elevations,
        min(elevations) - 50,
        alpha=0.3,
        color="sienna",
    )
    ax.plot([d / 1000 for d in distances], elevations, color="saddlebrown", linewidth=2)

    # Mark the rims
    ax.annotate(
        f"South Rim\n{south_elev:.0f}m",
        xy=(distances[0] / 1000, elevations[0]),
        xytext=(distances[0] / 1000 + 0.5, elevations[0] + 100),
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    ax.annotate(
        f"North Rim\n{north_elev:.0f}m",
        xy=(distances[-1] / 1000, elevations[-1]),
        xytext=(distances[-1] / 1000 - 3, elevations[-1] + 100),
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="black"),
    )

    # Find and mark the lowest point
    min_idx = elevations.index(min(elevations))
    min_elev = elevations[min_idx]
    min_dist = distances[min_idx] / 1000
    ax.annotate(
        f"River\n{min_elev:.0f}m",
        xy=(min_dist, min_elev),
        xytext=(min_dist, min_elev - 200),
        fontsize=10,
        fontweight="bold",
        color="blue",
        arrowprops=dict(arrowstyle="->", color="blue"),
    )

    # Stats text box
    stats_text = (
        f"Distance: {total_dist / 1000:.1f} km\n"
        f"Gain: {gain:.0f}m  Loss: {loss:.0f}m\n"
        f"Source: {SOURCE}"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    ax.set_xlabel("Distance (km)", fontsize=12)
    ax.set_ylabel("Elevation (m)", fontsize=12)
    ax.set_title(
        "Grand Canyon -- South Rim to North Rim Elevation Profile\n"
        f"Copernicus GLO-30 | {NUM_POINTS} sample points",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, total_dist / 1000)

    fig.tight_layout()
    output_path = OUTPUT_DIR / "grand_canyon_profile.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Demo complete!")
    print(f"  Profile: South Rim to North Rim ({total_dist / 1000:.1f} km)")
    print(f"  Canyon depth: ~{elev_range[1] - elev_range[0]:.0f}m")
    print(f"  Gain: {gain:.0f}m, Loss: {loss:.0f}m")
    print(f"\nOutput: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
