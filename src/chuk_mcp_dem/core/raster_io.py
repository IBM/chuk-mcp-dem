"""
Raster I/O operations for DEM data.

All functions are synchronous — callers wrap them in asyncio.to_thread().
Handles COG reading, tile merging, point sampling, terrain derivatives,
and output format conversion.
"""

import io
import logging
import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# Type aliases
FloatArray = NDArray[np.floating[Any]]
Transform = Any  # rasterio.Affine


# ---------------------------------------------------------------------------
# Retry decorator for network I/O
# ---------------------------------------------------------------------------

_retry_network = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    reraise=True,
)


# ---------------------------------------------------------------------------
# DEM tile reading
# ---------------------------------------------------------------------------


@_retry_network
def read_dem_tile(
    url: str,
    bbox: list[float] | None = None,
) -> tuple[FloatArray, Any, Transform]:
    """
    Read a single DEM tile from a COG URL.

    Args:
        url: COG URL (S3 or HTTPS)
        bbox: Optional [west, south, east, north] in EPSG:4326 to crop

    Returns:
        Tuple of (elevation_array, CRS, transform)
    """
    import rasterio
    from rasterio.windows import from_bounds

    with rasterio.open(url) as src:
        if bbox is not None:
            native_bbox = _reproject_bbox(bbox, src.crs)
            window = from_bounds(*native_bbox, transform=src.transform)
            data = src.read(1, window=window).astype(np.float32)
            transform = src.window_transform(window)
        else:
            data = src.read(1).astype(np.float32)
            transform = src.transform

        crs = src.crs
        nodata = src.nodata

    # Replace nodata with NaN
    if nodata is not None:
        data[data == nodata] = np.nan

    return data, crs, transform


def read_and_merge_tiles(
    urls: list[str],
    bbox: list[float] | None = None,
) -> tuple[FloatArray, Any, Transform]:
    """
    Read and merge multiple DEM tiles covering a bbox.

    Args:
        urls: List of COG URLs to read
        bbox: Optional crop bbox in EPSG:4326

    Returns:
        Tuple of (merged_elevation, CRS, transform)
    """
    import rasterio
    from rasterio.merge import merge

    if len(urls) == 1:
        return read_dem_tile(urls[0], bbox)

    datasets = []
    try:
        for url in urls:
            datasets.append(rasterio.open(url))

        merged, merged_transform = merge(datasets)
        crs = datasets[0].crs

        elevation = merged[0].astype(np.float32)

        nodata = datasets[0].nodata
        if nodata is not None:
            elevation[elevation == nodata] = np.nan

    finally:
        for ds in datasets:
            ds.close()

    if bbox is not None:
        elevation, merged_transform = _crop_to_bbox(elevation, merged_transform, crs, bbox)

    return elevation, crs, merged_transform


# ---------------------------------------------------------------------------
# Point sampling
# ---------------------------------------------------------------------------


def sample_elevation(
    elevation: FloatArray,
    transform: Transform,
    lon: float,
    lat: float,
    interpolation: str = "bilinear",
) -> float:
    """
    Sample elevation at a single point.

    Args:
        elevation: 2D elevation array
        transform: Affine transform
        lon: Longitude
        lat: Latitude
        interpolation: nearest, bilinear, or cubic

    Returns:
        Elevation value in metres
    """
    col_f, row_f = ~transform * (lon, lat)

    if interpolation == "nearest":
        row, col = int(round(row_f)), int(round(col_f))
        if 0 <= row < elevation.shape[0] and 0 <= col < elevation.shape[1]:
            val = elevation[row, col]
            return float(val) if not np.isnan(val) else float("nan")
        return float("nan")

    elif interpolation == "bilinear":
        return _bilinear_sample(elevation, row_f, col_f)

    elif interpolation == "cubic":
        return _cubic_sample(elevation, row_f, col_f)

    else:
        raise ValueError(f"Unknown interpolation: {interpolation}")


def sample_elevations(
    elevation: FloatArray,
    transform: Transform,
    points: list[list[float]],
    interpolation: str = "bilinear",
) -> list[float]:
    """Sample elevation at multiple points."""
    return [sample_elevation(elevation, transform, lon, lat, interpolation) for lon, lat in points]


# ---------------------------------------------------------------------------
# Void filling
# ---------------------------------------------------------------------------


def fill_voids(elevation: FloatArray, nodata: float = -9999.0) -> FloatArray:
    """
    Fill void pixels (NaN) using nearest-neighbour interpolation.

    Args:
        elevation: 2D array with NaN for voids
        nodata: Original nodata value (also treated as void)

    Returns:
        Array with voids filled
    """
    from scipy.ndimage import distance_transform_edt

    result = elevation.copy()

    invalid = np.isnan(result)
    if nodata != -9999.0:
        invalid |= result == nodata

    if not np.any(invalid):
        return result

    indices = distance_transform_edt(invalid, return_distances=False, return_indices=True)
    result = result[tuple(indices)]

    return result


# ---------------------------------------------------------------------------
# Output conversion
# ---------------------------------------------------------------------------


def arrays_to_geotiff(
    array: FloatArray,
    crs: Any,
    transform: Transform,
    dtype: str = "float32",
    nodata: float | None = None,
) -> bytes:
    """
    Convert a 2D NumPy array to GeoTIFF bytes.

    Args:
        array: 2D elevation array
        crs: Coordinate reference system
        transform: Affine transform
        dtype: Output data type
        nodata: Nodata value

    Returns:
        GeoTIFF bytes
    """
    from rasterio.io import MemoryFile

    if array.ndim == 2:
        height, width = array.shape
        count = 1
        write_data = array[np.newaxis, :]
    else:
        count, height, width = array.shape
        write_data = array

    memfile = MemoryFile()
    with memfile.open(
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(write_data.astype(dtype))

    return memfile.read()


def compute_hillshade(
    elevation: FloatArray,
    transform: Transform,
    azimuth: float = 315.0,
    altitude: float = 45.0,
    z_factor: float = 1.0,
) -> FloatArray:
    """
    Compute hillshade (shaded relief) from elevation data.

    Uses Horn's method (1981) for slope and aspect calculation.

    Args:
        elevation: 2D elevation array
        transform: Affine transform (for cell size)
        azimuth: Sun azimuth in degrees from north
        altitude: Sun altitude in degrees above horizon
        z_factor: Vertical exaggeration factor

    Returns:
        Hillshade array (0-255 range as float)
    """
    cellsize_x = abs(transform[0])
    cellsize_y = abs(transform[4])

    padded = np.pad(elevation, 1, mode="edge")
    padded = np.nan_to_num(padded, nan=0.0)

    # Horn's method: 3x3 gradient
    dz_dx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * cellsize_x)

    dz_dy = (
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
        - (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
    ) / (8.0 * cellsize_y)

    dz_dx *= z_factor
    dz_dy *= z_factor

    az_rad = math.radians(360.0 - azimuth + 90.0)
    alt_rad = math.radians(altitude)

    slope = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    aspect = np.arctan2(-dz_dy, dz_dx)

    hillshade = 255.0 * (
        math.cos(alt_rad) * np.cos(slope)
        + math.sin(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect)
    )

    hillshade = np.clip(hillshade, 0, 255)

    return hillshade


def elevation_to_hillshade_png(
    elevation: FloatArray,
    transform: Transform,
) -> bytes:
    """Generate a hillshade PNG from elevation data (used for auto-preview)."""
    hs = compute_hillshade(elevation, transform)
    hs_uint8 = hs.astype(np.uint8)

    img = Image.fromarray(hs_uint8, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def elevation_to_terrain_png(
    elevation: FloatArray,
) -> bytes:
    """Generate a terrain-coloured PNG from elevation data."""
    valid = elevation[~np.isnan(elevation)]
    if len(valid) == 0:
        img = Image.new("L", (elevation.shape[1], elevation.shape[0]), 0)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    vmin, vmax = float(np.nanmin(valid)), float(np.nanmax(valid))
    if vmax == vmin:
        vmax = vmin + 1.0

    norm = (elevation - vmin) / (vmax - vmin)
    norm = np.clip(np.nan_to_num(norm, nan=0.0), 0.0, 1.0)

    # Simple terrain colour ramp: green -> brown -> white
    r = np.clip(norm * 2.0, 0, 1) * 200 + 55
    g = np.clip(1.0 - norm * 0.5, 0, 1) * 200 + 55
    b = np.clip(norm * 1.5 - 0.5, 0, 1) * 200 + 55

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def slope_to_png(
    slope: FloatArray,
    units: str = "degrees",
) -> bytes:
    """Generate a slope PNG with green-yellow-red colour ramp.

    Args:
        slope: 2D slope array (degrees or percent)
        units: "degrees" or "percent"

    Returns:
        PNG bytes
    """
    max_val = 90.0 if units == "degrees" else 200.0
    norm = np.clip(np.nan_to_num(slope, nan=0.0) / max_val, 0.0, 1.0)

    # Green (flat) -> Yellow (moderate) -> Red (steep)
    r = np.clip(norm * 2.0, 0, 1) * 255
    g = np.clip(2.0 - norm * 2.0, 0, 1) * 255
    b = np.zeros_like(norm)

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def aspect_to_png(
    aspect: FloatArray,
    flat_value: float = -1.0,
) -> bytes:
    """Generate an aspect PNG with HSV colour wheel, flat areas grey.

    Args:
        aspect: 2D aspect array in degrees (0-360, flat_value for flat)
        flat_value: Value used for flat areas

    Returns:
        PNG bytes
    """
    h, w = aspect.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    flat_mask = aspect == flat_value
    valid_mask = ~flat_mask

    if np.any(valid_mask):
        hue = aspect[valid_mask] / 360.0
        # HSV to RGB: S=0.8, V=0.9
        hue6 = hue * 6.0
        sector = np.floor(hue6).astype(int) % 6
        f = hue6 - np.floor(hue6)
        v = 230  # 0.9 * 255
        p = int(230 * 0.2)  # v * (1 - s)
        q = (230 * (1.0 - 0.8 * f)).astype(np.uint8)
        t = (230 * (1.0 - 0.8 * (1.0 - f))).astype(np.uint8)

        r_vals = np.where(
            sector == 0,
            v,
            np.where(
                sector == 1,
                q,
                np.where(sector == 2, p, np.where(sector == 3, p, np.where(sector == 4, t, v))),
            ),
        )
        g_vals = np.where(
            sector == 0,
            t,
            np.where(
                sector == 1,
                v,
                np.where(sector == 2, v, np.where(sector == 3, q, np.where(sector == 4, p, p))),
            ),
        )
        b_vals = np.where(
            sector == 0,
            p,
            np.where(
                sector == 1,
                p,
                np.where(sector == 2, t, np.where(sector == 3, v, np.where(sector == 4, v, q))),
            ),
        )

        rgb[valid_mask, 0] = np.clip(r_vals, 0, 255).astype(np.uint8)
        rgb[valid_mask, 1] = np.clip(g_vals, 0, 255).astype(np.uint8)
        rgb[valid_mask, 2] = np.clip(b_vals, 0, 255).astype(np.uint8)

    # Flat areas are grey
    rgb[flat_mask] = [128, 128, 128]

    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Terrain derivatives
# ---------------------------------------------------------------------------


def compute_curvature(
    elevation: FloatArray,
    transform: Transform,
) -> FloatArray:
    """Compute profile curvature from elevation data.

    Profile curvature is the rate of change of slope along the direction
    of maximum slope. Positive values indicate convex surfaces (ridges),
    negative values indicate concave surfaces (valleys), and near-zero
    values indicate planar surfaces.

    Uses the second-order finite difference method on a 3x3 moving window.

    Args:
        elevation: 2D elevation array
        transform: Affine transform (for cell size)

    Returns:
        Curvature array (units: 1/m, float32)
    """
    cellsize_x = abs(transform[0])
    cellsize_y = abs(transform[4])

    padded = np.pad(elevation, 1, mode="edge")
    padded = np.nan_to_num(padded, nan=0.0)

    # Zevenbergen-Thorne (1987) second derivatives using 3x3 window
    # Neighbours: N, S, W, E, centre
    z_n = padded[:-2, 1:-1]
    z_w = padded[1:-1, :-2]
    z_c = padded[1:-1, 1:-1]
    z_e = padded[1:-1, 2:]
    z_s = padded[2:, 1:-1]

    # Second derivatives
    d2z_dx2 = (z_w + z_e - 2 * z_c) / (cellsize_x**2)
    d2z_dy2 = (z_n + z_s - 2 * z_c) / (cellsize_y**2)

    # Profile curvature (Laplacian approximation)
    curvature = d2z_dx2 + d2z_dy2

    return curvature.astype(np.float32)


def curvature_to_png(
    curvature: FloatArray,
) -> bytes:
    """Generate a curvature PNG with diverging blue-white-red colour ramp.

    Blue = concave (valleys), White = flat, Red = convex (ridges).

    Args:
        curvature: 2D curvature array

    Returns:
        PNG bytes
    """
    # Symmetric normalisation around zero
    abs_max = max(float(np.nanmax(np.abs(curvature))), 1e-10)
    norm = np.clip(np.nan_to_num(curvature, nan=0.0) / abs_max, -1.0, 1.0)

    # Diverging colour ramp: blue (-1) -> white (0) -> red (+1)
    r = np.where(norm >= 0, 255, (1 + norm) * 255).astype(np.uint8)
    g = np.where(norm >= 0, (1 - norm) * 255, (1 + norm) * 255).astype(np.uint8)
    b = np.where(norm <= 0, 255, (1 - norm) * 255).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_tri(
    elevation: FloatArray,
    transform: Transform,
) -> FloatArray:
    """Compute Terrain Ruggedness Index (TRI) from elevation data.

    TRI measures the mean absolute difference between a cell and its
    8 neighbours, following Riley et al. (1999).

    Classification:
        0-80m:    Level
        81-116m:  Nearly level
        117-161m: Slightly rugged
        162-239m: Intermediately rugged
        240-497m: Moderately rugged
        498-958m: Highly rugged
        959+m:    Extremely rugged

    Args:
        elevation: 2D elevation array
        transform: Affine transform (not used for TRI but kept for API consistency)

    Returns:
        TRI array (metres, float32)
    """
    padded = np.pad(elevation, 1, mode="edge")
    padded = np.nan_to_num(padded, nan=0.0)

    centre = padded[1:-1, 1:-1]

    # Sum of absolute differences with 8 neighbours
    tri = np.zeros_like(centre, dtype=np.float64)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            neighbour = padded[1 + dr : padded.shape[0] - 1 + dr, 1 + dc : padded.shape[1] - 1 + dc]
            tri += np.abs(neighbour - centre)

    tri /= 8.0  # mean absolute difference

    return tri.astype(np.float32)


def tri_to_png(
    tri: FloatArray,
) -> bytes:
    """Generate a TRI PNG with green-yellow-orange-red colour ramp.

    Green = level, Yellow = slightly rugged, Orange = moderate, Red = extreme.

    Args:
        tri: 2D TRI array (metres)

    Returns:
        PNG bytes
    """
    # Normalise to 0-500m range (covers level to moderately rugged)
    max_val = 500.0
    norm = np.clip(np.nan_to_num(tri, nan=0.0) / max_val, 0.0, 1.0)

    # Green -> Yellow -> Orange -> Red
    r = np.clip(norm * 3.0, 0, 1) * 255
    g = np.clip(1.0 - norm * 1.5, 0, 1) * 255
    b = np.zeros_like(norm)

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_contours(
    elevation: FloatArray,
    transform: Transform,
    interval_m: float,
    base_m: float | None = None,
) -> tuple[FloatArray, list[float]]:
    """Generate rasterised contour lines from elevation data.

    For each contour level, detects pixels where the elevation crosses that
    level by checking sign changes in (elevation - level) between adjacent
    pixels horizontally and vertically. Contour pixels are set to their
    elevation level value; non-contour pixels are NaN.

    Args:
        elevation: 2D elevation array
        transform: Affine transform (kept for API consistency)
        interval_m: Elevation interval between contour lines (metres)
        base_m: Base elevation for first contour (default: auto from data)

    Returns:
        Tuple of (contour raster, list of contour levels)
    """
    clean = np.nan_to_num(elevation, nan=0.0).astype(np.float64)
    elev_min = float(np.nanmin(elevation))
    elev_max = float(np.nanmax(elevation))

    if base_m is None:
        base_m = math.floor(elev_min / interval_m) * interval_m

    levels = []
    lvl = base_m
    while lvl <= elev_max:
        if lvl >= elev_min:
            levels.append(lvl)
        lvl += interval_m

    result = np.full_like(elevation, np.nan, dtype=np.float32)

    for level in levels:
        diff = clean - level
        # Horizontal crossings: sign change between adjacent columns
        h_cross = diff[:, :-1] * diff[:, 1:] <= 0
        # Vertical crossings: sign change between adjacent rows
        v_cross = diff[:-1, :] * diff[1:, :] <= 0
        # Mark crossing pixels
        mask = np.zeros(elevation.shape, dtype=bool)
        mask[:, :-1] |= h_cross
        mask[:, 1:] |= h_cross
        mask[:-1, :] |= v_cross
        mask[1:, :] |= v_cross
        result[mask] = level

    return result, levels


def contours_to_png(
    elevation: FloatArray,
    contours: FloatArray,
) -> bytes:
    """Generate a PNG with contour lines overlaid on terrain background.

    Background uses a green-brown-white terrain colour ramp.
    Contour lines are drawn in dark brown.

    Args:
        elevation: 2D elevation array (for background colouring)
        contours: 2D contour raster (NaN = no contour, value = contour level)

    Returns:
        PNG bytes
    """
    clean_elev = np.nan_to_num(elevation, nan=0.0)
    emin = float(np.nanmin(elevation))
    emax = float(np.nanmax(elevation))
    rng = max(emax - emin, 1e-10)
    norm = np.clip((clean_elev - emin) / rng, 0.0, 1.0)

    # Terrain colour ramp: green -> brown -> white
    r = np.clip(norm * 2.0, 0, 1) * 200 + 30
    g = np.clip(1.0 - norm * 0.5, 0, 1) * 200 + 30
    b = np.clip(norm - 0.5, 0, 0.5) * 200 + 30
    r = np.clip(r, 0, 255)
    g = np.clip(g, 0, 255)
    b = np.clip(b, 0, 255)

    # Overlay contour lines in dark brown
    contour_mask = ~np.isnan(contours)
    r[contour_mask] = 60
    g[contour_mask] = 30
    b[contour_mask] = 10

    rgb = np.stack([r, g, b], axis=-1).astype(np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_flow_accumulation(
    elevation: FloatArray,
    transform: Transform,
) -> FloatArray:
    """Compute flow accumulation using the D8 single-flow-direction algorithm.

    For each cell, water flows to the steepest downhill neighbour (D8).
    Flow accumulation counts the number of upstream cells draining through
    each cell. High values indicate streams and drainage channels.

    Args:
        elevation: 2D elevation array
        transform: Affine transform (kept for API consistency)

    Returns:
        Flow accumulation array (float32, units: contributing cells)
    """
    elev = np.nan_to_num(elevation, nan=0.0).astype(np.float64)
    rows, cols = elev.shape

    # D8 neighbour offsets: (row_offset, col_offset)
    d8_dr = np.array([-1, -1, 0, 1, 1, 1, 0, -1], dtype=np.intp)
    d8_dc = np.array([0, 1, 1, 1, 0, -1, -1, -1], dtype=np.intp)

    # Compute flow direction for each cell (index into d8 neighbours, -1 = pit)
    flow_dir = np.full((rows, cols), -1, dtype=np.intp)

    for r in range(rows):
        for c in range(cols):
            max_drop = 0.0
            best = -1
            for k in range(8):
                nr = r + d8_dr[k]
                nc = c + d8_dc[k]
                if 0 <= nr < rows and 0 <= nc < cols:
                    drop = elev[r, c] - elev[nr, nc]
                    if drop > max_drop:
                        max_drop = drop
                        best = k
            flow_dir[r, c] = best

    # Topological sort: process cells from highest to lowest elevation
    flat_indices = np.argsort(elev, axis=None)[::-1]

    accum = np.ones((rows, cols), dtype=np.float64)

    for idx in flat_indices:
        r = idx // cols
        c = idx % cols
        k = flow_dir[r, c]
        if k >= 0:
            nr = r + d8_dr[k]
            nc = c + d8_dc[k]
            accum[nr, nc] += accum[r, c]

    return accum.astype(np.float32)


def watershed_to_png(
    accumulation: FloatArray,
) -> bytes:
    """Generate a watershed PNG with log-scaled blue colour ramp.

    Low accumulation (hilltops) is light, high accumulation (streams) is
    dark blue. Uses log scaling to reveal the drainage network.

    Args:
        accumulation: 2D flow accumulation array (cell counts)

    Returns:
        PNG bytes
    """
    # Log-scale the accumulation (add 1 to avoid log(0))
    log_acc = np.log1p(np.nan_to_num(accumulation, nan=0.0))
    max_val = max(float(np.max(log_acc)), 1e-10)
    norm = np.clip(log_acc / max_val, 0.0, 1.0)

    # Light blue (low) -> Dark blue (high)
    r = ((1.0 - norm) * 200 + 20).astype(np.uint8)
    g = ((1.0 - norm) * 200 + 30).astype(np.uint8)
    b = (norm * 155 + 100).astype(np.uint8)

    rgb = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(rgb, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def compute_slope(
    elevation: FloatArray,
    transform: Transform,
    units: str = "degrees",
) -> FloatArray:
    """Compute slope from elevation data using Horn's method."""
    cellsize_x = abs(transform[0])
    cellsize_y = abs(transform[4])

    padded = np.pad(elevation, 1, mode="edge")
    padded = np.nan_to_num(padded, nan=0.0)

    dz_dx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * cellsize_x)

    dz_dy = (
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
        - (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
    ) / (8.0 * cellsize_y)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

    if units == "degrees":
        return np.degrees(slope_rad).astype(np.float32)
    elif units == "percent":
        return (np.tan(slope_rad) * 100.0).astype(np.float32)
    else:
        raise ValueError(f"Unknown slope units: {units}")


def compute_aspect(
    elevation: FloatArray,
    transform: Transform,
    flat_value: float = -1.0,
) -> FloatArray:
    """Compute aspect (slope direction) from elevation data."""
    cellsize_x = abs(transform[0])
    cellsize_y = abs(transform[4])

    padded = np.pad(elevation, 1, mode="edge")
    padded = np.nan_to_num(padded, nan=0.0)

    dz_dx = (
        (padded[:-2, 2:] + 2 * padded[1:-1, 2:] + padded[2:, 2:])
        - (padded[:-2, :-2] + 2 * padded[1:-1, :-2] + padded[2:, :-2])
    ) / (8.0 * cellsize_x)

    dz_dy = (
        (padded[:-2, :-2] + 2 * padded[:-2, 1:-1] + padded[:-2, 2:])
        - (padded[2:, :-2] + 2 * padded[2:, 1:-1] + padded[2:, 2:])
    ) / (8.0 * cellsize_y)

    aspect = np.degrees(np.arctan2(-dz_dy, dz_dx))
    aspect = (90.0 - aspect) % 360.0

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    flat_mask = np.degrees(slope_rad) < 1.0
    aspect[flat_mask] = flat_value

    return aspect.astype(np.float32)


# ---------------------------------------------------------------------------
# Profile & viewshed
# ---------------------------------------------------------------------------


def compute_profile_points(
    elevation: FloatArray,
    transform: Transform,
    start: list[float],
    end: list[float],
    num_points: int = 100,
    interpolation: str = "bilinear",
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Sample elevation along a line between two points.

    Args:
        elevation: 2D elevation array
        transform: Affine transform
        start: [lon, lat] start point
        end: [lon, lat] end point
        num_points: Number of sample points along the line
        interpolation: Interpolation method for sampling

    Returns:
        Tuple of (lons, lats, distances_m, elevations)
    """
    lons = np.linspace(start[0], end[0], num_points).tolist()
    lats = np.linspace(start[1], end[1], num_points).tolist()

    distances: list[float] = [0.0]
    for i in range(1, num_points):
        d = _haversine(lats[i - 1], lons[i - 1], lats[i], lons[i])
        distances.append(distances[-1] + d)

    elevations = [
        sample_elevation(elevation, transform, lons[i], lats[i], interpolation)
        for i in range(num_points)
    ]

    return lons, lats, distances, elevations


def compute_viewshed(
    elevation: FloatArray,
    transform: Transform,
    observer_lon: float,
    observer_lat: float,
    radius_m: float,
    observer_height_m: float = 1.8,
) -> FloatArray:
    """Compute viewshed (visible area) from an observer point.

    Uses DDA ray-casting to check line of sight from observer to every
    cell within the specified radius.

    Args:
        elevation: 2D elevation array
        transform: Affine transform
        observer_lon: Observer longitude
        observer_lat: Observer latitude
        radius_m: Maximum analysis radius in metres
        observer_height_m: Observer height above ground in metres

    Returns:
        2D array: 1.0=visible, 0.0=hidden, NaN=outside radius
    """
    h, w = elevation.shape

    col_f, row_f = ~transform * (observer_lon, observer_lat)
    obs_row = int(round(row_f))
    obs_col = int(round(col_f))

    cellsize_x = abs(transform[0])
    cellsize_y = abs(transform[4])

    mid_lat = observer_lat
    cellsize_x_m = cellsize_x * 111320.0 * math.cos(math.radians(mid_lat))
    cellsize_y_m = cellsize_y * 111320.0

    radius_cells_x = int(radius_m / cellsize_x_m) + 1
    radius_cells_y = int(radius_m / cellsize_y_m) + 1

    result = np.full((h, w), np.nan, dtype=np.float32)

    if 0 <= obs_row < h and 0 <= obs_col < w:
        obs_elev = elevation[obs_row, obs_col]
        if np.isnan(obs_elev):
            obs_elev = 0.0
        obs_height = obs_elev + observer_height_m
        result[obs_row, obs_col] = 1.0

        r_start = max(0, obs_row - radius_cells_y)
        r_end = min(h, obs_row + radius_cells_y + 1)
        c_start = max(0, obs_col - radius_cells_x)
        c_end = min(w, obs_col + radius_cells_x + 1)

        for r in range(r_start, r_end):
            for c in range(c_start, c_end):
                if r == obs_row and c == obs_col:
                    continue

                dr = (r - obs_row) * cellsize_y_m
                dc = (c - obs_col) * cellsize_x_m
                dist = math.sqrt(dr * dr + dc * dc)

                if dist > radius_m:
                    continue

                visible = _check_line_of_sight(
                    elevation,
                    obs_row,
                    obs_col,
                    obs_height,
                    r,
                    c,
                    cellsize_x_m,
                    cellsize_y_m,
                )
                result[r, c] = 1.0 if visible else 0.0

    return result


def _check_line_of_sight(
    elevation: FloatArray,
    r0: int,
    c0: int,
    obs_height: float,
    r1: int,
    c1: int,
    cellsize_x_m: float,
    cellsize_y_m: float,
) -> bool:
    """Check line of sight between observer and target using DDA.

    Args:
        elevation: 2D elevation array
        r0, c0: Observer row/col
        obs_height: Observer absolute height (ground + observer_height_m)
        r1, c1: Target row/col
        cellsize_x_m: Cell width in metres
        cellsize_y_m: Cell height in metres

    Returns:
        True if target is visible from observer
    """
    h, w = elevation.shape
    dr = r1 - r0
    dc = c1 - c0
    steps = max(abs(dr), abs(dc))

    if steps == 0:
        return True

    target_elev = elevation[r1, c1]
    if np.isnan(target_elev):
        target_elev = 0.0

    total_dx_m = dc * cellsize_x_m
    total_dy_m = dr * cellsize_y_m
    total_dist = math.sqrt(total_dx_m**2 + total_dy_m**2)

    if total_dist == 0:
        return True

    max_angle = -math.inf

    for step in range(1, steps):
        t = step / steps
        ri = int(round(r0 + dr * t))
        ci = int(round(c0 + dc * t))

        if ri < 0 or ri >= h or ci < 0 or ci >= w:
            continue

        cell_elev = elevation[ri, ci]
        if np.isnan(cell_elev):
            continue

        dx_m = (ci - c0) * cellsize_x_m
        dy_m = (ri - r0) * cellsize_y_m
        dist = math.sqrt(dx_m**2 + dy_m**2)

        if dist == 0:
            continue

        angle = (cell_elev - obs_height) / dist
        if angle > max_angle:
            max_angle = angle

    target_angle = (target_elev - obs_height) / total_dist
    return target_angle >= max_angle


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in metres.

    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)

    Returns:
        Distance in metres
    """
    r = 6371000.0  # Earth radius in metres
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return r * c


def _reproject_bbox(bbox_4326: list[float], dst_crs: Any) -> tuple[float, float, float, float]:
    """Reproject EPSG:4326 bbox to another CRS."""
    from pyproj import Transformer

    dst_crs_str = str(dst_crs)

    if dst_crs_str in ("EPSG:4326", "epsg:4326"):
        return (bbox_4326[0], bbox_4326[1], bbox_4326[2], bbox_4326[3])

    transformer = Transformer.from_crs("EPSG:4326", dst_crs_str, always_xy=True)
    west, south = transformer.transform(bbox_4326[0], bbox_4326[1])
    east, north = transformer.transform(bbox_4326[2], bbox_4326[3])

    return (
        min(west, east),
        min(south, north),
        max(west, east),
        max(south, north),
    )


def _crop_to_bbox(
    elevation: FloatArray,
    transform: Transform,
    crs: Any,
    bbox: list[float],
) -> tuple[FloatArray, Transform]:
    """Crop elevation array to bbox."""
    from rasterio.transform import Affine
    from rasterio.windows import from_bounds

    native_bbox = _reproject_bbox(bbox, crs)
    window = from_bounds(*native_bbox, transform=transform)

    row_start = max(0, int(window.row_off))
    row_stop = min(elevation.shape[0], int(window.row_off + window.height))
    col_start = max(0, int(window.col_off))
    col_stop = min(elevation.shape[1], int(window.col_off + window.width))

    cropped = elevation[row_start:row_stop, col_start:col_stop]

    new_transform = Affine(
        transform.a,
        transform.b,
        transform.c + col_start * transform.a,
        transform.d,
        transform.e,
        transform.f + row_start * transform.e,
    )

    return cropped, new_transform


def _bilinear_sample(array: FloatArray, row_f: float, col_f: float) -> float:
    """Bilinear interpolation at fractional pixel coordinates."""
    r0, c0 = int(math.floor(row_f)), int(math.floor(col_f))
    r1, c1 = r0 + 1, c0 + 1
    h, w = array.shape

    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return float("nan")

    dr = row_f - r0
    dc = col_f - c0

    v00 = array[r0, c0]
    v01 = array[r0, c1]
    v10 = array[r1, c0]
    v11 = array[r1, c1]

    if any(np.isnan(v) for v in [v00, v01, v10, v11]):
        return float("nan")

    val = v00 * (1 - dr) * (1 - dc) + v01 * (1 - dr) * dc + v10 * dr * (1 - dc) + v11 * dr * dc
    return float(val)


def _cubic_sample(array: FloatArray, row_f: float, col_f: float) -> float:
    """Bicubic interpolation at fractional pixel coordinates."""
    from scipy.interpolate import RectBivariateSpline

    r0 = int(math.floor(row_f))
    c0 = int(math.floor(col_f))
    h, w = array.shape

    r_start = max(0, r0 - 1)
    r_end = min(h, r0 + 3)
    c_start = max(0, c0 - 1)
    c_end = min(w, c0 + 3)

    if r_end - r_start < 4 or c_end - c_start < 4:
        return _bilinear_sample(array, row_f, col_f)

    patch = array[r_start:r_end, c_start:c_end]
    if np.any(np.isnan(patch)):
        return _bilinear_sample(array, row_f, col_f)

    rows = np.arange(r_start, r_end, dtype=float)
    cols = np.arange(c_start, c_end, dtype=float)

    spline = RectBivariateSpline(rows, cols, patch, kx=3, ky=3)
    val = spline(row_f, col_f)[0, 0]
    return float(val)
