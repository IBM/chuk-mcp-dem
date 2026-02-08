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


# ---------------------------------------------------------------------------
# Terrain derivatives
# ---------------------------------------------------------------------------


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
# Internal helpers
# ---------------------------------------------------------------------------


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
