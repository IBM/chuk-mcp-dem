"""
DEM Manager — central orchestrator for elevation data operations.

Manages tile caching, URL construction, download pipeline, and artifact storage.
All public async methods wrap synchronous rasterio I/O via asyncio.to_thread().
"""

import asyncio
import logging
import math
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constants import (
    DEFAULT_ALTITUDE,
    DEFAULT_AZIMUTH,
    DEFAULT_FLAT_VALUE,
    DEFAULT_INTERPOLATION,
    DEFAULT_NUM_POINTS,
    DEFAULT_OBSERVER_HEIGHT_M,
    DEFAULT_SOURCE,
    DEFAULT_Z_FACTOR,
    DEM_SOURCES,
    MAX_VIEWSHED_RADIUS_M,
    TILE_CACHE_MAX_BYTES,
    TILE_CACHE_MAX_ITEM,
    ErrorMessages,
)

logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    """Result of a DEM fetch operation."""

    artifact_ref: str
    preview_ref: str | None
    crs: str
    resolution_m: float
    shape: list[int]
    elevation_range: list[float]
    dtype: str
    nodata_pixels: int


@dataclass
class PointResult:
    """Result of a single-point elevation query."""

    elevation_m: float
    uncertainty_m: float


@dataclass
class MultiPointResult:
    """Result of a multi-point elevation query."""

    elevations: list[float]
    elevation_range: list[float]


@dataclass
class TerrainResult:
    """Result of a terrain derivative computation (hillshade/slope/aspect)."""

    artifact_ref: str
    preview_ref: str | None
    crs: str
    resolution_m: float
    shape: list[int]
    value_range: list[float]
    dtype: str


@dataclass
class ContourResult:
    """Result of a contour line generation."""

    artifact_ref: str
    preview_ref: str | None
    crs: str
    resolution_m: float
    shape: list[int]
    interval_m: float
    contour_count: int
    elevation_range: list[float]
    dtype: str


@dataclass
class ProfileResult:
    """Result of an elevation profile extraction."""

    longitudes: list[float]
    latitudes: list[float]
    distances_m: list[float]
    elevations: list[float]
    total_distance_m: float
    elevation_range: list[float]
    elevation_gain_m: float
    elevation_loss_m: float


@dataclass
class ViewshedResult:
    """Result of a viewshed analysis."""

    artifact_ref: str
    preview_ref: str | None
    crs: str
    resolution_m: float
    shape: list[int]
    visible_percentage: float
    observer_elevation_m: float
    radius_m: float


@dataclass
class TemporalChangeResult:
    """Result of a temporal elevation change analysis."""

    artifact_ref: str
    preview_ref: str | None
    crs: str
    resolution_m: float
    shape: list[int]
    significance_threshold_m: float
    volume_gained_m3: float
    volume_lost_m3: float
    significant_regions: list[dict]
    dtype: str


@dataclass
class LandformResult:
    """Result of a landform classification."""

    artifact_ref: str
    preview_ref: str | None
    crs: str
    resolution_m: float
    shape: list[int]
    class_distribution: dict[str, float]
    dominant_landform: str
    dtype: str


@dataclass
class AnomalyResult:
    """Result of an anomaly detection analysis."""

    artifact_ref: str
    preview_ref: str | None
    crs: str
    resolution_m: float
    shape: list[int]
    anomaly_count: int
    anomalies: list[dict]
    dtype: str


@dataclass
class FeatureResult:
    """Result of a terrain feature detection analysis."""

    artifact_ref: str
    preview_ref: str | None
    crs: str
    resolution_m: float
    shape: list[int]
    feature_count: int
    feature_summary: dict[str, int]
    features: list[dict]
    dtype: str


class DEMManager:
    """Central manager for DEM data operations."""

    def __init__(
        self,
        default_source: str = DEFAULT_SOURCE,
        progress_callback: object | None = None,
    ) -> None:
        self.default_source = default_source
        self.progress_callback = progress_callback

        # Tile LRU cache: key -> bytes
        self._tile_cache: dict[str, bytes] = {}
        self._tile_cache_sizes: dict[str, int] = {}
        self._tile_cache_total: int = 0

    # ------------------------------------------------------------------
    # Discovery (sync, no I/O)
    # ------------------------------------------------------------------

    def list_sources(self) -> list[dict]:
        """List all available DEM sources."""
        return [
            {
                "id": src["id"],
                "name": src["name"],
                "resolution_m": src["resolution_m"],
                "coverage": src["coverage"],
                "vertical_datum": src["vertical_datum"],
                "void_filled": src["void_filled"],
            }
            for src in DEM_SOURCES.values()
        ]

    def describe_source(self, source: str) -> dict:
        """Get detailed metadata for a DEM source."""
        if source not in DEM_SOURCES:
            raise ValueError(
                ErrorMessages.UNKNOWN_SOURCE.format(source, ", ".join(DEM_SOURCES.keys()))
            )
        return dict(DEM_SOURCES[source])

    def check_coverage(self, bbox: list[float], source: str) -> dict:
        """Check if a DEM source covers a given bounding box."""
        self._validate_bbox(bbox)
        src = self._get_source(source)

        bounds = src["coverage_bounds"]
        west, south, east, north = bbox

        overlap_west = max(west, bounds[0])
        overlap_south = max(south, bounds[1])
        overlap_east = min(east, bounds[2])
        overlap_north = min(north, bounds[3])

        if overlap_west >= overlap_east or overlap_south >= overlap_north:
            return {
                "fully_covered": False,
                "coverage_percentage": 0.0,
                "tiles_required": 0,
                "tile_ids": [],
                "estimated_size_mb": 0.0,
            }

        bbox_area = (east - west) * (north - south)
        overlap_area = (overlap_east - overlap_west) * (overlap_north - overlap_south)
        coverage_pct = (overlap_area / bbox_area) * 100.0 if bbox_area > 0 else 0.0

        tile_ids = self._get_tile_ids(source, bbox)

        mid_lat = (south + north) / 2.0
        meters_per_deg = 111320.0 * math.cos(math.radians(mid_lat))
        pixels_per_tile = int(src["tile_size_degrees"] * meters_per_deg / src["resolution_m"]) ** 2
        total_pixels = len(tile_ids) * pixels_per_tile
        estimated_mb = (total_pixels * 4) / (1024 * 1024)

        return {
            "fully_covered": coverage_pct >= 99.9,
            "coverage_percentage": round(coverage_pct, 1),
            "tiles_required": len(tile_ids),
            "tile_ids": tile_ids,
            "estimated_size_mb": round(estimated_mb, 1),
        }

    def estimate_size(
        self,
        bbox: list[float],
        source: str,
        resolution_m: float | None = None,
    ) -> dict:
        """Estimate download size for an area."""
        self._validate_bbox(bbox)
        src = self._get_source(source)

        native_res = src["resolution_m"]
        target_res = resolution_m or native_res

        west, south, east, north = bbox
        mid_lat = (south + north) / 2.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(mid_lat))
        meters_per_deg_lat = 111320.0

        width_m = (east - west) * meters_per_deg_lon
        height_m = (north - south) * meters_per_deg_lat

        n_cols = max(1, int(width_m / target_res))
        n_rows = max(1, int(height_m / target_res))
        total_pixels = n_rows * n_cols

        estimated_bytes = total_pixels * 4
        estimated_mb = estimated_bytes / (1024 * 1024)

        warning = None
        if estimated_bytes >= 1024 * 1024 * 1024:
            warning = (
                f"Large download: {estimated_mb:.0f} MB. "
                "Consider using a smaller bbox or coarser resolution."
            )
        elif estimated_bytes >= 500 * 1024 * 1024:
            warning = f"Download size {estimated_mb:.0f} MB may be slow."

        return {
            "native_resolution_m": native_res,
            "target_resolution_m": target_res,
            "dimensions": [n_rows, n_cols],
            "pixels": total_pixels,
            "dtype": src["dtype"],
            "estimated_bytes": estimated_bytes,
            "estimated_mb": round(estimated_mb, 1),
            "warning": warning,
        }

    # ------------------------------------------------------------------
    # Download (async)
    # ------------------------------------------------------------------

    async def fetch_elevation(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        resolution_m: float | None = None,
        output_crs: str | None = None,
        fill_voids: bool = True,
        output_format: str = "geotiff",
    ) -> FetchResult:
        """Download elevation data for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        if fill_voids and np.any(np.isnan(elevation)):
            elevation = await asyncio.to_thread(
                raster_io.fill_voids, elevation, src["nodata_value"]
            )

        valid = elevation[~np.isnan(elevation)]
        if len(valid) > 0:
            elev_min = float(np.min(valid))
            elev_max = float(np.max(valid))
        else:
            elev_min = 0.0
            elev_max = 0.0
        nodata_count = int(np.sum(np.isnan(elevation)))

        geotiff_bytes = await asyncio.to_thread(
            raster_io.arrays_to_geotiff,
            elevation,
            crs,
            transform,
            src["dtype"],
            float("nan"),
        )

        # Generate hillshade preview
        preview_ref = None
        try:
            preview_bytes = await asyncio.to_thread(
                raster_io.elevation_to_hillshade_png, elevation, transform
            )
            preview_ref = await self._store_raster(
                preview_bytes,
                {
                    "type": "dem_preview",
                    "source": source,
                    "format": "png",
                    "bbox": bbox,
                },
                suffix="_hillshade.png",
            )
        except Exception as e:
            logger.warning(f"Failed to generate hillshade preview: {e}")

        artifact_ref = await self._store_raster(
            geotiff_bytes,
            {
                "schema_version": "1.0",
                "type": "dem_raster",
                "source": source,
                "source_name": src["name"],
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(elevation.shape),
                "dtype": src["dtype"],
                "elevation_range": [elev_min, elev_max],
                "vertical_datum": src["vertical_datum"],
                "vertical_unit": src["vertical_unit"],
                "nodata_value": src["nodata_value"],
                "acquisition_period": src["acquisition_period"],
                "accuracy_vertical_m": src["accuracy_vertical_m"],
            },
            suffix=".tif",
        )

        return FetchResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(elevation.shape),
            elevation_range=[elev_min, elev_max],
            dtype=src["dtype"],
            nodata_pixels=nodata_count,
        )

    async def fetch_point(
        self,
        lon: float,
        lat: float,
        source: str = DEFAULT_SOURCE,
        interpolation: str = DEFAULT_INTERPOLATION,
    ) -> PointResult:
        """Get elevation at a single point."""
        from . import raster_io

        src = self._get_source(source)

        bbox = [
            math.floor(lon),
            math.floor(lat),
            math.floor(lon) + 1.0,
            math.floor(lat) + 1.0,
        ]
        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], f"({lon}, {lat})"))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_dem_tile, tile_urls[0], None
        )

        value = await asyncio.to_thread(
            raster_io.sample_elevation, elevation, transform, lon, lat, interpolation
        )

        return PointResult(
            elevation_m=value,
            uncertainty_m=src["accuracy_vertical_m"],
        )

    async def fetch_points(
        self,
        points: list[list[float]],
        source: str = DEFAULT_SOURCE,
        interpolation: str = DEFAULT_INTERPOLATION,
    ) -> MultiPointResult:
        """Get elevations at multiple points."""
        from . import raster_io

        src = self._get_source(source)

        lons = [p[0] for p in points]
        lats = [p[1] for p in points]
        bbox: list[float] = [
            float(math.floor(min(lons))),
            float(math.floor(min(lats))),
            float(math.ceil(max(lons))),
            float(math.ceil(max(lats))),
        ]

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, None
        )

        values = await asyncio.to_thread(
            raster_io.sample_elevations, elevation, transform, points, interpolation
        )

        valid_values = [v for v in values if not math.isnan(v)]
        if valid_values:
            elev_range = [min(valid_values), max(valid_values)]
        else:
            elev_range = [0.0, 0.0]

        return MultiPointResult(
            elevations=values,
            elevation_range=elev_range,
        )

    # ------------------------------------------------------------------
    # Terrain analysis (async)
    # ------------------------------------------------------------------

    async def fetch_hillshade(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        azimuth: float = DEFAULT_AZIMUTH,
        altitude: float = DEFAULT_ALTITUDE,
        z_factor: float = DEFAULT_Z_FACTOR,
        output_format: str = "geotiff",
    ) -> TerrainResult:
        """Compute hillshade for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        hillshade = await asyncio.to_thread(
            raster_io.compute_hillshade, elevation, transform, azimuth, altitude, z_factor
        )

        val_min = float(np.nanmin(hillshade))
        val_max = float(np.nanmax(hillshade))

        if output_format == "png":
            data_bytes = await asyncio.to_thread(
                raster_io.elevation_to_hillshade_png, elevation, transform
            )
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, hillshade, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(
                    raster_io.elevation_to_hillshade_png, elevation, transform
                )
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "hillshade_preview", "source": source, "format": "png"},
                    suffix="_hillshade.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate hillshade preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "hillshade",
                "source": source,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "azimuth": azimuth,
                "altitude": altitude,
                "z_factor": z_factor,
                "shape": list(hillshade.shape),
            },
            suffix=suffix,
        )

        return TerrainResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(hillshade.shape),
            value_range=[val_min, val_max],
            dtype="float32",
        )

    async def fetch_slope(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        units: str = "degrees",
        output_format: str = "geotiff",
    ) -> TerrainResult:
        """Compute slope for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        slope = await asyncio.to_thread(raster_io.compute_slope, elevation, transform, units)

        val_min = float(np.nanmin(slope))
        val_max = float(np.nanmax(slope))

        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.slope_to_png, slope, units)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, slope, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(raster_io.slope_to_png, slope, units)
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "slope_preview", "source": source, "format": "png"},
                    suffix="_slope.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate slope preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "slope",
                "source": source,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "units": units,
                "shape": list(slope.shape),
            },
            suffix=suffix,
        )

        return TerrainResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(slope.shape),
            value_range=[val_min, val_max],
            dtype="float32",
        )

    async def fetch_aspect(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        flat_value: float = DEFAULT_FLAT_VALUE,
        output_format: str = "geotiff",
    ) -> TerrainResult:
        """Compute aspect for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        aspect = await asyncio.to_thread(raster_io.compute_aspect, elevation, transform, flat_value)

        val_min = float(np.nanmin(aspect))
        val_max = float(np.nanmax(aspect))

        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.aspect_to_png, aspect, flat_value)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, aspect, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(raster_io.aspect_to_png, aspect, flat_value)
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "aspect_preview", "source": source, "format": "png"},
                    suffix="_aspect.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate aspect preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "aspect",
                "source": source,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "flat_value": flat_value,
                "shape": list(aspect.shape),
            },
            suffix=suffix,
        )

        return TerrainResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(aspect.shape),
            value_range=[val_min, val_max],
            dtype="float32",
        )

    async def fetch_curvature(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        output_format: str = "geotiff",
    ) -> TerrainResult:
        """Compute curvature for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        curvature = await asyncio.to_thread(raster_io.compute_curvature, elevation, transform)

        val_min = float(np.nanmin(curvature))
        val_max = float(np.nanmax(curvature))

        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.curvature_to_png, curvature)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, curvature, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(raster_io.curvature_to_png, curvature)
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "curvature_preview", "source": source, "format": "png"},
                    suffix="_curvature.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate curvature preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "curvature",
                "source": source,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(curvature.shape),
            },
            suffix=suffix,
        )

        return TerrainResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(curvature.shape),
            value_range=[val_min, val_max],
            dtype="float32",
        )

    async def fetch_tri(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        output_format: str = "geotiff",
    ) -> TerrainResult:
        """Compute Terrain Ruggedness Index for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        tri = await asyncio.to_thread(raster_io.compute_tri, elevation, transform)

        val_min = float(np.nanmin(tri))
        val_max = float(np.nanmax(tri))

        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.tri_to_png, tri)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, tri, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(raster_io.tri_to_png, tri)
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "tri_preview", "source": source, "format": "png"},
                    suffix="_tri.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate TRI preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "tri",
                "source": source,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(tri.shape),
            },
            suffix=suffix,
        )

        return TerrainResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(tri.shape),
            value_range=[val_min, val_max],
            dtype="float32",
        )

    async def fetch_contours(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        interval_m: float = 100.0,
        output_format: str = "geotiff",
    ) -> ContourResult:
        """Generate contour lines for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        elevation = await asyncio.to_thread(raster_io.fill_voids, elevation)

        contours, levels = await asyncio.to_thread(
            raster_io.compute_contours, elevation, transform, interval_m
        )

        elev_min = float(np.nanmin(elevation))
        elev_max = float(np.nanmax(elevation))

        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.contours_to_png, elevation, contours)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, contours, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(
                    raster_io.contours_to_png, elevation, contours
                )
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "contour_preview", "source": source, "format": "png"},
                    suffix="_contour.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate contour preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "contour",
                "source": source,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(contours.shape),
                "interval_m": interval_m,
                "contour_count": len(levels),
            },
            suffix=suffix,
        )

        return ContourResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(contours.shape),
            interval_m=interval_m,
            contour_count=len(levels),
            elevation_range=[elev_min, elev_max],
            dtype="float32",
        )

    async def fetch_watershed(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        output_format: str = "geotiff",
    ) -> TerrainResult:
        """Compute flow accumulation (watershed) for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        elevation = await asyncio.to_thread(raster_io.fill_voids, elevation)

        accumulation = await asyncio.to_thread(
            raster_io.compute_flow_accumulation, elevation, transform
        )

        val_min = float(np.nanmin(accumulation))
        val_max = float(np.nanmax(accumulation))

        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.watershed_to_png, accumulation)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, accumulation, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(raster_io.watershed_to_png, accumulation)
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "watershed_preview", "source": source, "format": "png"},
                    suffix="_watershed.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate watershed preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "watershed",
                "source": source,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(accumulation.shape),
            },
            suffix=suffix,
        )

        return TerrainResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(accumulation.shape),
            value_range=[val_min, val_max],
            dtype="float32",
        )

    # ------------------------------------------------------------------
    # Profile & viewshed (async)
    # ------------------------------------------------------------------

    async def fetch_profile(
        self,
        start: list[float],
        end: list[float],
        source: str = DEFAULT_SOURCE,
        num_points: int = DEFAULT_NUM_POINTS,
        interpolation: str = DEFAULT_INTERPOLATION,
    ) -> ProfileResult:
        """Extract an elevation profile between two points."""
        from . import raster_io

        src = self._get_source(source)

        all_lons = [start[0], end[0]]
        all_lats = [start[1], end[1]]
        bbox: list[float] = [
            float(math.floor(min(all_lons))),
            float(math.floor(min(all_lats))),
            float(math.ceil(max(all_lons))),
            float(math.ceil(max(all_lats))),
        ]

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, None
        )

        lons, lats, distances, elevations = await asyncio.to_thread(
            raster_io.compute_profile_points,
            elevation,
            transform,
            start,
            end,
            num_points,
            interpolation,
        )

        valid_elevs = [e for e in elevations if not math.isnan(e)]
        if valid_elevs:
            elev_range = [min(valid_elevs), max(valid_elevs)]
        else:
            elev_range = [0.0, 0.0]

        gain = 0.0
        loss = 0.0
        for i in range(1, len(elevations)):
            if math.isnan(elevations[i]) or math.isnan(elevations[i - 1]):
                continue
            diff = elevations[i] - elevations[i - 1]
            if diff > 0:
                gain += diff
            else:
                loss += abs(diff)

        total_distance = distances[-1] if distances else 0.0

        return ProfileResult(
            longitudes=lons,
            latitudes=lats,
            distances_m=distances,
            elevations=elevations,
            total_distance_m=total_distance,
            elevation_range=elev_range,
            elevation_gain_m=round(gain, 1),
            elevation_loss_m=round(loss, 1),
        )

    async def fetch_viewshed(
        self,
        observer: list[float],
        radius_m: float,
        source: str = DEFAULT_SOURCE,
        observer_height_m: float = DEFAULT_OBSERVER_HEIGHT_M,
        output_format: str = "geotiff",
    ) -> ViewshedResult:
        """Compute viewshed from an observer point."""
        from . import raster_io

        src = self._get_source(source)

        if radius_m > MAX_VIEWSHED_RADIUS_M:
            raise ValueError(ErrorMessages.RADIUS_TOO_LARGE.format(radius_m, MAX_VIEWSHED_RADIUS_M))

        observer_lon, observer_lat = observer
        mid_lat = observer_lat
        deg_per_m_lon = 1.0 / (111320.0 * math.cos(math.radians(mid_lat)))
        deg_per_m_lat = 1.0 / 111320.0

        buffer_lon = radius_m * deg_per_m_lon
        buffer_lat = radius_m * deg_per_m_lat

        bbox: list[float] = [
            float(math.floor(observer_lon - buffer_lon)),
            float(math.floor(observer_lat - buffer_lat)),
            float(math.ceil(observer_lon + buffer_lon)),
            float(math.ceil(observer_lat + buffer_lat)),
        ]

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, None
        )

        viewshed = await asyncio.to_thread(
            raster_io.compute_viewshed,
            elevation,
            transform,
            observer_lon,
            observer_lat,
            radius_m,
            observer_height_m,
        )

        analyzed = viewshed[~np.isnan(viewshed)]
        if len(analyzed) > 0:
            visible_pct = float(np.sum(analyzed == 1.0) / len(analyzed) * 100.0)
        else:
            visible_pct = 0.0

        # Get observer ground elevation
        col_f, row_f = ~transform * (observer_lon, observer_lat)
        obs_row = int(round(row_f))
        obs_col = int(round(col_f))
        h, w = elevation.shape
        if 0 <= obs_row < h and 0 <= obs_col < w:
            obs_elev = float(elevation[obs_row, obs_col])
            if math.isnan(obs_elev):
                obs_elev = 0.0
        else:
            obs_elev = 0.0

        if output_format == "png":
            vis_rgb = np.zeros((*viewshed.shape, 3), dtype=np.uint8)
            vis_rgb[viewshed == 1.0] = [0, 200, 0]
            vis_rgb[viewshed == 0.0] = [200, 0, 0]
            from PIL import Image
            import io as _io

            img = Image.fromarray(vis_rgb, mode="RGB")
            buf = _io.BytesIO()
            img.save(buf, format="PNG")
            data_bytes = buf.getvalue()
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, viewshed, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                vis_rgb = np.zeros((*viewshed.shape, 3), dtype=np.uint8)
                vis_rgb[viewshed == 1.0] = [0, 200, 0]
                vis_rgb[viewshed == 0.0] = [200, 0, 0]
                from PIL import Image
                import io as _io

                img = Image.fromarray(vis_rgb, mode="RGB")
                buf = _io.BytesIO()
                img.save(buf, format="PNG")
                preview_bytes = buf.getvalue()
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "viewshed_preview", "source": source, "format": "png"},
                    suffix="_viewshed.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate viewshed preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "viewshed",
                "source": source,
                "observer": observer,
                "radius_m": radius_m,
                "observer_height_m": observer_height_m,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(viewshed.shape),
                "visible_percentage": round(visible_pct, 1),
            },
            suffix=suffix,
        )

        return ViewshedResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(viewshed.shape),
            visible_percentage=round(visible_pct, 1),
            observer_elevation_m=obs_elev,
            radius_m=radius_m,
        )

    # ------------------------------------------------------------------
    # Phase 3.0: ML-enhanced terrain analysis (async)
    # ------------------------------------------------------------------

    async def fetch_temporal_change(
        self,
        bbox: list[float],
        before_source: str = "srtm",
        after_source: str = "cop30",
        significance_threshold_m: float = 1.0,
        output_format: str = "geotiff",
    ) -> TemporalChangeResult:
        """Compare elevation between two DEM sources/epochs."""
        from . import raster_io

        self._validate_bbox(bbox)
        src_before = self._get_source(before_source)
        src_after = self._get_source(after_source)

        # Fetch before tiles
        before_urls = self._get_tile_urls(before_source, bbox)
        if not before_urls:
            raise ValueError(
                ErrorMessages.COVERAGE_ERROR.format(src_before["name"], bbox)
            )
        before_elev, crs, tx_before = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, before_urls, bbox
        )

        # Fetch after tiles
        after_urls = self._get_tile_urls(after_source, bbox)
        if not after_urls:
            raise ValueError(
                ErrorMessages.COVERAGE_ERROR.format(src_after["name"], bbox)
            )
        after_elev, _, _ = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, after_urls, bbox
        )

        # Fill voids
        before_elev = await asyncio.to_thread(raster_io.fill_voids, before_elev)
        after_elev = await asyncio.to_thread(raster_io.fill_voids, after_elev)

        # Resample if shapes differ
        if before_elev.shape != after_elev.shape:
            from scipy.ndimage import zoom

            zoom_factors = (
                before_elev.shape[0] / after_elev.shape[0],
                before_elev.shape[1] / after_elev.shape[1],
            )
            after_elev = await asyncio.to_thread(zoom, after_elev, zoom_factors, order=1)

        # Compute change
        change, regions = await asyncio.to_thread(
            raster_io.compute_elevation_change,
            before_elev, after_elev, tx_before, significance_threshold_m,
        )

        # Compute volumes using approximate pixel area
        cellsize_x = abs(float(tx_before[0]))
        cellsize_y = abs(float(tx_before[4]))
        mid_lat = (bbox[1] + bbox[3]) / 2.0
        pixel_area_m2 = (
            cellsize_x * 111320.0 * math.cos(math.radians(mid_lat))
            * cellsize_y * 111320.0
        )
        positive = change[change > 0]
        negative = change[change < 0]
        vol_gained = float(np.sum(positive) * pixel_area_m2) if len(positive) > 0 else 0.0
        vol_lost = float(np.abs(np.sum(negative)) * pixel_area_m2) if len(negative) > 0 else 0.0

        # Output
        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.change_to_png, change)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, change, crs, tx_before, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(raster_io.change_to_png, change)
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "temporal_change_preview", "source": after_source, "format": "png"},
                    suffix="_change.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate change preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "temporal_change",
                "before_source": before_source,
                "after_source": after_source,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src_before["resolution_m"],
                "shape": list(change.shape),
            },
            suffix=suffix,
        )

        return TemporalChangeResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src_before["resolution_m"],
            shape=list(change.shape),
            significance_threshold_m=significance_threshold_m,
            volume_gained_m3=vol_gained,
            volume_lost_m3=vol_lost,
            significant_regions=regions,
            dtype="float32",
        )

    async def fetch_landforms(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        method: str = "rule_based",
        output_format: str = "geotiff",
    ) -> LandformResult:
        """Classify terrain into landform types for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        elevation = await asyncio.to_thread(raster_io.fill_voids, elevation)

        landforms = await asyncio.to_thread(
            raster_io.compute_landforms, elevation, transform, method
        )

        # Compute class distribution
        from ..constants import LANDFORM_CLASSES

        total_pixels = landforms.size
        distribution: dict[str, float] = {}
        for idx, cls_name in enumerate(LANDFORM_CLASSES):
            count = int(np.sum(landforms == idx))
            distribution[cls_name] = round(count / total_pixels * 100.0, 1)

        dominant = max(distribution, key=distribution.get)  # type: ignore[arg-type]

        if output_format == "png":
            data_bytes = await asyncio.to_thread(
                raster_io.landform_to_png, landforms
            )
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff,
                landforms.astype(np.float32), crs, transform, "float32",
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(
                    raster_io.landform_to_png, landforms
                )
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "landform_preview", "source": source, "format": "png"},
                    suffix="_landform.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate landform preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "landforms",
                "source": source,
                "method": method,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(landforms.shape),
            },
            suffix=suffix,
        )

        return LandformResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(landforms.shape),
            class_distribution=distribution,
            dominant_landform=dominant,
            dtype="uint8",
        )

    async def fetch_anomalies(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        sensitivity: float = 0.1,
        output_format: str = "geotiff",
    ) -> AnomalyResult:
        """Detect terrain anomalies for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        elevation = await asyncio.to_thread(raster_io.fill_voids, elevation)

        scores, anomalies = await asyncio.to_thread(
            raster_io.compute_anomaly_scores, elevation, transform, sensitivity
        )

        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.anomaly_to_png, scores)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, scores, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(
                    raster_io.anomaly_to_png, scores
                )
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "anomaly_preview", "source": source, "format": "png"},
                    suffix="_anomaly.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate anomaly preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "anomalies",
                "source": source,
                "sensitivity": sensitivity,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(scores.shape),
            },
            suffix=suffix,
        )

        return AnomalyResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(scores.shape),
            anomaly_count=len(anomalies),
            anomalies=anomalies,
            dtype="float32",
        )

    async def fetch_features(
        self,
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        method: str = "cnn_hillshade",
        output_format: str = "geotiff",
    ) -> FeatureResult:
        """Detect terrain features for a bounding box."""
        from . import raster_io

        self._validate_bbox(bbox)
        src = self._get_source(source)

        tile_urls = self._get_tile_urls(source, bbox)
        if not tile_urls:
            raise ValueError(ErrorMessages.COVERAGE_ERROR.format(src["name"], bbox))

        elevation, crs, transform = await asyncio.to_thread(
            raster_io.read_and_merge_tiles, tile_urls, bbox
        )

        elevation = await asyncio.to_thread(raster_io.fill_voids, elevation)

        feature_map, features_list = await asyncio.to_thread(
            raster_io.compute_feature_detection, elevation, transform, method
        )

        # Compute feature summary (count per type)
        feature_summary: dict[str, int] = {}
        for f in features_list:
            ftype = f["feature_type"]
            feature_summary[ftype] = feature_summary.get(ftype, 0) + 1

        if output_format == "png":
            data_bytes = await asyncio.to_thread(raster_io.feature_to_png, feature_map)
            suffix = ".png"
        else:
            data_bytes = await asyncio.to_thread(
                raster_io.arrays_to_geotiff, feature_map, crs, transform, "float32"
            )
            suffix = ".tif"

        preview_ref = None
        if output_format != "png":
            try:
                preview_bytes = await asyncio.to_thread(
                    raster_io.feature_to_png, feature_map
                )
                preview_ref = await self._store_raster(
                    preview_bytes,
                    {"type": "feature_preview", "source": source, "format": "png"},
                    suffix="_feature.png",
                )
            except Exception as e:
                logger.warning(f"Failed to generate feature preview: {e}")

        artifact_ref = await self._store_raster(
            data_bytes,
            {
                "schema_version": "1.0",
                "type": "features",
                "source": source,
                "method": method,
                "bbox": bbox,
                "crs": str(crs),
                "resolution_m": src["resolution_m"],
                "shape": list(feature_map.shape),
            },
            suffix=suffix,
        )

        return FeatureResult(
            artifact_ref=artifact_ref,
            preview_ref=preview_ref,
            crs=str(crs),
            resolution_m=src["resolution_m"],
            shape=list(feature_map.shape),
            feature_count=len(features_list),
            feature_summary=feature_summary,
            features=features_list,
            dtype="float32",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_source(self, source: str) -> dict:
        """Get source metadata, raising ValueError if unknown."""
        if source not in DEM_SOURCES:
            raise ValueError(
                ErrorMessages.UNKNOWN_SOURCE.format(source, ", ".join(DEM_SOURCES.keys()))
            )
        return DEM_SOURCES[source]

    def _validate_bbox(self, bbox: list[float]) -> None:
        """Validate bounding box."""
        if len(bbox) != 4:
            raise ValueError(ErrorMessages.INVALID_BBOX)
        west, south, east, north = bbox
        if west >= east:
            raise ValueError(ErrorMessages.INVALID_BBOX_VALUES.format(west, east))
        if south >= north:
            raise ValueError(ErrorMessages.INVALID_BBOX_LAT.format(south, north))

    def _get_tile_ids(self, source: str, bbox: list[float]) -> list[str]:
        """Get tile IDs covering a bbox."""
        src = self._get_source(source)
        tile_size = int(src["tile_size_degrees"])

        west, south, east, north = bbox
        tiles = []

        lat = math.floor(south)
        while lat < north:
            lon = math.floor(west)
            while lon < east:
                tile_id = self._make_tile_id(source, lat, lon)
                tiles.append(tile_id)
                lon += tile_size
            lat += tile_size

        return tiles

    def _get_tile_urls(self, source: str, bbox: list[float]) -> list[str]:
        """Get tile URLs covering a bbox."""
        src = self._get_source(source)
        tile_size = int(src["tile_size_degrees"])

        west, south, east, north = bbox
        urls = []

        lat = math.floor(south)
        while lat < north:
            lon = math.floor(west)
            while lon < east:
                url = self._make_tile_url(source, lat, lon)
                if url:
                    urls.append(url)
                lon += tile_size
            lat += tile_size

        return urls

    def _make_tile_id(self, source: str, lat: int, lon: int) -> str:
        """Construct a tile ID for a given lat/lon."""
        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        abs_lat = abs(lat)
        abs_lon = abs(lon)

        if source in ("cop30", "cop90"):
            return f"Copernicus_DSM_COG_10_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00"
        else:
            return f"{ns}{abs_lat:02d}{ew}{abs_lon:03d}"

    def _make_tile_url(self, source: str, lat: int, lon: int) -> str | None:
        """Construct the URL for a DEM tile."""
        src = DEM_SOURCES.get(source)
        if not src:
            return None

        ns = "N" if lat >= 0 else "S"
        ew = "E" if lon >= 0 else "W"
        abs_lat = abs(lat)
        abs_lon = abs(lon)

        if source == "cop30":
            tile_name = f"Copernicus_DSM_COG_10_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00_DEM"
            return f"https://copernicus-dem-30m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"
        elif source == "cop90":
            tile_name = f"Copernicus_DSM_COG_30_{ns}{abs_lat:02d}_00_{ew}{abs_lon:03d}_00_DEM"
            return f"https://copernicus-dem-90m.s3.amazonaws.com/{tile_name}/{tile_name}.tif"
        elif source == "srtm":
            tile_id = f"{ns}{abs_lat:02d}{ew}{abs_lon:03d}"
            return (
                f"https://elevation-tiles-prod.s3.amazonaws.com"
                f"/skadi/{ns}{abs_lat:02d}/{tile_id}.hgt.gz"
            )
        elif source == "3dep":
            n_s = "n" if lat >= 0 else "s"
            e_w = "e" if lon >= 0 else "w"
            tile_id = f"{n_s}{abs_lat:02d}{e_w}{abs_lon:03d}"
            return (
                f"https://prd-tnm.s3.amazonaws.com"
                f"/StagedProducts/Elevation/13/TIFF/current"
                f"/{tile_id}/USGS_13_{tile_id}.tif"
            )
        elif source == "fabdem":
            tile_id = f"{ns}{abs_lat:02d}{ew}{abs_lon:03d}"
            return (
                f"https://data.bris.ac.uk/data/dataset"
                f"/s5hqmjcdj8yo2ibzi9b4ew3sn"
                f"/{tile_id}_FABDEM_V1-2.tif"
            )
        elif source == "aster":
            return None  # Requires NASA Earthdata auth
        else:
            return None

    def _get_store(self) -> Any:
        """Get the artifact store instance."""
        from chuk_mcp_server import get_artifact_store

        store = get_artifact_store()
        if store is None:
            raise RuntimeError(ErrorMessages.NO_ARTIFACT_STORE)
        return store

    async def _store_raster(
        self,
        data: bytes,
        metadata: dict,
        suffix: str = ".tif",
    ) -> str:
        """Store raster data in the artifact store."""
        try:
            store = self._get_store()
            mime = "image/tiff" if suffix.endswith(".tif") else "image/png"
            filename = f"dem_{uuid.uuid4().hex[:12]}{suffix}"

            ref = await store.store(
                data,
                mime=mime,
                meta=metadata,
                filename=filename,
                summary=f"DEM data ({metadata.get('type', 'unknown')})",
            )
            return ref
        except Exception as e:
            logger.error(f"Failed to store raster: {e}")
            raise

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _cache_tile(self, key: str, data: bytes) -> None:
        """Cache tile data with LRU eviction."""
        size = len(data)
        if size > TILE_CACHE_MAX_ITEM:
            return

        while self._tile_cache_total + size > TILE_CACHE_MAX_BYTES and self._tile_cache:
            oldest_key = next(iter(self._tile_cache))
            evicted_size = self._tile_cache_sizes.pop(oldest_key, 0)
            del self._tile_cache[oldest_key]
            self._tile_cache_total -= evicted_size

        self._tile_cache[key] = data
        self._tile_cache_sizes[key] = size
        self._tile_cache_total += size

    def _get_cached_tile(self, key: str) -> bytes | None:
        """Get cached tile data, moving to end of LRU."""
        if key not in self._tile_cache:
            return None
        data = self._tile_cache.pop(key)
        size = self._tile_cache_sizes.pop(key)
        self._tile_cache[key] = data
        self._tile_cache_sizes[key] = size
        return data
