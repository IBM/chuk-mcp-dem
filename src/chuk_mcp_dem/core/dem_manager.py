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
    DEFAULT_INTERPOLATION,
    DEFAULT_SOURCE,
    DEM_SOURCES,
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
            ref = f"dem/{uuid.uuid4().hex[:12]}{suffix}"
            mime = "image/tiff" if suffix.endswith(".tif") else "image/png"

            await store.store(
                ref,
                data,
                mime_type=mime,
                metadata=metadata,
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
