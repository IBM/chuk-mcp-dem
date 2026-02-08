"""
Download tools — DEM fetch, point elevation, coverage check, size estimation.

These tools perform network I/O to download DEM tiles and store results
in the artifact store.
"""

import logging

from ...constants import (
    DEFAULT_INTERPOLATION,
    DEFAULT_SOURCE,
    INTERPOLATION_METHODS,
    ErrorMessages,
    SuccessMessages,
)
from ...models.responses import (
    CoverageCheckResponse,
    ErrorResponse,
    FetchResponse,
    MultiPointResponse,
    PointElevationResponse,
    PointInfo,
    SizeEstimateResponse,
    format_response,
)

logger = logging.getLogger(__name__)


def register_download_tools(mcp, manager):
    """Register download tools with the MCP server."""

    @mcp.tool()
    async def dem_fetch(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        resolution_m: float | None = None,
        output_crs: str | None = None,
        fill_voids: bool = True,
        output_mode: str = "json",
    ) -> str:
        """Download elevation data for a bounding box. Returns a GeoTIFF artifact
        with auto-generated hillshade PNG preview.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            resolution_m: Target resolution in metres (None = native)
            output_crs: Output CRS (None = native EPSG:4326)
            fill_voids: Interpolate nodata pixels (default True)
            output_mode: "json" or "text"

        Returns:
            Artifact reference with elevation range and preview
        """
        try:
            result = await manager.fetch_elevation(
                bbox=bbox,
                source=source,
                resolution_m=resolution_m,
                output_crs=output_crs,
                fill_voids=fill_voids,
            )

            estimated_mb = (result.shape[0] * result.shape[1] * 4) / (1024 * 1024)

            response = FetchResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                crs=result.crs,
                resolution_m=int(result.resolution_m),
                shape=result.shape,
                elevation_range=result.elevation_range,
                dtype=result.dtype,
                nodata_pixels=result.nodata_pixels,
                message=SuccessMessages.FETCH_COMPLETE.format(estimated_mb),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_fetch failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_fetch_point(
        lon: float,
        lat: float,
        source: str = DEFAULT_SOURCE,
        interpolation: str = DEFAULT_INTERPOLATION,
        output_mode: str = "json",
    ) -> str:
        """Get elevation at a single geographic point.

        Args:
            lon: Longitude (-180 to 180)
            lat: Latitude (-90 to 90)
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            interpolation: Sampling method (nearest, bilinear, cubic)
            output_mode: "json" or "text"

        Returns:
            Elevation in metres with uncertainty estimate
        """
        try:
            if interpolation not in INTERPOLATION_METHODS:
                raise ValueError(
                    ErrorMessages.INVALID_INTERPOLATION.format(
                        interpolation, ", ".join(INTERPOLATION_METHODS)
                    )
                )

            result = await manager.fetch_point(
                lon=lon,
                lat=lat,
                source=source,
                interpolation=interpolation,
            )

            response = PointElevationResponse(
                lon=lon,
                lat=lat,
                source=source,
                elevation_m=result.elevation_m,
                interpolation=interpolation,
                uncertainty_m=result.uncertainty_m,
                message=SuccessMessages.POINT_ELEVATION.format(
                    result.elevation_m, result.uncertainty_m
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_fetch_point failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_fetch_points(
        points: list[list[float]],
        source: str = DEFAULT_SOURCE,
        interpolation: str = DEFAULT_INTERPOLATION,
        output_mode: str = "json",
    ) -> str:
        """Get elevations at multiple geographic points in a single request.

        Args:
            points: List of [lon, lat] coordinate pairs
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            interpolation: Sampling method (nearest, bilinear, cubic)
            output_mode: "json" or "text"

        Returns:
            Elevation for each point with range statistics
        """
        try:
            if interpolation not in INTERPOLATION_METHODS:
                raise ValueError(
                    ErrorMessages.INVALID_INTERPOLATION.format(
                        interpolation, ", ".join(INTERPOLATION_METHODS)
                    )
                )

            result = await manager.fetch_points(
                points=points,
                source=source,
                interpolation=interpolation,
            )

            point_infos = [
                PointInfo(lon=p[0], lat=p[1], elevation_m=e)
                for p, e in zip(points, result.elevations)
            ]

            response = MultiPointResponse(
                source=source,
                point_count=len(points),
                points=point_infos,
                elevation_range=result.elevation_range,
                interpolation=interpolation,
                message=SuccessMessages.POINTS_ELEVATION.format(len(points)),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_fetch_points failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_check_coverage(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        output_mode: str = "json",
    ) -> str:
        """Check if a DEM source fully covers a bounding box before downloading.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source to check
            output_mode: "json" or "text"

        Returns:
            Coverage percentage, tile count, and estimated download size
        """
        try:
            result = manager.check_coverage(bbox, source)

            if result["fully_covered"]:
                msg = SuccessMessages.COVERAGE_FULL
            else:
                msg = SuccessMessages.COVERAGE_PARTIAL.format(result["coverage_percentage"])

            response = CoverageCheckResponse(
                source=source,
                bbox=bbox,
                fully_covered=result["fully_covered"],
                coverage_percentage=result["coverage_percentage"],
                tiles_required=result["tiles_required"],
                tile_ids=result["tile_ids"],
                estimated_size_mb=result["estimated_size_mb"],
                message=msg,
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_check_coverage failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_estimate_size(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        resolution_m: float | None = None,
        output_mode: str = "json",
    ) -> str:
        """Estimate download size for a DEM area before fetching.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source
            resolution_m: Target resolution (None = native)
            output_mode: "json" or "text"

        Returns:
            Estimated dimensions, pixel count, and download size
        """
        try:
            result = manager.estimate_size(bbox, source, resolution_m)

            megapixels = result["pixels"] / 1_000_000

            response = SizeEstimateResponse(
                source=source,
                bbox=bbox,
                native_resolution_m=result["native_resolution_m"],
                target_resolution_m=int(result["target_resolution_m"]),
                dimensions=result["dimensions"],
                pixels=result["pixels"],
                dtype=result["dtype"],
                estimated_bytes=result["estimated_bytes"],
                estimated_mb=result["estimated_mb"],
                warning=result["warning"],
                message=SuccessMessages.SIZE_ESTIMATE.format(result["estimated_mb"], megapixels),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_estimate_size failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)
