"""
Analysis tools — terrain derivatives, elevation profile, and viewshed.

Phase 1.1: hillshade, slope, aspect
Phase 1.2: profile, viewshed
"""

import logging

from ...constants import (
    DEFAULT_ALTITUDE,
    DEFAULT_AZIMUTH,
    DEFAULT_FLAT_VALUE,
    DEFAULT_INTERPOLATION,
    DEFAULT_NUM_POINTS,
    DEFAULT_OBSERVER_HEIGHT_M,
    DEFAULT_SOURCE,
    DEFAULT_Z_FACTOR,
    INTERPOLATION_METHODS,
    MAX_VIEWSHED_RADIUS_M,
    OUTPUT_FORMATS,
    SLOPE_UNITS,
    ErrorMessages,
    SuccessMessages,
)
from ...models.responses import (
    AspectResponse,
    ErrorResponse,
    HillshadeResponse,
    ProfilePointInfo,
    ProfileResponse,
    SlopeResponse,
    ViewshedResponse,
    format_response,
)

logger = logging.getLogger(__name__)


def register_analysis_tools(mcp, manager):
    """Register analysis tools with the MCP server."""

    @mcp.tool()
    async def dem_hillshade(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        azimuth: float = DEFAULT_AZIMUTH,
        altitude: float = DEFAULT_ALTITUDE,
        z_factor: float = DEFAULT_Z_FACTOR,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Compute hillshade (shaded relief) for a bounding box using Horn's method.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            azimuth: Sun azimuth in degrees from north (default 315)
            altitude: Sun altitude in degrees above horizon (default 45)
            z_factor: Vertical exaggeration factor (default 1.0)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Hillshade raster artifact with value range 0-255
        """
        try:
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_hillshade(
                bbox=bbox,
                source=source,
                azimuth=azimuth,
                altitude=altitude,
                z_factor=z_factor,
                output_format=output_format,
            )

            response = HillshadeResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                azimuth=azimuth,
                altitude=altitude,
                z_factor=z_factor,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                value_range=result.value_range,
                output_format=output_format,
                message=SuccessMessages.HILLSHADE_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}", azimuth, altitude
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_hillshade failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_slope(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        units: str = "degrees",
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Compute slope steepness for a bounding box.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            units: Slope units (degrees or percent)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Slope raster artifact with value range
        """
        try:
            if units not in SLOPE_UNITS:
                raise ValueError(
                    ErrorMessages.INVALID_SLOPE_UNITS.format(units, ", ".join(SLOPE_UNITS))
                )
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_slope(
                bbox=bbox,
                source=source,
                units=units,
                output_format=output_format,
            )

            response = SlopeResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                units=units,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                value_range=result.value_range,
                output_format=output_format,
                message=SuccessMessages.SLOPE_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}", units
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_slope failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_aspect(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        flat_value: float = DEFAULT_FLAT_VALUE,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Compute aspect (slope direction) for a bounding box.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            flat_value: Value for flat areas (default -1)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Aspect raster artifact with values in degrees (0-360)
        """
        try:
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_aspect(
                bbox=bbox,
                source=source,
                flat_value=flat_value,
                output_format=output_format,
            )

            response = AspectResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                flat_value=flat_value,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                value_range=result.value_range,
                output_format=output_format,
                message=SuccessMessages.ASPECT_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}", flat_value
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_aspect failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_profile(
        start: list[float],
        end: list[float],
        source: str = DEFAULT_SOURCE,
        num_points: int = DEFAULT_NUM_POINTS,
        interpolation: str = DEFAULT_INTERPOLATION,
        output_mode: str = "json",
    ) -> str:
        """Extract an elevation profile along a line between two points.

        Args:
            start: Start point [lon, lat]
            end: End point [lon, lat]
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            num_points: Number of sample points (default 100, minimum 2)
            interpolation: Sampling method (nearest, bilinear, cubic)
            output_mode: "json" or "text"

        Returns:
            Elevation profile with distance, gain, loss, and per-point data
        """
        try:
            if num_points < 2:
                raise ValueError(ErrorMessages.INVALID_NUM_POINTS.format(num_points))
            if interpolation not in INTERPOLATION_METHODS:
                raise ValueError(
                    ErrorMessages.INVALID_INTERPOLATION.format(
                        interpolation, ", ".join(INTERPOLATION_METHODS)
                    )
                )

            result = await manager.fetch_profile(
                start=start,
                end=end,
                source=source,
                num_points=num_points,
                interpolation=interpolation,
            )

            point_infos = [
                ProfilePointInfo(
                    lon=result.longitudes[i],
                    lat=result.latitudes[i],
                    distance_m=result.distances_m[i],
                    elevation_m=result.elevations[i],
                )
                for i in range(len(result.longitudes))
            ]

            response = ProfileResponse(
                source=source,
                start=start,
                end=end,
                num_points=num_points,
                points=point_infos,
                total_distance_m=result.total_distance_m,
                elevation_range=result.elevation_range,
                elevation_gain_m=result.elevation_gain_m,
                elevation_loss_m=result.elevation_loss_m,
                interpolation=interpolation,
                message=SuccessMessages.PROFILE_COMPLETE.format(
                    num_points, result.total_distance_m
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_profile failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_viewshed(
        observer: list[float],
        radius_m: float,
        source: str = DEFAULT_SOURCE,
        observer_height_m: float = DEFAULT_OBSERVER_HEIGHT_M,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Compute viewshed (visible area) from an observer point.

        Args:
            observer: Observer location [lon, lat]
            radius_m: Analysis radius in metres (max 50000)
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            observer_height_m: Observer height above ground in metres (default 1.8)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Viewshed raster artifact with visible percentage
        """
        try:
            if radius_m <= 0:
                raise ValueError(ErrorMessages.INVALID_RADIUS.format(radius_m))
            if radius_m > MAX_VIEWSHED_RADIUS_M:
                raise ValueError(
                    ErrorMessages.RADIUS_TOO_LARGE.format(radius_m, MAX_VIEWSHED_RADIUS_M)
                )
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_viewshed(
                observer=observer,
                radius_m=radius_m,
                source=source,
                observer_height_m=observer_height_m,
                output_format=output_format,
            )

            response = ViewshedResponse(
                source=source,
                observer=observer,
                radius_m=result.radius_m,
                observer_height_m=observer_height_m,
                observer_elevation_m=result.observer_elevation_m,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                visible_percentage=result.visible_percentage,
                output_format=output_format,
                message=SuccessMessages.VIEWSHED_COMPLETE.format(
                    result.visible_percentage, result.radius_m
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_viewshed failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)
