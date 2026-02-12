"""
Analysis tools — terrain derivatives, elevation profile, viewshed,
and Phase 3.0 ML-enhanced terrain analysis.
"""

import logging

from ...constants import (
    DEFAULT_ALTITUDE,
    DEFAULT_ANOMALY_SENSITIVITY,
    DEFAULT_AZIMUTH,
    DEFAULT_CONTOUR_INTERVAL_M,
    DEFAULT_FLAT_VALUE,
    DEFAULT_INTERPOLATION,
    DEFAULT_NUM_POINTS,
    DEFAULT_OBSERVER_HEIGHT_M,
    DEFAULT_SIGNIFICANCE_THRESHOLD_M,
    DEFAULT_SOURCE,
    DEFAULT_Z_FACTOR,
    FEATURE_METHODS,
    INTERPRETATION_CONTEXTS,
    INTERPOLATION_METHODS,
    LANDFORM_METHODS,
    MAX_VIEWSHED_RADIUS_M,
    OUTPUT_FORMATS,
    SLOPE_UNITS,
    ErrorMessages,
    SuccessMessages,
    get_license_warning,
)
from ...models.responses import (
    AnomalyResponse,
    AspectResponse,
    ChangeRegion,
    ContourResponse,
    CurvatureResponse,
    ErrorResponse,
    FeatureDetectionResponse,
    HillshadeResponse,
    InterpretResponse,
    LandformResponse,
    ProfilePointInfo,
    ProfileResponse,
    TerrainFeature,
    SlopeResponse,
    TemporalChangeResponse,
    TerrainAnomaly,
    TRIResponse,
    ViewshedResponse,
    WatershedResponse,
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
                license_warning=get_license_warning(source),
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
                license_warning=get_license_warning(source),
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
                license_warning=get_license_warning(source),
                message=SuccessMessages.ASPECT_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}", flat_value
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_aspect failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_curvature(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Compute surface curvature (profile curvature) for a bounding box.

        Curvature measures the rate of change of slope. Positive values indicate
        convex surfaces (ridges), negative values indicate concave surfaces (valleys),
        and near-zero values indicate planar surfaces.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Curvature raster artifact with value range in 1/m
        """
        try:
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_curvature(
                bbox=bbox,
                source=source,
                output_format=output_format,
            )

            response = CurvatureResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                value_range=result.value_range,
                output_format=output_format,
                license_warning=get_license_warning(source),
                message=SuccessMessages.CURVATURE_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}"
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_curvature failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_terrain_ruggedness(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Compute Terrain Ruggedness Index (TRI) for a bounding box.

        TRI measures the mean absolute elevation difference between a cell and
        its 8 neighbours (Riley et al. 1999). Values are in metres.

        Classification: 0-80m level, 81-116m nearly level, 117-161m slightly
        rugged, 162-239m intermediately rugged, 240-497m moderately rugged,
        498-958m highly rugged, 959+m extremely rugged.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            TRI raster artifact with values in metres
        """
        try:
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_tri(
                bbox=bbox,
                source=source,
                output_format=output_format,
            )

            response = TRIResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                value_range=result.value_range,
                output_format=output_format,
                license_warning=get_license_warning(source),
                message=SuccessMessages.TRI_COMPLETE.format(f"{result.shape[0]}x{result.shape[1]}"),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_terrain_ruggedness failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_contour(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        interval_m: float = DEFAULT_CONTOUR_INTERVAL_M,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Generate contour lines at specified elevation intervals for a bounding box.

        Contour lines are rasterised: contour pixels contain their elevation level
        value, non-contour pixels are NaN. PNG output shows contours overlaid on
        a terrain-coloured background.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            interval_m: Elevation interval between contour lines in metres (default 100)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Contour raster artifact with contour count and elevation range
        """
        try:
            if interval_m <= 0:
                raise ValueError(ErrorMessages.INVALID_CONTOUR_INTERVAL.format(interval_m))
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_contours(
                bbox=bbox,
                source=source,
                interval_m=interval_m,
                output_format=output_format,
            )

            response = ContourResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                interval_m=result.interval_m,
                contour_count=result.contour_count,
                elevation_range=result.elevation_range,
                output_format=output_format,
                license_warning=get_license_warning(source),
                message=SuccessMessages.CONTOUR_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}",
                    interval_m,
                    result.contour_count,
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_contour failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_watershed(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Compute watershed flow accumulation for a bounding box.

        Uses the D8 algorithm to compute flow direction and accumulation.
        High accumulation values indicate streams and drainage channels.
        Useful for hydrological analysis and flood risk assessment.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Flow accumulation raster artifact with value range in cells
        """
        try:
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_watershed(
                bbox=bbox,
                source=source,
                output_format=output_format,
            )

            response = WatershedResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                value_range=result.value_range,
                output_format=output_format,
                license_warning=get_license_warning(source),
                message=SuccessMessages.WATERSHED_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}", result.value_range[1]
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_watershed failed: {e}")
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
                license_warning=get_license_warning(source),
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
                license_warning=get_license_warning(source),
                message=SuccessMessages.VIEWSHED_COMPLETE.format(
                    result.visible_percentage, result.radius_m
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_viewshed failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    # ------------------------------------------------------------------
    # Phase 3.0: ML-enhanced terrain analysis tools
    # ------------------------------------------------------------------

    @mcp.tool()
    async def dem_compare_temporal(
        bbox: list[float],
        before_source: str = "srtm",
        after_source: str = "cop30",
        significance_threshold_m: float = DEFAULT_SIGNIFICANCE_THRESHOLD_M,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Compare elevation between two DEM sources/epochs to detect change.

        Computes the difference (after - before) and identifies significant
        change regions. Useful for detecting erosion, deposition, construction,
        or mining activity.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            before_source: DEM source for earlier epoch (default: srtm, year 2000)
            after_source: DEM source for later epoch (default: cop30, year 2021)
            significance_threshold_m: Minimum change to flag (default 1.0m)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Change map with volume statistics and significant regions
        """
        try:
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_temporal_change(
                bbox=bbox,
                before_source=before_source,
                after_source=after_source,
                significance_threshold_m=significance_threshold_m,
                output_format=output_format,
            )

            regions = [ChangeRegion(**r) for r in result.significant_regions]

            response = TemporalChangeResponse(
                before_source=before_source,
                after_source=after_source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                significance_threshold_m=significance_threshold_m,
                volume_gained_m3=result.volume_gained_m3,
                volume_lost_m3=result.volume_lost_m3,
                significant_regions=regions,
                output_format=output_format,
                license_warning=get_license_warning(after_source),
                message=SuccessMessages.TEMPORAL_CHANGE_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}",
                    result.volume_gained_m3,
                    result.volume_lost_m3,
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_compare_temporal failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_classify_landforms(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        method: str = "rule_based",
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Classify terrain into landform types (ridge, valley, plateau, etc.).

        Rule-based classification using slope, curvature, and terrain ruggedness.
        Landform classes: plain, ridge, valley, plateau, escarpment, depression,
        saddle, terrace, alluvial_fan.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            method: Classification method (currently "rule_based" only)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Landform classification map with class distribution
        """
        try:
            if method not in LANDFORM_METHODS:
                raise ValueError(
                    ErrorMessages.INVALID_LANDFORM_METHOD.format(
                        method, ", ".join(LANDFORM_METHODS)
                    )
                )
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_landforms(
                bbox=bbox,
                source=source,
                method=method,
                output_format=output_format,
            )

            response = LandformResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                method=method,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                class_distribution=result.class_distribution,
                dominant_landform=result.dominant_landform,
                output_format=output_format,
                license_warning=get_license_warning(source),
                message=SuccessMessages.LANDFORM_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}", result.dominant_landform
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_classify_landforms failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_detect_anomalies(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        sensitivity: float = DEFAULT_ANOMALY_SENSITIVITY,
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Detect terrain anomalies that don't fit the natural landscape.

        Uses Isolation Forest on slope/curvature/TRI feature vectors to find
        outlier terrain. Catches anthropogenic features, data artefacts, or
        unusual geological features.

        Requires scikit-learn: pip install chuk-mcp-dem[ml]

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            sensitivity: Detection sensitivity 0-1 (lower = fewer, default 0.1)
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Anomaly score map with detected anomaly regions
        """
        try:
            if not 0 < sensitivity < 1:
                raise ValueError(ErrorMessages.INVALID_SENSITIVITY.format(sensitivity))
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_anomalies(
                bbox=bbox,
                source=source,
                sensitivity=sensitivity,
                output_format=output_format,
            )

            anomaly_models = [TerrainAnomaly(**a) for a in result.anomalies]

            response = AnomalyResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                sensitivity=sensitivity,
                anomaly_count=result.anomaly_count,
                anomalies=anomaly_models,
                output_format=output_format,
                license_warning=get_license_warning(source),
                message=SuccessMessages.ANOMALY_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}", result.anomaly_count
                ),
            )
            return format_response(response, output_mode)

        except ImportError:
            return format_response(
                ErrorResponse(error=ErrorMessages.SKLEARN_NOT_AVAILABLE), output_mode
            )
        except Exception as e:
            logger.error(f"dem_detect_anomalies failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    @mcp.tool()
    async def dem_detect_features(
        bbox: list[float],
        source: str = DEFAULT_SOURCE,
        method: str = "cnn_hillshade",
        output_format: str = "geotiff",
        output_mode: str = "json",
    ) -> str:
        """Detect discrete terrain features using CNN-inspired multi-angle hillshade.

        Generates 8 hillshades at different sun angles, applies Sobel edge filters
        to each (mimicking CNN convolutional layers), then classifies pixels into
        feature types based on edge consensus, slope, and curvature.

        Feature types: peak, ridge, valley, cliff, saddle, channel.

        Args:
            bbox: Bounding box [west, south, east, north] in EPSG:4326
            source: DEM source (cop30, cop90, srtm, aster, 3dep, fabdem)
            method: Detection method (currently "cnn_hillshade")
            output_format: Output format (geotiff or png)
            output_mode: "json" or "text"

        Returns:
            Feature map artifact with detected feature list and summary
        """
        try:
            if method not in FEATURE_METHODS:
                raise ValueError(
                    ErrorMessages.INVALID_FEATURE_METHOD.format(
                        method, ", ".join(FEATURE_METHODS)
                    )
                )
            if output_format not in OUTPUT_FORMATS:
                raise ValueError(
                    ErrorMessages.INVALID_OUTPUT_FORMAT.format(
                        output_format, ", ".join(OUTPUT_FORMATS)
                    )
                )

            result = await manager.fetch_features(
                bbox=bbox,
                source=source,
                method=method,
                output_format=output_format,
            )

            feature_models = [TerrainFeature(**f) for f in result.features]

            response = FeatureDetectionResponse(
                source=source,
                bbox=bbox,
                artifact_ref=result.artifact_ref,
                preview_ref=result.preview_ref,
                method=method,
                crs=result.crs,
                resolution_m=result.resolution_m,
                shape=result.shape,
                feature_count=result.feature_count,
                feature_summary=result.feature_summary,
                features=feature_models,
                output_format=output_format,
                license_warning=get_license_warning(source),
                message=SuccessMessages.FEATURE_DETECT_COMPLETE.format(
                    f"{result.shape[0]}x{result.shape[1]}", result.feature_count
                ),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_detect_features failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)

    # ------------------------------------------------------------------
    # Phase 3.1: LLM terrain interpretation via MCP sampling
    # ------------------------------------------------------------------

    @mcp.tool()
    async def dem_interpret(
        artifact_ref: str,
        context: str = "general",
        question: str = "",
        output_mode: str = "json",
    ) -> str:
        """Interpret a terrain artifact using the calling LLM via MCP sampling.

        Sends a rendered PNG of any terrain artifact to the client LLM for
        natural language interpretation. The LLM receives the image and a
        context-specific prompt, then returns its analysis.

        Requires an MCP client that supports the sampling/createMessage
        capability (e.g., Claude Desktop).

        Available contexts: general, archaeological_survey, flood_risk,
        geological, military_history, urban_planning.

        Args:
            artifact_ref: Reference to any stored terrain artifact
            context: Interpretation context (default "general")
            question: Optional specific question about the terrain
            output_mode: "json" or "text"

        Returns:
            LLM interpretation with identified features
        """
        try:
            # Validate context
            if context not in INTERPRETATION_CONTEXTS:
                raise ValueError(
                    ErrorMessages.INVALID_INTERPRETATION_CONTEXT.format(
                        context, ", ".join(INTERPRETATION_CONTEXTS)
                    )
                )

            # Prepare PNG from artifact
            try:
                prep = await manager.prepare_for_interpretation(artifact_ref)
            except Exception as e:
                raise ValueError(
                    ErrorMessages.INVALID_ARTIFACT_REF.format(artifact_ref)
                ) from e

            # Base64-encode the PNG
            import base64
            png_b64 = base64.b64encode(prep.png_bytes).decode()

            # Build system context
            context_descriptions = {
                "general": "terrain and elevation analysis",
                "archaeological_survey": (
                    "archaeological landscape survey, identifying earthworks, "
                    "barrows, enclosures, field systems, and anthropogenic terrain "
                    "modifications"
                ),
                "flood_risk": (
                    "flood risk assessment, identifying drainage channels, "
                    "flood plains, catchment areas, and vulnerable low-lying terrain"
                ),
                "geological": (
                    "geological analysis, identifying rock formations, fault lines, "
                    "erosion patterns, and geomorphological processes"
                ),
                "military_history": (
                    "military terrain analysis, identifying defensive positions, "
                    "line-of-sight advantages, natural barriers, and strategic terrain"
                ),
                "urban_planning": (
                    "urban planning and development, identifying buildable terrain, "
                    "slope constraints, drainage patterns, and natural hazards"
                ),
            }
            system_prompt = (
                f"You are a terrain analysis expert specialising in "
                f"{context_descriptions.get(context, context)}. "
                f"Analyse the terrain image provided and describe what you observe. "
                f"Identify notable features, patterns, and any anomalies."
            )

            # Build user content
            user_text = "Analyse this terrain image."
            if question:
                user_text = question

            # Add metadata context if available
            meta = prep.artifact_metadata
            if meta.get("type"):
                user_text += f"\n\nArtifact type: {meta['type']}"
            if meta.get("source"):
                user_text += f", Source: {meta['source']}"
            if meta.get("bbox"):
                user_text += f", Bbox: {meta['bbox']}"

            # Call MCP sampling
            try:
                from mcp.server.lowlevel.server import request_ctx
                from mcp.types import (
                    ImageContent,
                    SamplingMessage,
                    TextContent,
                )

                messages = [
                    SamplingMessage(
                        role="user",
                        content=TextContent(type="text", text=user_text),
                    ),
                    SamplingMessage(
                        role="user",
                        content=ImageContent(
                            type="image",
                            data=png_b64,
                            mimeType="image/png",
                        ),
                    ),
                ]

                ctx = request_ctx.get()
                result = await ctx.session.create_message(
                    messages=messages,
                    max_tokens=2000,
                    system_prompt=system_prompt,
                )

                # Extract text from result
                interpretation = ""
                model_name = getattr(result, "model", "unknown")
                if hasattr(result, "content"):
                    if isinstance(result.content, str):
                        interpretation = result.content
                    elif hasattr(result.content, "text"):
                        interpretation = result.content.text
                    elif isinstance(result.content, list):
                        parts = []
                        for block in result.content:
                            if hasattr(block, "text"):
                                parts.append(block.text)
                        interpretation = "\n".join(parts)

            except (LookupError, ImportError):
                return format_response(
                    ErrorResponse(error=ErrorMessages.SAMPLING_NOT_SUPPORTED),
                    output_mode,
                )

            response = InterpretResponse(
                artifact_ref=artifact_ref,
                context=context,
                question=question or None,
                interpretation=interpretation,
                model=str(model_name),
                features_identified=[],
                message=SuccessMessages.INTERPRET_COMPLETE.format(artifact_ref),
            )
            return format_response(response, output_mode)

        except Exception as e:
            logger.error(f"dem_interpret failed: {e}")
            return format_response(ErrorResponse(error=str(e)), output_mode)
