"""
Response models for chuk-mcp-dem tools.

All tool responses are Pydantic models for type safety and consistent API.
"""

from pydantic import BaseModel, ConfigDict, Field


def format_response(model: BaseModel, output_mode: str = "json") -> str:
    """Format a response model as JSON or human-readable text.

    Args:
        model: Pydantic response model instance
        output_mode: "json" (default) or "text"

    Returns:
        Formatted string
    """
    if output_mode == "text" and hasattr(model, "to_text"):
        return str(model.to_text())
    return str(model.model_dump_json())


class ErrorResponse(BaseModel):
    """Error response model for tool failures."""

    model_config = ConfigDict(extra="forbid")

    error: str = Field(..., description="Error message describing what went wrong")

    def to_text(self) -> str:
        return f"Error: {self.error}"


class SourceInfo(BaseModel):
    """Summary information about a DEM source."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Source identifier (e.g., cop30)")
    name: str = Field(..., description="Human-readable source name")
    resolution_m: int = Field(..., description="Native resolution in metres")
    coverage: str = Field(..., description="Coverage description (e.g., global, USA)")
    vertical_datum: str = Field(..., description="Vertical datum (e.g., EGM2008)")
    void_filled: bool = Field(..., description="Whether voids have been filled")

    def to_text(self) -> str:
        voids = "void-filled" if self.void_filled else "may have voids"
        return f"{self.id}: {self.name} ({self.resolution_m}m, {self.coverage}, {voids})"


class SourcesResponse(BaseModel):
    """Response model for listing available DEM sources."""

    model_config = ConfigDict(extra="forbid")

    sources: list[SourceInfo] = Field(..., description="Available DEM sources")
    default: str = Field(..., description="Default source identifier")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        lines = [self.message, f"Default: {self.default}", ""]
        for s in self.sources:
            lines.append(f"  {s.to_text()}")
        return "\n".join(lines)


class SourceDetailResponse(BaseModel):
    """Response model for detailed DEM source description."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Source identifier")
    name: str = Field(..., description="Human-readable source name")
    resolution_m: int = Field(..., description="Native resolution in metres")
    coverage: str = Field(..., description="Coverage description")
    coverage_bounds: list[float] = Field(
        ..., description="Coverage bounding box [west, south, east, north]"
    )
    vertical_datum: str = Field(..., description="Vertical datum")
    vertical_unit: str = Field(..., description="Vertical measurement unit")
    horizontal_crs: str = Field(..., description="Horizontal coordinate reference system")
    tile_size_degrees: float = Field(..., description="Tile size in degrees")
    dtype: str = Field(..., description="Data type (e.g., float32, int16)")
    nodata_value: float = Field(..., description="NoData sentinel value")
    void_filled: bool = Field(..., description="Whether voids have been filled")
    acquisition_period: str = Field(..., description="Data acquisition period")
    source_sensors: list[str] = Field(..., description="Source sensor names")
    accuracy_vertical_m: float = Field(..., description="Vertical accuracy in metres")
    access_url: str = Field(..., description="Data access URL")
    license: str = Field(..., description="Data license")
    llm_guidance: str = Field(..., description="LLM-friendly usage guidance")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        voids = "void-filled" if self.void_filled else "may have voids"
        bounds = ", ".join(f"{b:.1f}" for b in self.coverage_bounds)
        lines = [
            f"{self.name} ({self.id})",
            f"Resolution: {self.resolution_m}m",
            f"Coverage: {self.coverage} [{bounds}]",
            f"Vertical datum: {self.vertical_datum} ({self.vertical_unit})",
            f"CRS: {self.horizontal_crs}",
            f"Data type: {self.dtype}, nodata: {self.nodata_value}",
            f"Accuracy: +/-{self.accuracy_vertical_m}m vertical",
            f"Voids: {voids}",
            f"Sensors: {', '.join(self.source_sensors)}",
            f"Period: {self.acquisition_period}",
            f"License: {self.license}",
            f"Guidance: {self.llm_guidance}",
        ]
        return "\n".join(lines)


class CoverageCheckResponse(BaseModel):
    """Response model for checking DEM coverage over an area."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source checked")
    bbox: list[float] = Field(..., description="Requested bounding box [west, south, east, north]")
    fully_covered: bool = Field(..., description="Whether the bbox is fully covered")
    coverage_percentage: float = Field(..., description="Percentage of bbox covered", ge=0, le=100)
    tiles_required: int = Field(..., description="Number of tiles needed", ge=0)
    tile_ids: list[str] = Field(..., description="Tile identifiers required")
    estimated_size_mb: float = Field(..., description="Estimated download size in megabytes")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        status = (
            "fully covered" if self.fully_covered else f"{self.coverage_percentage:.1f}% covered"
        )
        bbox_str = ", ".join(f"{b:.4f}" for b in self.bbox)
        lines = [
            f"Coverage check: {self.source}",
            f"Area: [{bbox_str}]",
            f"Status: {status}",
            f"Tiles required: {self.tiles_required}",
            f"Estimated size: {self.estimated_size_mb:.1f} MB",
        ]
        if self.tile_ids:
            lines.append(f"Tile IDs: {', '.join(self.tile_ids[:10])}")
            if len(self.tile_ids) > 10:
                lines.append(f"  ... and {len(self.tile_ids) - 10} more")
        return "\n".join(lines)


class SizeEstimateResponse(BaseModel):
    """Response model for download size estimation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source")
    bbox: list[float] = Field(..., description="Bounding box [west, south, east, north]")
    native_resolution_m: int = Field(..., description="Native source resolution in metres")
    target_resolution_m: int = Field(..., description="Target output resolution in metres")
    dimensions: list[int] = Field(..., description="Output dimensions [width, height]")
    pixels: int = Field(..., description="Total pixel count")
    dtype: str = Field(..., description="Data type")
    estimated_bytes: int = Field(..., description="Estimated size in bytes")
    estimated_mb: float = Field(..., description="Estimated size in megabytes")
    warning: str | None = Field(None, description="Size warning if area is large")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        dims = f"{self.dimensions[0]}x{self.dimensions[1]}"
        megapixels = self.pixels / 1_000_000
        lines = [
            f"Size estimate: {self.source}",
            f"Dimensions: {dims} ({megapixels:.1f} megapixels)",
            f"Resolution: {self.target_resolution_m}m (native: {self.native_resolution_m}m)",
            f"Data type: {self.dtype}",
            f"Estimated size: {self.estimated_mb:.1f} MB",
        ]
        if self.warning:
            lines.append(f"WARNING: {self.warning}")
        return "\n".join(lines)


class FetchResponse(BaseModel):
    """Response model for DEM data fetch/download."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    bbox: list[float] = Field(..., description="Fetched bounding box [west, south, east, north]")
    artifact_ref: str = Field(..., description="Artifact store reference for the data")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    crs: str = Field(..., description="Coordinate reference system of output")
    resolution_m: int = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    elevation_range: list[float] = Field(..., description="[min, max] elevation in metres")
    dtype: str = Field(..., description="Data type of the array")
    nodata_pixels: int = Field(..., description="Number of nodata pixels", ge=0)
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        elev_min, elev_max = self.elevation_range
        lines = [
            f"Fetched DEM: {self.source}",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Elevation range: {elev_min:.1f}m to {elev_max:.1f}m",
            f"Data type: {self.dtype}",
            f"NoData pixels: {self.nodata_pixels}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class PointElevationResponse(BaseModel):
    """Response model for single-point elevation query."""

    model_config = ConfigDict(extra="forbid")

    lon: float = Field(..., description="Longitude of the query point")
    lat: float = Field(..., description="Latitude of the query point")
    source: str = Field(..., description="DEM source used")
    elevation_m: float = Field(..., description="Elevation in metres")
    interpolation: str = Field(..., description="Interpolation method used")
    uncertainty_m: float = Field(..., description="Estimated vertical uncertainty in metres")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        lines = [
            f"Elevation at ({self.lon:.6f}, {self.lat:.6f}): {self.elevation_m:.1f}m",
            f"Source: {self.source}",
            f"Interpolation: {self.interpolation}",
            f"Uncertainty: +/-{self.uncertainty_m:.1f}m",
        ]
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class PointInfo(BaseModel):
    """Elevation data for a single point in a multi-point query."""

    model_config = ConfigDict(extra="forbid")

    lon: float = Field(..., description="Longitude")
    lat: float = Field(..., description="Latitude")
    elevation_m: float = Field(..., description="Elevation in metres")


class MultiPointResponse(BaseModel):
    """Response model for multi-point elevation query."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    point_count: int = Field(..., description="Number of points queried", ge=1)
    points: list[PointInfo] = Field(..., description="Elevation results per point")
    elevation_range: list[float] = Field(..., description="[min, max] elevation across all points")
    interpolation: str = Field(..., description="Interpolation method used")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        elev_min, elev_max = self.elevation_range
        lines = [
            f"Elevation for {self.point_count} point(s) from {self.source}",
            f"Interpolation: {self.interpolation}",
            f"Range: {elev_min:.1f}m to {elev_max:.1f}m",
            "",
        ]
        for p in self.points:
            lines.append(f"  ({p.lon:.6f}, {p.lat:.6f}): {p.elevation_m:.1f}m")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class StatusResponse(BaseModel):
    """Response model for server status queries."""

    model_config = ConfigDict(extra="forbid")

    server: str = Field(default="chuk-mcp-dem", description="Server name")
    version: str = Field(default="0.1.0", description="Server version")
    default_source: str = Field(..., description="Default DEM source")
    available_sources: list[str] = Field(..., description="Available DEM source identifiers")
    storage_provider: str = Field(..., description="Active storage provider (memory/filesystem/s3)")
    artifact_store_available: bool = Field(
        default=False, description="Whether artifact store is available"
    )
    cache_size_mb: float = Field(default=0.0, description="Current tile cache size in megabytes")

    def to_text(self) -> str:
        store_status = "available" if self.artifact_store_available else "not available"
        lines = [
            f"{self.server} v{self.version}",
            f"Default source: {self.default_source}",
            f"Sources: {', '.join(self.available_sources)}",
            f"Storage: {self.storage_provider}",
            f"Artifact store: {store_status}",
            f"Cache: {self.cache_size_mb:.1f} MB",
        ]
        return "\n".join(lines)


class CapabilitiesResponse(BaseModel):
    """Response model for server capabilities listing."""

    model_config = ConfigDict(extra="forbid")

    server: str = Field(..., description="Server name")
    version: str = Field(..., description="Server version")
    sources: list[SourceInfo] = Field(..., description="Available DEM sources")
    default_source: str = Field(..., description="Default DEM source identifier")
    terrain_derivatives: list[str] = Field(..., description="Available terrain derivative types")
    analysis_tools: list[str] = Field(..., description="Available analysis tool types")
    output_formats: list[str] = Field(..., description="Supported output formats")
    tool_count: int = Field(..., description="Number of available tools", ge=0)
    llm_guidance: str = Field(..., description="LLM-friendly usage guidance for the server")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        lines = [
            f"{self.server} v{self.version}",
            f"Tools: {self.tool_count}",
            f"Default source: {self.default_source}",
            f"Sources: {', '.join(s.id for s in self.sources)}",
            f"Terrain derivatives: {', '.join(self.terrain_derivatives)}",
            f"Analysis tools: {', '.join(self.analysis_tools)}",
            f"Output formats: {', '.join(self.output_formats)}",
            f"Guidance: {self.llm_guidance}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Terrain analysis responses (Phase 1.1)
# ---------------------------------------------------------------------------


class HillshadeResponse(BaseModel):
    """Response model for hillshade computation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    bbox: list[float] = Field(..., description="Bounding box [west, south, east, north]")
    artifact_ref: str = Field(..., description="Artifact store reference for hillshade data")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    azimuth: float = Field(..., description="Sun azimuth in degrees from north")
    altitude: float = Field(..., description="Sun altitude in degrees above horizon")
    z_factor: float = Field(..., description="Vertical exaggeration factor")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: float = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    value_range: list[float] = Field(..., description="[min, max] hillshade values")
    output_format: str = Field(..., description="Output format (geotiff or png)")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        lines = [
            f"Hillshade: {self.source}",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Azimuth: {self.azimuth:.0f}, Altitude: {self.altitude:.0f}",
            f"Z-factor: {self.z_factor}",
            f"Value range: {self.value_range[0]:.1f} to {self.value_range[1]:.1f}",
            f"Format: {self.output_format}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class SlopeResponse(BaseModel):
    """Response model for slope computation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    bbox: list[float] = Field(..., description="Bounding box [west, south, east, north]")
    artifact_ref: str = Field(..., description="Artifact store reference for slope data")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    units: str = Field(..., description="Slope units (degrees or percent)")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: float = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    value_range: list[float] = Field(..., description="[min, max] slope values")
    output_format: str = Field(..., description="Output format (geotiff or png)")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        lines = [
            f"Slope: {self.source} ({self.units})",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Value range: {self.value_range[0]:.1f} to {self.value_range[1]:.1f}",
            f"Format: {self.output_format}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class AspectResponse(BaseModel):
    """Response model for aspect computation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    bbox: list[float] = Field(..., description="Bounding box [west, south, east, north]")
    artifact_ref: str = Field(..., description="Artifact store reference for aspect data")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    flat_value: float = Field(..., description="Value used for flat areas")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: float = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    value_range: list[float] = Field(..., description="[min, max] aspect values")
    output_format: str = Field(..., description="Output format (geotiff or png)")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        lines = [
            f"Aspect: {self.source}",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Flat value: {self.flat_value}",
            f"Value range: {self.value_range[0]:.1f} to {self.value_range[1]:.1f}",
            f"Format: {self.output_format}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class CurvatureResponse(BaseModel):
    """Response model for curvature computation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    bbox: list[float] = Field(..., description="Bounding box [west, south, east, north]")
    artifact_ref: str = Field(..., description="Artifact store reference for curvature data")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: float = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    value_range: list[float] = Field(..., description="[min, max] curvature values (1/m)")
    output_format: str = Field(..., description="Output format (geotiff or png)")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        lines = [
            f"Curvature: {self.source}",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Value range: {self.value_range[0]:.6f} to {self.value_range[1]:.6f} (1/m)",
            f"Format: {self.output_format}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class TRIResponse(BaseModel):
    """Response model for Terrain Ruggedness Index computation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    bbox: list[float] = Field(..., description="Bounding box [west, south, east, north]")
    artifact_ref: str = Field(..., description="Artifact store reference for TRI data")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: float = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    value_range: list[float] = Field(..., description="[min, max] TRI values (metres)")
    output_format: str = Field(..., description="Output format (geotiff or png)")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        lines = [
            f"Terrain Ruggedness: {self.source}",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Value range: {self.value_range[0]:.1f}m to {self.value_range[1]:.1f}m",
            f"Format: {self.output_format}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class ContourResponse(BaseModel):
    """Response model for contour line generation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    bbox: list[float] = Field(..., description="Bounding box [west, south, east, north]")
    artifact_ref: str = Field(..., description="Artifact store reference for contour data")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: float = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    interval_m: float = Field(..., description="Contour interval in metres")
    contour_count: int = Field(..., description="Number of contour levels generated")
    elevation_range: list[float] = Field(..., description="[min, max] elevation in metres")
    output_format: str = Field(..., description="Output format (geotiff or png)")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        elev_min, elev_max = self.elevation_range
        lines = [
            f"Contours: {self.source}",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Interval: {self.interval_m}m ({self.contour_count} levels)",
            f"Elevation: {elev_min:.1f}m to {elev_max:.1f}m",
            f"Format: {self.output_format}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Profile & viewshed responses (Phase 1.2)
# ---------------------------------------------------------------------------


class ProfilePointInfo(BaseModel):
    """Elevation data for a single point in a profile."""

    model_config = ConfigDict(extra="forbid")

    lon: float = Field(..., description="Longitude")
    lat: float = Field(..., description="Latitude")
    distance_m: float = Field(..., description="Distance from start in metres")
    elevation_m: float = Field(..., description="Elevation in metres")


class ProfileResponse(BaseModel):
    """Response model for elevation profile extraction."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    start: list[float] = Field(..., description="Start point [lon, lat]")
    end: list[float] = Field(..., description="End point [lon, lat]")
    num_points: int = Field(..., description="Number of profile points")
    points: list[ProfilePointInfo] = Field(..., description="Profile points with elevation")
    total_distance_m: float = Field(..., description="Total profile distance in metres")
    elevation_range: list[float] = Field(..., description="[min, max] elevation in metres")
    elevation_gain_m: float = Field(..., description="Total elevation gain in metres")
    elevation_loss_m: float = Field(..., description="Total elevation loss in metres")
    interpolation: str = Field(..., description="Interpolation method used")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        elev_min, elev_max = self.elevation_range
        lines = [
            f"Profile: {self.source}",
            f"Start: ({self.start[0]:.6f}, {self.start[1]:.6f})",
            f"End: ({self.end[0]:.6f}, {self.end[1]:.6f})",
            f"Distance: {self.total_distance_m:.1f}m ({self.num_points} points)",
            f"Elevation range: {elev_min:.1f}m to {elev_max:.1f}m",
            f"Gain: {self.elevation_gain_m:.1f}m, Loss: {self.elevation_loss_m:.1f}m",
            f"Interpolation: {self.interpolation}",
        ]
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


class ViewshedResponse(BaseModel):
    """Response model for viewshed analysis."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    observer: list[float] = Field(..., description="Observer point [lon, lat]")
    radius_m: float = Field(..., description="Analysis radius in metres")
    observer_height_m: float = Field(..., description="Observer height above ground in metres")
    observer_elevation_m: float = Field(..., description="Ground elevation at observer in metres")
    artifact_ref: str = Field(..., description="Artifact store reference for viewshed raster")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: float = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    visible_percentage: float = Field(..., description="Percentage of area visible", ge=0, le=100)
    output_format: str = Field(..., description="Output format (geotiff or png)")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        lines = [
            f"Viewshed: {self.source}",
            f"Observer: ({self.observer[0]:.6f}, {self.observer[1]:.6f})",
            f"Radius: {self.radius_m:.0f}m, Height: {self.observer_height_m:.1f}m",
            f"Observer elevation: {self.observer_elevation_m:.1f}m",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Visible: {self.visible_percentage:.1f}%",
            f"Format: {self.output_format}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Watershed response (Phase 2.1)
# ---------------------------------------------------------------------------


class WatershedResponse(BaseModel):
    """Response model for watershed (flow accumulation) computation."""

    model_config = ConfigDict(extra="forbid")

    source: str = Field(..., description="DEM source used")
    bbox: list[float] = Field(..., description="Bounding box [west, south, east, north]")
    artifact_ref: str = Field(..., description="Artifact store reference for watershed data")
    preview_ref: str | None = Field(None, description="PNG preview artifact reference")
    crs: str = Field(..., description="Coordinate reference system")
    resolution_m: float = Field(..., description="Output resolution in metres")
    shape: list[int] = Field(..., description="Array shape [height, width]")
    value_range: list[float] = Field(
        ..., description="[min, max] flow accumulation (contributing cells)"
    )
    output_format: str = Field(..., description="Output format (geotiff or png)")
    license_warning: str | None = Field(None, description="License restriction warning")
    message: str = Field(..., description="Operation result message")

    def to_text(self) -> str:
        shape_str = f"{self.shape[0]}x{self.shape[1]}"
        lines = [
            f"Watershed: {self.source}",
            f"Artifact: {self.artifact_ref}",
            f"Shape: {shape_str} ({self.crs}, {self.resolution_m}m)",
            f"Flow accumulation: {self.value_range[0]:.0f} to {self.value_range[1]:.0f} cells",
            f"Format: {self.output_format}",
        ]
        if self.preview_ref:
            lines.append(f"Preview: {self.preview_ref}")
        if self.license_warning:
            lines.append(f"WARNING: {self.license_warning}")
        return "\n".join(lines)
