# chuk-mcp-dem Specification

Version 0.1.0

## Overview

chuk-mcp-dem is an MCP (Model Context Protocol) server that provides digital
elevation model (DEM) discovery, retrieval, and terrain analysis.

- **23 tools** for discovering DEM sources, fetching elevation data, computing terrain derivatives, generating profiles/viewsheds, ML-enhanced terrain analysis, and LLM terrain interpretation
- **Dual output mode** -- all tools return JSON (default) or human-readable text via `output_mode` parameter
- **Async-first** -- tool entry points are async; sync rasterio I/O runs in thread pools
- **Pluggable storage** -- raster data stored via chuk-artifacts (memory, filesystem, S3)

All 23 tools are implemented: discovery (4), download (5), terrain analysis (7), profile/viewshed (2), ML-enhanced analysis (4), and LLM interpretation (1).

---

## DEM Sources

| Source ID | Name | Resolution | Coverage | Vertical Datum | Void-Filled | License |
|-----------|------|------------|----------|----------------|-------------|---------|
| `cop30` (default) | Copernicus GLO-30 | 30m | Global | EGM2008 | Yes | CC-BY-4.0 |
| `cop90` | Copernicus GLO-90 | 90m | Global | EGM2008 | Yes | CC-BY-4.0 |
| `srtm` | SRTM v3 | 30m | 60N-56S | EGM96 | No | Public Domain |
| `aster` | ASTER GDEM v3 | 30m | 83N-83S | EGM96 | No | Public Domain |
| `3dep` | 3DEP | 10m | USA | NAVD88 | Yes | Public Domain |
| `fabdem` | FABDEM | 30m | Global (bare-earth) | EGM2008 | Yes | CC-BY-NC-SA-4.0 |

### Source Selection Guidance

- **General global terrain**: Use `cop30` (default). Void-filled, 30m, good accuracy.
- **Large-area overviews**: Use `cop90`. 9x faster downloads at 90m resolution.
- **US sites requiring detail**: Use `3dep`. LiDAR-derived, 1-10m accuracy.
- **Bare-earth / flood modelling**: Use `fabdem`. Buildings and forests removed.
- **High-latitude coverage**: Use `aster` (83N-83S) over `srtm` (60N-56S).
- **Historical reference**: Use `srtm` for year-2000 baseline comparisons.

---

## Tools

### Common Parameter

All tools accept the following optional parameter:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_mode` | `str` | `json` | Response format: `json` (structured) or `text` (human-readable) |

---

### Discovery Tools

#### `dem_list_sources`

List all available DEM sources with summary metadata.

**Parameters:** `output_mode` only

**Response:** `SourcesResponse`

| Field | Type | Description |
|-------|------|-------------|
| `sources` | `SourceInfo[]` | Available DEM sources |
| `default` | `str` | Default source identifier |
| `message` | `str` | Result message |

---

#### `dem_describe_source`

Get detailed metadata for a specific DEM source.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `source` | `str` | `cop30` | Source identifier (e.g., `cop30`, `srtm`) |

**Response:** `SourceDetailResponse`

| Field | Type | Description |
|-------|------|-------------|
| `id` | `str` | Source identifier |
| `name` | `str` | Human-readable source name |
| `resolution_m` | `int` | Native resolution in metres |
| `coverage` | `str` | Coverage description |
| `coverage_bounds` | `float[]` | Coverage bbox [west, south, east, north] |
| `vertical_datum` | `str` | Vertical datum (e.g., EGM2008) |
| `vertical_unit` | `str` | Vertical measurement unit |
| `horizontal_crs` | `str` | Horizontal CRS |
| `tile_size_degrees` | `float` | Tile size in degrees |
| `dtype` | `str` | Data type (e.g., float32) |
| `nodata_value` | `float` | NoData sentinel value |
| `void_filled` | `bool` | Whether voids have been filled |
| `acquisition_period` | `str` | Data acquisition period |
| `source_sensors` | `str[]` | Source sensor names |
| `accuracy_vertical_m` | `float` | Vertical accuracy in metres |
| `access_url` | `str` | Data access URL |
| `license` | `str` | Data license |
| `llm_guidance` | `str` | LLM-friendly usage guidance |
| `message` | `str` | Result message |

---

#### `dem_status`

Get server status and configuration.

**Parameters:** `output_mode` only

**Response:** `StatusResponse`

| Field | Type | Description |
|-------|------|-------------|
| `server` | `str` | Server name |
| `version` | `str` | Server version |
| `default_source` | `str` | Default DEM source |
| `available_sources` | `str[]` | Available source identifiers |
| `storage_provider` | `str` | Active storage backend |
| `artifact_store_available` | `bool` | Whether store is ready |
| `cache_size_mb` | `float` | Current tile cache size in MB |

---

#### `dem_capabilities`

List full server capabilities for LLM workflow planning.

**Parameters:** `output_mode` only

**Response:** `CapabilitiesResponse`

| Field | Type | Description |
|-------|------|-------------|
| `server` | `str` | Server name |
| `version` | `str` | Server version |
| `sources` | `SourceInfo[]` | Available DEM sources |
| `default_source` | `str` | Default source identifier |
| `terrain_derivatives` | `str[]` | Available terrain derivative types |
| `analysis_tools` | `str[]` | Available analysis tool types |
| `output_formats` | `str[]` | Supported output formats |
| `tool_count` | `int` | Number of registered tools |
| `llm_guidance` | `str` | LLM-friendly usage guidance |
| `message` | `str` | Result message |

---

### Download Tools

#### `dem_fetch`

Download elevation data for a bounding box.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` EPSG:4326 |
| `source` | `str?` | `cop30` | DEM source identifier |
| `resolution_m` | `float?` | `None` | Target resolution (native if omitted) |
| `output_crs` | `str?` | `None` | Output CRS (source CRS if omitted) |
| `fill_voids` | `bool` | `true` | Fill void pixels via nearest-neighbour interpolation |

**Response:** `FetchResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `bbox` | `float[]` | Fetched bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | Hillshade PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `int` | Output resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `elevation_range` | `float[]` | [min, max] elevation in metres |
| `dtype` | `str` | Data type |
| `nodata_pixels` | `int` | Number of nodata pixels |
| `message` | `str` | Result message |

---

#### `dem_fetch_point`

Get elevation at a single geographic coordinate.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `lon` | `float` | *required* | Longitude (EPSG:4326) |
| `lat` | `float` | *required* | Latitude (EPSG:4326) |
| `source` | `str?` | `cop30` | DEM source identifier |
| `interpolation` | `str` | `bilinear` | Interpolation method: `nearest`, `bilinear`, or `cubic` |

**Response:** `PointElevationResponse`

| Field | Type | Description |
|-------|------|-------------|
| `lon` | `float` | Query longitude |
| `lat` | `float` | Query latitude |
| `source` | `str` | DEM source used |
| `elevation_m` | `float` | Elevation in metres |
| `interpolation` | `str` | Interpolation method used |
| `uncertainty_m` | `float` | Estimated vertical uncertainty |
| `message` | `str` | Result message |

---

#### `dem_fetch_points`

Get elevations at multiple geographic coordinates in one call.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `points` | `float[N][2]` | *required* | Nested array of [lon, lat] numeric pairs, e.g. `[[9.19, 45.62], [9.28, 45.63]]`. String and flat-list formats are auto-normalized. |
| `source` | `str?` | `cop30` | DEM source identifier |
| `interpolation` | `str` | `bilinear` | Interpolation method |

**Response:** `MultiPointResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `point_count` | `int` | Number of points queried |
| `points` | `PointInfo[]` | Per-point elevation results |
| `elevation_range` | `float[]` | [min, max] elevation across all points |
| `interpolation` | `str` | Interpolation method used |
| `message` | `str` | Result message |

---

#### `dem_check_coverage`

Check if a DEM source covers a given bounding box and estimate tile requirements.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |

**Response:** `CoverageCheckResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source checked |
| `bbox` | `float[]` | Requested bounding box |
| `fully_covered` | `bool` | Whether the bbox is fully covered |
| `coverage_percentage` | `float` | Percentage of bbox covered |
| `tiles_required` | `int` | Number of tiles needed |
| `tile_ids` | `str[]` | Tile identifiers required |
| `estimated_size_mb` | `float` | Estimated download size in MB |
| `message` | `str` | Result message |

---

#### `dem_estimate_size`

Estimate download size for an area without fetching pixel data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `resolution_m` | `float?` | `None` | Target resolution (native if omitted) |

**Response:** `SizeEstimateResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source |
| `bbox` | `float[]` | Bounding box |
| `native_resolution_m` | `int` | Native source resolution |
| `target_resolution_m` | `int` | Target output resolution |
| `dimensions` | `int[]` | Output dimensions [rows, cols] |
| `pixels` | `int` | Total pixel count |
| `dtype` | `str` | Data type |
| `estimated_bytes` | `int` | Estimated size in bytes |
| `estimated_mb` | `float` | Estimated size in MB |
| `warning` | `str?` | Size warning for large areas |
| `message` | `str` | Result message |

---

### Terrain Analysis Tools

#### `dem_hillshade`

Compute hillshade (shaded relief) from elevation data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `azimuth` | `float` | `315.0` | Sun azimuth in degrees from north |
| `altitude` | `float` | `45.0` | Sun altitude in degrees above horizon |
| `z_factor` | `float` | `1.0` | Vertical exaggeration factor |
| `output_format` | `str` | `geotiff` | Output format |

---

#### `dem_slope`

Compute slope (steepness) from elevation data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `units` | `str` | `degrees` | Slope units: `degrees` or `percent` |
| `output_format` | `str` | `geotiff` | Output format |

---

#### `dem_aspect`

Compute aspect (slope direction) from elevation data.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `flat_value` | `float` | `-1.0` | Value assigned to flat areas |
| `output_format` | `str` | `geotiff` | Output format |

---

#### `dem_curvature`

Compute surface curvature (profile curvature) from elevation data. Positive values
indicate convex surfaces (ridges), negative values indicate concave surfaces (valleys),
and near-zero values indicate planar surfaces. Uses Zevenbergen-Thorne (1987)
second-order finite difference method.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `output_format` | `str` | `geotiff` | Output format |

**Response:** `CurvatureResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `bbox` | `float[]` | Bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `float` | Resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `value_range` | `float[]` | [min, max] curvature in 1/m |
| `output_format` | `str` | Output format used |
| `message` | `str` | Result message |

---

#### `dem_terrain_ruggedness`

Compute Terrain Ruggedness Index (TRI) from elevation data. TRI measures the mean
absolute elevation difference between a cell and its 8 neighbours (Riley et al. 1999).
Values are in metres.

Classification: 0-80m level, 81-116m nearly level, 117-161m slightly rugged,
162-239m intermediately rugged, 240-497m moderately rugged, 498-958m highly rugged,
959+m extremely rugged.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `output_format` | `str` | `geotiff` | Output format |

**Response:** `TRIResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `bbox` | `float[]` | Bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `float` | Resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `value_range` | `float[]` | [min, max] TRI in metres |
| `output_format` | `str` | Output format used |
| `message` | `str` | Result message |

---

#### `dem_contour`

Generate contour lines from elevation data at a specified interval.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `interval_m` | `float` | `100.0` | Contour interval in metres |
| `output_format` | `str` | `geotiff` | Output format |

**Response:** `ContourResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `bbox` | `float[]` | Bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `float` | Resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `interval_m` | `float` | Contour interval used (metres) |
| `contour_count` | `int` | Number of contour levels generated |
| `elevation_range` | `float[]` | [min, max] elevation in metres |
| `output_format` | `str` | Output format used |
| `message` | `str` | Result message |

---

#### `dem_watershed`

Compute flow accumulation (watershed) using the D8 algorithm. High accumulation
values indicate streams and drainage channels. Useful for hydrological analysis
and flood risk assessment.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `output_format` | `str` | `geotiff` | Output format |

**Response:** `WatershedResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `bbox` | `float[]` | Bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `float` | Resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `value_range` | `float[]` | [min, max] flow accumulation in contributing cells |
| `output_format` | `str` | Output format used |
| `license_warning` | `str?` | License restriction warning (FABDEM only) |
| `message` | `str` | Result message |

---

### Profile & Viewshed Tools

#### `dem_profile`

Extract an elevation profile along a line between two points.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `start` | `float[2]` | *required* | Start point [lon, lat] |
| `end` | `float[2]` | *required* | End point [lon, lat] |
| `source` | `str?` | `cop30` | DEM source identifier |
| `num_points` | `int` | `100` | Number of sample points along profile |
| `interpolation` | `str` | `bilinear` | Interpolation method |

---

#### `dem_viewshed`

Compute visible area from an observer point.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `observer` | `float[2]` | *required* | Observer location [lon, lat] |
| `radius_m` | `float` | *required* | Maximum viewing radius in metres |
| `source` | `str?` | `cop30` | DEM source identifier |
| `observer_height_m` | `float` | `1.8` | Observer height above ground |
| `output_format` | `str` | `geotiff` | Output format |

---

### ML-Enhanced Analysis Tools

#### `dem_classify_landforms`

Classify terrain into geomorphological landform types using slope, curvature, and TRI thresholds.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `method` | `str` | `rule_based` | Classification method: `rule_based` |
| `output_format` | `str` | `geotiff` | Output format: `geotiff` or `png` |

**Response:** `LandformResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `bbox` | `float[]` | Bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `float` | Resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `dominant_landform` | `str` | Most common landform class |
| `class_distribution` | `dict` | Percentage of each landform class |
| `output_format` | `str` | Output format used |
| `license_warning` | `str?` | License restriction warning |
| `message` | `str` | Result message |

Landform classes: plain (0), ridge (1), valley (2), plateau (3), escarpment (4), depression (5), saddle (6), terrace (7), alluvial_fan (8).

---

#### `dem_detect_anomalies`

Detect terrain anomalies using Isolation Forest on terrain feature vectors (slope, curvature, TRI, roughness). Requires `pip install chuk-mcp-dem[ml]` for scikit-learn.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `sensitivity` | `float` | `0.1` | Isolation Forest contamination parameter (0.0-1.0) |
| `output_format` | `str` | `geotiff` | Output format: `geotiff` or `png` |

**Response:** `AnomalyDetectionResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `bbox` | `float[]` | Bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `float` | Resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `anomaly_count` | `int` | Number of anomaly regions detected |
| `anomalies` | `TerrainAnomaly[]` | Anomaly region details |
| `output_format` | `str` | Output format used |
| `license_warning` | `str?` | License restriction warning |
| `message` | `str` | Result message |

TerrainAnomaly fields: `bbox`, `area_m2`, `confidence`, `mean_anomaly_score`.

---

#### `dem_compare_temporal`

Compute elevation change between two DEM sources by pixel subtraction with significance thresholding and change region detection.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `before_source` | `str` | *required* | Earlier DEM source (e.g. `cop90`) |
| `after_source` | `str` | *required* | Later DEM source (e.g. `cop30`) |
| `significance_threshold_m` | `float` | `1.0` | Minimum change in metres to flag as significant |
| `output_format` | `str` | `geotiff` | Output format: `geotiff` or `png` |

**Response:** `TemporalChangeResponse`

| Field | Type | Description |
|-------|------|-------------|
| `before_source` | `str` | Earlier DEM source |
| `after_source` | `str` | Later DEM source |
| `bbox` | `float[]` | Bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `float` | Resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `volume_gained_m3` | `float` | Total volume gain in cubic metres |
| `volume_lost_m3` | `float` | Total volume loss in cubic metres |
| `significant_regions` | `ChangeRegion[]` | Significant change regions |
| `output_format` | `str` | Output format used |
| `message` | `str` | Result message |

ChangeRegion fields: `bbox`, `area_m2`, `mean_change_m`, `max_change_m`, `change_type` (gain/loss).

---

#### `dem_detect_features`

Detect geomorphological features using CNN-inspired multi-angle hillshade with Sobel edge filters. Classifies pixels into: peak, ridge, valley, cliff, saddle, channel.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `bbox` | `float[4]` | *required* | Bounding box `[west, south, east, north]` |
| `source` | `str?` | `cop30` | DEM source identifier |
| `method` | `str` | `cnn_hillshade` | Feature detection method |
| `output_format` | `str` | `geotiff` | Output format: `geotiff` or `png` |

**Response:** `FeatureDetectionResponse`

| Field | Type | Description |
|-------|------|-------------|
| `source` | `str` | DEM source used |
| `bbox` | `float[]` | Bounding box |
| `artifact_ref` | `str` | Artifact store reference |
| `preview_ref` | `str?` | PNG preview reference |
| `crs` | `str` | Output CRS |
| `resolution_m` | `float` | Resolution in metres |
| `shape` | `int[]` | Array shape [height, width] |
| `feature_count` | `int` | Number of feature regions detected |
| `feature_summary` | `dict` | Count per feature type |
| `features` | `TerrainFeature[]` | Feature region details |
| `output_format` | `str` | Output format used |
| `license_warning` | `str?` | License restriction warning |
| `message` | `str` | Result message |

Feature classes: none (0), peak (1), ridge (2), valley (3), cliff (4), saddle (5), channel (6).

TerrainFeature fields: `bbox`, `area_m2`, `feature_type`, `confidence`.

**Algorithm:** Generates 8 hillshades at azimuths [0, 45, 90, 135, 180, 225, 270, 315], applies Sobel edge filter to each channel, computes edge consensus (mean across angles), then classifies pixels using slope, curvature, and edge thresholds. Morphological cleanup with binary opening/closing, followed by connected component labelling for region statistics.

---

### LLM Interpretation Tool

#### `dem_interpret`

Send any terrain artifact to the calling LLM via MCP sampling (`sampling/createMessage`) for natural language interpretation. Requires an MCP client that supports sampling (e.g., Claude Desktop). No additional dependencies.

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `artifact_ref` | `str` | *required* | Reference to any stored terrain artifact |
| `context` | `str` | `general` | Interpretation context: `general`, `archaeological_survey`, `flood_risk`, `geological`, `military_history`, `urban_planning` |
| `question` | `str` | `""` | Optional specific question about the terrain |

**Response:** `InterpretResponse`

| Field | Type | Description |
|-------|------|-------------|
| `artifact_ref` | `str` | The artifact that was interpreted |
| `context` | `str` | Interpretation context used |
| `question` | `str?` | Specific question asked |
| `interpretation` | `str` | The LLM's terrain interpretation |
| `model` | `str` | Model that generated the interpretation |
| `features_identified` | `str[]` | Notable features mentioned |
| `message` | `str` | Result message |

**MCP Sampling:** The tool converts the artifact to PNG, base64-encodes it, and calls `ServerSession.create_message()` via the MCP SDK `request_ctx` contextvar. If the client doesn't support sampling, a graceful error message is returned suggesting manual inspection of the artifact.

---

## Build Phases

| Phase | Version | Tools | Focus |
|-------|---------|-------|-------|
| **1.0** | v0.1.0 | 9 tools | Core Fetch: scaffold, discovery (4), download (5), tests, CI/CD |
| **1.1** | v0.2.0 | +5 tools | Terrain Analysis: hillshade, slope, aspect, curvature, TRI |
| **1.2** | v0.3.0 | +2 tools | Advanced: profile, viewshed |
| **1.2+2.0** | v0.4.0 | +1 tool | Contour lines + SRTM/3DEP/FABDEM download integration |
| **2.1** | v0.5.0 | +1 tool | Watershed analysis + FABDEM license warnings |
| **2.2** | v0.5.1 | -- | LLM input normalization for `dem_fetch_points` |
| **3.0** | v0.6.0 | +4 tools | ML-enhanced analysis: landforms, anomalies, temporal change, feature detection |
| **3.1** | v0.7.0 | +1 tool | LLM terrain interpretation via MCP sampling |

---

## Output Formats

### GeoTIFF (default)

Single-band float32 GeoTIFF with embedded CRS and transform metadata. Nodata pixels
use NaN. Suitable for GIS analysis and further processing.

### PNG

Hillshade or terrain-coloured PNG for visual inspection. LLMs with vision can render
these inline. Auto-preview generates a hillshade PNG alongside every GeoTIFF fetch.

---

## Interpolation Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `nearest` | Nearest-neighbour | Fast, preserves original values |
| `bilinear` | Bilinear (default) | Good balance of speed and accuracy |
| `cubic` | Bicubic spline | Smoothest, uses 4x4 neighbourhood |

---

## Artifact Storage

Downloaded elevation data is stored via chuk-artifacts with enriched metadata:

```json
{
  "schema_version": "1.0",
  "type": "dem_raster",
  "source": "cop30",
  "source_name": "Copernicus GLO-30",
  "bbox": [-121.8, 46.7, -121.6, 46.9],
  "crs": "EPSG:4326",
  "resolution_m": 30,
  "shape": [667, 667],
  "dtype": "float32",
  "elevation_range": [1200.5, 4392.1],
  "vertical_datum": "EGM2008",
  "vertical_unit": "metres",
  "nodata_value": -9999.0,
  "acquisition_period": "2011-2015",
  "accuracy_vertical_m": 4.0
}
```

### Storage Providers

| Provider | Env Variable | Value |
|----------|-------------|-------|
| Memory (default) | `CHUK_ARTIFACTS_PROVIDER` | `memory` |
| Filesystem | `CHUK_ARTIFACTS_PROVIDER` | `filesystem` |
| Amazon S3 | `CHUK_ARTIFACTS_PROVIDER` | `s3` |

Additional environment variables for S3: `BUCKET_NAME`, `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL_S3`.

---

## Error Handling

All tools return `ErrorResponse` on failure:

```json
{
  "error": "Unknown DEM source 'invalid'. Available: cop30, cop90, srtm, aster, 3dep, fabdem"
}
```

### Common Error Scenarios

| Scenario | Error Message |
|----------|---------------|
| Invalid bbox length | "Invalid bounding box: must be [west, south, east, north]" |
| Invalid bbox values | "Invalid bounding box values: west (...) must be < east (...)" |
| Unknown source | "Unknown DEM source '...'. Available: cop30, cop90, srtm, aster, 3dep, fabdem" |
| No coverage | "... does not cover the requested area (...)" |
| No artifact store | "No artifact store available. Configure CHUK_ARTIFACTS_PROVIDER..." |
| Area too large | "Requested area (... MB) exceeds limit. Add bbox or use ..." |
| Network failure | "Failed to fetch tile after 3 retries: ..." |
| Invalid interpolation | "Invalid interpolation '...'. Available: nearest, bilinear, cubic" |

---

## Planned Demo Scripts

| Demo | Location | What It Shows |
|------|----------|---------------|
| `mount_rainier_demo.py` | Mount Rainier, WA | Elevation fetch, hillshade preview, terrain stats |
| `grand_canyon_demo.py` | Grand Canyon, AZ | Cross-section profile, multi-source comparison |
| `mt_everest_demo.py` | Mt Everest, Nepal | High-altitude DEM, void handling, cop30 vs aster |
| `coastal_flood_demo.py` | Coastal area | FABDEM bare-earth, flood zone elevation query |

---

## Note on Geocoding

Geocoding (place name to bbox) is planned as a **separate MCP server**, not part
of chuk-mcp-dem. This server accepts only coordinate-based bounding boxes and points.
