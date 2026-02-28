# Chuk MCP DEM

**Digital Elevation Model Discovery, Retrieval & Terrain Analysis MCP Server** -- A comprehensive Model Context Protocol (MCP) server for querying DEM sources, fetching elevation data, and computing terrain derivatives.

> This is a demonstration project provided as-is for learning and testing purposes.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Tests: 1231 passed](https://img.shields.io/badge/tests-1231%20passed-brightgreen.svg)]()

## Features

This MCP server provides access to global elevation data through 6 DEM sources via 25 tools.

**All tools return fully-typed Pydantic v2 models** for type safety, validation, and excellent IDE support. All tools support `output_mode="text"` for human-readable output alongside the default JSON.

### DEM Sources

| Source | Resolution | Coverage | Notes |
|--------|-----------|----------|-------|
| **Copernicus GLO-30** (default) | 30m | Global | Void-filled, CC-BY-4.0 |
| **Copernicus GLO-90** | 90m | Global | 9x faster downloads, CC-BY-4.0 |
| **SRTM v3** | 30m | 60N-56S | May have voids, Public Domain |
| **ASTER GDEM v3** | 30m | 83N-83S | Wider latitude than SRTM, Public Domain |
| **3DEP** | 10m | USA | LiDAR-derived, highest resolution, Public Domain |
| **FABDEM** | 30m | Global | Bare-earth (buildings/forests removed), CC-BY-NC-SA-4.0 (non-commercial) |

### Discovery Tools

#### 1. List Sources (`dem_list_sources`)
List all available DEM sources:
- 6 sources with resolution, coverage, and void-fill status
- Default source identification (Copernicus GLO-30)

#### 2. Describe Source (`dem_describe_source`)
Get detailed metadata for a DEM source:
- Resolution, coverage bounds, vertical datum, accuracy
- Acquisition period, sensors, license
- LLM-friendly usage guidance

#### 3. Server Status (`dem_status`)
Check server configuration:
- Server version, default source, available sources
- Storage provider and artifact store availability
- Current tile cache size

#### 4. Capabilities (`dem_capabilities`)
List full server capabilities for LLM workflow planning:
- All sources with metadata
- Available terrain derivatives and analysis tools
- Output formats, tool count, usage guidance

### Download Tools

#### 5. Fetch Elevation (`dem_fetch`)
Download elevation data for a bounding box:
- Any of the 6 DEM sources
- Multi-tile merging for areas spanning multiple tiles
- Automatic void filling (nearest-neighbour interpolation)
- Hillshade auto-preview alongside GeoTIFF output
- Stored in chuk-artifacts with enriched metadata

#### 6. Point Elevation (`dem_fetch_point`)
Get elevation at a single geographic coordinate:
- Three interpolation methods: nearest, bilinear (default), cubic
- Returns elevation with vertical uncertainty estimate
- Fast single-tile read

#### 7. Multi-Point Elevation (`dem_fetch_points`)
Get elevations at multiple coordinates in one call:
- Batch processing with single tile merge
- Returns per-point elevations and overall range
- Configurable interpolation method
- Automatic input normalization for LLM callers (handles string and flat-list formats)

#### 8. Coverage Check (`dem_check_coverage`)
Check if a DEM source covers a bounding box:
- Coverage percentage calculation
- Tile enumeration and size estimation
- No pixel data transferred -- pure geometry computation

#### 9. Size Estimation (`dem_estimate_size`)
Estimate download size before fetching:
- Pixel count and byte size calculation
- Warnings for large downloads (>=500 MB, >=1 GB)
- Optional custom resolution target

### Terrain Analysis Tools

#### 10. Hillshade (`dem_hillshade`)
Compute shaded relief from elevation data:
- Horn's method (1981) with configurable sun azimuth and altitude
- Vertical exaggeration via z-factor
- GeoTIFF or PNG output with auto-preview

#### 11. Slope (`dem_slope`)
Compute slope steepness:
- Output in degrees or percent
- Horn's method gradient computation
- GeoTIFF or PNG output with green-yellow-red colour ramp

#### 12. Aspect (`dem_aspect`)
Compute slope direction:
- 0-360 degrees from north, flat areas marked with configurable value
- GeoTIFF or PNG output with HSV colour wheel

#### 13. Curvature (`dem_curvature`)
Compute surface curvature (rate of change of slope):
- Positive values = convex surfaces (ridges), negative = concave (valleys)
- Zevenbergen-Thorne second derivative method
- GeoTIFF or PNG output with diverging blue-white-red ramp

#### 14. Terrain Ruggedness (`dem_terrain_ruggedness`)
Compute Terrain Ruggedness Index (TRI):
- Mean absolute elevation difference from 8 neighbours (Riley et al. 1999)
- Values in metres with standard classification (level to extremely rugged)
- GeoTIFF or PNG output with green-yellow-red ramp

#### 15. Contours (`dem_contour`)
Generate contour lines at specified elevation intervals:
- Configurable contour interval (default 100m)
- Sign-change detection for precise contour placement
- GeoTIFF or PNG output with terrain background and contour overlay

#### 16. Watershed (`dem_watershed`)
Compute flow accumulation for hydrological analysis:
- D8 single-flow-direction algorithm
- High accumulation values indicate streams and drainage channels
- GeoTIFF or PNG output with log-scaled blue ramp for stream visualisation

### Profile & Viewshed Tools

#### 17. Elevation Profile (`dem_profile`)
Extract elevation cross-section between two points:
- Configurable number of sample points (default 100)
- Returns per-point elevation, distance, total gain and loss
- Three interpolation methods

#### 18. Viewshed (`dem_viewshed`)
Compute visible area from an observer point:
- DDA ray-casting algorithm for line-of-sight analysis
- Configurable observer height and radius (up to 50 km)
- Returns visibility percentage and raster artifact

### View Tools

Interactive visualisation tools that return structured view content for MCP clients that support
rich UI rendering (e.g. Claude Desktop with mcp-views). These tools return `structuredContent`
rather than JSON strings and do not accept `output_mode`.

#### 19. Elevation Profile Chart (`dem_profile_chart`)
Render an elevation profile as an interactive chart:
- Same profile computation as `dem_profile` with gain/loss statistics in the title
- Returns a `ProfileContent` view with labelled x/y axes and filled area
- Rendered as an interactive line chart in supporting MCP clients

#### 20. DEM Map (`dem_map`)
Display a DEM analysis area on an interactive map:
- Renders the bounding box as a GeoJSON polygon overlay on a tiled basemap
- Configurable basemap: terrain (default), satellite, osm, dark
- Auto-calculates zoom level from bbox extent
- Useful for verifying coverage before fetching elevation data

### ML-Enhanced Analysis Tools

Tier 2 tools that *recognise* terrain features using existing terrain derivatives as input. Install with `pip install chuk-mcp-dem[ml]` for Isolation Forest support.

#### 21. Landform Classification (`dem_classify_landforms`)
Classify terrain into geomorphological types:
- 9 landform classes: plain, ridge, valley, plateau, escarpment, depression, saddle, terrace, alluvial fan
- Rule-based classification using slope, curvature, and TRI thresholds
- Returns dominant landform and class distribution percentages

#### 22. Anomaly Detection (`dem_detect_anomalies`)
Detect terrain anomalies using Isolation Forest:
- Stacks slope, curvature, TRI, and roughness into feature vectors
- scikit-learn Isolation Forest identifies outlier terrain
- Returns anomaly score map and labelled anomaly regions with confidence

#### 23. Temporal Change (`dem_compare_temporal`)
Compute elevation change between two DEM sources:
- Pixel-aligned subtraction with significance thresholding
- Volume gained/lost calculation in cubic metres
- Labelled change regions with mean/max change and gain/loss type

#### 24. Feature Detection (`dem_detect_features`)
CNN-inspired geomorphological feature detection:
- Multi-angle hillshade (8 azimuths) with Sobel edge filters
- Classifies pixels into peak, ridge, valley, cliff, saddle, channel
- Morphological cleanup and connected component labelling
- No torch required -- uses scipy convolutional filters only

### LLM Interpretation Tool

#### 25. Terrain Interpretation (`dem_interpret`)
Send any terrain artifact to the calling LLM via MCP sampling for interpretation:
- Converts GeoTIFF or PNG artifacts to images for the LLM
- 6 interpretation contexts: general, archaeological survey, flood risk, geological, military history, urban planning
- Optional specific question about the terrain
- Requires an MCP client that supports sampling (e.g., Claude Desktop)
- No additional dependencies -- uses the MCP SDK already installed

## Installation

### Using uv (Recommended)

```bash
# Install from PyPI
uv pip install chuk-mcp-dem

# Or clone and install from source
git clone <repository-url>
cd chuk-mcp-dem
uv sync --dev
```

### Using pip

```bash
pip install chuk-mcp-dem
```

## Environment Setup

```bash
# Storage backend (optional, defaults to memory)
export CHUK_ARTIFACTS_PROVIDER=memory    # or: filesystem, s3

# For filesystem storage
export CHUK_ARTIFACTS_PATH=/tmp/dem-artifacts

# For S3 storage
export BUCKET_NAME=my-bucket
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
```

## Usage

### With Claude Desktop

**macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows**: `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "dem": {
      "command": "uvx",
      "args": ["chuk-mcp-dem"]
    }
  }
}
```

Or if installed locally:

```json
{
  "mcpServers": {
    "dem": {
      "command": "chuk-mcp-dem"
    }
  }
}
```

### Standalone

```bash
# STDIO mode (default, for MCP clients)
uv run chuk-mcp-dem

# HTTP mode (for web access)
uv run chuk-mcp-dem http
```

### Example Queries

Once configured, you can ask Claude questions like:

- "What DEM sources are available?"
- "What is the elevation of Mount Rainier's summit?"
- "Download elevation data for the Grand Canyon area"
- "Check if Copernicus 30m covers this bounding box"
- "How big would a 30m DEM download be for the entire state of Colorado?"
- "Get elevation for these 5 mountain peaks"
- "Describe the FABDEM source -- how is it different from Copernicus?"
- "Generate a hillshade map for Mount Rainier"
- "What's the slope along the trail from A to B?"
- "Show me what's visible from this viewpoint within 5 km"
- "Show me an elevation profile chart for a hike from A to B"
- "Show the Grand Canyon bounding box on a map before I download it"
- "Compute watershed flow accumulation for this mountain area"
- "Classify the landforms in this bounding box"
- "Detect terrain anomalies near Hoover Dam"
- "Compare elevation between GLO-90 and GLO-30 for Mount St. Helens"
- "Detect geomorphological features in the Grand Canyon"

## Tool Reference

All tools accept an optional `output_mode` parameter (`"json"` default, or `"text"` for human-readable output).

| Tool | Description | Key Parameters |
|------|-------------|----------------|
| `dem_list_sources` | List all DEM sources | -- |
| `dem_describe_source` | Detailed source metadata | `source` |
| `dem_status` | Server status | -- |
| `dem_capabilities` | Full capabilities listing | -- |
| `dem_fetch` | Download elevation for a bbox | `bbox`, `source`, `fill_voids` |
| `dem_fetch_point` | Elevation at a point | `lon`, `lat`, `interpolation` |
| `dem_fetch_points` | Elevation at multiple points | `points`, `interpolation` |
| `dem_check_coverage` | Check source coverage | `bbox`, `source` |
| `dem_estimate_size` | Estimate download size | `bbox`, `source`, `resolution_m` |
| `dem_hillshade` | Compute shaded relief | `bbox`, `azimuth`, `altitude`, `z_factor` |
| `dem_slope` | Compute slope steepness | `bbox`, `units`, `output_format` |
| `dem_aspect` | Compute slope direction | `bbox`, `flat_value`, `output_format` |
| `dem_curvature` | Compute surface curvature | `bbox`, `source`, `output_format` |
| `dem_terrain_ruggedness` | Terrain Ruggedness Index | `bbox`, `source`, `output_format` |
| `dem_contour` | Generate contour lines | `bbox`, `source`, `interval_m`, `output_format` |
| `dem_watershed` | Flow accumulation analysis | `bbox`, `source`, `output_format` |
| `dem_profile` | Elevation cross-section | `start`, `end`, `num_points` |
| `dem_viewshed` | Visibility analysis | `observer`, `radius_m`, `observer_height_m` |
| `dem_profile_chart` | Interactive elevation profile chart (view) | `start`, `end`, `num_points` |
| `dem_map` | Interactive DEM analysis area map (view) | `bbox`, `source`, `basemap` |
| `dem_classify_landforms` | Landform classification | `bbox`, `source`, `method` |
| `dem_detect_anomalies` | Anomaly detection (Isolation Forest) | `bbox`, `source`, `sensitivity` |
| `dem_compare_temporal` | Elevation change detection | `bbox`, `before_source`, `after_source` |
| `dem_detect_features` | Feature detection (CNN-inspired) | `bbox`, `source`, `method` |
| `dem_interpret` | LLM terrain interpretation (MCP sampling) | `artifact_ref`, `context`, `question` |

### dem_fetch

```python
{
  "bbox": [-121.8, 46.7, -121.6, 46.9],     # [west, south, east, north]
  "source": "cop30",                          # optional (default: cop30)
  "fill_voids": true,                         # optional (default: true)
  "resolution_m": null,                       # optional: target resolution in metres
  "output_crs": null                          # optional: output CRS
}
```

### dem_fetch_point

```python
{
  "lon": -121.7603,                           # longitude
  "lat": 46.8523,                             # latitude
  "source": "cop30",                          # optional
  "interpolation": "bilinear"                 # optional: nearest, bilinear, cubic
}
```

### dem_fetch_points

```python
{
  "points": [                                 # nested array of [lon, lat] numeric pairs
    [-121.7603, 46.8523],
    [-105.6836, 40.2548],
    [-112.1871, 36.0544]
  ],
  "source": "cop30",                          # optional
  "interpolation": "bilinear"                 # optional
}
```

### dem_check_coverage

```python
{
  "bbox": [-121.8, 46.7, -121.6, 46.9],
  "source": "srtm"                            # check SRTM coverage for this area
}
```

### dem_estimate_size

```python
{
  "bbox": [-109.05, 37.0, -102.05, 41.0],    # Colorado
  "source": "cop30",
  "resolution_m": 90                          # optional: coarser for faster download
}
```

## Architecture

```
server.py                         # CLI entry point (sync)
  +-- async_server.py             # Async server setup, tool registration
       +-- tools/discovery/       # list_sources, describe, status, capabilities
       +-- tools/download/        # fetch, fetch_point, fetch_points, coverage, size
       +-- tools/analysis/        # hillshade, slope, aspect, curvature, tri, contour, watershed, profile, viewshed, profile_chart, map, landforms, anomalies, temporal, features
       +-- core/dem_manager.py    # Cache, URL construction, download pipeline
            +-- core/raster_io.py # COG read, merge, sample, terrain, PNG
```

Built on top of chuk-mcp-server, this server uses:

- **Async-First**: Native async/await with sync rasterio wrapped in `asyncio.to_thread()`
- **Type-Safe**: Pydantic v2 models with `extra="forbid"` for all responses
- **Efficient I/O**: Cloud-Optimized GeoTIFF (COG) reading via S3
- **Smart Caching**: LRU tile cache (100 MB total, 10 MB per item)
- **Multi-Tile Merging**: Automatic merge for bounding boxes spanning multiple 1-degree tiles
- **Void Filling**: Nearest-neighbour interpolation for SRTM/ASTER voids
- **Interpolation**: Nearest, bilinear, and bicubic methods for point queries
- **Auto-Preview**: Hillshade PNG auto-generated alongside every GeoTIFF fetch
- **Artifact Storage**: Pluggable storage via chuk-artifacts (memory, filesystem, S3)
- **Retry with Backoff**: Tenacity-based retry (3 attempts, exponential backoff)
- **Dual Output**: All tools support `output_mode="text"` for human-readable responses
- **License Warnings**: Automatic warnings when using FABDEM (CC-BY-NC-SA-4.0 non-commercial)
- **View Tools**: Interactive profile charts and maps via `chuk-view-schemas` for MCP clients that support rich UI

See [ARCHITECTURE.md](ARCHITECTURE.md) for design principles and data flow diagrams.
See [SPEC.md](SPEC.md) for the full tool specification with parameter tables.
See [ROADMAP.md](ROADMAP.md) for development status and planned features.

## Development

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd chuk-mcp-dem

# Install with uv (recommended)
uv sync --dev

# Or with pip
pip install -e ".[dev]"
```

### Running Tests

```bash
make test              # Run 1231 tests
make test-cov          # Run tests with coverage
make coverage-report   # Show coverage report
```

### Code Quality

```bash
make lint      # Run linters
make format    # Auto-format code
make typecheck # Run type checking
make security  # Run security checks
make check     # Run all checks (lint, typecheck, security, test)
```

### Building

```bash
make build         # Build package
make docker-build  # Build Docker image
make docker-run    # Run Docker container
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

Apache License 2.0 -- See [LICENSE](LICENSE) for details.

## Acknowledgments

- [Copernicus DEM](https://spacedata.copernicus.eu/collections/copernicus-digital-elevation-model) for the GLO-30 and GLO-90 datasets
- [rasterio](https://rasterio.readthedocs.io/) for raster data I/O
- [Model Context Protocol](https://modelcontextprotocol.io/) for the MCP specification
- [Anthropic](https://www.anthropic.com/) for Claude and MCP support
