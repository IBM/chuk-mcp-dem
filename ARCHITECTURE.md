# Architecture

This document describes the design principles, module structure, and key patterns
used in chuk-mcp-dem.

## Design Principles

### 1. Async-First

All tool entry points are `async`. Synchronous I/O (rasterio COG reads, GDAL HTTP)
is wrapped in `asyncio.to_thread()` so the event loop is never blocked.

### 2. Single Responsibility -- Tools Never Handle Bytes

Tool functions validate inputs, call `DEMManager`, and format JSON responses.
They never touch raw raster bytes. `DEMManager.fetch_elevation()` owns the full
I/O pipeline: tile URL construction, COG read, tile merging, void filling,
optional preview generation, and artifact storage.

### 3. Pydantic v2 Native -- No Dict Goop

All responses use Pydantic models with `model_config = ConfigDict(extra="forbid")`.
This catches typos at serialisation time rather than silently passing unknown fields.

### 4. No Magic Strings

Every repeated string lives in `constants.py` as a class attribute or
module-level constant. Source identifiers, error messages, cache limits,
terrain defaults, and format options are all constants -- never inline strings.

### 5. Pluggable Storage via chuk-artifacts

Downloaded elevation data is stored through the `chuk-artifacts` abstraction layer.
Supported backends (memory, filesystem, S3) are selected via the
`CHUK_ARTIFACTS_PROVIDER` environment variable. The artifact store is initialised
at server startup, not at module import time.

### 6. Test Coverage >90% per File, 95%+ Overall

813 tests across 8 test files. Overall project coverage is 95.43%.
Every source module maintains at least 78% line coverage (async_server.py,
which has a short `if __name__` block), with most modules at 95-100%.
Tests mock at the `DEMManager` level for tool tests, and at the rasterio
layer for raster I/O unit tests.

### 7. Graceful Degradation

Errors return structured JSON (`{"error": "..."}`) -- never unhandled exceptions
or stack traces. Network failures are retried (tenacity, 3 attempts with exponential
backoff). Missing capabilities (no artifact store, unsupported source) produce clear
error messages explaining what to configure.

---

## Module Dependency Graph

```
server.py                         # CLI entry point (sync)
  +-- async_server.py             # Async server setup, tool registration
       +-- tools/discovery/api.py       # list_sources, describe_source, status, capabilities
       +-- tools/download/api.py        # fetch, fetch_point, fetch_points, check_coverage, estimate_size
       +-- tools/analysis/api.py        # hillshade, slope, aspect, profile, viewshed
       +-- core/dem_manager.py          # Cache, download pipeline, terrain, artifact storage
            +-- core/raster_io.py       # COG reading, merging, sampling, terrain, profile, viewshed, PNG

models/responses.py               # Pydantic response models (extra="forbid")
constants.py                      # Source metadata, cache limits, error messages
```

---

## Component Responsibilities

### `server.py`

Synchronous entry point. Parses environment variables, configures the artifact store
provider, and calls `asyncio.run()` on the async server. This is the only file that
touches `sys.argv` or `os.environ` directly.

### `async_server.py`

Creates the `chuk-mcp-server` MCP instance, instantiates `DEMManager`, and
registers all tool modules. Each tool module receives the MCP instance and the
shared `DEMManager`.

### `core/dem_manager.py`

The central orchestrator. Manages:
- **Tile cache**: LRU dict with byte-size tracking (100 MB total, 10 MB per item)
- **Tile URL construction**: Source-specific URL builders for Copernicus S3 COGs
- **Download pipeline**: `fetch_elevation()`, `fetch_point()`, `fetch_points()`
- **Terrain analysis**: `fetch_hillshade()`, `fetch_slope()`, `fetch_aspect()`
- **Profile extraction**: `fetch_profile()` with haversine distances and gain/loss
- **Viewshed analysis**: `fetch_viewshed()` with DDA ray-casting visibility
- **Void filling**: Nearest-neighbour interpolation for NaN pixels
- **Artifact storage**: `_store_raster()` writes bytes + metadata to chuk-artifacts
- **Coverage checking**: `check_coverage()` and `estimate_size()` without I/O
- **Discovery**: `list_sources()` and `describe_source()` from constant metadata

Result dataclasses: `ElevationResult`, `TerrainResult`, `ProfileResult`, `ViewshedResult`.

### `core/raster_io.py`

Pure I/O layer -- all functions are synchronous (called via `to_thread()`):
- `read_dem_tile()`: Single tile COG read with optional bbox crop
- `read_and_merge_tiles()`: Multi-tile merging via `rasterio.merge`
- `sample_elevation()`: Point query with nearest/bilinear/cubic interpolation
- `sample_elevations()`: Multi-point query
- `fill_voids()`: NaN filling via `scipy.ndimage.distance_transform_edt`
- `compute_hillshade()`: Horn's method (1981) shaded relief
- `compute_slope()`: Horn's method slope in degrees or percent
- `compute_aspect()`: Slope direction with flat-area handling
- `compute_profile_points()`: Line sampling between two points with haversine distances
- `compute_viewshed()`: DDA ray-casting visibility analysis from observer point
- `arrays_to_geotiff()`: NumPy array to GeoTIFF bytes via MemoryFile
- `elevation_to_hillshade_png()`: Hillshade preview for auto-preview
- `elevation_to_terrain_png()`: Terrain-coloured PNG with green-brown-white ramp
- `slope_to_png()`: Green-yellow-red ramp for slope visualisation
- `aspect_to_png()`: HSV colour wheel for aspect visualisation

### `models/responses.py`

Pydantic v2 response models for every tool. All use `extra="forbid"` to catch
serialisation errors early. Includes `SourcesResponse`, `FetchResponse`,
`PointElevationResponse`, `MultiPointResponse`, `CoverageCheckResponse`,
`SizeEstimateResponse`, `StatusResponse`, `CapabilitiesResponse`,
`HillshadeResponse`, `SlopeResponse`, `AspectResponse`, `ProfilePointInfo`,
`ProfileResponse`, `ViewshedResponse`.

Every response model implements `to_text()` for human-readable output mode.

### `constants.py`

All magic strings, source metadata, and configuration values. Includes:
- `ServerConfig` -- server name and version
- `DEMSource` -- source identifier constants (`cop30`, `cop90`, `srtm`, etc.)
- `DEM_SOURCES` -- full metadata dict per source (resolution, coverage, accuracy, URLs)
- `StorageProvider`, `SessionProvider`, `EnvVar` -- environment configuration
- `ErrorMessages`, `SuccessMessages` -- format-string message templates
- `TERRAIN_DERIVATIVES`, `ANALYSIS_TOOLS`, `OUTPUT_FORMATS` -- capability lists
- Cache limits: `TILE_CACHE_MAX_BYTES` (100 MB), `TILE_CACHE_MAX_ITEM` (10 MB)
- Terrain defaults: azimuth, altitude, z-factor, window size
- Profile/viewshed defaults: `DEFAULT_NUM_POINTS` (100), `DEFAULT_OBSERVER_HEIGHT_M` (1.8), `MAX_VIEWSHED_RADIUS_M` (50 km)

---

## Data Flows

### Fetch Elevation (bbox)

```
1. dem_fetch(bbox, source, ...)
   +-- manager.fetch_elevation()
       +-- _validate_bbox(bbox)
       +-- _get_tile_urls(source, bbox)          <-- tile URL construction
       +-- to_thread(raster_io.read_and_merge_tiles)
       |     +-- read_dem_tile() per tile        <-- COG read with retry
       |     +-- rasterio.merge.merge()          <-- multi-tile merge
       |     +-- _crop_to_bbox()                 <-- optional crop
       +-- to_thread(raster_io.fill_voids)       <-- nearest-neighbour void fill
       +-- to_thread(raster_io.arrays_to_geotiff) <-- GeoTIFF output
       +-- to_thread(raster_io.elevation_to_hillshade_png) <-- auto-preview
       +-- _store_raster()                       <-- artifact storage
```

### Point Elevation Query

```
2. dem_fetch_point(lon, lat, source, interpolation)
   +-- manager.fetch_point()
       +-- _get_tile_urls(source, point_bbox)    <-- single tile for point
       +-- to_thread(raster_io.read_dem_tile)    <-- COG read
       +-- to_thread(raster_io.sample_elevation) <-- interpolated sampling
```

### Multi-Point Query

```
3. dem_fetch_points(points, source, interpolation)
   +-- manager.fetch_points()
       +-- _get_tile_urls(source, envelope_bbox) <-- tiles covering all points
       +-- to_thread(raster_io.read_and_merge_tiles) <-- merge tiles
       +-- to_thread(raster_io.sample_elevations)    <-- batch sampling
```

### Coverage Check (no I/O)

```
4. dem_check_coverage(bbox, source)
   +-- manager.check_coverage()
       +-- _validate_bbox(bbox)
       +-- coverage_bounds intersection          <-- pure geometry
       +-- _get_tile_ids(source, bbox)           <-- tile enumeration
       +-- pixel/size estimation                 <-- arithmetic only
```

### Terrain Analysis (hillshade, slope, aspect)

```
5. dem_hillshade(bbox, source, azimuth, altitude, z_factor, ...)
   +-- manager.fetch_hillshade()
       +-- _validate_bbox(bbox)
       +-- _get_tile_urls(source, bbox)
       +-- to_thread(raster_io.read_and_merge_tiles)   <-- fetch elevation
       +-- to_thread(raster_io.fill_voids)             <-- void fill
       +-- to_thread(raster_io.compute_hillshade)      <-- Horn's method
       +-- to_thread(raster_io.arrays_to_geotiff)      <-- GeoTIFF output
       |   or to_thread(raster_io.elevation_to_hillshade_png) <-- PNG output
       +-- _store_raster()                             <-- artifact storage
```

Slope and aspect follow the same pattern, substituting `compute_slope()` or
`compute_aspect()` and their respective PNG renderers (`slope_to_png()`,
`aspect_to_png()`).

### Elevation Profile

```
6. dem_profile(start, end, source, num_points, interpolation)
   +-- manager.fetch_profile()
       +-- _get_tile_urls(source, envelope_bbox)       <-- tiles covering line
       +-- to_thread(raster_io.read_and_merge_tiles)   <-- merge tiles
       +-- to_thread(raster_io.fill_voids)             <-- void fill
       +-- to_thread(raster_io.compute_profile_points) <-- line sampling
       |     +-- _haversine() per segment              <-- cumulative distance
       |     +-- sample_elevation() per point          <-- interpolated elevation
       +-- compute gain/loss from elevation deltas     <-- arithmetic
```

### Viewshed Analysis

```
7. dem_viewshed(observer, radius_m, source, observer_height_m, ...)
   +-- manager.fetch_viewshed()
       +-- _radius_to_bbox(observer, radius_m)         <-- bbox from radius
       +-- _get_tile_urls(source, bbox)
       +-- to_thread(raster_io.read_and_merge_tiles)   <-- merge tiles
       +-- to_thread(raster_io.fill_voids)             <-- void fill
       +-- to_thread(raster_io.compute_viewshed)       <-- DDA ray-casting
       |     +-- _check_line_of_sight() per edge pixel <-- line-of-sight rays
       +-- to_thread(raster_io.arrays_to_geotiff)      <-- visibility raster
       +-- _store_raster()                             <-- artifact storage
```

---

## Key Patterns

### Tile URL Construction

Copernicus DEM tiles follow a deterministic naming convention based on latitude
and longitude. The `_make_tile_url()` method constructs S3 COG URLs:

```
cop30: https://copernicus-dem-30m.s3.amazonaws.com/
       Copernicus_DSM_COG_10_{NS}{LAT}_00_{EW}{LON}_00_DEM/
       Copernicus_DSM_COG_10_{NS}{LAT}_00_{EW}{LON}_00_DEM.tif

cop90: https://copernicus-dem-90m.s3.amazonaws.com/
       Copernicus_DSM_COG_30_{NS}{LAT}_00_{EW}{LON}_00_DEM/
       Copernicus_DSM_COG_30_{NS}{LAT}_00_{EW}{LON}_00_DEM.tif
```

Tiles are 1-degree squares. A bbox spanning multiple degrees fetches multiple
tiles that are merged with `rasterio.merge.merge()`.

### LRU Tile Cache

`DEMManager._tile_cache` is a plain `dict` used as an ordered map (Python 3.7+
insertion order). The cache tracks total byte size across all entries:

- **Maximum total size**: 100 MB (`TILE_CACHE_MAX_BYTES`)
- **Maximum per item**: 10 MB (`TILE_CACHE_MAX_ITEM`)
- **Eviction**: When adding an item would exceed the total limit, the oldest
  (first) entries are evicted until there is room.
- **LRU access**: On cache hit, the entry is deleted and re-inserted to move it
  to the end of the dict.

### Void Filling

SRTM and ASTER DEMs may contain void pixels (areas where radar returns were
insufficient). Void filling uses `scipy.ndimage.distance_transform_edt` to find
the nearest valid pixel for each void, then copies those values.

Void filling is enabled by default on `dem_fetch` and can be disabled with
`fill_voids=false`.

### Retry with Backoff

Network operations use `tenacity` with:
- 3 attempts
- Exponential backoff (1s min, 10s max)
- Retry only on `ConnectionError`, `TimeoutError`, `OSError`
- Reraise on non-retryable errors

### Hillshade Auto-Preview

Every `dem_fetch` GeoTIFF download automatically generates a hillshade PNG preview.
The preview uses Horn's method with default parameters (azimuth 315, altitude 45)
and is stored alongside the main artifact. Preview generation is non-fatal -- if
it fails, `preview_ref` is null and the download still succeeds.

### Interpolation Methods

Point elevation queries support three interpolation methods:
- **nearest**: Rounds to nearest pixel. Fast, preserves original values.
- **bilinear**: 2x2 weighted average. Good accuracy/speed balance (default).
- **cubic**: 4x4 bicubic spline via `scipy.interpolate.RectBivariateSpline`.
  Falls back to bilinear if the neighbourhood contains NaN or is at the edge.

### DDA Ray-Casting (Viewshed)

Viewshed analysis uses the Digital Differential Analyzer (DDA) algorithm to cast
rays from the observer to every pixel on the raster edge. For each ray:

1. Step along the ray in the direction of greatest axis change (DDA)
2. At each step, compute the line-of-sight elevation angle from observer to
   the current cell
3. Track the maximum elevation angle seen so far along the ray
4. A cell is visible if its elevation angle exceeds the running maximum

Pixels outside the specified radius are marked NaN. The observer pixel is
always visible. Result is a binary float array (1.0 = visible, 0.0 = hidden).

### Haversine Distance (Profile)

Elevation profiles compute cumulative ground distance using the haversine
formula for great-circle distance on a sphere (R = 6,371,000 m). Each segment
between consecutive sample points contributes to the running total. The formula
handles zero-distance edge cases (same point) gracefully.

### CRS Handling

All DEM sources use EPSG:4326 as the native horizontal CRS. When a source uses
a different CRS internally, bounding boxes are reprojected via `pyproj.Transformer`
before windowed reads. The `_reproject_bbox()` helper handles this transparently.
