# chuk-mcp-dem Roadmap

## Current State (v0.3.0)

**Working:** 14 tools functional with full DEM discovery, coverage check, fetch, point query, terrain analysis, profile, and viewshed pipeline.

**Test Stats:** 813 tests, 95.43% coverage. All checks pass (ruff, mypy, bandit, pytest).

**Infrastructure:** Project scaffold, pyproject.toml, Makefile, CI/CD (GitHub Actions), Dockerfile.

**Implemented:** 6 DEM sources (Copernicus GLO-30/90, SRTM, ASTER, 3DEP, FABDEM), tile URL construction, multi-tile merging, void filling, point sampling (nearest/bilinear/cubic), hillshade auto-preview, coverage checking, size estimation, LRU tile cache (100 MB), artifact storage via chuk-artifacts, Pydantic v2 response models, dual output mode, terrain derivatives (hillshade/slope/aspect), elevation profiles, viewshed analysis.

---

## Phase 1.0: Core Fetch (v0.1.0) -- COMPLETE

### 1.0.1 Project Infrastructure

- [x] Initialize git repo with `.gitignore`
- [x] Create `pyproject.toml` with all dependencies
- [x] Create `Makefile` with full target set (test, lint, format, typecheck, security, check)
- [x] Add `LICENSE` file (Apache-2.0)
- [x] Add `.github/workflows/test.yml` -- lint, typecheck, security, tests
- [x] Add `.github/workflows/publish.yml` -- PyPI publishing
- [x] Add `.github/workflows/release.yml` -- auto-generate changelog and GitHub release
- [x] Add `.github/workflows/fly-deploy.yml` -- deploy to Fly.io on push to main

### 1.0.2 Core Modules

- [x] `constants.py` -- 6 DEM sources with full metadata, cache limits, error messages
- [x] `models/responses.py` -- Pydantic v2 response models with `extra="forbid"` and `to_text()`
- [x] `core/raster_io.py` -- COG reading, merging, sampling, terrain derivatives, PNG output
- [x] `core/dem_manager.py` -- tile cache, URL construction, download pipeline, artifact storage

### 1.0.3 Discovery Tools (4 tools)

- [x] `dem_list_sources` -- list all 6 DEM sources with summary metadata
- [x] `dem_describe_source` -- detailed source metadata with LLM guidance
- [x] `dem_status` -- server status, version, storage provider, cache size
- [x] `dem_capabilities` -- full capabilities listing for LLM workflow planning

### 1.0.4 Download Tools (5 tools)

- [x] `dem_fetch` -- download elevation data for a bounding box, auto-preview
- [x] `dem_fetch_point` -- single-point elevation query with interpolation
- [x] `dem_fetch_points` -- multi-point elevation query
- [x] `dem_check_coverage` -- verify source coverage before download
- [x] `dem_estimate_size` -- size estimation without fetching pixel data

### 1.0.5 Tests & Documentation

- [x] 813 tests with 95.43% overall coverage
- [x] `SPEC.md` -- full tool specification
- [x] `ARCHITECTURE.md` -- design principles and data flows
- [x] `ROADMAP.md` -- this document
- [x] `README.md` -- overview, install, quick start, tool reference

---

## Phase 1.1: Terrain Analysis (v0.2.0) -- COMPLETE

### 1.1.1 Terrain Derivative Tools (3 tools)

- [x] `dem_hillshade` -- shaded relief via Horn's method (azimuth, altitude, z-factor)
- [x] `dem_slope` -- slope steepness in degrees or percent
- [x] `dem_aspect` -- slope direction with flat-area handling

### 1.1.2 Terrain Infrastructure

- [x] Terrain tool response models (`HillshadeResponse`, `SlopeResponse`, `AspectResponse`)
- [x] Terrain-coloured PNG output for slope/aspect visualisation
- [x] DEMManager `fetch_hillshade()`, `fetch_slope()`, `fetch_aspect()` async methods

### 1.1.3 Tests

- [x] Unit tests for `compute_hillshade()`, `compute_slope()`, `compute_aspect()`
- [x] Unit tests for `slope_to_png()`, `aspect_to_png()`
- [x] Integration tests for terrain tool endpoints
- [x] Edge cases: flat terrain, steep cliffs, nodata handling

---

## Phase 1.2: Advanced (v0.3.0) -- COMPLETE

### 1.2.1 Profile & Viewshed Tools (2 tools)

- [x] `dem_profile` -- elevation profile along a line between two points
- [x] `dem_viewshed` -- visibility analysis from an observer point
- [ ] `dem_contour` -- generate contour lines at specified intervals (stretch goal)

### 1.2.2 Profile Infrastructure

- [x] Line sampling with configurable point count
- [x] Distance calculation (Haversine)
- [x] Profile response model with distance/elevation arrays and gain/loss
- [x] Viewshed raster output (binary visible/not-visible) via DDA ray-casting
- [x] DEMManager `fetch_profile()`, `fetch_viewshed()` async methods

### 1.2.3 Tests

- [x] Profile accuracy validation (haversine distances, monotonic, gain/loss)
- [x] Viewshed validation (flat surface 100% visible, ridge blocking, NaN outside radius)
- [x] Line-of-sight tests (clear LOS, blocked by ridge, NaN obstacles)

---

## Phase 2.0: Extended Sources (v0.4.0)

### 2.0.1 SRTM Download Integration

- [ ] SRTM v3 tile URL construction and download
- [ ] SRTM HGT file reading via rasterio
- [ ] Automatic void detection and reporting

### 2.0.2 3DEP Download Integration

- [ ] 3DEP tile URL construction (USGS TNM API)
- [ ] Multi-resolution support (1m, 3m, 10m, 30m)
- [ ] US-only coverage validation

### 2.0.3 FABDEM Download Integration

- [ ] FABDEM tile access
- [ ] Bare-earth vs DSM comparison capability
- [ ] Non-commercial license enforcement/warning

---

## Future Considerations

### Potential Features

- **Bathymetry**: GEBCO ocean depth data as an additional source
- **Planetary DEMs**: Mars MOLA, Moon LOLA via PDS archives
- **Difference maps**: Compute elevation change between two DEM sources/dates
- **Watershed delineation**: Flow direction and accumulation from elevation
- **Cross-section export**: Export profiles as CSV/GeoJSON for external tools

### Not in Scope (for now)

- **LiDAR point clouds** -- different data model, separate server
- **Geocoding** -- planned as a separate MCP server (chuk-mcp-geo)
- **3D visualisation** -- client-side concern, not server-side
- **Real-time terrain streaming** -- outside MCP request/response model

---

## Version Summary

| Version | Phase | Focus | Key Deliverables |
|---------|-------|-------|------------------|
| 0.1.0 | 1.0 | Core Fetch | 9 tools, 6 sources, 596 tests (97% coverage), tile cache, auto-preview, CI/CD |
| 0.2.0 | 1.1 | Terrain Analysis | +3 tools (hillshade, slope, aspect), terrain PNG output |
| 0.3.0 | 1.2 | Advanced | +2 tools (profile, viewshed), haversine distance, DDA ray-casting, 813 tests (95% coverage) |
| 0.4.0 | 2.0 | Extended Sources | SRTM/3DEP/FABDEM download integration |

---

*Last updated: 2026-02*
