# chuk-mcp-dem Roadmap

## Current State (v0.7.0)

**Working:** 23 tools functional with full DEM discovery, coverage check, fetch, point query, terrain analysis (hillshade/slope/aspect/curvature/TRI/contour/watershed), profile, viewshed, ML-enhanced terrain analysis (landform classification, anomaly detection, temporal change, feature detection), and LLM terrain interpretation via MCP sampling.

**Test Stats:** 1205 tests. All checks pass (ruff, mypy, bandit, pytest).

**Infrastructure:** Project scaffold, pyproject.toml, Makefile, CI/CD (GitHub Actions), Dockerfile.

**Implemented:** 6 DEM sources (Copernicus GLO-30/90, SRTM, ASTER, 3DEP, FABDEM), tile URL construction for Copernicus/SRTM/3DEP/FABDEM, multi-tile merging, void filling, point sampling (nearest/bilinear/cubic), hillshade auto-preview, coverage checking, size estimation, LRU tile cache (100 MB), artifact storage via chuk-artifacts, Pydantic v2 response models, dual output mode, terrain derivatives (hillshade/slope/aspect/curvature/TRI/contour/watershed), elevation profiles, viewshed analysis, FABDEM license warnings, LLM input normalization for `dem_fetch_points`, rule-based landform classification, Isolation Forest anomaly detection, temporal elevation change detection, CNN-inspired multi-angle hillshade feature detection, LLM terrain interpretation via MCP sampling.

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

- [x] 1006 tests with 95% overall coverage
- [x] `SPEC.md` -- full tool specification
- [x] `ARCHITECTURE.md` -- design principles and data flows
- [x] `ROADMAP.md` -- this document
- [x] `README.md` -- overview, install, quick start, tool reference

---

## Phase 1.1: Terrain Analysis (v0.2.0) -- COMPLETE

### 1.1.1 Terrain Derivative Tools (5 tools)

- [x] `dem_hillshade` -- shaded relief via Horn's method (azimuth, altitude, z-factor)
- [x] `dem_slope` -- slope steepness in degrees or percent
- [x] `dem_aspect` -- slope direction with flat-area handling
- [x] `dem_curvature` -- surface curvature via Zevenbergen-Thorne (ridges/valleys)
- [x] `dem_terrain_ruggedness` -- Terrain Ruggedness Index (Riley et al. 1999)

### 1.1.2 Terrain Infrastructure

- [x] Terrain tool response models (`HillshadeResponse`, `SlopeResponse`, `AspectResponse`, `CurvatureResponse`, `TRIResponse`)
- [x] Terrain-coloured PNG output for slope/aspect/curvature/TRI visualisation
- [x] DEMManager `fetch_hillshade()`, `fetch_slope()`, `fetch_aspect()`, `fetch_curvature()`, `fetch_tri()` async methods

### 1.1.3 Tests

- [x] Unit tests for `compute_hillshade()`, `compute_slope()`, `compute_aspect()`, `compute_curvature()`, `compute_tri()`
- [x] Unit tests for `slope_to_png()`, `aspect_to_png()`, `curvature_to_png()`, `tri_to_png()`
- [x] Integration tests for terrain tool endpoints
- [x] Edge cases: flat terrain, steep cliffs, nodata handling

---

## Phase 1.2: Advanced (v0.3.0) -- COMPLETE

### 1.2.1 Profile & Viewshed Tools (2 tools)

- [x] `dem_profile` -- elevation profile along a line between two points
- [x] `dem_viewshed` -- visibility analysis from an observer point
- [x] `dem_contour` -- generate contour lines at specified intervals

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

## Phase 2.0: Extended Sources (v0.4.0) -- COMPLETE

### 2.0.1 SRTM Download Integration

- [x] SRTM v3 tile URL construction and download (AWS Open Data skadi format)
- [x] SRTM HGT file reading via rasterio
- [x] Automatic void detection and reporting

### 2.0.2 3DEP Download Integration

- [x] 3DEP tile URL construction (USGS S3 1/3 arc-second)
- [ ] Multi-resolution support (1m, 3m, 10m, 30m) -- future enhancement
- [x] US-only coverage validation

### 2.0.3 FABDEM Download Integration

- [x] FABDEM tile access (Bristol University public endpoint)
- [ ] Bare-earth vs DSM comparison capability -- future enhancement
- [x] Non-commercial license warning on all FABDEM responses

### 2.0.4 ASTER

- [ ] ASTER GDEM v3 download -- requires NASA Earthdata authentication (deferred)

---

## Phase 2.1: Watershed & License (v0.5.0) -- COMPLETE

### 2.1.1 Watershed Analysis (1 tool)

- [x] `dem_watershed` -- D8 flow direction and flow accumulation
- [x] `compute_flow_accumulation()` -- D8 single-flow-direction with topological sort
- [x] `watershed_to_png()` -- log-scaled blue ramp for stream network visualisation
- [x] `WatershedResponse` Pydantic v2 response model
- [x] `DEMManager.fetch_watershed()` async method

### 2.1.2 FABDEM License Warning

- [x] `FABDEM_LICENSE_WARNING` constant and `get_license_warning()` helper
- [x] `license_warning` field on all response models that accept a `source` parameter
- [x] Automatic warning in both JSON and text output modes for FABDEM source

### 2.1.3 Tests

- [x] Unit tests for `compute_flow_accumulation()`, `watershed_to_png()`
- [x] Integration tests for `dem_watershed` tool endpoint
- [x] License warning tests across all terrain and download tools

---

## Phase 2.2: LLM Robustness (v0.5.1) -- COMPLETE

### 2.2.1 Input Normalization for `dem_fetch_points`

- [x] `_normalize_points()` helper handles malformed LLM inputs (string elements, JSON string elements, flat numeric lists)
- [x] Improved docstring with explicit JSON example for better schema inference by LLMs
- [x] 13 new tests (10 unit + 3 integration) covering normalization edge cases

**Context:** GPT-5.2 failed to call `dem_fetch_points` correctly, sending strings like `"9.28,45.62"` instead of nested arrays `[[9.28, 45.62]]`. The type annotation was correct (`list[list[float]]`) but the docstring description was ambiguous for models. Fix: clearer docstring with concrete example + defensive input coercion.

---

## Phase 3.0: ML-Enhanced Terrain Analysis (v0.6.0) -- COMPLETE

Tier 2 upgrade: the existing 18 tools compute terrain derivatives but don't *recognise* anything. Phase 3.0 adds classification, anomaly detection, feature detection, and temporal comparison. All new tools consume existing tier 1 outputs (slope, curvature, TRI, hillshade) as inputs.

**Optional dependency:** `pip install chuk-mcp-dem[ml]` adds scikit-learn. Base install unchanged. No torch/torchvision required -- feature detection uses scipy convolutional filters (Sobel on multi-angle hillshade).

### 3.0.1 Landform Classification (1 tool)

- [x] `dem_classify_landforms` -- classify each pixel into geomorphological types (ridge, valley, plateau, escarpment, depression, saddle, plain, terrace, alluvial fan)
- [x] `compute_landforms()` in `raster_io.py` -- rule-based (threshold on slope + curvature + TRI)
- [x] `landform_to_png()` -- categorical colourmap for landform classes
- [x] `LandformResponse` Pydantic v2 response model (`artifact_ref`, `class_distribution`, `dominant_landform`)
- [x] `DEMManager.fetch_landforms()` async method

### 3.0.2 Terrain Anomaly Detection (1 tool)

- [x] `dem_detect_anomalies` -- detect terrain features that don't fit the natural landscape pattern (anthropogenic features, earthworks, quarries, old roads)
- [x] `compute_anomaly_scores()` in `raster_io.py` -- stack slope/curvature/TRI/roughness into per-pixel feature vectors, fit Isolation Forest, threshold on anomaly scores
- [x] `anomaly_to_png()` -- heatmap of anomaly scores
- [x] `AnomalyDetectionResponse` Pydantic v2 response model (`artifact_ref`, `anomaly_count`, `anomalies: list[TerrainAnomaly]`)
- [x] `TerrainAnomaly` model (bbox, area_m2, confidence, mean_anomaly_score)
- [x] `DEMManager.fetch_anomalies()` async method

**Implementation note:** Uses Isolation Forest (scikit-learn) on terrain feature vectors already computed by existing functions. No per-region training, no model weights. Autoencoder-based detection is a v2.0 upgrade path if needed.

### 3.0.3 Temporal Elevation Change (1 tool)

- [x] `dem_compare_temporal` -- compute elevation change between two DEM sources or acquisition dates
- [x] `compute_elevation_change()` in `raster_io.py` -- aligned pixel subtraction with significance thresholding
- [x] `change_to_png()` -- diverging colourmap (blue = loss, red = gain)
- [x] `TemporalChangeResponse` Pydantic v2 response model (`artifact_ref`, `volume_gained_m3`, `volume_lost_m3`, `significant_regions: list[ChangeRegion]`)
- [x] `ChangeRegion` model (bbox, area_m2, mean_change_m, max_change_m, change_type)
- [x] `DEMManager.fetch_temporal_change()` async method

### 3.0.4 Feature Detection (1 tool)

- [x] `dem_detect_features` -- CNN-inspired detection of geomorphological features using multi-angle hillshade with Sobel edge filters
- [x] `compute_feature_detection()` in `raster_io.py` -- generate 8-angle hillshade stack, apply Sobel edge filters, compute edge consensus, classify features (peak, ridge, valley, cliff, saddle, channel) using slope/curvature/edge thresholds, morphological cleanup, connected component labelling
- [x] `feature_to_png()` -- categorical colourmap (red=peak, brown=ridge, green=valley, orange=cliff, gold=saddle, blue=channel)
- [x] `TerrainFeature` model (bbox, area_m2, feature_type, confidence)
- [x] `FeatureDetectionResponse` Pydantic v2 response model (`artifact_ref`, `feature_count`, `feature_summary`, `features: list[TerrainFeature]`)
- [x] `DEMManager.fetch_features()` async method

**Implementation note:** Uses scipy convolutional filters (Sobel) on multi-angle hillshade -- same mathematical approach as CNNs but with hand-crafted kernels. Only needs scipy + numpy (already dependencies). No torch/torchvision required, no model weight downloads.

### 3.0.5 Tests

- [x] Unit tests for `compute_landforms()`, `compute_anomaly_scores()`, `compute_elevation_change()`, `compute_feature_detection()`
- [x] Unit tests for `landform_to_png()`, `anomaly_to_png()`, `change_to_png()`, `feature_to_png()`
- [x] Integration tests for all 4 new tool endpoints
- [x] Edge cases: flat terrain (no anomalies/features), identical sources (zero change), missing `[ml]` extra (graceful error)
- [x] 1167 tests total, all passing

### 3.0.6 Examples

- [x] `landform_classification_demo.py` -- Matterhorn landform classification
- [x] `anomaly_detection_demo.py` -- Hoover Dam anomaly detection (Isolation Forest)
- [x] `temporal_change_demo.py` -- Mount St. Helens elevation change (GLO-90 vs GLO-30)
- [x] `feature_detection_demo.py` -- Grand Canyon CNN-inspired feature detection
- [x] `ml_terrain_analysis_demo.py` -- Combined Grand Canyon showcase of all 4 Phase 3.0 tools

---

## Phase 3.1: LLM Terrain Interpretation via MCP Sampling (v0.7.0) -- COMPLETE

Tier 3 upgrade: contextual reasoning about terrain. Takes any terrain derivative or analysis output and sends it to the calling LLM via MCP sampling (`sampling/createMessage`) for natural language interpretation.

**No new dependencies.** Uses the MCP SDK's `ServerSession.create_message()` accessible via the `request_ctx` contextvar. The calling LLM (e.g., Claude Desktop) receives the terrain image and returns its interpretation. No chuk-llm or external API keys required.

### 3.1.1 Terrain Interpretation (1 tool)

- [x] `dem_interpret` -- send any terrain artifact to the client LLM via MCP sampling for interpretation
- [x] `raster_to_png()` in `raster_io.py` -- converts GeoTIFF or PNG artifact bytes to PNG (passthrough if already PNG)
- [x] `InterpretResponse` Pydantic v2 response model (`artifact_ref`, `context`, `question`, `interpretation`, `model`, `features_identified`, `message`)
- [x] `InterpretResult` dataclass and `DEMManager.prepare_for_interpretation()` async method
- [x] 6 interpretation contexts: general, archaeological_survey, flood_risk, geological, military_history, urban_planning
- [x] Graceful fallback when MCP client doesn't support sampling (returns helpful error message)

**MCP Sampling Call:** `from mcp.server.lowlevel.server import request_ctx` → `ctx.session.create_message(messages=[ImageContent + TextContent], max_tokens=2000, system_prompt=...)`

**Example pipeline:** `dem_fetch` → `dem_hillshade` → `dem_detect_anomalies` → `dem_interpret` → "Three circular features visible on the south-facing slope, consistent with Bronze Age barrow cemetery."

### 3.1.2 Tests

- [x] Unit tests for `raster_to_png()` (PNG passthrough, GeoTIFF conversion, nodata handling)
- [x] `InterpretResponse` model tests (creation, extra fields rejected, to_text, format_response)
- [x] `InterpretResult` dataclass tests
- [x] `prepare_for_interpretation()` tests (success, metadata, missing artifact)
- [x] Integration tests with mocked MCP sampling (success, text mode, invalid context, sampling unavailable)
- [x] 1205 tests total, all passing

---

## Dependency Tiers

| Install | Adds | Tier |
|---------|------|------|
| `chuk-mcp-dem` | numpy, rasterio, scipy | Tier 1: deterministic compute (18 tools) + feature detection |
| `chuk-mcp-dem[ml]` | + scikit-learn | Tier 2: recognition (4 tools -- landforms, anomalies, temporal, features) |
| *no extra needed* | MCP SDK (already installed) | Tier 3: reasoning (1 tool -- dem_interpret via MCP sampling) |

Note: Feature detection (`dem_detect_features`) uses scipy convolutional filters only -- no torch/torchvision required, no model weight downloads. Anomaly detection (`dem_detect_anomalies`) requires scikit-learn for Isolation Forest. Landform classification and temporal change work with base install. `dem_interpret` uses MCP sampling (callback to client LLM) -- no additional dependencies, but requires an MCP client that supports `sampling/createMessage`.

---

## Future Considerations

### Potential Features

- **Bathymetry**: GEBCO ocean depth data as an additional source
- **Planetary DEMs**: Mars MOLA, Moon LOLA via PDS archives
- **Cross-section export**: Export profiles as CSV/GeoJSON for external tools
- **Autoencoder anomaly detection**: Replace Isolation Forest with trained autoencoder for higher sensitivity (v2.0 upgrade path for `dem_detect_anomalies`)

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
| 0.2.0 | 1.1 | Terrain Analysis | +5 tools (hillshade, slope, aspect, curvature, TRI), terrain PNG output |
| 0.3.0 | 1.2 | Advanced | +2 tools (profile, viewshed), haversine distance, DDA ray-casting, 883 tests (95% coverage) |
| 0.4.0 | 1.2+2.0 | Contour + Sources | +1 tool (contour), SRTM/3DEP/FABDEM download integration, 932 tests (95% coverage) |
| 0.5.0 | 2.1 | Watershed + License | +1 tool (watershed), FABDEM license warnings, 993 tests (95% coverage) |
| 0.5.1 | 2.2 | LLM Robustness | `dem_fetch_points` input normalization, improved schema docstrings, 1006 tests (95% coverage) |
| 0.6.0 | 3.0 | ML Terrain Analysis | +4 tools (landforms, anomalies, temporal change, feature detection), `[ml]` optional extra, 1167 tests |
| 0.7.0 | 3.1 | LLM Interpretation | +1 tool (interpret), MCP sampling, no new deps, 1205 tests |

---

*Last updated: 2026-02*
