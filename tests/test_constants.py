"""Comprehensive tests for chuk_mcp_dem.constants module."""

import pytest

from chuk_mcp_dem.constants import (
    ALL_SOURCE_IDS,
    ANALYSIS_TOOLS,
    DEFAULT_ALTITUDE,
    DEFAULT_ANOMALY_SENSITIVITY,
    DEFAULT_AZIMUTH,
    DEFAULT_CONTOUR_INTERVAL_M,
    DEFAULT_FLAT_VALUE,
    DEFAULT_INTERPOLATION,
    DEFAULT_SIGNIFICANCE_THRESHOLD_M,
    DEFAULT_SOURCE,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_Z_FACTOR,
    DEM_SOURCES,
    DEMSource,
    FABDEM_LICENSE_WARNING,
    INTERPOLATION_METHODS,
    LANDFORM_CLASSES,
    LANDFORM_METHODS,
    MAX_DOWNLOAD_BYTES,
    OUTPUT_FORMATS,
    RETRY_ATTEMPTS,
    RETRY_WAIT_MAX,
    RETRY_WAIT_MIN,
    SIZE_LARGE_BYTES,
    SIZE_WARNING_BYTES,
    TERRAIN_DERIVATIVES,
    TILE_CACHE_MAX_BYTES,
    TILE_CACHE_MAX_ITEM,
    EnvVar,
    ErrorMessages,
    ServerConfig,
    SessionProvider,
    StorageProvider,
    SuccessMessages,
    get_license_warning,
)

# ── ServerConfig ────────────────────────────────────────────────────


class TestServerConfig:
    """Tests for ServerConfig class."""

    def test_name(self):
        assert ServerConfig.NAME == "chuk-mcp-dem"

    def test_version(self):
        assert ServerConfig.VERSION == "0.1.0"

    def test_description_is_nonempty(self):
        assert isinstance(ServerConfig.DESCRIPTION, str)
        assert len(ServerConfig.DESCRIPTION) > 10


# ── StorageProvider / SessionProvider ───────────────────────────────


class TestStorageProvider:
    def test_memory(self):
        assert StorageProvider.MEMORY == "memory"

    def test_s3(self):
        assert StorageProvider.S3 == "s3"

    def test_filesystem(self):
        assert StorageProvider.FILESYSTEM == "filesystem"


class TestSessionProvider:
    def test_memory(self):
        assert SessionProvider.MEMORY == "memory"

    def test_redis(self):
        assert SessionProvider.REDIS == "redis"


# ── EnvVar ──────────────────────────────────────────────────────────


class TestEnvVar:
    def test_artifacts_provider(self):
        assert EnvVar.ARTIFACTS_PROVIDER == "CHUK_ARTIFACTS_PROVIDER"

    def test_mcp_stdio(self):
        assert EnvVar.MCP_STDIO == "MCP_STDIO"


# ── DEMSource ───────────────────────────────────────────────────────


class TestDEMSource:
    """Tests for DEMSource enum-like class."""

    def test_cop30(self):
        assert DEMSource.COP30 == "cop30"

    def test_cop90(self):
        assert DEMSource.COP90 == "cop90"

    def test_srtm(self):
        assert DEMSource.SRTM == "srtm"

    def test_aster(self):
        assert DEMSource.ASTER == "aster"

    def test_tdep(self):
        assert DEMSource.TDEP == "3dep"

    def test_fabdem(self):
        assert DEMSource.FABDEM == "fabdem"

    def test_all_six_values_are_unique(self):
        values = [
            DEMSource.COP30,
            DEMSource.COP90,
            DEMSource.SRTM,
            DEMSource.ASTER,
            DEMSource.TDEP,
            DEMSource.FABDEM,
        ]
        assert len(values) == len(set(values)) == 6


# ── DEM_SOURCES dict ───────────────────────────────────────────────


class TestDEMSources:
    """Tests for the DEM_SOURCES metadata dictionary."""

    def test_has_six_entries(self):
        assert len(DEM_SOURCES) == 6

    def test_keys_match_demsource_values(self):
        expected = {
            DEMSource.COP30,
            DEMSource.COP90,
            DEMSource.SRTM,
            DEMSource.ASTER,
            DEMSource.TDEP,
            DEMSource.FABDEM,
        }
        assert set(DEM_SOURCES.keys()) == expected

    REQUIRED_KEYS = [
        "id",
        "name",
        "resolution_m",
        "coverage",
        "coverage_bounds",
        "vertical_datum",
        "vertical_unit",
        "horizontal_crs",
        "tile_size_degrees",
        "dtype",
        "nodata_value",
        "void_filled",
        "acquisition_period",
        "source_sensors",
        "accuracy_vertical_m",
        "access_url",
        "license",
        "llm_guidance",
    ]

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_source_has_all_required_keys(self, source_id):
        src = DEM_SOURCES[source_id]
        for key in self.REQUIRED_KEYS:
            assert key in src, f"Source '{source_id}' missing key '{key}'"

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_id_matches_key(self, source_id):
        assert DEM_SOURCES[source_id]["id"] == source_id

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_resolution_m_is_positive(self, source_id):
        assert DEM_SOURCES[source_id]["resolution_m"] > 0

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_coverage_bounds_has_four_elements(self, source_id):
        bounds = DEM_SOURCES[source_id]["coverage_bounds"]
        assert len(bounds) == 4

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_coverage_bounds_west_lt_east(self, source_id):
        bounds = DEM_SOURCES[source_id]["coverage_bounds"]
        west, south, east, north = bounds
        assert west < east, f"west ({west}) must be < east ({east})"

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_coverage_bounds_south_lt_north(self, source_id):
        bounds = DEM_SOURCES[source_id]["coverage_bounds"]
        west, south, east, north = bounds
        assert south < north, f"south ({south}) must be < north ({north})"

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_nodata_value_is_numeric(self, source_id):
        assert isinstance(DEM_SOURCES[source_id]["nodata_value"], (int, float))

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_void_filled_is_bool(self, source_id):
        assert isinstance(DEM_SOURCES[source_id]["void_filled"], bool)

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_source_sensors_is_nonempty_list(self, source_id):
        sensors = DEM_SOURCES[source_id]["source_sensors"]
        assert isinstance(sensors, list)
        assert len(sensors) >= 1

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_accuracy_vertical_m_is_positive(self, source_id):
        assert DEM_SOURCES[source_id]["accuracy_vertical_m"] > 0

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_access_url_starts_with_https(self, source_id):
        assert DEM_SOURCES[source_id]["access_url"].startswith("https://")

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_llm_guidance_is_nonempty_string(self, source_id):
        guidance = DEM_SOURCES[source_id]["llm_guidance"]
        assert isinstance(guidance, str)
        assert len(guidance) > 10

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_horizontal_crs_is_epsg4326(self, source_id):
        assert DEM_SOURCES[source_id]["horizontal_crs"] == "EPSG:4326"

    @pytest.mark.parametrize("source_id", list(DEM_SOURCES.keys()))
    def test_dtype_is_valid(self, source_id):
        assert DEM_SOURCES[source_id]["dtype"] in ("float32", "int16")


# ── Specific source spot checks ────────────────────────────────────


class TestSpecificSources:
    def test_cop30_resolution(self):
        assert DEM_SOURCES["cop30"]["resolution_m"] == 30

    def test_cop90_resolution(self):
        assert DEM_SOURCES["cop90"]["resolution_m"] == 90

    def test_srtm_coverage_bounds(self):
        assert DEM_SOURCES["srtm"]["coverage_bounds"] == [-180, -56, 180, 60]

    def test_aster_coverage_bounds(self):
        assert DEM_SOURCES["aster"]["coverage_bounds"] == [-180, -83, 180, 83]

    def test_tdep_is_usa_only(self):
        assert DEM_SOURCES["3dep"]["coverage"] == "USA"

    def test_tdep_resolution_10m(self):
        assert DEM_SOURCES["3dep"]["resolution_m"] == 10

    def test_fabdem_license_noncommercial(self):
        assert "NC" in DEM_SOURCES["fabdem"]["license"]


# ── DEFAULT_SOURCE ──────────────────────────────────────────────────


class TestDefaultSource:
    def test_default_source_is_cop30(self):
        assert DEFAULT_SOURCE == "cop30"

    def test_default_source_exists_in_dem_sources(self):
        assert DEFAULT_SOURCE in DEM_SOURCES


# ── ALL_SOURCE_IDS ──────────────────────────────────────────────────


class TestAllSourceIDs:
    def test_is_list(self):
        assert isinstance(ALL_SOURCE_IDS, list)

    def test_has_six_entries(self):
        assert len(ALL_SOURCE_IDS) == 6

    def test_matches_dem_sources_keys(self):
        assert set(ALL_SOURCE_IDS) == set(DEM_SOURCES.keys())


# ── Interpolation ──────────────────────────────────────────────────


class TestInterpolation:
    def test_has_three_methods(self):
        assert len(INTERPOLATION_METHODS) == 3

    def test_contains_nearest(self):
        assert "nearest" in INTERPOLATION_METHODS

    def test_contains_bilinear(self):
        assert "bilinear" in INTERPOLATION_METHODS

    def test_contains_cubic(self):
        assert "cubic" in INTERPOLATION_METHODS

    def test_default_interpolation_is_bilinear(self):
        assert DEFAULT_INTERPOLATION == "bilinear"

    def test_default_interpolation_in_methods(self):
        assert DEFAULT_INTERPOLATION in INTERPOLATION_METHODS


# ── Terrain derivatives & analysis ─────────────────────────────────


class TestTerrainDerivatives:
    def test_has_seven_entries(self):
        assert len(TERRAIN_DERIVATIVES) == 7

    def test_contains_hillshade(self):
        assert "hillshade" in TERRAIN_DERIVATIVES

    def test_contains_slope(self):
        assert "slope" in TERRAIN_DERIVATIVES

    def test_contains_aspect(self):
        assert "aspect" in TERRAIN_DERIVATIVES

    def test_contains_curvature(self):
        assert "curvature" in TERRAIN_DERIVATIVES

    def test_contains_tri(self):
        assert "tri" in TERRAIN_DERIVATIVES

    def test_contains_contour(self):
        assert "contour" in TERRAIN_DERIVATIVES

    def test_contains_watershed(self):
        assert "watershed" in TERRAIN_DERIVATIVES


class TestAnalysisTools:
    def test_has_six_entries(self):
        assert len(ANALYSIS_TOOLS) == 6

    def test_contains_profile(self):
        assert "profile" in ANALYSIS_TOOLS

    def test_contains_viewshed(self):
        assert "viewshed" in ANALYSIS_TOOLS

    def test_contains_temporal_change(self):
        assert "temporal_change" in ANALYSIS_TOOLS

    def test_contains_landforms(self):
        assert "landforms" in ANALYSIS_TOOLS

    def test_contains_anomalies(self):
        assert "anomalies" in ANALYSIS_TOOLS

    def test_contains_features(self):
        assert "features" in ANALYSIS_TOOLS


class TestOutputFormats:
    def test_has_two_entries(self):
        assert len(OUTPUT_FORMATS) == 2

    def test_contains_geotiff(self):
        assert "geotiff" in OUTPUT_FORMATS

    def test_contains_png(self):
        assert "png" in OUTPUT_FORMATS


# ── Terrain defaults ───────────────────────────────────────────────


class TestTerrainDefaults:
    def test_default_azimuth(self):
        assert DEFAULT_AZIMUTH == 315.0

    def test_default_altitude(self):
        assert DEFAULT_ALTITUDE == 45.0

    def test_default_z_factor(self):
        assert DEFAULT_Z_FACTOR == 1.0

    def test_default_flat_value(self):
        assert DEFAULT_FLAT_VALUE == -1.0

    def test_default_window_size(self):
        assert DEFAULT_WINDOW_SIZE == 3

    def test_default_contour_interval(self):
        assert DEFAULT_CONTOUR_INTERVAL_M == 100.0


# ── Cache constants ────────────────────────────────────────────────


class TestCacheConstants:
    def test_tile_cache_max_bytes(self):
        assert TILE_CACHE_MAX_BYTES == 100 * 1024 * 1024

    def test_tile_cache_max_item(self):
        assert TILE_CACHE_MAX_ITEM == 10 * 1024 * 1024

    def test_max_item_less_than_total(self):
        assert TILE_CACHE_MAX_ITEM < TILE_CACHE_MAX_BYTES


# ── Retry constants ────────────────────────────────────────────────


class TestRetryConstants:
    def test_retry_attempts(self):
        assert RETRY_ATTEMPTS == 3

    def test_retry_wait_min(self):
        assert RETRY_WAIT_MIN == 1

    def test_retry_wait_max(self):
        assert RETRY_WAIT_MAX == 10

    def test_min_less_than_max(self):
        assert RETRY_WAIT_MIN < RETRY_WAIT_MAX


# ── Size limits ────────────────────────────────────────────────────


class TestSizeLimits:
    def test_max_download_bytes(self):
        assert MAX_DOWNLOAD_BYTES == 2 * 1024 * 1024 * 1024

    def test_size_warning_bytes(self):
        assert SIZE_WARNING_BYTES == 500 * 1024 * 1024

    def test_size_large_bytes(self):
        assert SIZE_LARGE_BYTES == 1024 * 1024 * 1024

    def test_warning_less_than_large(self):
        assert SIZE_WARNING_BYTES < SIZE_LARGE_BYTES

    def test_large_less_than_max(self):
        assert SIZE_LARGE_BYTES < MAX_DOWNLOAD_BYTES


# ── ErrorMessages ──────────────────────────────────────────────────


class TestErrorMessages:
    def test_unknown_source_format(self):
        msg = ErrorMessages.UNKNOWN_SOURCE.format("bogus", "cop30, cop90")
        assert "bogus" in msg
        assert "cop30, cop90" in msg

    def test_invalid_bbox(self):
        assert "bounding box" in ErrorMessages.INVALID_BBOX.lower()

    def test_invalid_bbox_values_format(self):
        msg = ErrorMessages.INVALID_BBOX_VALUES.format(10, 5)
        assert "10" in msg and "5" in msg

    def test_invalid_bbox_lat_format(self):
        msg = ErrorMessages.INVALID_BBOX_LAT.format(50, 30)
        assert "50" in msg and "30" in msg

    def test_coverage_error_format(self):
        msg = ErrorMessages.COVERAGE_ERROR.format("SRTM", "70N 0E")
        assert "SRTM" in msg and "70N 0E" in msg

    def test_no_artifact_store(self):
        assert "CHUK_ARTIFACTS_PROVIDER" in ErrorMessages.NO_ARTIFACT_STORE

    def test_area_too_large_format(self):
        msg = ErrorMessages.AREA_TOO_LARGE.format(3000.5, "cop90")
        assert "3000.5" in msg

    def test_network_error_format(self):
        msg = ErrorMessages.NETWORK_ERROR.format(3, "timeout")
        assert "3" in msg and "timeout" in msg

    def test_invalid_interpolation_format(self):
        msg = ErrorMessages.INVALID_INTERPOLATION.format("quadratic", "nearest, bilinear")
        assert "quadratic" in msg

    def test_invalid_output_format_format(self):
        msg = ErrorMessages.INVALID_OUTPUT_FORMAT.format("bmp", "geotiff, png")
        assert "bmp" in msg

    def test_invalid_contour_interval_format(self):
        msg = ErrorMessages.INVALID_CONTOUR_INTERVAL.format(-50)
        assert "-50" in msg

    def test_source_not_downloadable_format(self):
        msg = ErrorMessages.SOURCE_NOT_DOWNLOADABLE.format("ASTER")
        assert "ASTER" in msg


# ── SuccessMessages ────────────────────────────────────────────────


class TestSuccessMessages:
    def test_sources_list_format(self):
        msg = SuccessMessages.SOURCES_LIST.format(6)
        assert "6" in msg

    def test_source_describe_format(self):
        msg = SuccessMessages.SOURCE_DESCRIBE.format("Copernicus GLO-30", 30, "global")
        assert "Copernicus GLO-30" in msg

    def test_coverage_full(self):
        assert "coverage" in SuccessMessages.COVERAGE_FULL.lower()

    def test_coverage_partial_format(self):
        msg = SuccessMessages.COVERAGE_PARTIAL.format(75.3)
        assert "75.3" in msg

    def test_size_estimate_format(self):
        msg = SuccessMessages.SIZE_ESTIMATE.format(25.6, 1.2)
        assert "25.6" in msg and "1.2" in msg

    def test_fetch_complete_format(self):
        msg = SuccessMessages.FETCH_COMPLETE.format(12.5)
        assert "12.5" in msg

    def test_point_elevation_format(self):
        msg = SuccessMessages.POINT_ELEVATION.format(1234.5, 4.0)
        assert "1234.5" in msg and "4.0" in msg

    def test_points_elevation_format(self):
        msg = SuccessMessages.POINTS_ELEVATION.format(5)
        assert "5" in msg

    def test_status_format(self):
        msg = SuccessMessages.STATUS.format("0.1.0", 6, "memory")
        assert "0.1.0" in msg and "6" in msg and "memory" in msg

    def test_contour_complete_format(self):
        msg = SuccessMessages.CONTOUR_COMPLETE.format("100x100", 50, 8)
        assert "100x100" in msg and "50" in msg and "8" in msg

    def test_watershed_complete_format(self):
        msg = SuccessMessages.WATERSHED_COMPLETE.format("100x100", 5000)
        assert "100x100" in msg and "5000" in msg


# ── License Warning ───────────────────────────────────────────────


class TestLicenseWarning:
    def test_fabdem_returns_warning(self):
        warning = get_license_warning("fabdem")
        assert warning is not None
        assert "CC-BY-NC-SA-4.0" in warning

    def test_fabdem_warning_matches_constant(self):
        assert get_license_warning("fabdem") == FABDEM_LICENSE_WARNING

    def test_cop30_returns_none(self):
        assert get_license_warning("cop30") is None

    def test_cop90_returns_none(self):
        assert get_license_warning("cop90") is None

    def test_srtm_returns_none(self):
        assert get_license_warning("srtm") is None

    def test_aster_returns_none(self):
        assert get_license_warning("aster") is None

    def test_3dep_returns_none(self):
        assert get_license_warning("3dep") is None

    def test_unknown_source_returns_none(self):
        assert get_license_warning("bogus") is None

    def test_fabdem_warning_mentions_noncommercial(self):
        warning = get_license_warning("fabdem")
        assert "non-commercial" in warning.lower()

    def test_fabdem_warning_mentions_bristol(self):
        warning = get_license_warning("fabdem")
        assert "Bristol" in warning


# ── ML / Phase 3.0 constants ──────────────────────────────────────


class TestTemporalChangeDefaults:
    def test_default_significance_threshold(self):
        assert DEFAULT_SIGNIFICANCE_THRESHOLD_M == 1.0

    def test_significance_threshold_positive(self):
        assert DEFAULT_SIGNIFICANCE_THRESHOLD_M > 0


class TestLandformConstants:
    def test_landform_classes_has_nine_entries(self):
        assert len(LANDFORM_CLASSES) == 9

    def test_landform_classes_contains_plain(self):
        assert "plain" in LANDFORM_CLASSES

    def test_landform_classes_contains_ridge(self):
        assert "ridge" in LANDFORM_CLASSES

    def test_landform_classes_contains_valley(self):
        assert "valley" in LANDFORM_CLASSES

    def test_landform_classes_contains_escarpment(self):
        assert "escarpment" in LANDFORM_CLASSES

    def test_landform_methods_contains_rule_based(self):
        assert "rule_based" in LANDFORM_METHODS


class TestAnomalyDefaults:
    def test_default_anomaly_sensitivity(self):
        assert DEFAULT_ANOMALY_SENSITIVITY == 0.1

    def test_sensitivity_between_zero_and_one(self):
        assert 0 < DEFAULT_ANOMALY_SENSITIVITY < 1


class TestPhase3ErrorMessages:
    def test_invalid_sensitivity_format(self):
        msg = ErrorMessages.INVALID_SENSITIVITY.format(1.5)
        assert "1.5" in msg

    def test_invalid_landform_method_format(self):
        msg = ErrorMessages.INVALID_LANDFORM_METHOD.format("cnn", "rule_based")
        assert "cnn" in msg and "rule_based" in msg

    def test_sklearn_not_available(self):
        assert "scikit-learn" in ErrorMessages.SKLEARN_NOT_AVAILABLE

    def test_feature_detection_not_available(self):
        assert "not yet available" in ErrorMessages.FEATURE_DETECTION_NOT_AVAILABLE

    def test_invalid_feature_method_format(self):
        msg = ErrorMessages.INVALID_FEATURE_METHOD.format("bad", "cnn_hillshade")
        assert "bad" in msg and "cnn_hillshade" in msg


class TestFeatureConstants:
    def test_feature_classes_has_seven_entries(self):
        from chuk_mcp_dem.constants import FEATURE_CLASSES

        assert len(FEATURE_CLASSES) == 7

    def test_feature_classes_starts_with_none(self):
        from chuk_mcp_dem.constants import FEATURE_CLASSES

        assert FEATURE_CLASSES[0] == "none"

    def test_feature_methods_has_one_entry(self):
        from chuk_mcp_dem.constants import FEATURE_METHODS

        assert len(FEATURE_METHODS) == 1

    def test_feature_methods_contains_cnn_hillshade(self):
        from chuk_mcp_dem.constants import FEATURE_METHODS

        assert "cnn_hillshade" in FEATURE_METHODS


class TestPhase3SuccessMessages:
    def test_temporal_change_complete_format(self):
        msg = SuccessMessages.TEMPORAL_CHANGE_COMPLETE.format("100x100", 5000.0, 3000.0)
        assert "100x100" in msg and "5000.0" in msg

    def test_landform_complete_format(self):
        msg = SuccessMessages.LANDFORM_COMPLETE.format("100x100", "plain")
        assert "100x100" in msg and "plain" in msg

    def test_anomaly_complete_format(self):
        msg = SuccessMessages.ANOMALY_COMPLETE.format("100x100", 5)
        assert "100x100" in msg and "5" in msg

    def test_feature_detect_complete_format(self):
        msg = SuccessMessages.FEATURE_DETECT_COMPLETE.format("100x100", 3)
        assert "100x100" in msg and "3" in msg


# ── Phase 3.1: Interpretation constants ────────────────────────────


class TestInterpretationContexts:
    """Tests for INTERPRETATION_CONTEXTS list."""

    def test_has_expected_entries(self):
        from chuk_mcp_dem.constants import INTERPRETATION_CONTEXTS

        assert "general" in INTERPRETATION_CONTEXTS
        assert "archaeological_survey" in INTERPRETATION_CONTEXTS
        assert "flood_risk" in INTERPRETATION_CONTEXTS
        assert "geological" in INTERPRETATION_CONTEXTS
        assert "military_history" in INTERPRETATION_CONTEXTS
        assert "urban_planning" in INTERPRETATION_CONTEXTS

    def test_has_six_entries(self):
        from chuk_mcp_dem.constants import INTERPRETATION_CONTEXTS

        assert len(INTERPRETATION_CONTEXTS) == 6

    def test_general_is_first(self):
        from chuk_mcp_dem.constants import INTERPRETATION_CONTEXTS

        assert INTERPRETATION_CONTEXTS[0] == "general"


class TestPhase31ErrorMessages:
    """Tests for Phase 3.1 error message format strings."""

    def test_invalid_interpretation_context_format(self):
        msg = ErrorMessages.INVALID_INTERPRETATION_CONTEXT.format(
            "bad_ctx", "general, archaeological_survey"
        )
        assert "bad_ctx" in msg
        assert "general, archaeological_survey" in msg

    def test_invalid_artifact_ref_format(self):
        msg = ErrorMessages.INVALID_ARTIFACT_REF.format("ref/missing123")
        assert "ref/missing123" in msg

    def test_sampling_not_supported_mentions_mcp(self):
        assert "MCP" in ErrorMessages.SAMPLING_NOT_SUPPORTED
        assert "sampling" in ErrorMessages.SAMPLING_NOT_SUPPORTED.lower()


class TestPhase31SuccessMessages:
    """Tests for Phase 3.1 success message format strings."""

    def test_interpret_complete_format(self):
        msg = SuccessMessages.INTERPRET_COMPLETE.format("dem/abc123.tif")
        assert "dem/abc123.tif" in msg
        assert "interpretation" in msg.lower() or "Terrain" in msg
