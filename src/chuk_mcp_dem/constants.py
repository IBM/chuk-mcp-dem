"""
Constants for chuk-mcp-dem server.

All magic strings, source metadata, and configuration values live here.
"""


class ServerConfig:
    NAME = "chuk-mcp-dem"
    VERSION = "0.1.0"
    DESCRIPTION = "Digital Elevation Model Discovery, Retrieval & Terrain Analysis MCP Server"


class StorageProvider:
    MEMORY = "memory"
    S3 = "s3"
    FILESYSTEM = "filesystem"


class SessionProvider:
    MEMORY = "memory"
    REDIS = "redis"


class EnvVar:
    ARTIFACTS_PROVIDER = "CHUK_ARTIFACTS_PROVIDER"
    BUCKET_NAME = "BUCKET_NAME"
    REDIS_URL = "REDIS_URL"
    ARTIFACTS_PATH = "CHUK_ARTIFACTS_PATH"
    AWS_ACCESS_KEY_ID = "AWS_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY = "AWS_SECRET_ACCESS_KEY"
    AWS_ENDPOINT_URL_S3 = "AWS_ENDPOINT_URL_S3"
    MCP_STDIO = "MCP_STDIO"


class DEMSource:
    COP30 = "cop30"
    COP90 = "cop90"
    SRTM = "srtm"
    ASTER = "aster"
    TDEP = "3dep"
    FABDEM = "fabdem"


DEFAULT_SOURCE = DEMSource.COP30

# Full source metadata
DEM_SOURCES: dict[str, dict] = {
    DEMSource.COP30: {
        "id": DEMSource.COP30,
        "name": "Copernicus GLO-30",
        "resolution_m": 30,
        "coverage": "global",
        "coverage_bounds": [-180, -90, 180, 90],
        "vertical_datum": "EGM2008",
        "vertical_unit": "metres",
        "horizontal_crs": "EPSG:4326",
        "tile_size_degrees": 1.0,
        "dtype": "float32",
        "nodata_value": -9999.0,
        "void_filled": True,
        "acquisition_period": "2011-2015",
        "source_sensors": ["TanDEM-X", "SRTM"],
        "accuracy_vertical_m": 4.0,
        "access_url": "https://copernicus-dem-30m.s3.amazonaws.com",
        "license": "CC-BY-4.0",
        "llm_guidance": (
            "Best choice for global terrain analysis. 30m resolution sufficient for "
            "regional studies. Use cop90 for faster downloads over large areas. "
            "Use 3dep for US sites requiring sub-10m detail."
        ),
    },
    DEMSource.COP90: {
        "id": DEMSource.COP90,
        "name": "Copernicus GLO-90",
        "resolution_m": 90,
        "coverage": "global",
        "coverage_bounds": [-180, -90, 180, 90],
        "vertical_datum": "EGM2008",
        "vertical_unit": "metres",
        "horizontal_crs": "EPSG:4326",
        "tile_size_degrees": 1.0,
        "dtype": "float32",
        "nodata_value": -9999.0,
        "void_filled": True,
        "acquisition_period": "2011-2015",
        "source_sensors": ["TanDEM-X", "SRTM"],
        "accuracy_vertical_m": 4.0,
        "access_url": "https://copernicus-dem-90m.s3.amazonaws.com",
        "license": "CC-BY-4.0",
        "llm_guidance": (
            "3x coarser than GLO-30 but 9x faster to download. Good for large-area "
            "overviews and continental-scale analysis. Use cop30 for detailed studies."
        ),
    },
    DEMSource.SRTM: {
        "id": DEMSource.SRTM,
        "name": "SRTM v3",
        "resolution_m": 30,
        "coverage": "60N-56S",
        "coverage_bounds": [-180, -56, 180, 60],
        "vertical_datum": "EGM96",
        "vertical_unit": "metres",
        "horizontal_crs": "EPSG:4326",
        "tile_size_degrees": 1.0,
        "dtype": "int16",
        "nodata_value": -32768.0,
        "void_filled": False,
        "acquisition_period": "2000",
        "source_sensors": ["STS-99 Shuttle Radar"],
        "accuracy_vertical_m": 16.0,
        "access_url": "https://elevation-tiles-prod.s3.amazonaws.com",
        "license": "Public Domain",
        "llm_guidance": (
            "Classic 30m DEM covering 60N-56S. May have voids in mountainous areas. "
            "Prefer cop30 for void-free global coverage."
        ),
    },
    DEMSource.ASTER: {
        "id": DEMSource.ASTER,
        "name": "ASTER GDEM v3",
        "resolution_m": 30,
        "coverage": "83N-83S",
        "coverage_bounds": [-180, -83, 180, 83],
        "vertical_datum": "EGM96",
        "vertical_unit": "metres",
        "horizontal_crs": "EPSG:4326",
        "tile_size_degrees": 1.0,
        "dtype": "int16",
        "nodata_value": -9999.0,
        "void_filled": False,
        "acquisition_period": "2000-2011",
        "source_sensors": ["ASTER"],
        "accuracy_vertical_m": 17.0,
        "access_url": "https://gdemdl.aster.jspacesystems.or.jp",
        "license": "Public Domain",
        "llm_guidance": (
            "Wider latitude coverage than SRTM (83N-83S vs 60N-56S). "
            "Lower accuracy than Copernicus. Prefer cop30 for most use cases."
        ),
    },
    DEMSource.TDEP: {
        "id": DEMSource.TDEP,
        "name": "3DEP",
        "resolution_m": 10,
        "coverage": "USA",
        "coverage_bounds": [-180, 17, -64, 72],
        "vertical_datum": "NAVD88",
        "vertical_unit": "metres",
        "horizontal_crs": "EPSG:4326",
        "tile_size_degrees": 1.0,
        "dtype": "float32",
        "nodata_value": -9999.0,
        "void_filled": True,
        "acquisition_period": "2000-2024",
        "source_sensors": ["LiDAR", "IfSAR"],
        "accuracy_vertical_m": 1.0,
        "access_url": "https://prd-tnm.s3.amazonaws.com",
        "license": "Public Domain",
        "llm_guidance": (
            "Highest resolution (1-10m) DEM for the USA. LiDAR-derived. "
            "Use for US sites requiring sub-10m detail. Not available outside USA."
        ),
    },
    DEMSource.FABDEM: {
        "id": DEMSource.FABDEM,
        "name": "FABDEM",
        "resolution_m": 30,
        "coverage": "global",
        "coverage_bounds": [-180, -90, 180, 90],
        "vertical_datum": "EGM2008",
        "vertical_unit": "metres",
        "horizontal_crs": "EPSG:4326",
        "tile_size_degrees": 1.0,
        "dtype": "float32",
        "nodata_value": -9999.0,
        "void_filled": True,
        "acquisition_period": "2011-2015",
        "source_sensors": ["TanDEM-X (processed)"],
        "accuracy_vertical_m": 4.0,
        "access_url": "https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn",
        "license": "CC-BY-NC-SA-4.0",
        "llm_guidance": (
            "Copernicus GLO-30 with forests and buildings removed. Shows bare-earth "
            "elevation. Better for hydrological modelling and flood risk. "
            "Non-commercial license."
        ),
    },
}

ALL_SOURCE_IDS = list(DEM_SOURCES.keys())

# Interpolation methods
INTERPOLATION_METHODS = ["nearest", "bilinear", "cubic"]
DEFAULT_INTERPOLATION = "bilinear"

# Terrain defaults
DEFAULT_AZIMUTH = 315.0
DEFAULT_ALTITUDE = 45.0
DEFAULT_Z_FACTOR = 1.0
DEFAULT_FLAT_VALUE = -1.0
DEFAULT_WINDOW_SIZE = 3

# Terrain derivatives (available analysis types)
TERRAIN_DERIVATIVES = ["hillshade", "slope", "aspect", "curvature", "tri", "contour", "watershed"]
ANALYSIS_TOOLS = [
    "profile", "viewshed", "temporal_change", "landforms", "anomalies", "features",
]
OUTPUT_FORMATS = ["geotiff", "png"]
SLOPE_UNITS = ["degrees", "percent"]

# Contour defaults
DEFAULT_CONTOUR_INTERVAL_M = 100.0

# Profile & viewshed defaults
DEFAULT_NUM_POINTS = 100
DEFAULT_OBSERVER_HEIGHT_M = 1.8
MAX_VIEWSHED_RADIUS_M = 50000.0  # 50 km max radius

# Temporal change defaults
DEFAULT_SIGNIFICANCE_THRESHOLD_M = 1.0

# Landform classification
LANDFORM_CLASSES = [
    "plain", "ridge", "valley", "plateau", "escarpment",
    "depression", "saddle", "terrace", "alluvial_fan",
]
LANDFORM_METHODS = ["rule_based"]

# Feature detection
FEATURE_CLASSES = ["none", "peak", "ridge", "valley", "cliff", "saddle", "channel"]
FEATURE_METHODS = ["cnn_hillshade"]

# Anomaly detection defaults
DEFAULT_ANOMALY_SENSITIVITY = 0.1  # Isolation Forest contamination parameter

# Interpretation contexts
INTERPRETATION_CONTEXTS = [
    "general",
    "archaeological_survey",
    "flood_risk",
    "geological",
    "military_history",
    "urban_planning",
]

# Cache & retry
TILE_CACHE_MAX_BYTES = 100 * 1024 * 1024  # 100 MB total
TILE_CACHE_MAX_ITEM = 10 * 1024 * 1024  # 10 MB per item
RETRY_ATTEMPTS = 3
RETRY_WAIT_MIN = 1
RETRY_WAIT_MAX = 10

# Size limits
MAX_DOWNLOAD_BYTES = 2 * 1024 * 1024 * 1024  # 2 GB
SIZE_WARNING_BYTES = 500 * 1024 * 1024  # 500 MB
SIZE_LARGE_BYTES = 1024 * 1024 * 1024  # 1 GB


FABDEM_LICENSE_WARNING = (
    "FABDEM is licensed under CC-BY-NC-SA-4.0 (non-commercial). "
    "Commercial use requires permission from the University of Bristol."
)


def get_license_warning(source: str) -> str | None:
    """Return a license warning string if the source has restrictions, else None."""
    if source == DEMSource.FABDEM:
        return FABDEM_LICENSE_WARNING
    return None


class ErrorMessages:
    UNKNOWN_SOURCE = "Unknown DEM source '{}'. Available: {}"
    INVALID_BBOX = "Invalid bounding box: must be [west, south, east, north]"
    INVALID_BBOX_VALUES = "Invalid bounding box values: west ({}) must be < east ({})"
    INVALID_BBOX_LAT = "Invalid bounding box values: south ({}) must be < north ({})"
    COVERAGE_ERROR = "{} does not cover the requested area ({})"
    NO_ARTIFACT_STORE = (
        "No artifact store available. Configure CHUK_ARTIFACTS_PROVIDER "
        "environment variable (memory, filesystem, or s3)."
    )
    AREA_TOO_LARGE = "Requested area ({:.1f} MB) exceeds limit. Add bbox or use {}"
    NETWORK_ERROR = "Failed to fetch tile after {} retries: {}"
    INVALID_INTERPOLATION = "Invalid interpolation '{}'. Available: {}"
    INVALID_OUTPUT_FORMAT = "Invalid output format '{}'. Available: {}"
    INVALID_SLOPE_UNITS = "Invalid slope units '{}'. Available: {}"
    INVALID_NUM_POINTS = "num_points must be >= 2, got {}"
    INVALID_RADIUS = "radius_m must be > 0, got {}"
    RADIUS_TOO_LARGE = "radius_m ({:.0f}) exceeds maximum ({:.0f})"
    INVALID_CONTOUR_INTERVAL = "interval_m must be > 0, got {}"
    SOURCE_NOT_DOWNLOADABLE = "{} does not support direct download (authentication required)"
    INVALID_SENSITIVITY = "sensitivity must be between 0 and 1, got {}"
    INVALID_LANDFORM_METHOD = "Invalid landform method '{}'. Available: {}"
    INVALID_FEATURE_METHOD = "Invalid feature method '{}'. Available: {}"
    SKLEARN_NOT_AVAILABLE = (
        "scikit-learn is required for anomaly detection. "
        "Install with: pip install chuk-mcp-dem[ml]"
    )
    FEATURE_DETECTION_NOT_AVAILABLE = (
        "CNN feature detection is not yet available. "
        "Trained models are required but not yet released. "
        "Multi-angle hillshade generation is functional."
    )
    SAMPLING_NOT_SUPPORTED = (
        "MCP sampling is not supported by this client. "
        "dem_interpret requires an MCP client that supports the sampling/createMessage "
        "capability (e.g., Claude Desktop). The terrain artifact is still available "
        "at the provided artifact_ref for manual inspection."
    )
    INVALID_INTERPRETATION_CONTEXT = "Invalid interpretation context '{}'. Available: {}"
    INVALID_ARTIFACT_REF = "Artifact '{}' not found or could not be retrieved"


class SuccessMessages:
    SOURCES_LIST = "{} DEM sources available"
    SOURCE_DESCRIBE = "Source: {} ({}m, {})"
    COVERAGE_FULL = "Full coverage available"
    COVERAGE_PARTIAL = "Partial coverage: {:.1f}%"
    SIZE_ESTIMATE = "Estimated {:.1f} MB for {:.1f} megapixels"
    FETCH_COMPLETE = "Downloaded {:.1f} MB elevation data"
    POINT_ELEVATION = "Elevation at point: {:.1f}m (±{:.1f}m)"
    POINTS_ELEVATION = "Retrieved elevation for {} points"
    STATUS = "DEM MCP Server v{} ({} sources, storage: {})"
    HILLSHADE_COMPLETE = "Hillshade computed ({} shape, azimuth {:.0f}, altitude {:.0f})"
    SLOPE_COMPLETE = "Slope computed ({} shape, units: {})"
    ASPECT_COMPLETE = "Aspect computed ({} shape, flat value: {})"
    CURVATURE_COMPLETE = "Curvature computed ({} shape)"
    TRI_COMPLETE = "Terrain Ruggedness Index computed ({} shape)"
    CONTOUR_COMPLETE = "Contours generated ({} shape, interval {}m, {} contour levels)"
    PROFILE_COMPLETE = "Profile extracted: {} points over {:.1f}m"
    VIEWSHED_COMPLETE = "Viewshed computed: {:.1f}% visible within {:.0f}m radius"
    WATERSHED_COMPLETE = "Watershed computed ({} shape, max accumulation {:.0f} cells)"
    TEMPORAL_CHANGE_COMPLETE = (
        "Elevation change computed ({} shape, {:.1f} m³ gained, {:.1f} m³ lost)"
    )
    LANDFORM_COMPLETE = "Landforms classified ({} shape, dominant: {})"
    ANOMALY_COMPLETE = "Anomaly detection complete ({} shape, {} anomalies found)"
    FEATURE_DETECT_COMPLETE = "Feature detection complete ({} shape, {} features found)"
    INTERPRET_COMPLETE = "Terrain interpretation complete for artifact {}"
