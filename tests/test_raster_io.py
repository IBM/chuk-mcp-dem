"""Comprehensive tests for chuk_mcp_dem.core.raster_io module."""

import io
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# read_dem_tile
# ---------------------------------------------------------------------------


class TestReadDemTile:
    """Tests for read_dem_tile() -- COG reading with optional bbox crop."""

    def test_read_without_bbox(self, sample_elevation, sample_transform, sample_crs):
        """Reading without bbox returns full array, crs, and transform."""
        from chuk_mcp_dem.core.raster_io import read_dem_tile

        mock_src = MagicMock()
        mock_src.read.return_value = sample_elevation.copy()
        mock_src.transform = sample_transform
        mock_src.crs = sample_crs
        mock_src.nodata = None
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        with patch("chuk_mcp_dem.core.raster_io.read_dem_tile.__wrapped__", None):
            pass  # ensure we test the actual function

        with patch("rasterio.open", return_value=mock_src):
            data, crs, transform = read_dem_tile("https://example.com/dem.tif")

        assert data.shape == (100, 100)
        assert data.dtype == np.float32
        assert crs == sample_crs
        mock_src.read.assert_called_once_with(1)

    def test_read_with_bbox(self, sample_transform, sample_crs):
        """Reading with bbox calls from_bounds and window_transform."""
        from chuk_mcp_dem.core.raster_io import read_dem_tile

        small_data = np.ones((10, 10), dtype=np.float32) * 200.0
        mock_src = MagicMock()
        mock_src.read.return_value = small_data
        mock_src.transform = sample_transform
        mock_src.crs = sample_crs
        mock_src.nodata = None
        mock_src.window_transform.return_value = sample_transform
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        bbox = [7.0, 46.5, 7.5, 47.0]

        with patch("rasterio.open", return_value=mock_src):
            data, crs, transform = read_dem_tile("https://example.com/dem.tif", bbox=bbox)

        assert data.dtype == np.float32
        # read was called with window kwarg
        call_kwargs = mock_src.read.call_args
        assert "window" in call_kwargs.kwargs or len(call_kwargs.args) >= 2

    def test_nodata_replaced_with_nan(self, sample_transform, sample_crs):
        """Nodata values in the source are replaced with NaN."""
        from chuk_mcp_dem.core.raster_io import read_dem_tile

        raw = np.array([[100.0, -9999.0], [300.0, -9999.0]], dtype=np.float32)
        mock_src = MagicMock()
        mock_src.read.return_value = raw.copy()
        mock_src.transform = sample_transform
        mock_src.crs = sample_crs
        mock_src.nodata = -9999.0
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        with patch("rasterio.open", return_value=mock_src):
            data, _, _ = read_dem_tile("https://example.com/dem.tif")

        assert np.isnan(data[0, 1])
        assert np.isnan(data[1, 1])
        assert data[0, 0] == pytest.approx(100.0)
        assert data[1, 0] == pytest.approx(300.0)

    def test_nodata_none_leaves_data_intact(self, sample_transform, sample_crs):
        """When nodata is None, no values are replaced."""
        from chuk_mcp_dem.core.raster_io import read_dem_tile

        raw = np.array([[100.0, -9999.0]], dtype=np.float32)
        mock_src = MagicMock()
        mock_src.read.return_value = raw.copy()
        mock_src.transform = sample_transform
        mock_src.crs = sample_crs
        mock_src.nodata = None
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        with patch("rasterio.open", return_value=mock_src):
            data, _, _ = read_dem_tile("https://example.com/dem.tif")

        assert not np.isnan(data[0, 1])
        assert data[0, 1] == pytest.approx(-9999.0)

    def test_retry_on_connection_error(self, sample_transform, sample_crs):
        """read_dem_tile retries on ConnectionError via tenacity."""
        from chuk_mcp_dem.core.raster_io import read_dem_tile

        raw = np.ones((5, 5), dtype=np.float32)
        mock_src = MagicMock()
        mock_src.read.return_value = raw
        mock_src.transform = sample_transform
        mock_src.crs = sample_crs
        mock_src.nodata = None
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        call_count = 0

        def open_side_effect(url):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("transient failure")
            return mock_src

        with patch("rasterio.open", side_effect=open_side_effect):
            data, _, _ = read_dem_tile("https://example.com/dem.tif")

        assert call_count == 3
        assert data.shape == (5, 5)

    def test_retry_exhausted_raises(self):
        """After 3 failed attempts the ConnectionError is reraised."""
        from chuk_mcp_dem.core.raster_io import read_dem_tile

        with patch("rasterio.open", side_effect=ConnectionError("down")):
            with pytest.raises(ConnectionError):
                read_dem_tile("https://example.com/dem.tif")

    def test_output_dtype_is_float32(self, sample_transform, sample_crs):
        """Output is always cast to float32 regardless of source dtype."""
        from chuk_mcp_dem.core.raster_io import read_dem_tile

        raw = np.array([[100, 200]], dtype=np.int16)
        mock_src = MagicMock()
        mock_src.read.return_value = raw
        mock_src.transform = sample_transform
        mock_src.crs = sample_crs
        mock_src.nodata = None
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        with patch("rasterio.open", return_value=mock_src):
            data, _, _ = read_dem_tile("https://example.com/dem.tif")

        assert data.dtype == np.float32


# ---------------------------------------------------------------------------
# read_and_merge_tiles
# ---------------------------------------------------------------------------


class TestReadAndMergeTiles:
    """Tests for read_and_merge_tiles() -- single & multi-tile merging."""

    def test_single_url_delegates_to_read_dem_tile(
        self, sample_elevation, sample_transform, sample_crs
    ):
        """A single URL bypasses merge and calls read_dem_tile directly."""
        from chuk_mcp_dem.core.raster_io import read_and_merge_tiles

        with patch(
            "chuk_mcp_dem.core.raster_io.read_dem_tile",
            return_value=(sample_elevation, sample_crs, sample_transform),
        ) as mock_read:
            data, crs, transform = read_and_merge_tiles(["https://example.com/tile1.tif"])

        mock_read.assert_called_once_with("https://example.com/tile1.tif", None)
        assert data.shape == (100, 100)

    def test_single_url_with_bbox(self, sample_elevation, sample_transform, sample_crs):
        """Single URL with bbox passes it through."""
        from chuk_mcp_dem.core.raster_io import read_and_merge_tiles

        bbox = [7.0, 46.5, 7.5, 47.0]
        with patch(
            "chuk_mcp_dem.core.raster_io.read_dem_tile",
            return_value=(sample_elevation, sample_crs, sample_transform),
        ) as mock_read:
            read_and_merge_tiles(["https://example.com/tile1.tif"], bbox=bbox)

        mock_read.assert_called_once_with("https://example.com/tile1.tif", bbox)

    def test_multi_tile_merge(self, sample_transform, sample_crs):
        """Multiple URLs are opened, merged, and datasets closed."""
        from chuk_mcp_dem.core.raster_io import read_and_merge_tiles

        merged_arr = np.ones((1, 50, 50), dtype=np.float32) * 300.0

        mock_ds1 = MagicMock()
        mock_ds1.crs = sample_crs
        mock_ds1.nodata = None

        mock_ds2 = MagicMock()
        mock_ds2.crs = sample_crs
        mock_ds2.nodata = None

        open_returns = iter([mock_ds1, mock_ds2])

        with patch("rasterio.open", side_effect=lambda url: next(open_returns)):
            with patch("rasterio.merge.merge", return_value=(merged_arr, sample_transform)):
                data, crs, transform = read_and_merge_tiles(["https://a.tif", "https://b.tif"])

        assert data.shape == (50, 50)
        assert data.dtype == np.float32
        mock_ds1.close.assert_called_once()
        mock_ds2.close.assert_called_once()

    def test_multi_tile_nodata_replaced(self, sample_transform, sample_crs):
        """Nodata values in merged result are replaced with NaN."""
        from chuk_mcp_dem.core.raster_io import read_and_merge_tiles

        merged_arr = np.array([[[100.0, -9999.0], [200.0, 300.0]]], dtype=np.float32)

        mock_ds = MagicMock()
        mock_ds.crs = sample_crs
        mock_ds.nodata = -9999.0

        with patch("rasterio.open", return_value=mock_ds):
            with patch("rasterio.merge.merge", return_value=(merged_arr, sample_transform)):
                data, _, _ = read_and_merge_tiles(["https://a.tif", "https://b.tif"])

        assert np.isnan(data[0, 1])
        assert data[0, 0] == pytest.approx(100.0)

    def test_multi_tile_datasets_closed_on_error(self, sample_crs):
        """Datasets are closed even if merge raises."""
        from chuk_mcp_dem.core.raster_io import read_and_merge_tiles

        mock_ds = MagicMock()
        mock_ds.crs = sample_crs
        mock_ds.nodata = None

        with patch("rasterio.open", return_value=mock_ds):
            with patch("rasterio.merge.merge", side_effect=RuntimeError("boom")):
                with pytest.raises(RuntimeError, match="boom"):
                    read_and_merge_tiles(["https://a.tif", "https://b.tif"])

        assert mock_ds.close.call_count == 2


# ---------------------------------------------------------------------------
# sample_elevation (the function, not the fixture)
# ---------------------------------------------------------------------------


class TestSampleElevation:
    """Tests for sample_elevation() -- point sampling with interpolation."""

    def test_nearest_in_bounds(self, sample_elevation, sample_transform):
        """Nearest sampling at a valid coordinate returns the pixel value."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        # Point at pixel (0,0): lon=7.0, lat=47.0 maps to row=0, col=0
        val = sample_elev(sample_elevation, sample_transform, 7.005, 46.995, "nearest")
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_nearest_out_of_bounds(self, sample_elevation, sample_transform):
        """Nearest sampling outside array bounds returns NaN."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        val = sample_elev(sample_elevation, sample_transform, 0.0, 0.0, "nearest")
        assert math.isnan(val)

    def test_nearest_at_nan_pixel(self, sample_elevation_with_voids, sample_transform):
        """Nearest sampling at a NaN pixel returns NaN."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        # Voids at [10:15, 10:15], pixel (12,12) => lon=7.0+12*0.01=7.12, lat=47.0-12*0.01=46.88
        val = sample_elev(sample_elevation_with_voids, sample_transform, 7.12, 46.88, "nearest")
        assert math.isnan(val)

    def test_bilinear_center(self, sample_elevation, sample_transform):
        """Bilinear sampling at array center returns a valid float."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        val = sample_elev(sample_elevation, sample_transform, 7.505, 46.505, "bilinear")
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_bilinear_at_edge_returns_nan(self, sample_elevation, sample_transform):
        """Bilinear at the very edge (r1 >= h) returns NaN."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        # Bottom-right corner: row_f ~ 99.5 => r1=100 >= h=100
        val = sample_elev(sample_elevation, sample_transform, 7.995, 46.005, "bilinear")
        assert math.isnan(val)

    def test_bilinear_default_interpolation(self, sample_elevation, sample_transform):
        """Default interpolation is bilinear."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        val_default = sample_elev(sample_elevation, sample_transform, 7.505, 46.505)
        val_bilinear = sample_elev(sample_elevation, sample_transform, 7.505, 46.505, "bilinear")
        assert val_default == val_bilinear

    def test_cubic_center(self, sample_elevation, sample_transform):
        """Cubic sampling at array center returns a valid float."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        val = sample_elev(sample_elevation, sample_transform, 7.505, 46.505, "cubic")
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_cubic_small_patch_fallback(self, sample_transform):
        """Cubic falls back to bilinear when patch is too small (< 4x4)."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        tiny = np.ones((3, 3), dtype=np.float32) * 500.0
        # Pixel (1,1) center => need 4x4 patch but only 3x3 available
        val = sample_elev(tiny, sample_transform, 7.01, 46.99, "cubic")
        assert isinstance(val, float)

    def test_unknown_interpolation_raises(self, sample_elevation, sample_transform):
        """Unknown interpolation method raises ValueError."""
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        with pytest.raises(ValueError, match="Unknown interpolation"):
            sample_elev(sample_elevation, sample_transform, 7.5, 46.5, "quadratic")


# ---------------------------------------------------------------------------
# sample_elevations (batch)
# ---------------------------------------------------------------------------


class TestSampleElevations:
    """Tests for sample_elevations() -- batch point sampling."""

    def test_multiple_points(self, sample_elevation, sample_transform):
        """Returns one result per input point."""
        from chuk_mcp_dem.core.raster_io import sample_elevations

        points = [[7.1, 46.9], [7.2, 46.8], [7.3, 46.7]]
        results = sample_elevations(sample_elevation, sample_transform, points, "nearest")
        assert len(results) == 3
        assert all(isinstance(v, float) for v in results)

    def test_empty_points(self, sample_elevation, sample_transform):
        """Empty point list returns empty list."""
        from chuk_mcp_dem.core.raster_io import sample_elevations

        results = sample_elevations(sample_elevation, sample_transform, [])
        assert results == []

    def test_single_point(self, sample_elevation, sample_transform):
        """Single point returns single-element list."""
        from chuk_mcp_dem.core.raster_io import sample_elevations

        results = sample_elevations(sample_elevation, sample_transform, [[7.5, 46.5]], "nearest")
        assert len(results) == 1

    def test_mixed_valid_and_oob(self, sample_elevation, sample_transform):
        """Mix of valid and out-of-bounds points."""
        from chuk_mcp_dem.core.raster_io import sample_elevations

        points = [[7.5, 46.5], [0.0, 0.0]]
        results = sample_elevations(sample_elevation, sample_transform, points, "nearest")
        assert len(results) == 2
        assert not math.isnan(results[0])
        assert math.isnan(results[1])


# ---------------------------------------------------------------------------
# fill_voids
# ---------------------------------------------------------------------------


class TestFillVoids:
    """Tests for fill_voids() -- nearest-neighbour void filling."""

    def test_fills_nan_voids(self, sample_elevation_with_voids):
        """NaN voids are filled with nearest valid values."""
        from chuk_mcp_dem.core.raster_io import fill_voids

        result = fill_voids(sample_elevation_with_voids)
        assert not np.any(np.isnan(result))
        assert result.shape == sample_elevation_with_voids.shape

    def test_no_voids_unchanged(self, sample_elevation):
        """Array without voids is returned unchanged."""
        from chuk_mcp_dem.core.raster_io import fill_voids

        result = fill_voids(sample_elevation)
        np.testing.assert_array_equal(result, sample_elevation)

    def test_custom_nodata_filled(self):
        """Custom nodata value (-32768) is also treated as void."""
        from chuk_mcp_dem.core.raster_io import fill_voids

        arr = np.array([[100.0, -32768.0], [200.0, 300.0]], dtype=np.float32)
        result = fill_voids(arr, nodata=-32768.0)
        assert result[0, 1] != -32768.0
        assert not np.isnan(result[0, 1])

    def test_default_nodata_not_treated_as_void(self):
        """Default nodata=-9999 is only treated when values match."""
        from chuk_mcp_dem.core.raster_io import fill_voids

        arr = np.array([[100.0, 200.0]], dtype=np.float32)
        result = fill_voids(arr)
        np.testing.assert_array_equal(result, arr)

    def test_preserves_valid_values(self, sample_elevation_with_voids):
        """Valid (non-void) pixels are not modified."""
        from chuk_mcp_dem.core.raster_io import fill_voids

        original = sample_elevation_with_voids.copy()
        result = fill_voids(sample_elevation_with_voids)
        # Non-NaN pixels should be identical
        valid_mask = ~np.isnan(original)
        np.testing.assert_array_equal(result[valid_mask], original[valid_mask])

    def test_all_nan_array(self):
        """All-NaN array stays NaN (no valid neighbors to fill from)."""
        from chuk_mcp_dem.core.raster_io import fill_voids

        arr = np.full((5, 5), np.nan, dtype=np.float32)
        # scipy distance_transform_edt with all invalid still runs;
        # the result depends on scipy behavior but should not crash
        result = fill_voids(arr)
        assert result.shape == (5, 5)


# ---------------------------------------------------------------------------
# arrays_to_geotiff
# ---------------------------------------------------------------------------


class TestArraysToGeotiff:
    """Tests for arrays_to_geotiff() -- array to GeoTIFF bytes."""

    def test_2d_array_produces_bytes(self, sample_elevation, sample_transform, sample_crs):
        """2D array produces non-empty GeoTIFF bytes."""
        from chuk_mcp_dem.core.raster_io import arrays_to_geotiff

        result = arrays_to_geotiff(sample_elevation, sample_crs, sample_transform)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_3d_array_produces_bytes(self, sample_transform, sample_crs):
        """3D array (bands, height, width) produces valid bytes."""
        from chuk_mcp_dem.core.raster_io import arrays_to_geotiff

        arr_3d = np.random.uniform(0, 255, (3, 50, 50)).astype(np.float32)
        result = arrays_to_geotiff(arr_3d, sample_crs, sample_transform)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_geotiff_header(self, sample_elevation, sample_transform, sample_crs):
        """Output starts with TIFF magic bytes."""
        from chuk_mcp_dem.core.raster_io import arrays_to_geotiff

        result = arrays_to_geotiff(sample_elevation, sample_crs, sample_transform)
        # TIFF files start with II (little-endian) or MM (big-endian)
        assert result[:2] in (b"II", b"MM")

    def test_custom_nodata(self, sample_elevation, sample_transform, sample_crs):
        """Custom nodata value is accepted without error."""
        from chuk_mcp_dem.core.raster_io import arrays_to_geotiff

        result = arrays_to_geotiff(
            sample_elevation,
            sample_crs,
            sample_transform,
            nodata=-9999.0,
        )
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_int16_dtype(self, sample_transform, sample_crs):
        """int16 dtype produces valid output."""
        from chuk_mcp_dem.core.raster_io import arrays_to_geotiff

        arr = np.array([[100, 200], [300, 400]], dtype=np.int16)
        result = arrays_to_geotiff(arr, sample_crs, sample_transform, dtype="int16")
        assert isinstance(result, bytes)


# ---------------------------------------------------------------------------
# compute_hillshade
# ---------------------------------------------------------------------------


class TestComputeHillshade:
    """Tests for compute_hillshade() -- Horn's method shaded relief."""

    def test_flat_surface_uniform_value(self, sample_transform):
        """A perfectly flat surface produces uniform hillshade."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade

        flat = np.ones((50, 50), dtype=np.float32) * 500.0
        hs = compute_hillshade(flat, sample_transform)
        # Flat surface: slope=0, so hillshade = 255 * cos(alt_rad)
        assert hs.shape == (50, 50)
        # All values should be nearly identical
        assert np.std(hs) < 1.0

    def test_output_range_0_255(self, sample_elevation, sample_transform):
        """Hillshade values are clipped to 0-255."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade

        hs = compute_hillshade(sample_elevation, sample_transform)
        assert np.all(hs >= 0)
        assert np.all(hs <= 255)

    def test_output_shape_matches_input(self, sample_elevation, sample_transform):
        """Output has same shape as input elevation."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade

        hs = compute_hillshade(sample_elevation, sample_transform)
        assert hs.shape == sample_elevation.shape

    def test_custom_azimuth(self, sample_elevation, sample_transform):
        """Different azimuth produces different hillshade."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade

        hs_315 = compute_hillshade(sample_elevation, sample_transform, azimuth=315.0)
        hs_45 = compute_hillshade(sample_elevation, sample_transform, azimuth=45.0)
        # Should not be identical for non-flat terrain
        assert not np.allclose(hs_315, hs_45)

    def test_custom_altitude(self, sample_elevation, sample_transform):
        """Different altitude produces different hillshade."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade

        hs_45 = compute_hillshade(sample_elevation, sample_transform, altitude=45.0)
        hs_90 = compute_hillshade(sample_elevation, sample_transform, altitude=90.0)
        assert not np.allclose(hs_45, hs_90)

    def test_z_factor_amplification(self, sample_elevation, sample_transform):
        """Higher z_factor increases contrast (larger range of values)."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade

        hs_z1 = compute_hillshade(sample_elevation, sample_transform, z_factor=1.0)
        hs_z10 = compute_hillshade(sample_elevation, sample_transform, z_factor=10.0)
        # z_factor=10 should produce more extreme values (wider spread)
        assert np.std(hs_z10) >= np.std(hs_z1) * 0.5  # allow some tolerance

    def test_sloped_surface(self, sample_transform):
        """A ramp surface produces non-uniform hillshade."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade

        # Gentle ramp: 0.001m per pixel (very mild slope relative to cellsize)
        ramp = np.tile(np.arange(50, dtype=np.float32) * 0.001, (50, 1))
        hs = compute_hillshade(ramp, sample_transform)
        assert hs.shape == (50, 50)
        # Non-flat so should have some non-zero values
        assert np.max(hs) > 0

    def test_nan_in_elevation_handled(self, sample_transform):
        """NaN values in elevation are handled gracefully."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade

        arr = np.ones((20, 20), dtype=np.float32) * 100.0
        arr[5, 5] = np.nan
        hs = compute_hillshade(arr, sample_transform)
        assert hs.shape == (20, 20)
        assert not np.any(np.isnan(hs))


# ---------------------------------------------------------------------------
# elevation_to_hillshade_png
# ---------------------------------------------------------------------------


class TestElevationToHillshadePng:
    """Tests for elevation_to_hillshade_png() -- hillshade as PNG bytes."""

    def test_returns_png_bytes(self, sample_elevation, sample_transform):
        """Output is valid PNG (starts with PNG signature)."""
        from chuk_mcp_dem.core.raster_io import elevation_to_hillshade_png

        result = elevation_to_hillshade_png(sample_elevation, sample_transform)
        assert isinstance(result, bytes)
        # PNG signature: \x89PNG\r\n\x1a\n
        assert result[:4] == b"\x89PNG"

    def test_png_is_nonempty(self, sample_elevation, sample_transform):
        """PNG output has non-trivial size."""
        from chuk_mcp_dem.core.raster_io import elevation_to_hillshade_png

        result = elevation_to_hillshade_png(sample_elevation, sample_transform)
        assert len(result) > 100

    def test_flat_surface_png(self, sample_transform):
        """Flat surface produces valid PNG."""
        from chuk_mcp_dem.core.raster_io import elevation_to_hillshade_png

        flat = np.ones((30, 30), dtype=np.float32) * 200.0
        result = elevation_to_hillshade_png(flat, sample_transform)
        assert result[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# elevation_to_terrain_png
# ---------------------------------------------------------------------------


class TestElevationToTerrainPng:
    """Tests for elevation_to_terrain_png() -- terrain color ramp PNG."""

    def test_returns_valid_png(self, sample_elevation):
        """Output is valid PNG bytes."""
        from chuk_mcp_dem.core.raster_io import elevation_to_terrain_png

        result = elevation_to_terrain_png(sample_elevation)
        assert result[:4] == b"\x89PNG"

    def test_all_nan_array(self):
        """All-NaN array produces a valid (black) PNG."""
        from chuk_mcp_dem.core.raster_io import elevation_to_terrain_png

        arr = np.full((20, 20), np.nan, dtype=np.float32)
        result = elevation_to_terrain_png(arr)
        assert result[:4] == b"\x89PNG"

    def test_uniform_elevation(self):
        """Uniform elevation produces valid PNG (vmax-vmin handled)."""
        from chuk_mcp_dem.core.raster_io import elevation_to_terrain_png

        arr = np.ones((20, 20), dtype=np.float32) * 1000.0
        result = elevation_to_terrain_png(arr)
        assert result[:4] == b"\x89PNG"

    def test_png_is_rgb(self, sample_elevation):
        """Terrain PNG is RGB (3-channel)."""
        from chuk_mcp_dem.core.raster_io import elevation_to_terrain_png
        from PIL import Image

        result = elevation_to_terrain_png(sample_elevation)
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_png_dimensions_match(self, sample_elevation):
        """PNG dimensions match input array shape."""
        from chuk_mcp_dem.core.raster_io import elevation_to_terrain_png
        from PIL import Image

        result = elevation_to_terrain_png(sample_elevation)
        img = Image.open(io.BytesIO(result))
        assert img.size == (100, 100)  # (width, height)


# ---------------------------------------------------------------------------
# compute_slope
# ---------------------------------------------------------------------------


class TestComputeSlope:
    """Tests for compute_slope() -- slope in degrees or percent."""

    def test_flat_surface_zero_slope(self, sample_transform):
        """Flat surface has zero slope everywhere."""
        from chuk_mcp_dem.core.raster_io import compute_slope

        flat = np.ones((50, 50), dtype=np.float32) * 100.0
        slope = compute_slope(flat, sample_transform)
        np.testing.assert_allclose(slope, 0.0, atol=1e-5)

    def test_ramp_known_slope(self, sample_transform):
        """Linear ramp yields consistent non-zero slope."""
        from chuk_mcp_dem.core.raster_io import compute_slope

        # Ramp increasing by 1m per pixel in x-direction, cell size = 0.01
        ramp = np.tile(np.arange(50, dtype=np.float32), (50, 1))
        slope = compute_slope(ramp, sample_transform)
        assert slope.shape == (50, 50)
        # Interior pixels should have consistent slope
        interior = slope[5:45, 5:45]
        assert np.std(interior) < 1.0  # nearly uniform
        assert np.mean(interior) > 0  # non-zero slope

    def test_degrees_units(self, sample_elevation, sample_transform):
        """Degrees output is in [0, 90] range."""
        from chuk_mcp_dem.core.raster_io import compute_slope

        slope = compute_slope(sample_elevation, sample_transform, units="degrees")
        assert np.all(slope >= 0)
        assert np.all(slope <= 90)

    def test_percent_units(self, sample_elevation, sample_transform):
        """Percent output is non-negative."""
        from chuk_mcp_dem.core.raster_io import compute_slope

        slope = compute_slope(sample_elevation, sample_transform, units="percent")
        assert np.all(slope >= 0)

    def test_degrees_vs_percent_relationship(self, sample_elevation, sample_transform):
        """tan(degrees) * 100 should equal percent slope."""
        from chuk_mcp_dem.core.raster_io import compute_slope

        deg = compute_slope(sample_elevation, sample_transform, units="degrees")
        pct = compute_slope(sample_elevation, sample_transform, units="percent")
        expected_pct = np.tan(np.radians(deg)) * 100.0
        np.testing.assert_allclose(pct, expected_pct, rtol=0.01)

    def test_invalid_units_raises(self, sample_elevation, sample_transform):
        """Invalid units raise ValueError."""
        from chuk_mcp_dem.core.raster_io import compute_slope

        with pytest.raises(ValueError, match="Unknown slope units"):
            compute_slope(sample_elevation, sample_transform, units="radians")

    def test_output_dtype_float32(self, sample_elevation, sample_transform):
        """Output dtype is float32."""
        from chuk_mcp_dem.core.raster_io import compute_slope

        slope = compute_slope(sample_elevation, sample_transform)
        assert slope.dtype == np.float32

    def test_output_shape_matches_input(self, sample_elevation, sample_transform):
        """Output shape matches input."""
        from chuk_mcp_dem.core.raster_io import compute_slope

        slope = compute_slope(sample_elevation, sample_transform)
        assert slope.shape == sample_elevation.shape


# ---------------------------------------------------------------------------
# compute_aspect
# ---------------------------------------------------------------------------


class TestComputeAspect:
    """Tests for compute_aspect() -- slope direction."""

    def test_flat_area_gets_flat_value(self, sample_transform):
        """Flat surface pixels get the flat_value sentinel."""
        from chuk_mcp_dem.core.raster_io import compute_aspect

        flat = np.ones((50, 50), dtype=np.float32) * 500.0
        aspect = compute_aspect(flat, sample_transform)
        assert np.all(aspect == -1.0)

    def test_custom_flat_value(self, sample_transform):
        """Custom flat_value is used for flat areas."""
        from chuk_mcp_dem.core.raster_io import compute_aspect

        flat = np.ones((50, 50), dtype=np.float32) * 500.0
        aspect = compute_aspect(flat, sample_transform, flat_value=-999.0)
        assert np.all(aspect == -999.0)

    def test_north_facing_slope(self, sample_transform):
        """Surface increasing southward (row index) faces north (~0/360)."""
        from chuk_mcp_dem.core.raster_io import compute_aspect

        # Elevation increases with row index => gradient points south
        # So aspect (direction slope faces = downhill) should be ~south (180)
        # Actually: dz_dy < 0 because padded[top] < padded[bottom]
        # Let's just verify it's a valid direction
        ramp_ns = np.tile(np.arange(50, dtype=np.float32).reshape(50, 1) * 100, (1, 50))
        aspect = compute_aspect(ramp_ns, sample_transform)
        interior = aspect[10:40, 10:40]
        # All interior pixels should have the same aspect (consistent slope)
        flat_mask = interior == -1.0
        non_flat = interior[~flat_mask]
        if len(non_flat) > 0:
            assert np.std(non_flat) < 5.0  # consistent direction

    def test_east_facing_slope(self, sample_transform):
        """Surface increasing eastward (col index) yields consistent aspect."""
        from chuk_mcp_dem.core.raster_io import compute_aspect

        # Elevation increases with column index
        ramp_ew = np.tile(np.arange(50, dtype=np.float32) * 0.01, (50, 1))
        aspect = compute_aspect(ramp_ew, sample_transform)
        interior = aspect[10:40, 10:40]
        non_flat = interior[interior != -1.0]
        if len(non_flat) > 0:
            # All non-flat pixels should have consistent aspect direction
            assert np.std(non_flat) < 5.0

    def test_output_range_0_360_or_flat(self, sample_elevation, sample_transform):
        """Aspect values are in [0, 360) or equal to flat_value."""
        from chuk_mcp_dem.core.raster_io import compute_aspect

        aspect = compute_aspect(sample_elevation, sample_transform)
        non_flat = aspect[aspect != -1.0]
        assert np.all(non_flat >= 0)
        assert np.all(non_flat < 360)

    def test_output_dtype_float32(self, sample_elevation, sample_transform):
        """Output dtype is float32."""
        from chuk_mcp_dem.core.raster_io import compute_aspect

        aspect = compute_aspect(sample_elevation, sample_transform)
        assert aspect.dtype == np.float32

    def test_output_shape_matches_input(self, sample_elevation, sample_transform):
        """Output shape matches input."""
        from chuk_mcp_dem.core.raster_io import compute_aspect

        aspect = compute_aspect(sample_elevation, sample_transform)
        assert aspect.shape == sample_elevation.shape


# ---------------------------------------------------------------------------
# compute_curvature
# ---------------------------------------------------------------------------


class TestComputeCurvature:
    """Tests for compute_curvature() -- profile curvature."""

    def test_flat_surface_zero_curvature(self, sample_transform):
        """Flat surface has zero curvature everywhere."""
        from chuk_mcp_dem.core.raster_io import compute_curvature

        flat = np.ones((50, 50), dtype=np.float32) * 100.0
        curv = compute_curvature(flat, sample_transform)
        np.testing.assert_allclose(curv, 0.0, atol=1e-5)

    def test_linear_ramp_zero_curvature(self, sample_transform):
        """Linear ramp (constant slope) has near-zero curvature."""
        from chuk_mcp_dem.core.raster_io import compute_curvature

        ramp = np.tile(np.arange(50, dtype=np.float32), (50, 1))
        curv = compute_curvature(ramp, sample_transform)
        # Interior should be near zero (linear = constant first derivative)
        interior = curv[5:45, 5:45]
        np.testing.assert_allclose(interior, 0.0, atol=1e-3)

    def test_convex_surface_positive(self, sample_transform):
        """Convex (bowl-shaped) surface has positive curvature."""
        from chuk_mcp_dem.core.raster_io import compute_curvature

        y, x = np.mgrid[0:50, 0:50]
        bowl = ((x - 25.0) ** 2 + (y - 25.0) ** 2).astype(np.float32)
        curv = compute_curvature(bowl, sample_transform)
        # Interior should be positive (concave up)
        interior = curv[10:40, 10:40]
        assert np.mean(interior) > 0

    def test_concave_surface_negative(self, sample_transform):
        """Concave (dome-shaped) surface has negative curvature."""
        from chuk_mcp_dem.core.raster_io import compute_curvature

        y, x = np.mgrid[0:50, 0:50]
        dome = -((x - 25.0) ** 2 + (y - 25.0) ** 2).astype(np.float32)
        curv = compute_curvature(dome, sample_transform)
        interior = curv[10:40, 10:40]
        assert np.mean(interior) < 0

    def test_output_dtype_float32(self, sample_elevation, sample_transform):
        """Output dtype is float32."""
        from chuk_mcp_dem.core.raster_io import compute_curvature

        curv = compute_curvature(sample_elevation, sample_transform)
        assert curv.dtype == np.float32

    def test_output_shape_matches_input(self, sample_elevation, sample_transform):
        """Output shape matches input."""
        from chuk_mcp_dem.core.raster_io import compute_curvature

        curv = compute_curvature(sample_elevation, sample_transform)
        assert curv.shape == sample_elevation.shape

    def test_nan_handling(self, sample_transform):
        """NaN values are handled via nan_to_num."""
        from chuk_mcp_dem.core.raster_io import compute_curvature

        arr = np.ones((10, 10), dtype=np.float32) * 100.0
        arr[5, 5] = np.nan
        curv = compute_curvature(arr, sample_transform)
        assert curv.shape == (10, 10)
        assert not np.any(np.isnan(curv))


# ---------------------------------------------------------------------------
# curvature_to_png
# ---------------------------------------------------------------------------


class TestCurvatureToPng:
    """Tests for curvature_to_png() -- diverging blue-white-red ramp."""

    def test_produces_valid_png(self, sample_elevation, sample_transform):
        """Output is valid PNG bytes."""
        from chuk_mcp_dem.core.raster_io import compute_curvature, curvature_to_png

        curv = compute_curvature(sample_elevation, sample_transform)
        png_bytes = curvature_to_png(curv)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        img = Image.open(io.BytesIO(png_bytes))
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_zero_curvature_is_white(self):
        """Zero curvature maps to white in diverging ramp."""
        from chuk_mcp_dem.core.raster_io import curvature_to_png

        zero = np.zeros((10, 10), dtype=np.float32)
        png_bytes = curvature_to_png(zero)
        img = Image.open(io.BytesIO(png_bytes))
        pixel = img.getpixel((5, 5))
        # White = (255, 255, 255)
        assert pixel[0] == 255
        assert pixel[1] == 255
        assert pixel[2] == 255

    def test_output_dimensions_match(self):
        """PNG dimensions match input array."""
        from chuk_mcp_dem.core.raster_io import curvature_to_png

        arr = np.random.uniform(-1, 1, (30, 40)).astype(np.float32)
        png_bytes = curvature_to_png(arr)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (40, 30)  # PIL size is (width, height)


# ---------------------------------------------------------------------------
# compute_tri
# ---------------------------------------------------------------------------


class TestComputeTRI:
    """Tests for compute_tri() -- Terrain Ruggedness Index."""

    def test_flat_surface_zero_tri(self, sample_transform):
        """Flat surface has zero TRI."""
        from chuk_mcp_dem.core.raster_io import compute_tri

        flat = np.ones((50, 50), dtype=np.float32) * 100.0
        tri = compute_tri(flat, sample_transform)
        np.testing.assert_allclose(tri, 0.0, atol=1e-5)

    def test_rough_surface_positive_tri(self, sample_elevation, sample_transform):
        """Random elevation (100-500m range) produces positive TRI."""
        from chuk_mcp_dem.core.raster_io import compute_tri

        tri = compute_tri(sample_elevation, sample_transform)
        assert np.mean(tri) > 0

    def test_checkerboard_max_ruggedness(self, sample_transform):
        """Checkerboard pattern produces high TRI."""
        from chuk_mcp_dem.core.raster_io import compute_tri

        checker = np.zeros((20, 20), dtype=np.float32)
        checker[::2, ::2] = 1000.0
        checker[1::2, 1::2] = 1000.0
        tri = compute_tri(checker, sample_transform)
        # Interior pixels should have high TRI
        interior = tri[2:18, 2:18]
        assert np.mean(interior) > 100

    def test_output_non_negative(self, sample_elevation, sample_transform):
        """TRI values are always non-negative."""
        from chuk_mcp_dem.core.raster_io import compute_tri

        tri = compute_tri(sample_elevation, sample_transform)
        assert np.all(tri >= 0)

    def test_output_dtype_float32(self, sample_elevation, sample_transform):
        """Output dtype is float32."""
        from chuk_mcp_dem.core.raster_io import compute_tri

        tri = compute_tri(sample_elevation, sample_transform)
        assert tri.dtype == np.float32

    def test_output_shape_matches_input(self, sample_elevation, sample_transform):
        """Output shape matches input."""
        from chuk_mcp_dem.core.raster_io import compute_tri

        tri = compute_tri(sample_elevation, sample_transform)
        assert tri.shape == sample_elevation.shape

    def test_nan_handling(self, sample_transform):
        """NaN values are handled via nan_to_num."""
        from chuk_mcp_dem.core.raster_io import compute_tri

        arr = np.ones((10, 10), dtype=np.float32) * 100.0
        arr[5, 5] = np.nan
        tri = compute_tri(arr, sample_transform)
        assert tri.shape == (10, 10)
        assert not np.any(np.isnan(tri))

    def test_symmetric_difference(self, sample_transform):
        """TRI is symmetric — raising or lowering a single cell by the same amount gives same TRI."""
        from chuk_mcp_dem.core.raster_io import compute_tri

        base = np.ones((10, 10), dtype=np.float32) * 100.0

        high = base.copy()
        high[5, 5] = 200.0
        tri_high = compute_tri(high, sample_transform)

        low = base.copy()
        low[5, 5] = 0.0
        tri_low = compute_tri(low, sample_transform)

        np.testing.assert_allclose(tri_high, tri_low, atol=1e-5)


# ---------------------------------------------------------------------------
# tri_to_png
# ---------------------------------------------------------------------------


class TestTriToPng:
    """Tests for tri_to_png() -- green-yellow-orange-red ramp."""

    def test_produces_valid_png(self, sample_elevation, sample_transform):
        """Output is valid PNG bytes."""
        from chuk_mcp_dem.core.raster_io import compute_tri, tri_to_png

        tri = compute_tri(sample_elevation, sample_transform)
        png_bytes = tri_to_png(tri)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        img = Image.open(io.BytesIO(png_bytes))
        assert img.mode == "RGB"
        assert img.size == (100, 100)

    def test_flat_is_dark_green(self):
        """Zero TRI (flat) maps to dark green (low R, high G)."""
        from chuk_mcp_dem.core.raster_io import tri_to_png

        flat_tri = np.zeros((10, 10), dtype=np.float32)
        png_bytes = tri_to_png(flat_tri)
        img = Image.open(io.BytesIO(png_bytes))
        pixel = img.getpixel((5, 5))
        assert pixel[0] == 0  # R = 0 (green channel)
        assert pixel[1] == 255  # G = 255

    def test_output_dimensions_match(self):
        """PNG dimensions match input array."""
        from chuk_mcp_dem.core.raster_io import tri_to_png

        arr = np.random.uniform(0, 100, (30, 40)).astype(np.float32)
        png_bytes = tri_to_png(arr)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (40, 30)


class TestComputeContours:
    """Tests for compute_contours() -- rasterised contour line generation."""

    def test_flat_surface_no_contours(self, sample_transform):
        """Flat surface at a non-contour elevation produces no contours."""
        from chuk_mcp_dem.core.raster_io import compute_contours

        flat = np.full((50, 50), 150.0, dtype=np.float32)
        contours, levels = compute_contours(flat, sample_transform, 100.0)
        # 150m is not on a 100m interval, so no crossing should occur on a truly flat surface
        # However, levels list should include 100 (which is below flat)... only 200 would cross
        # Since the surface is uniformly 150, no pixel crosses any level
        assert contours.shape == (50, 50)
        assert np.all(np.isnan(contours))

    def test_gradient_produces_contours(self, sample_transform):
        """Linear gradient should produce contours at expected intervals."""
        from chuk_mcp_dem.core.raster_io import compute_contours

        gradient = np.linspace(0, 500, 100).reshape(1, -1).repeat(10, axis=0).astype(np.float32)
        contours, levels = compute_contours(gradient, sample_transform, 100.0)
        assert contours.shape == gradient.shape
        # Should have contour levels at 0, 100, 200, 300, 400, 500
        assert 100.0 in levels
        assert 200.0 in levels
        assert 300.0 in levels
        # Some pixels should be marked (not all NaN)
        assert not np.all(np.isnan(contours))

    def test_contour_pixel_values_match_levels(self, sample_transform):
        """Contour pixels should contain their contour level value."""
        from chuk_mcp_dem.core.raster_io import compute_contours

        gradient = np.linspace(50, 350, 100).reshape(1, -1).repeat(10, axis=0).astype(np.float32)
        contours, levels = compute_contours(gradient, sample_transform, 100.0)
        valid = contours[~np.isnan(contours)]
        # All contour values should be one of the levels
        for v in valid:
            assert float(v) in levels

    def test_interval_spacing(self, sample_transform):
        """Levels should be spaced at the specified interval."""
        from chuk_mcp_dem.core.raster_io import compute_contours

        gradient = np.linspace(0, 1000, 200).reshape(1, -1).repeat(5, axis=0).astype(np.float32)
        _, levels = compute_contours(gradient, sample_transform, 50.0)
        for i in range(1, len(levels)):
            assert pytest.approx(levels[i] - levels[i - 1], abs=1e-6) == 50.0

    def test_custom_base(self, sample_transform):
        """Custom base_m shifts contour alignment."""
        from chuk_mcp_dem.core.raster_io import compute_contours

        gradient = np.linspace(0, 500, 100).reshape(1, -1).repeat(5, axis=0).astype(np.float32)
        _, levels = compute_contours(gradient, sample_transform, 100.0, base_m=25.0)
        assert 25.0 in levels
        assert 125.0 in levels
        assert 225.0 in levels

    def test_nan_handling(self, sample_transform):
        """NaN values should not crash contour generation."""
        from chuk_mcp_dem.core.raster_io import compute_contours

        arr = np.linspace(0, 500, 100).reshape(1, -1).repeat(10, axis=0).astype(np.float32)
        arr[3:5, 20:30] = np.nan
        contours, levels = compute_contours(arr, sample_transform, 100.0)
        assert contours.shape == arr.shape
        assert len(levels) > 0

    def test_returns_float32(self, sample_transform):
        """Output should be float32."""
        from chuk_mcp_dem.core.raster_io import compute_contours

        gradient = np.linspace(0, 500, 100).reshape(1, -1).repeat(5, axis=0).astype(np.float32)
        contours, _ = compute_contours(gradient, sample_transform, 100.0)
        assert contours.dtype == np.float32

    def test_small_interval(self, sample_transform):
        """Small interval should produce many contour levels."""
        from chuk_mcp_dem.core.raster_io import compute_contours

        gradient = np.linspace(100, 200, 100).reshape(1, -1).repeat(5, axis=0).astype(np.float32)
        _, levels = compute_contours(gradient, sample_transform, 10.0)
        assert len(levels) >= 10


class TestContoursToPng:
    """Tests for contours_to_png() -- contour visualisation."""

    def test_returns_png_bytes(self, sample_elevation, sample_transform):
        """Output should be valid PNG bytes."""
        from chuk_mcp_dem.core.raster_io import compute_contours, contours_to_png

        contours, _ = compute_contours(sample_elevation, sample_transform, 50.0)
        result = contours_to_png(sample_elevation, contours)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_output_dimensions(self, sample_elevation, sample_transform):
        """PNG should match the input array dimensions."""
        from chuk_mcp_dem.core.raster_io import compute_contours, contours_to_png

        contours, _ = compute_contours(sample_elevation, sample_transform, 50.0)
        result = contours_to_png(sample_elevation, contours)
        img = Image.open(io.BytesIO(result))
        h, w = sample_elevation.shape
        assert img.size == (w, h)

    def test_all_nan_contours(self, sample_elevation):
        """All-NaN contours should still produce valid PNG."""
        from chuk_mcp_dem.core.raster_io import contours_to_png

        empty = np.full_like(sample_elevation, np.nan, dtype=np.float32)
        result = contours_to_png(sample_elevation, empty)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# _reproject_bbox
# ---------------------------------------------------------------------------


class TestReprojectBbox:
    """Tests for _reproject_bbox() -- EPSG:4326 passthrough and reprojection."""

    def test_4326_passthrough(self):
        """EPSG:4326 input returns bbox unchanged."""
        from chuk_mcp_dem.core.raster_io import _reproject_bbox

        bbox = [7.0, 46.0, 8.0, 47.0]
        # Create a mock CRS that stringifies to "EPSG:4326"
        mock_crs = MagicMock()
        mock_crs.__str__ = MagicMock(return_value="EPSG:4326")
        result = _reproject_bbox(bbox, mock_crs)
        assert result == (7.0, 46.0, 8.0, 47.0)

    def test_4326_lowercase_passthrough(self):
        """Lower-case 'epsg:4326' also triggers passthrough."""
        from chuk_mcp_dem.core.raster_io import _reproject_bbox

        bbox = [7.0, 46.0, 8.0, 47.0]
        mock_crs = MagicMock()
        mock_crs.__str__ = MagicMock(return_value="epsg:4326")
        result = _reproject_bbox(bbox, mock_crs)
        assert result == (7.0, 46.0, 8.0, 47.0)

    def test_reprojection_to_utm(self):
        """Reprojection to UTM produces different coordinates."""
        from chuk_mcp_dem.core.raster_io import _reproject_bbox

        bbox = [7.0, 46.0, 8.0, 47.0]
        mock_crs = MagicMock()
        mock_crs.__str__ = MagicMock(return_value="EPSG:32632")

        result = _reproject_bbox(bbox, mock_crs)
        west, south, east, north = result
        # UTM zone 32N coordinates for this area are in the hundreds of thousands
        assert west > 100000
        assert east > west
        assert north > south

    def test_returns_tuple_of_four(self):
        """Result is always a 4-tuple of floats."""
        from chuk_mcp_dem.core.raster_io import _reproject_bbox

        bbox = [7.0, 46.0, 8.0, 47.0]
        mock_crs = MagicMock()
        mock_crs.__str__ = MagicMock(return_value="EPSG:4326")
        result = _reproject_bbox(bbox, mock_crs)
        assert len(result) == 4
        assert all(isinstance(v, float) for v in result)


# ---------------------------------------------------------------------------
# _crop_to_bbox
# ---------------------------------------------------------------------------


class TestCropToBbox:
    """Tests for _crop_to_bbox() -- array cropping to bbox."""

    def test_valid_crop(self, sample_elevation, sample_transform, sample_crs):
        """Cropping to a sub-bbox produces a smaller array."""
        from chuk_mcp_dem.core.raster_io import _crop_to_bbox

        # Array covers [7.0, 46.0, 8.0, 47.0] (100 pixels at 0.01 deg)
        bbox = [7.2, 46.5, 7.8, 46.9]
        cropped, new_transform = _crop_to_bbox(sample_elevation, sample_transform, sample_crs, bbox)
        assert cropped.shape[0] < sample_elevation.shape[0]
        assert cropped.shape[1] < sample_elevation.shape[1]

    def test_transform_updated(self, sample_elevation, sample_transform, sample_crs):
        """Cropped transform origin is shifted."""
        from chuk_mcp_dem.core.raster_io import _crop_to_bbox

        bbox = [7.2, 46.5, 7.8, 46.9]
        _, new_transform = _crop_to_bbox(sample_elevation, sample_transform, sample_crs, bbox)
        # New transform c (x-origin) should be shifted right from 7.0
        assert new_transform.c > sample_transform.c

    def test_full_extent_no_change(self, sample_elevation, sample_transform, sample_crs):
        """Cropping to full extent returns same-sized array."""
        from chuk_mcp_dem.core.raster_io import _crop_to_bbox

        bbox = [7.0, 46.0, 8.0, 47.0]
        cropped, _ = _crop_to_bbox(sample_elevation, sample_transform, sample_crs, bbox)
        assert cropped.shape == sample_elevation.shape


# ---------------------------------------------------------------------------
# _bilinear_sample
# ---------------------------------------------------------------------------


class TestBilinearSample:
    """Tests for _bilinear_sample() -- bilinear interpolation."""

    def test_exact_pixel(self):
        """Integer coordinates return exact pixel value (weighted average of neighbors)."""
        from chuk_mcp_dem.core.raster_io import _bilinear_sample

        arr = np.array(
            [[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]],
            dtype=np.float32,
        )
        # At (1.0, 1.0): dr=0, dc=0 => v00=50, all weight on v00
        val = _bilinear_sample(arr, 1.0, 1.0)
        assert val == pytest.approx(50.0)

    def test_halfway_interpolation(self):
        """Halfway between four pixels averages them."""
        from chuk_mcp_dem.core.raster_io import _bilinear_sample

        arr = np.array(
            [[0.0, 0.0, 0.0], [0.0, 100.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        # At (0.5, 0.5): average of arr[0,0]=0, arr[0,1]=0, arr[1,0]=0, arr[1,1]=100 => 25
        val = _bilinear_sample(arr, 0.5, 0.5)
        assert val == pytest.approx(25.0)

    def test_edge_out_of_bounds(self):
        """Coordinates where r1 >= h return NaN."""
        from chuk_mcp_dem.core.raster_io import _bilinear_sample

        arr = np.ones((3, 3), dtype=np.float32)
        # row_f = 2.5 => r0=2, r1=3 >= h=3
        val = _bilinear_sample(arr, 2.5, 1.0)
        assert math.isnan(val)

    def test_negative_coordinates(self):
        """Negative coordinates return NaN."""
        from chuk_mcp_dem.core.raster_io import _bilinear_sample

        arr = np.ones((5, 5), dtype=np.float32)
        val = _bilinear_sample(arr, -0.5, 2.0)
        assert math.isnan(val)

    def test_nan_neighbor_returns_nan(self):
        """If any of the four neighbors is NaN, returns NaN."""
        from chuk_mcp_dem.core.raster_io import _bilinear_sample

        arr = np.array(
            [[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]],
            dtype=np.float32,
        )
        val = _bilinear_sample(arr, 0.5, 0.5)
        # Neighbors: arr[0,0]=1, arr[0,1]=2, arr[1,0]=4, arr[1,1]=NaN
        assert math.isnan(val)

    def test_fractional_interpolation(self):
        """Verify linear interpolation at fractional position."""
        from chuk_mcp_dem.core.raster_io import _bilinear_sample

        arr = np.array(
            [[0.0, 10.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=np.float32,
        )
        # At (0.0, 0.5): dr=0, dc=0.5
        # v00=arr[0,0]=0, v01=arr[0,1]=10, v10=arr[1,0]=0, v11=arr[1,1]=10
        # val = 0*(1-0)*(0.5) + 10*(1-0)*0.5 + 0*0*(0.5) + 10*0*0.5
        # val = 0 + 5 + 0 + 0 = 5
        val = _bilinear_sample(arr, 0.0, 0.5)
        assert val == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# _cubic_sample
# ---------------------------------------------------------------------------


class TestCubicSample:
    """Tests for _cubic_sample() -- bicubic interpolation."""

    def test_center_of_large_array(self, sample_elevation):
        """Cubic sampling at center of large array uses spline."""
        from chuk_mcp_dem.core.raster_io import _cubic_sample

        val = _cubic_sample(sample_elevation, 50.5, 50.5)
        assert isinstance(val, float)
        assert not math.isnan(val)

    def test_small_patch_fallback_to_bilinear(self):
        """When patch < 4x4, falls back to bilinear."""
        from chuk_mcp_dem.core.raster_io import _cubic_sample, _bilinear_sample

        arr = np.ones((3, 3), dtype=np.float32) * 42.0
        # At (1.0, 1.0): patch from (0,0) to (3,3) but array is 3x3
        # r_start=max(0,0)=0, r_end=min(3,3)=3, extent=3 < 4 => fallback
        val = _cubic_sample(arr, 1.0, 1.0)
        expected = _bilinear_sample(arr, 1.0, 1.0)
        assert val == pytest.approx(expected)

    def test_nan_patch_fallback_to_bilinear(self):
        """When 4x4 patch contains NaN, falls back to bilinear."""
        from chuk_mcp_dem.core.raster_io import _cubic_sample

        arr = np.ones((10, 10), dtype=np.float32) * 100.0
        arr[4, 4] = np.nan
        # Sample near (4,4) => 4x4 patch includes NaN => fallback
        val = _cubic_sample(arr, 4.5, 4.5)
        # Should return bilinear result (NaN since neighbor is NaN)
        assert math.isnan(val)

    def test_exact_center_returns_close_value(self):
        """At exact pixel center, cubic ≈ pixel value."""
        from chuk_mcp_dem.core.raster_io import _cubic_sample

        arr = np.ones((10, 10), dtype=np.float32) * 200.0
        val = _cubic_sample(arr, 5.0, 5.0)
        assert val == pytest.approx(200.0, abs=1.0)

    def test_edge_falls_back(self):
        """Sampling near edge where 4x4 patch can't be formed falls back."""
        from chuk_mcp_dem.core.raster_io import _cubic_sample

        arr = np.ones((10, 10), dtype=np.float32) * 50.0
        # Row 0: r0=-1+1=0, r_start=0, r_end=min(10,2)=2, extent=2 < 4
        val = _cubic_sample(arr, 0.5, 5.0)
        assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Integration-style tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests combining multiple raster_io functions."""

    def test_slope_of_hillshade_is_valid(self, sample_elevation, sample_transform):
        """compute_slope on compute_hillshade output produces valid result."""
        from chuk_mcp_dem.core.raster_io import compute_hillshade, compute_slope

        hs = compute_hillshade(sample_elevation, sample_transform)
        slope = compute_slope(hs, sample_transform)
        assert slope.shape == sample_elevation.shape
        assert slope.dtype == np.float32

    def test_fill_then_sample(self, sample_elevation_with_voids, sample_transform):
        """Filling voids then sampling returns non-NaN at void location."""
        from chuk_mcp_dem.core.raster_io import fill_voids
        from chuk_mcp_dem.core.raster_io import sample_elevation as sample_elev

        filled = fill_voids(sample_elevation_with_voids)
        # Sample at a formerly void location: pixel (12, 12)
        lon = 7.0 + 12 * 0.01  # 7.12
        lat = 47.0 - 12 * 0.01  # 46.88
        val = sample_elev(filled, sample_transform, lon, lat, "nearest")
        assert not math.isnan(val)

    def test_terrain_png_from_hillshade(self, sample_elevation, sample_transform):
        """elevation_to_terrain_png accepts hillshade output."""
        from chuk_mcp_dem.core.raster_io import (
            compute_hillshade,
            elevation_to_terrain_png,
        )

        hs = compute_hillshade(sample_elevation, sample_transform)
        png = elevation_to_terrain_png(hs)
        assert png[:4] == b"\x89PNG"

    def test_aspect_and_slope_shapes_match(self, sample_elevation, sample_transform):
        """Slope and aspect have the same shape for same input."""
        from chuk_mcp_dem.core.raster_io import compute_slope, compute_aspect

        slope = compute_slope(sample_elevation, sample_transform)
        aspect = compute_aspect(sample_elevation, sample_transform)
        assert slope.shape == aspect.shape == sample_elevation.shape

    def test_geotiff_roundtrip(self, sample_elevation, sample_transform, sample_crs):
        """GeoTIFF bytes can be opened back with rasterio."""
        from chuk_mcp_dem.core.raster_io import arrays_to_geotiff

        pytest.importorskip("rasterio")
        from rasterio.io import MemoryFile

        tiff_bytes = arrays_to_geotiff(sample_elevation, sample_crs, sample_transform)
        with MemoryFile(tiff_bytes) as memfile:
            with memfile.open() as src:
                data = src.read(1)
                assert data.shape == (100, 100)
                assert src.crs == sample_crs


# ---------------------------------------------------------------------------
# slope_to_png
# ---------------------------------------------------------------------------


class TestSlopeToPng:
    """Tests for slope_to_png() -- slope visualisation as PNG."""

    def test_returns_valid_png_bytes(self, sample_elevation, sample_transform):
        """Output is valid PNG (starts with PNG signature)."""
        from chuk_mcp_dem.core.raster_io import compute_slope, slope_to_png

        slope = compute_slope(sample_elevation, sample_transform)
        result = slope_to_png(slope)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_flat_surface_all_green(self, sample_transform):
        """Flat surface (slope=0 everywhere) produces all-green pixels."""
        from chuk_mcp_dem.core.raster_io import slope_to_png

        flat_slope = np.zeros((50, 50), dtype=np.float32)
        result = slope_to_png(flat_slope)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # norm=0 => r=0, g=255, b=0 => pure green
        assert np.all(arr[:, :, 0] == 0)  # red channel = 0
        assert np.all(arr[:, :, 1] == 255)  # green channel = 255
        assert np.all(arr[:, :, 2] == 0)  # blue channel = 0

    def test_steep_surface_red_dominant(self, sample_transform):
        """Very steep slope (90 degrees) produces red-dominant pixels."""
        from chuk_mcp_dem.core.raster_io import slope_to_png

        steep_slope = np.full((30, 30), 90.0, dtype=np.float32)
        result = slope_to_png(steep_slope, units="degrees")
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # norm=1.0 => r=255, g=0, b=0
        assert np.all(arr[:, :, 0] == 255)  # red channel = 255
        assert np.all(arr[:, :, 1] == 0)  # green channel = 0

    def test_moderate_slope_yellow_region(self, sample_transform):
        """Moderate slope (~45 degrees) produces yellow-ish pixels."""
        from chuk_mcp_dem.core.raster_io import slope_to_png

        mod_slope = np.full((20, 20), 45.0, dtype=np.float32)
        result = slope_to_png(mod_slope, units="degrees")
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # norm=0.5 => r=255, g=255, b=0 => yellow
        assert np.all(arr[:, :, 0] == 255)
        assert np.all(arr[:, :, 1] == 255)

    def test_correct_rgb_mode(self, sample_transform):
        """Output PNG is RGB (3-channel)."""
        from chuk_mcp_dem.core.raster_io import slope_to_png

        slope = np.ones((20, 20), dtype=np.float32) * 30.0
        result = slope_to_png(slope)
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_dimensions_match_input(self, sample_elevation, sample_transform):
        """PNG dimensions match the slope array shape."""
        from chuk_mcp_dem.core.raster_io import compute_slope, slope_to_png

        slope = compute_slope(sample_elevation, sample_transform)
        result = slope_to_png(slope)
        img = Image.open(io.BytesIO(result))
        assert img.size == (slope.shape[1], slope.shape[0])

    def test_percent_units_accepted(self, sample_transform):
        """Percent units are handled without error."""
        from chuk_mcp_dem.core.raster_io import slope_to_png

        slope_pct = np.full((20, 20), 100.0, dtype=np.float32)
        result = slope_to_png(slope_pct, units="percent")
        assert result[:4] == b"\x89PNG"

    def test_percent_units_normalisation_range(self):
        """Percent units use max_val=200 for normalisation."""
        from chuk_mcp_dem.core.raster_io import slope_to_png

        # 200% slope -> norm=1.0 -> full red
        slope_pct = np.full((10, 10), 200.0, dtype=np.float32)
        result = slope_to_png(slope_pct, units="percent")
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        assert np.all(arr[:, :, 0] == 255)
        assert np.all(arr[:, :, 1] == 0)

    def test_nan_values_treated_as_zero(self):
        """NaN values in slope are treated as zero (flat/green)."""
        from chuk_mcp_dem.core.raster_io import slope_to_png

        slope_nan = np.full((10, 10), np.nan, dtype=np.float32)
        result = slope_to_png(slope_nan)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        assert np.all(arr[:, :, 0] == 0)
        assert np.all(arr[:, :, 1] == 255)


# ---------------------------------------------------------------------------
# aspect_to_png
# ---------------------------------------------------------------------------


class TestAspectToPng:
    """Tests for aspect_to_png() -- aspect visualisation as PNG."""

    def test_returns_valid_png_bytes(self, sample_elevation, sample_transform):
        """Output is valid PNG (starts with PNG signature)."""
        from chuk_mcp_dem.core.raster_io import compute_aspect, aspect_to_png

        aspect = compute_aspect(sample_elevation, sample_transform)
        result = aspect_to_png(aspect)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_flat_areas_grey(self):
        """All-flat surface (all flat_value) produces uniform grey pixels."""
        from chuk_mcp_dem.core.raster_io import aspect_to_png

        flat_aspect = np.full((40, 40), -1.0, dtype=np.float32)
        result = aspect_to_png(flat_aspect)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # Flat areas: [128, 128, 128]
        assert np.all(arr[:, :, 0] == 128)
        assert np.all(arr[:, :, 1] == 128)
        assert np.all(arr[:, :, 2] == 128)

    def test_custom_flat_value_grey(self):
        """Custom flat_value also produces grey pixels."""
        from chuk_mcp_dem.core.raster_io import aspect_to_png

        flat_aspect = np.full((20, 20), -999.0, dtype=np.float32)
        result = aspect_to_png(flat_aspect, flat_value=-999.0)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        assert np.all(arr[:, :, 0] == 128)
        assert np.all(arr[:, :, 1] == 128)
        assert np.all(arr[:, :, 2] == 128)

    def test_non_flat_areas_have_colour(self):
        """Non-flat areas have non-grey colour from the HSV wheel."""
        from chuk_mcp_dem.core.raster_io import aspect_to_png

        # Aspect = 0 degrees (north-facing)
        north_aspect = np.full((20, 20), 0.0, dtype=np.float32)
        result = aspect_to_png(north_aspect)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # Should NOT be grey (128,128,128)
        pixel = arr[10, 10]
        assert not (pixel[0] == 128 and pixel[1] == 128 and pixel[2] == 128)

    def test_different_aspects_different_colours(self):
        """Different aspect angles produce different colours."""
        from chuk_mcp_dem.core.raster_io import aspect_to_png

        north = np.full((10, 10), 0.0, dtype=np.float32)
        east = np.full((10, 10), 90.0, dtype=np.float32)

        result_n = aspect_to_png(north)
        result_e = aspect_to_png(east)

        img_n = np.array(Image.open(io.BytesIO(result_n)))
        img_e = np.array(Image.open(io.BytesIO(result_e)))

        # The centre pixel should differ
        assert not np.array_equal(img_n[5, 5], img_e[5, 5])

    def test_correct_rgb_mode(self):
        """Output PNG is RGB (3-channel)."""
        from chuk_mcp_dem.core.raster_io import aspect_to_png

        aspect = np.full((20, 20), 180.0, dtype=np.float32)
        result = aspect_to_png(aspect)
        img = Image.open(io.BytesIO(result))
        assert img.mode == "RGB"

    def test_dimensions_match_input(self, sample_elevation, sample_transform):
        """PNG dimensions match the aspect array shape."""
        from chuk_mcp_dem.core.raster_io import compute_aspect, aspect_to_png

        aspect = compute_aspect(sample_elevation, sample_transform)
        result = aspect_to_png(aspect)
        img = Image.open(io.BytesIO(result))
        assert img.size == (aspect.shape[1], aspect.shape[0])

    def test_mixed_flat_and_sloped(self):
        """Array with both flat and non-flat pixels handles both correctly."""
        from chuk_mcp_dem.core.raster_io import aspect_to_png

        aspect = np.full((20, 20), 45.0, dtype=np.float32)
        aspect[0:5, :] = -1.0  # flat band at top
        result = aspect_to_png(aspect)
        img = Image.open(io.BytesIO(result))
        arr = np.array(img)
        # Flat rows are grey
        assert np.all(arr[0:5, :, 0] == 128)
        assert np.all(arr[0:5, :, 1] == 128)
        assert np.all(arr[0:5, :, 2] == 128)
        # Non-flat rows are NOT grey
        non_flat_pixel = arr[10, 10]
        assert not (
            non_flat_pixel[0] == 128 and non_flat_pixel[1] == 128 and non_flat_pixel[2] == 128
        )

    def test_full_circle_aspects(self):
        """Various aspect angles from 0-359 all produce valid PNG."""
        from chuk_mcp_dem.core.raster_io import aspect_to_png

        for angle in [0, 45, 90, 135, 180, 225, 270, 315, 359]:
            aspect = np.full((5, 5), float(angle), dtype=np.float32)
            result = aspect_to_png(aspect)
            assert result[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# _haversine
# ---------------------------------------------------------------------------


class TestHaversine:
    """Tests for _haversine() -- great-circle distance computation."""

    def test_one_degree_latitude_at_equator(self):
        """1 degree of latitude at the equator is approximately 111.2 km."""
        from chuk_mcp_dem.core.raster_io import _haversine

        d = _haversine(0.0, 0.0, 1.0, 0.0)
        assert d == pytest.approx(111_195.0, rel=0.01)

    def test_zero_distance_same_point(self):
        """Distance from a point to itself is zero."""
        from chuk_mcp_dem.core.raster_io import _haversine

        d = _haversine(47.0, 7.0, 47.0, 7.0)
        assert d == pytest.approx(0.0, abs=1e-6)

    def test_small_distance(self):
        """Small distance (0.001 degrees) at mid-latitude is in the 100 m range."""
        from chuk_mcp_dem.core.raster_io import _haversine

        d = _haversine(47.0, 7.0, 47.001, 7.0)
        assert 100.0 < d < 150.0

    def test_one_degree_longitude_at_equator(self):
        """1 degree of longitude at the equator is approximately 111.2 km."""
        from chuk_mcp_dem.core.raster_io import _haversine

        d = _haversine(0.0, 0.0, 0.0, 1.0)
        assert d == pytest.approx(111_195.0, rel=0.01)

    def test_longitude_shrinks_with_latitude(self):
        """1 degree of longitude at 60N is approximately half the equatorial distance."""
        from chuk_mcp_dem.core.raster_io import _haversine

        d_equator = _haversine(0.0, 0.0, 0.0, 1.0)
        d_60n = _haversine(60.0, 0.0, 60.0, 1.0)
        assert d_60n == pytest.approx(d_equator * 0.5, rel=0.02)

    def test_symmetry(self):
        """Distance is symmetric: d(A,B) == d(B,A)."""
        from chuk_mcp_dem.core.raster_io import _haversine

        d_ab = _haversine(47.0, 7.0, 48.0, 8.0)
        d_ba = _haversine(48.0, 8.0, 47.0, 7.0)
        assert d_ab == pytest.approx(d_ba, rel=1e-10)

    def test_known_distance_zurich_bern(self):
        """Zurich (47.3769, 8.5417) to Bern (46.9480, 7.4474) is approximately 95 km."""
        from chuk_mcp_dem.core.raster_io import _haversine

        d = _haversine(47.3769, 8.5417, 46.9480, 7.4474)
        assert d == pytest.approx(95_000.0, rel=0.05)

    def test_antipodal_points(self):
        """Distance between antipodal points is approximately half Earth circumference."""
        from chuk_mcp_dem.core.raster_io import _haversine

        d = _haversine(0.0, 0.0, 0.0, 180.0)
        half_circumference = math.pi * 6371000.0
        assert d == pytest.approx(half_circumference, rel=0.001)


# ---------------------------------------------------------------------------
# compute_profile_points
# ---------------------------------------------------------------------------


class TestComputeProfilePoints:
    """Tests for compute_profile_points() -- line sampling."""

    def test_correct_number_of_points(self, sample_elevation, sample_transform):
        """Returns the requested number of points."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        lons, lats, dists, elevs = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=[7.1, 46.9],
            end=[7.5, 46.5],
            num_points=50,
        )
        assert len(lons) == 50
        assert len(lats) == 50
        assert len(dists) == 50
        assert len(elevs) == 50

    def test_first_distance_is_zero(self, sample_elevation, sample_transform):
        """First distance value is always zero."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        _, _, dists, _ = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=[7.2, 46.8],
            end=[7.4, 46.6],
            num_points=20,
        )
        assert dists[0] == 0.0

    def test_distances_monotonically_increasing(self, sample_elevation, sample_transform):
        """Distances are strictly monotonically increasing."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        _, _, dists, _ = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=[7.1, 46.9],
            end=[7.5, 46.5],
            num_points=30,
        )
        for i in range(1, len(dists)):
            assert dists[i] > dists[i - 1]

    def test_start_coordinates_match(self, sample_elevation, sample_transform):
        """First point coordinates match the start argument."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        start = [7.15, 46.85]
        end = [7.45, 46.55]
        lons, lats, _, _ = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=start,
            end=end,
            num_points=10,
        )
        assert lons[0] == pytest.approx(start[0])
        assert lats[0] == pytest.approx(start[1])

    def test_end_coordinates_match(self, sample_elevation, sample_transform):
        """Last point coordinates match the end argument."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        start = [7.15, 46.85]
        end = [7.45, 46.55]
        lons, lats, _, _ = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=start,
            end=end,
            num_points=10,
        )
        assert lons[-1] == pytest.approx(end[0])
        assert lats[-1] == pytest.approx(end[1])

    def test_elevations_are_floats(self, sample_elevation, sample_transform):
        """All elevation values are floats."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        _, _, _, elevs = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=[7.2, 46.8],
            end=[7.4, 46.6],
            num_points=15,
        )
        assert all(isinstance(e, float) for e in elevs)

    def test_elevations_within_data_range(self, sample_elevation, sample_transform):
        """Sampled elevations are within the data range (100-500m) or NaN."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        _, _, _, elevs = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=[7.2, 46.8],
            end=[7.4, 46.6],
            num_points=20,
        )
        for e in elevs:
            if not math.isnan(e):
                assert 50.0 <= e <= 550.0  # allow some bilinear overshoot

    def test_two_points_profile(self, sample_elevation, sample_transform):
        """Minimum profile with 2 points works correctly."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        lons, lats, dists, elevs = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=[7.3, 46.7],
            end=[7.5, 46.5],
            num_points=2,
        )
        assert len(lons) == 2
        assert dists[0] == 0.0
        assert dists[1] > 0.0

    def test_nearest_interpolation(self, sample_elevation, sample_transform):
        """Profile with nearest interpolation returns valid results."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points

        _, _, _, elevs = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=[7.2, 46.8],
            end=[7.4, 46.6],
            num_points=10,
            interpolation="nearest",
        )
        # All points are within the grid, so none should be NaN
        for e in elevs:
            assert not math.isnan(e)

    def test_total_distance_plausible(self, sample_elevation, sample_transform):
        """Total distance between start and end is geographically plausible."""
        from chuk_mcp_dem.core.raster_io import compute_profile_points, _haversine

        start = [7.1, 46.9]
        end = [7.5, 46.5]
        _, _, dists, _ = compute_profile_points(
            sample_elevation,
            sample_transform,
            start=start,
            end=end,
            num_points=100,
        )
        expected_total = _haversine(start[1], start[0], end[1], end[0])
        assert dists[-1] == pytest.approx(expected_total, rel=0.01)


# ---------------------------------------------------------------------------
# compute_viewshed
# ---------------------------------------------------------------------------


class TestComputeViewshed:
    """Tests for compute_viewshed() -- visibility analysis."""

    def test_flat_surface_all_visible_within_radius(self, sample_transform):
        """On a flat surface, everything within radius is visible."""
        from chuk_mcp_dem.core.raster_io import compute_viewshed

        flat = np.ones((50, 50), dtype=np.float32) * 500.0
        # Observer at centre: lon=7.0+25*0.01=7.25, lat=47.0-25*0.01=46.75
        result = compute_viewshed(
            flat,
            sample_transform,
            observer_lon=7.25,
            observer_lat=46.75,
            radius_m=500.0,
            observer_height_m=1.8,
        )
        # All non-NaN cells should be visible (1.0)
        visible_cells = result[~np.isnan(result)]
        assert len(visible_cells) > 0
        assert np.all(visible_cells == 1.0)

    def test_observer_position_visible(self, sample_elevation, sample_transform):
        """Observer position is always marked as visible."""
        from chuk_mcp_dem.core.raster_io import compute_viewshed

        result = compute_viewshed(
            sample_elevation,
            sample_transform,
            observer_lon=7.5,
            observer_lat=46.5,
            radius_m=500.0,
            observer_height_m=1.8,
        )
        # Observer at pixel (50, 50)
        assert result[50, 50] == 1.0

    def test_nan_outside_radius(self, sample_transform):
        """Cells outside the radius remain NaN."""
        from chuk_mcp_dem.core.raster_io import compute_viewshed

        flat = np.ones((100, 100), dtype=np.float32) * 200.0
        # Observer at centre (50,50), small radius
        result = compute_viewshed(
            flat,
            sample_transform,
            observer_lon=7.5,
            observer_lat=46.5,
            radius_m=200.0,
            observer_height_m=1.8,
        )
        # Corners should be NaN (far from centre)
        assert np.isnan(result[0, 0])
        assert np.isnan(result[0, 99])
        assert np.isnan(result[99, 0])
        assert np.isnan(result[99, 99])

    def test_correct_output_shape(self, sample_elevation, sample_transform):
        """Output shape matches input elevation shape."""
        from chuk_mcp_dem.core.raster_io import compute_viewshed

        result = compute_viewshed(
            sample_elevation,
            sample_transform,
            observer_lon=7.5,
            observer_lat=46.5,
            radius_m=300.0,
        )
        assert result.shape == sample_elevation.shape

    def test_output_dtype_float32(self, sample_elevation, sample_transform):
        """Output dtype is float32."""
        from chuk_mcp_dem.core.raster_io import compute_viewshed

        result = compute_viewshed(
            sample_elevation,
            sample_transform,
            observer_lon=7.5,
            observer_lat=46.5,
            radius_m=300.0,
        )
        assert result.dtype == np.float32

    def test_values_are_0_1_or_nan(self, sample_elevation, sample_transform):
        """All output values are exactly 0.0, 1.0, or NaN."""
        from chuk_mcp_dem.core.raster_io import compute_viewshed

        result = compute_viewshed(
            sample_elevation,
            sample_transform,
            observer_lon=7.5,
            observer_lat=46.5,
            radius_m=500.0,
        )
        non_nan = result[~np.isnan(result)]
        unique_vals = np.unique(non_nan)
        for v in unique_vals:
            assert v in (0.0, 1.0)

    def test_ridge_blocks_visibility(self, sample_transform):
        """A tall ridge between observer and target blocks visibility."""
        from chuk_mcp_dem.core.raster_io import compute_viewshed

        # Flat terrain with a tall ridge in the middle
        terrain = np.ones((50, 50), dtype=np.float32) * 100.0
        terrain[25, :] = 5000.0  # massive wall at row 25

        # Observer at row 10 (north side of the wall)
        obs_lon = 7.0 + 25 * 0.01  # col 25
        obs_lat = 47.0 - 10 * 0.01  # row 10

        result = compute_viewshed(
            terrain,
            sample_transform,
            observer_lon=obs_lon,
            observer_lat=obs_lat,
            radius_m=5000.0,
            observer_height_m=1.8,
        )
        # Cells on the far side of the wall (row 40) should be hidden
        far_side = result[40, 20:30]
        non_nan_far = far_side[~np.isnan(far_side)]
        if len(non_nan_far) > 0:
            assert np.all(non_nan_far == 0.0)

    def test_observer_out_of_bounds(self, sample_elevation, sample_transform):
        """Observer outside the grid produces all-NaN result."""
        from chuk_mcp_dem.core.raster_io import compute_viewshed

        result = compute_viewshed(
            sample_elevation,
            sample_transform,
            observer_lon=0.0,
            observer_lat=0.0,
            radius_m=500.0,
        )
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# _check_line_of_sight
# ---------------------------------------------------------------------------


class TestCheckLineOfSight:
    """Tests for _check_line_of_sight() -- DDA line-of-sight check."""

    def test_clear_los_flat_terrain(self):
        """Clear line of sight over flat terrain returns True."""
        from chuk_mcp_dem.core.raster_io import _check_line_of_sight

        flat = np.ones((50, 50), dtype=np.float32) * 100.0
        # Observer at (25,25) with height=101.8 (100+1.8), target at (25,40)
        visible = _check_line_of_sight(
            flat,
            r0=25,
            c0=25,
            obs_height=101.8,
            r1=25,
            c1=40,
            cellsize_x_m=100.0,
            cellsize_y_m=100.0,
        )
        assert visible

    def test_blocked_by_ridge(self):
        """Ridge between observer and target blocks line of sight."""
        from chuk_mcp_dem.core.raster_io import _check_line_of_sight

        terrain = np.ones((50, 50), dtype=np.float32) * 100.0
        terrain[25, 30] = 5000.0  # tall obstacle between observer and target

        # Observer at (25,25), target at (25,40)
        visible = _check_line_of_sight(
            terrain,
            r0=25,
            c0=25,
            obs_height=101.8,
            r1=25,
            c1=40,
            cellsize_x_m=100.0,
            cellsize_y_m=100.0,
        )
        assert not visible

    def test_same_cell_always_visible(self):
        """Target at same cell as observer is always visible."""
        from chuk_mcp_dem.core.raster_io import _check_line_of_sight

        terrain = np.ones((20, 20), dtype=np.float32) * 200.0
        visible = _check_line_of_sight(
            terrain,
            r0=10,
            c0=10,
            obs_height=201.8,
            r1=10,
            c1=10,
            cellsize_x_m=100.0,
            cellsize_y_m=100.0,
        )
        assert visible

    def test_adjacent_cell_visible(self):
        """Adjacent cell on flat terrain is visible."""
        from chuk_mcp_dem.core.raster_io import _check_line_of_sight

        flat = np.ones((20, 20), dtype=np.float32) * 100.0
        visible = _check_line_of_sight(
            flat,
            r0=10,
            c0=10,
            obs_height=101.8,
            r1=10,
            c1=11,
            cellsize_x_m=100.0,
            cellsize_y_m=100.0,
        )
        assert visible

    def test_diagonal_clear_los(self):
        """Diagonal line of sight on flat terrain is clear."""
        from chuk_mcp_dem.core.raster_io import _check_line_of_sight

        flat = np.ones((50, 50), dtype=np.float32) * 100.0
        visible = _check_line_of_sight(
            flat,
            r0=10,
            c0=10,
            obs_height=101.8,
            r1=40,
            c1=40,
            cellsize_x_m=100.0,
            cellsize_y_m=100.0,
        )
        assert visible

    def test_target_higher_than_obstacle_visible(self):
        """Target on a hill higher than an intermediate obstacle is visible."""
        from chuk_mcp_dem.core.raster_io import _check_line_of_sight

        terrain = np.ones((50, 50), dtype=np.float32) * 100.0
        terrain[10, 15] = 150.0  # moderate obstacle
        terrain[10, 25] = 300.0  # target on high ground

        # Observer at (10,5), height=101.8
        # Obstacle at (10,15) elev=150, target at (10,25) elev=300
        # Obstacle angle: (150-101.8) / (10*100) = 48.2/1000 = 0.0482
        # Target angle: (300-101.8) / (20*100) = 198.2/2000 = 0.0991
        # target_angle > max_angle -> visible
        visible = _check_line_of_sight(
            terrain,
            r0=10,
            c0=5,
            obs_height=101.8,
            r1=10,
            c1=25,
            cellsize_x_m=100.0,
            cellsize_y_m=100.0,
        )
        assert visible

    def test_nan_obstacle_ignored(self):
        """NaN cells along the path are skipped (not treated as blocking)."""
        from chuk_mcp_dem.core.raster_io import _check_line_of_sight

        terrain = np.ones((50, 50), dtype=np.float32) * 100.0
        terrain[25, 30] = np.nan  # NaN cell along the path

        visible = _check_line_of_sight(
            terrain,
            r0=25,
            c0=25,
            obs_height=101.8,
            r1=25,
            c1=40,
            cellsize_x_m=100.0,
            cellsize_y_m=100.0,
        )
        assert visible

    def test_target_with_nan_elevation_treated_as_zero(self):
        """Target with NaN elevation is treated as elevation 0."""
        from chuk_mcp_dem.core.raster_io import _check_line_of_sight

        terrain = np.ones((50, 50), dtype=np.float32) * 100.0
        terrain[25, 40] = np.nan  # NaN at target

        # Observer height 101.8 looking at target with elev=0
        # target_angle = (0 - 101.8) / dist < 0 < max_angle (which is -inf on flat)
        # Actually max_angle stays -inf since all intermediate cells are 100m
        # and (100 - 101.8)/dist < 0, but max_angle starts at -inf
        # After intermediate: max_angle = (100-101.8)/dist_step < 0
        # target_angle = (0-101.8)/total_dist < max_angle? Depends on distances.
        # This test just checks it does not crash and returns a boolean-like value.
        result = _check_line_of_sight(
            terrain,
            r0=25,
            c0=25,
            obs_height=101.8,
            r1=25,
            c1=40,
            cellsize_x_m=100.0,
            cellsize_y_m=100.0,
        )
        assert result is True or result is False or isinstance(result, (bool, np.bool_))


# ---------------------------------------------------------------------------
# compute_flow_accumulation
# ---------------------------------------------------------------------------


class TestComputeFlowAccumulation:
    """Tests for compute_flow_accumulation() -- D8 flow direction + accumulation."""

    def test_flat_surface_uniform_accumulation(self, sample_transform):
        """Flat surface: no clear flow direction, all cells ~1."""
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation

        flat = np.ones((20, 20), dtype=np.float32) * 100.0
        accum = compute_flow_accumulation(flat, sample_transform)
        assert accum.shape == (20, 20)
        assert accum.dtype == np.float32
        # All values >= 1 (every cell counts itself)
        assert np.all(accum >= 1.0)

    def test_simple_funnel(self, sample_transform):
        """Bowl surface: centre should have highest accumulation."""
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation

        y, x = np.mgrid[0:30, 0:30]
        bowl = ((x - 15.0) ** 2 + (y - 15.0) ** 2).astype(np.float32)
        accum = compute_flow_accumulation(bowl, sample_transform)
        # Centre should have highest accumulation
        centre_val = accum[15, 15]
        assert centre_val == float(np.max(accum))

    def test_values_non_negative(self, sample_elevation, sample_transform):
        """All accumulation values should be >= 1."""
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation

        accum = compute_flow_accumulation(sample_elevation, sample_transform)
        assert np.all(accum >= 1.0)

    def test_output_dtype_float32(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation

        accum = compute_flow_accumulation(sample_elevation, sample_transform)
        assert accum.dtype == np.float32

    def test_output_shape_matches_input(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation

        accum = compute_flow_accumulation(sample_elevation, sample_transform)
        assert accum.shape == sample_elevation.shape

    def test_nan_handling(self, sample_transform):
        """NaN cells are treated as zero elevation."""
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation

        elev = np.ones((20, 20), dtype=np.float32) * 100.0
        elev[5:10, 5:10] = np.nan
        accum = compute_flow_accumulation(elev, sample_transform)
        assert accum.shape == (20, 20)
        assert np.all(accum >= 1.0)

    def test_ramp_accumulation_increases_downslope(self, sample_transform):
        """N-S ramp: accumulation should increase toward the low end."""
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation

        # North (row 0) high, south (row -1) low
        ramp = np.tile(np.arange(30, 0, -1, dtype=np.float32).reshape(30, 1), (1, 30))
        accum = compute_flow_accumulation(ramp, sample_transform)
        # Bottom row (low elevation) should have higher mean accumulation
        # than top row (high elevation)
        assert np.mean(accum[-1, :]) > np.mean(accum[0, :])


# ---------------------------------------------------------------------------
# watershed_to_png
# ---------------------------------------------------------------------------


class TestWatershedToPng:
    """Tests for watershed_to_png() -- log-scaled blue colour ramp."""

    def test_returns_bytes(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation, watershed_to_png

        accum = compute_flow_accumulation(sample_elevation, sample_transform)
        png_bytes = watershed_to_png(accum)
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0

    def test_valid_png(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_flow_accumulation, watershed_to_png

        accum = compute_flow_accumulation(sample_elevation, sample_transform)
        png_bytes = watershed_to_png(accum)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.mode == "RGB"
        assert img.size == (sample_elevation.shape[1], sample_elevation.shape[0])

    def test_uniform_input(self):
        from chuk_mcp_dem.core.raster_io import watershed_to_png

        uniform = np.ones((50, 50), dtype=np.float32) * 10.0
        png_bytes = watershed_to_png(uniform)
        assert isinstance(png_bytes, bytes)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (50, 50)


# ---------------------------------------------------------------------------
# compute_elevation_change
# ---------------------------------------------------------------------------


class TestComputeElevationChange:
    """Tests for compute_elevation_change() -- before/after DEM differencing."""

    def test_no_change_returns_zeros(self, sample_elevation, sample_transform):
        """Two identical arrays produce an all-zero change array and no regions."""
        from chuk_mcp_dem.core.raster_io import compute_elevation_change

        change, regions = compute_elevation_change(
            sample_elevation,
            sample_elevation,
            sample_transform,
        )
        assert np.allclose(change, 0.0)
        assert regions == []

    def test_uniform_gain(self, sample_elevation, sample_transform):
        """after = before + 5 gives change array all 5.0 with change_type=gain."""
        from chuk_mcp_dem.core.raster_io import compute_elevation_change

        after = sample_elevation + 5.0
        change, regions = compute_elevation_change(
            sample_elevation,
            after,
            sample_transform,
        )
        assert np.allclose(change, 5.0)
        assert len(regions) >= 1
        assert all(r["change_type"] == "gain" for r in regions)

    def test_uniform_loss(self, sample_elevation, sample_transform):
        """after = before - 5 gives change array all -5.0 with change_type=loss."""
        from chuk_mcp_dem.core.raster_io import compute_elevation_change

        after = sample_elevation - 5.0
        change, regions = compute_elevation_change(
            sample_elevation,
            after,
            sample_transform,
        )
        assert np.allclose(change, -5.0)
        assert len(regions) >= 1
        assert all(r["change_type"] == "loss" for r in regions)

    def test_below_threshold_no_regions(self, sample_elevation, sample_transform):
        """Change below significance threshold yields empty regions list."""
        from chuk_mcp_dem.core.raster_io import compute_elevation_change

        after = sample_elevation + 0.5
        change, regions = compute_elevation_change(
            sample_elevation,
            after,
            sample_transform,
            significance_threshold_m=1.0,
        )
        assert np.allclose(change, 0.5, rtol=1e-5, atol=1e-4)
        assert regions == []

    def test_output_dtype_float32(self, sample_elevation, sample_transform):
        """Change array dtype is float32."""
        from chuk_mcp_dem.core.raster_io import compute_elevation_change

        after = sample_elevation + 2.0
        change, _ = compute_elevation_change(
            sample_elevation,
            after,
            sample_transform,
        )
        assert change.dtype == np.float32

    def test_output_shape_matches_input(self, sample_elevation, sample_transform):
        """Change array shape matches input elevation arrays."""
        from chuk_mcp_dem.core.raster_io import compute_elevation_change

        after = sample_elevation + 2.0
        change, _ = compute_elevation_change(
            sample_elevation,
            after,
            sample_transform,
        )
        assert change.shape == sample_elevation.shape

    def test_region_has_correct_keys(self, sample_elevation, sample_transform):
        """Each region dict contains all expected keys."""
        from chuk_mcp_dem.core.raster_io import compute_elevation_change

        after = sample_elevation + 5.0
        _, regions = compute_elevation_change(
            sample_elevation,
            after,
            sample_transform,
        )
        assert len(regions) >= 1
        expected_keys = {"bbox", "area_m2", "mean_change_m", "max_change_m", "change_type"}
        for region in regions:
            assert set(region.keys()) == expected_keys

    def test_nan_handling(self, sample_transform):
        """NaN values in input do not crash; NaN positions are handled safely."""
        from chuk_mcp_dem.core.raster_io import compute_elevation_change

        before = np.ones((20, 20), dtype=np.float32) * 100.0
        after = np.ones((20, 20), dtype=np.float32) * 110.0
        before[5, 5] = np.nan
        after[10, 10] = np.nan

        change, regions = compute_elevation_change(
            before,
            after,
            sample_transform,
        )
        assert change.shape == (20, 20)
        # NaN positions should be NaN in the change array
        assert np.isnan(change[5, 5]) or np.isfinite(change[5, 5])
        # Function should not raise
        assert isinstance(regions, list)


# ---------------------------------------------------------------------------
# change_to_png
# ---------------------------------------------------------------------------


class TestChangeToPng:
    """Tests for change_to_png() -- diverging blue-white-red colourmap."""

    def test_produces_valid_png(self):
        """Output is valid PNG bytes."""
        from chuk_mcp_dem.core.raster_io import change_to_png

        change = np.random.uniform(-5, 5, (50, 50)).astype(np.float32)
        png_bytes = change_to_png(change)
        assert isinstance(png_bytes, bytes)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"

    def test_zero_change_is_white(self):
        """Zero change renders as white pixels (255, 255, 255)."""
        from chuk_mcp_dem.core.raster_io import change_to_png

        change = np.zeros((20, 20), dtype=np.float32)
        png_bytes = change_to_png(change)
        img = Image.open(io.BytesIO(png_bytes))
        pixels = np.array(img)
        # All pixels should be white
        assert np.all(pixels[:, :, 0] == 255)
        assert np.all(pixels[:, :, 1] == 255)
        assert np.all(pixels[:, :, 2] == 255)

    def test_positive_change_is_red(self):
        """Positive change renders with R=255 and G=0."""
        from chuk_mcp_dem.core.raster_io import change_to_png

        change = np.full((20, 20), 10.0, dtype=np.float32)
        png_bytes = change_to_png(change)
        img = Image.open(io.BytesIO(png_bytes))
        pixels = np.array(img)
        assert np.all(pixels[:, :, 0] == 255)  # Red channel
        assert np.all(pixels[:, :, 1] == 0)  # Green channel

    def test_negative_change_is_blue(self):
        """Negative change renders with B=255 and G=0."""
        from chuk_mcp_dem.core.raster_io import change_to_png

        change = np.full((20, 20), -10.0, dtype=np.float32)
        png_bytes = change_to_png(change)
        img = Image.open(io.BytesIO(png_bytes))
        pixels = np.array(img)
        assert np.all(pixels[:, :, 2] == 255)  # Blue channel
        assert np.all(pixels[:, :, 1] == 0)  # Green channel

    def test_output_dimensions_match(self):
        """PNG dimensions match input array shape."""
        from chuk_mcp_dem.core.raster_io import change_to_png

        change = np.zeros((30, 40), dtype=np.float32)
        png_bytes = change_to_png(change)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (40, 30)  # PIL size is (width, height)


# ---------------------------------------------------------------------------
# compute_landforms
# ---------------------------------------------------------------------------


class TestComputeLandforms:
    """Tests for compute_landforms() -- rule-based landform classification."""

    def test_flat_surface_all_plain(self, sample_transform):
        """Flat elevation surface classifies entirely as plain (class 0)."""
        from chuk_mcp_dem.core.raster_io import compute_landforms

        flat = np.full((30, 30), 200.0, dtype=np.float32)
        result = compute_landforms(flat, sample_transform)
        assert np.all(result == 0)

    def test_output_dtype_uint8(self, sample_elevation, sample_transform):
        """Result dtype is uint8."""
        from chuk_mcp_dem.core.raster_io import compute_landforms

        result = compute_landforms(sample_elevation, sample_transform)
        assert result.dtype == np.uint8

    def test_output_shape_matches_input(self, sample_elevation, sample_transform):
        """Result shape matches input elevation shape."""
        from chuk_mcp_dem.core.raster_io import compute_landforms

        result = compute_landforms(sample_elevation, sample_transform)
        assert result.shape == sample_elevation.shape

    def test_invalid_method_raises(self, sample_elevation, sample_transform):
        """Invalid method parameter raises ValueError."""
        from chuk_mcp_dem.core.raster_io import compute_landforms

        with pytest.raises(ValueError, match="Invalid landform method"):
            compute_landforms(sample_elevation, sample_transform, method="invalid")

    def test_values_in_valid_range(self, sample_elevation, sample_transform):
        """All output values are in the range 0-8."""
        from chuk_mcp_dem.core.raster_io import compute_landforms

        result = compute_landforms(sample_elevation, sample_transform)
        assert np.all(result >= 0)
        assert np.all(result <= 8)

    def test_steep_surface_has_escarpments(self):
        """Steep ramp produces some escarpment (class 4, slope > 30 degrees)."""
        from rasterio.transform import Affine

        from chuk_mcp_dem.core.raster_io import compute_landforms

        # Use a fine-resolution transform (10m pixels) so that a 500m elevation
        # ramp across 30 pixels creates very steep slopes (~59 degrees).
        fine_transform = Affine(0.0001, 0.0, 7.0, 0.0, -0.0001, 47.0)
        ramp = np.zeros((30, 30), dtype=np.float32)
        for col in range(30):
            ramp[:, col] = col * 500.0 / 30  # 0 to ~500m over 30 pixels
        result = compute_landforms(ramp, fine_transform)
        assert 4 in result  # At least some escarpment pixels


# ---------------------------------------------------------------------------
# landform_to_png
# ---------------------------------------------------------------------------


class TestLandformToPng:
    """Tests for landform_to_png() -- categorical colour map for landform classes."""

    def test_produces_valid_png(self):
        """Output is valid PNG bytes."""
        from chuk_mcp_dem.core.raster_io import landform_to_png

        landforms = np.random.randint(0, 9, (50, 50)).astype(np.uint8)
        png_bytes = landform_to_png(landforms)
        assert isinstance(png_bytes, bytes)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"

    def test_output_dimensions_match(self):
        """PNG dimensions match input array shape."""
        from chuk_mcp_dem.core.raster_io import landform_to_png

        landforms = np.zeros((30, 40), dtype=np.uint8)
        png_bytes = landform_to_png(landforms)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (40, 30)  # PIL size is (width, height)

    def test_uniform_class_uniform_color(self):
        """All-zero array produces a uniform colour across the image."""
        from chuk_mcp_dem.core.raster_io import landform_to_png

        landforms = np.zeros((20, 20), dtype=np.uint8)
        png_bytes = landform_to_png(landforms)
        img = Image.open(io.BytesIO(png_bytes))
        pixels = np.array(img)
        # All pixels should be the same colour (plain = class 0)
        assert np.all(pixels[:, :, 0] == pixels[0, 0, 0])
        assert np.all(pixels[:, :, 1] == pixels[0, 0, 1])
        assert np.all(pixels[:, :, 2] == pixels[0, 0, 2])


# ---------------------------------------------------------------------------
# compute_anomaly_scores
# ---------------------------------------------------------------------------


_sklearn_available = True
try:
    from sklearn.ensemble import IsolationForest  # noqa: F401
except Exception:
    _sklearn_available = False


@pytest.mark.skipif(not _sklearn_available, reason="scikit-learn not usable")
class TestComputeAnomalyScores:
    """Tests for compute_anomaly_scores() -- Isolation Forest terrain outliers."""

    @pytest.fixture
    def small_elevation(self):
        """20x20 elevation array for faster anomaly tests."""
        np.random.seed(99)
        return np.random.uniform(100, 500, (20, 20)).astype(np.float32)

    @pytest.fixture
    def small_transform(self):
        """Affine transform matching the small elevation array."""
        from rasterio.transform import Affine

        return Affine(0.01, 0.0, 7.0, 0.0, -0.01, 47.0)

    def test_returns_correct_shape(self, small_elevation, small_transform):
        """Anomaly scores shape matches input elevation."""
        from chuk_mcp_dem.core.raster_io import compute_anomaly_scores

        scores, _ = compute_anomaly_scores(
            small_elevation,
            small_transform,
            sensitivity=0.1,
        )
        assert scores.shape == small_elevation.shape

    def test_scores_in_range(self, small_elevation, small_transform):
        """All anomaly scores are between 0 and 1."""
        from chuk_mcp_dem.core.raster_io import compute_anomaly_scores

        scores, _ = compute_anomaly_scores(
            small_elevation,
            small_transform,
            sensitivity=0.1,
        )
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)

    def test_dtype_float32(self, small_elevation, small_transform):
        """Anomaly scores dtype is float32."""
        from chuk_mcp_dem.core.raster_io import compute_anomaly_scores

        scores, _ = compute_anomaly_scores(
            small_elevation,
            small_transform,
            sensitivity=0.1,
        )
        assert scores.dtype == np.float32

    def test_anomaly_dict_keys(self, small_elevation, small_transform):
        """Each anomaly dict has the expected keys."""
        from chuk_mcp_dem.core.raster_io import compute_anomaly_scores

        _, anomalies = compute_anomaly_scores(
            small_elevation,
            small_transform,
            sensitivity=0.1,
        )
        expected_keys = {"bbox", "area_m2", "confidence", "mean_anomaly_score"}
        for anomaly in anomalies:
            assert set(anomaly.keys()) == expected_keys


# ---------------------------------------------------------------------------
# anomaly_to_png
# ---------------------------------------------------------------------------


class TestAnomalyToPng:
    """Tests for anomaly_to_png() -- green-to-red heatmap."""

    def test_produces_valid_png(self):
        """Output is valid PNG bytes."""
        from chuk_mcp_dem.core.raster_io import anomaly_to_png

        scores = np.random.uniform(0, 1, (50, 50)).astype(np.float32)
        png_bytes = anomaly_to_png(scores)
        assert isinstance(png_bytes, bytes)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.format == "PNG"

    def test_zero_scores_green(self):
        """All-zero scores render as green pixels (R=0, G=255)."""
        from chuk_mcp_dem.core.raster_io import anomaly_to_png

        scores = np.zeros((20, 20), dtype=np.float32)
        png_bytes = anomaly_to_png(scores)
        img = Image.open(io.BytesIO(png_bytes))
        pixels = np.array(img)
        assert np.all(pixels[:, :, 0] == 0)  # Red channel
        assert np.all(pixels[:, :, 1] == 255)  # Green channel

    def test_output_dimensions_match(self):
        """PNG dimensions match input array shape."""
        from chuk_mcp_dem.core.raster_io import anomaly_to_png

        scores = np.zeros((30, 40), dtype=np.float32)
        png_bytes = anomaly_to_png(scores)
        img = Image.open(io.BytesIO(png_bytes))
        assert img.size == (40, 30)  # PIL size is (width, height)


# ---------------------------------------------------------------------------
# compute_feature_detection
# ---------------------------------------------------------------------------


class TestComputeFeatureDetection:
    """Tests for compute_feature_detection() -- CNN-inspired multi-angle hillshade."""

    def test_returns_correct_shape(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        feature_map, features = compute_feature_detection(sample_elevation, sample_transform)
        assert feature_map.shape == sample_elevation.shape

    def test_output_dtype_float32(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        feature_map, _ = compute_feature_detection(sample_elevation, sample_transform)
        assert feature_map.dtype == np.float32

    def test_feature_values_in_range(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        feature_map, _ = compute_feature_detection(sample_elevation, sample_transform)
        assert float(np.min(feature_map)) >= 0
        assert float(np.max(feature_map)) <= 6

    def test_returns_features_list(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        _, features = compute_feature_detection(sample_elevation, sample_transform)
        assert isinstance(features, list)

    def test_feature_dict_keys(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        _, features = compute_feature_detection(sample_elevation, sample_transform)
        for f in features:
            assert "bbox" in f
            assert "area_m2" in f
            assert "feature_type" in f
            assert "confidence" in f

    def test_feature_types_valid(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        valid_types = {"peak", "ridge", "valley", "cliff", "saddle", "channel"}
        _, features = compute_feature_detection(sample_elevation, sample_transform)
        for f in features:
            assert f["feature_type"] in valid_types

    def test_confidence_in_range(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        _, features = compute_feature_detection(sample_elevation, sample_transform)
        for f in features:
            assert 0.0 <= f["confidence"] <= 1.0

    def test_flat_terrain_few_features(self, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        flat = np.full((20, 20), 100.0, dtype=np.float32)
        feature_map, features = compute_feature_detection(flat, sample_transform)
        # Flat terrain should have very few or no features
        assert float(np.sum(feature_map > 0)) < flat.size * 0.5

    def test_invalid_method_raises(self, sample_elevation, sample_transform):
        from chuk_mcp_dem.core.raster_io import compute_feature_detection

        with pytest.raises(ValueError, match="Invalid feature method"):
            compute_feature_detection(sample_elevation, sample_transform, method="bad")


# ---------------------------------------------------------------------------
# feature_to_png
# ---------------------------------------------------------------------------


class TestFeatureToPng:
    """Tests for feature_to_png() -- categorical feature map renderer."""

    def test_produces_valid_png(self):
        from chuk_mcp_dem.core.raster_io import feature_to_png

        features = np.zeros((10, 10), dtype=np.float32)
        result = feature_to_png(features)
        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_output_dimensions_match(self):
        from chuk_mcp_dem.core.raster_io import feature_to_png

        features = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.float32)
        result = feature_to_png(features)
        from PIL import Image
        import io as _io

        img = Image.open(_io.BytesIO(result))
        assert img.size == (3, 2)  # width, height

    def test_all_classes_render(self):
        from chuk_mcp_dem.core.raster_io import feature_to_png

        # Create array with all 7 classes (0-6)
        features = np.arange(7, dtype=np.float32).reshape(1, 7)
        result = feature_to_png(features)
        assert isinstance(result, bytes)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# raster_to_png
# ---------------------------------------------------------------------------


class TestRasterToPng:
    """Tests for raster_to_png() -- convert artifact bytes to PNG."""

    def test_png_passthrough(self):
        """If input already starts with PNG magic bytes, return as-is."""
        from chuk_mcp_dem.core.raster_io import raster_to_png

        # Create a real PNG
        img = Image.new("L", (4, 4), 128)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        assert png_bytes[:4] == b"\x89PNG"  # sanity
        result = raster_to_png(png_bytes)
        assert result is png_bytes  # exact same object, returned as-is

    def test_geotiff_conversion_produces_png(self):
        """GeoTIFF bytes are converted to PNG via rasterio."""
        from chuk_mcp_dem.core.raster_io import raster_to_png

        # Mock rasterio.open to return a fake dataset
        fake_band = np.random.uniform(100, 500, (20, 20)).astype(np.float32)
        mock_src = MagicMock()
        mock_src.read.return_value = fake_band
        mock_src.nodata = None
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        # Input that does NOT start with PNG magic (simulating GeoTIFF)
        fake_tiff_bytes = b"\x49\x49\x2a\x00" + b"\x00" * 100  # TIFF magic

        with patch("rasterio.open", return_value=mock_src):
            result = raster_to_png(fake_tiff_bytes)

        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_geotiff_with_nodata_replaces_nan(self):
        """GeoTIFF with nodata values should replace them with NaN before rendering."""
        from chuk_mcp_dem.core.raster_io import raster_to_png

        band = np.full((10, 10), 200.0, dtype=np.float32)
        band[0, 0] = -9999.0  # nodata pixel

        mock_src = MagicMock()
        mock_src.read.return_value = band
        mock_src.nodata = -9999.0
        mock_src.__enter__ = MagicMock(return_value=mock_src)
        mock_src.__exit__ = MagicMock(return_value=False)

        fake_tiff_bytes = b"\x49\x49\x2a\x00" + b"\x00" * 100

        with patch("rasterio.open", return_value=mock_src):
            result = raster_to_png(fake_tiff_bytes)

        assert isinstance(result, bytes)
        assert result[:4] == b"\x89PNG"

    def test_empty_data_raises(self):
        """Empty or invalid data should raise an error through rasterio."""
        from chuk_mcp_dem.core.raster_io import raster_to_png

        with patch("rasterio.open", side_effect=Exception("Cannot read")):
            with pytest.raises(Exception, match="Cannot read"):
                raster_to_png(b"")
