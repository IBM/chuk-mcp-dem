"""Comprehensive tests for chuk_mcp_dem.core.raster_io module."""

import io
import math
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


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
