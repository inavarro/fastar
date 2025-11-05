import pytest
import jax.numpy as jnp
from fastar.interpolate.color import color_interpolation


@pytest.mark.unit
class TestColorInterpolation:
    """Test suite for color interpolation."""

    def test_color_interpolation_returns_array(self):
        """Test that color_interpolation returns an array."""
        # Create simple test grids
        logg_array = jnp.array([3.0, 4.0, 5.0])
        teff_array = jnp.array([4000.0, 5000.0, 6000.0])
        fmet_array = jnp.array([-1.0, 0.0, 0.5])

        # Create a simple color grid
        color_grid = jnp.ones(
            (len(logg_array), len(teff_array), len(fmet_array))
        )

        # Test values
        logg = jnp.array([4.0, 4.5])
        teff = jnp.array([5000.0, 5500.0])
        fmet = jnp.array([0.0, 0.25])

        result = color_interpolation(
            logg, teff, fmet, logg_array, teff_array, fmet_array, color_grid
        )

        assert isinstance(result, jnp.ndarray), 'Should return a JAX array'

    def test_color_interpolation_finite_values(self):
        """Test that color_interpolation returns finite values."""
        logg_array = jnp.array([3.0, 4.0, 5.0])
        teff_array = jnp.array([4000.0, 5000.0, 6000.0])
        fmet_array = jnp.array([-1.0, 0.0, 0.5])

        color_grid = jnp.linspace(
            0, 1, len(logg_array) * len(teff_array) * len(fmet_array)
        ).reshape(len(logg_array), len(teff_array), len(fmet_array))

        logg = jnp.array([3.5, 4.5])
        teff = jnp.array([4500.0, 5500.0])
        fmet = jnp.array([-0.5, 0.25])

        result = color_interpolation(
            logg, teff, fmet, logg_array, teff_array, fmet_array, color_grid
        )

        assert jnp.all(jnp.isfinite(result)), (
            'Interpolated values should be finite'
        )

    def test_color_interpolation_output_shape(self):
        """Test that output shape matches input shape."""
        logg_array = jnp.array([3.0, 4.0, 5.0])
        teff_array = jnp.array([4000.0, 5000.0, 6000.0])
        fmet_array = jnp.array([-1.0, 0.0, 0.5])

        color_grid = jnp.ones(
            (len(logg_array), len(teff_array), len(fmet_array))
        )

        n_stars = 10
        logg = jnp.ones(n_stars) * 4.0
        teff = jnp.ones(n_stars) * 5000.0
        fmet = jnp.ones(n_stars) * 0.0

        result = color_interpolation(
            logg, teff, fmet, logg_array, teff_array, fmet_array, color_grid
        )

        assert result.shape == (n_stars,), (
            f'Expected shape ({n_stars},), got {result.shape}'
        )
