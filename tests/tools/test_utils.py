import pytest
import jax.numpy as jnp
from fastar.tools.utils import compute_ab_magnitudes, flux_sum


@pytest.mark.unit
class TestPhotometricUtils:
    """Test suite for photometric utility functions."""

    def test_compute_ab_magnitudes_returns_finite(self, wavelength_grid):
        """Test that compute_ab_magnitudes returns finite values."""
        # Create simple test spectrum and filter
        spectrum = jnp.ones_like(wavelength_grid) * 1e-17
        filter_response = jnp.exp(
            -((wavelength_grid - 5500) ** 2) / (2 * 500**2)
        )

        mag = compute_ab_magnitudes(wavelength_grid, spectrum, filter_response)
        assert jnp.isfinite(mag), 'AB magnitude should be finite'

    def test_compute_ab_magnitudes_is_scalar(self, wavelength_grid):
        """Test that compute_ab_magnitudes returns a scalar."""
        spectrum = jnp.ones_like(wavelength_grid) * 1e-17
        filter_response = jnp.exp(
            -((wavelength_grid - 5500) ** 2) / (2 * 500**2)
        )

        mag = compute_ab_magnitudes(wavelength_grid, spectrum, filter_response)
        assert jnp.ndim(mag) == 0, 'AB magnitude should be a scalar'

    def test_compute_ab_magnitudes_reasonable_range(self, wavelength_grid):
        """Test that AB magnitudes are in a reasonable range."""
        spectrum = jnp.ones_like(wavelength_grid) * 1e-17
        filter_response = jnp.exp(
            -((wavelength_grid - 5500) ** 2) / (2 * 500**2)
        )

        mag = compute_ab_magnitudes(wavelength_grid, spectrum, filter_response)
        # AB magnitudes for typical sources should be in range -30 to 40
        assert -30 < mag < 40, (
            f'AB magnitude {mag} is outside reasonable range'
        )

    def test_flux_sum_positive(self, wavelength_grid):
        """Test that flux_sum returns positive values for positive flux."""
        flux = jnp.ones_like(wavelength_grid)
        bend = wavelength_grid[10]
        rend = wavelength_grid[-10]

        total_flux = flux_sum(wavelength_grid, flux, bend, rend)
        assert total_flux > 0, 'Flux sum should be positive for positive flux'

    def test_flux_sum_zero_for_zero_flux(self, wavelength_grid):
        """Test that flux_sum returns zero for zero flux."""
        flux = jnp.zeros_like(wavelength_grid)
        bend = wavelength_grid[10]
        rend = wavelength_grid[-10]

        total_flux = flux_sum(wavelength_grid, flux, bend, rend)
        assert bool(jnp.isclose(total_flux, 0.0)), (
            'Flux sum should be zero for zero flux'
        )

    def test_flux_sum_reasonable_value(self, wavelength_grid):
        """Test that flux_sum returns reasonable values."""
        flux = jnp.ones_like(wavelength_grid)
        bend = wavelength_grid[100]
        rend = wavelength_grid[200]

        total_flux = flux_sum(wavelength_grid, flux, bend, rend)
        assert 50 < total_flux < 150, (
            f'Flux sum {total_flux} is outside expected range'
        )
