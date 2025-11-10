import pytest
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from fastar.imf.named_imf.chabrier import chabrier_imf_raw, chabrier


@pytest.mark.unit
@pytest.mark.imf
class TestChabrierIMF:
    """Test suite for Chabrier IMF."""

    def test_chabrier_returns_finite_values(self, mass_array):
        """Test that Chabrier IMF returns finite values."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = chabrier_imf_raw(mass_array, **params)
        assert jnp.all(jnp.isfinite(imf_values)), (
            'IMF contains non-finite values'
        )

    def test_chabrier_normalization(self, mass_array):
        """Test that Chabrier IMF is properly normalized."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = chabrier_imf_raw(mass_array, **params)
        integral = trapezoid(imf_values * mass_array, x=mass_array)
        assert bool(jnp.isclose(integral, 1.0, rtol=1e-2)), (
            f'IMF not normalized: integral = {integral}'
        )

    def test_chabrier_positive_values(self, mass_array):
        """Test that Chabrier IMF returns non-negative values."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = chabrier_imf_raw(mass_array, **params)
        assert jnp.all(imf_values >= 0), 'IMF contains negative values'

    def test_chabrier_peak_near_expected(self, mass_array):
        """Test that Chabrier IMF peaks near expected mass (~0.2-0.3 solar masses)."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = chabrier_imf_raw(mass_array, **params)
        peak_idx = jnp.argmax(imf_values)
        peak_mass = mass_array[peak_idx]
        # Peak should be roughly around 0.2-0.4 solar masses for log-normal part
        assert 0.05 < peak_mass < 0.5, f'Peak at unexpected mass: {peak_mass}'

    def test_chabrier_wrapper_function(self, mass_array):
        """Test that chabrier wrapper function works correctly."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = chabrier(mass_array, params)
        assert jnp.all(jnp.isfinite(imf_values)), 'Wrapper function failed'
        assert imf_values.shape == mass_array.shape, 'Output shape mismatch'

    def test_chabrier_outside_mass_range(self):
        """Test that Chabrier IMF returns zero outside mass range."""
        mass_out_of_range = jnp.array([0.01, 200.0])
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = chabrier_imf_raw(mass_out_of_range, **params)
        assert bool(jnp.allclose(imf_values, 0.0)), (
            'IMF should be zero outside mass range'
        )
