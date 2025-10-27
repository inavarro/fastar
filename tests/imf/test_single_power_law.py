"""
Tests for Single Power Law IMF implementation.
"""
import pytest
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from fastar.imf.named_imf.single_power_law import single_power_law_raw, single_power_law


@pytest.mark.unit
@pytest.mark.imf
class TestSinglePowerLawIMF:
    """Test suite for Single Power Law (Salpeter-like) IMF."""

    def test_salpeter_returns_finite_values(self, mass_array):
        """Test that single power law IMF returns finite values."""
        params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}
        imf_values = single_power_law_raw(mass_array, **params)
        assert jnp.all(jnp.isfinite(imf_values)), "IMF contains non-finite values"

    def test_salpeter_normalization(self, mass_array):
        """Test that single power law IMF is properly normalized."""
        params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}
        imf_values = single_power_law_raw(mass_array, **params)
        integral = trapezoid(imf_values * mass_array, x=mass_array)
        assert bool(jnp.isclose(integral, 1.0, rtol=1e-2)), f"IMF not normalized: integral = {integral}"

    def test_salpeter_positive_values(self, mass_array):
        """Test that single power law IMF returns non-negative values."""
        params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}
        imf_values = single_power_law_raw(mass_array, **params)
        assert jnp.all(imf_values >= 0), "IMF contains negative values"

    def test_different_alpha_values(self, mass_array):
        """Test that different alpha values produce different IMFs."""
        params1 = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.0}
        params2 = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.7}
        imf1 = single_power_law_raw(mass_array, **params1)
        imf2 = single_power_law_raw(mass_array, **params2)
        assert not bool(jnp.allclose(imf1, imf2)), "Different alphas should produce different IMFs"

    def test_wrapper_function(self, mass_array):
        """Test that wrapper function works correctly."""
        params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}
        imf_values = single_power_law(mass_array, params)
        assert jnp.all(jnp.isfinite(imf_values)), "Wrapper function failed"
        assert imf_values.shape == mass_array.shape, "Output shape mismatch"

    def test_outside_mass_range(self):
        """Test that IMF returns zero outside mass range."""
        mass_out_of_range = jnp.array([0.01, 200.0])
        params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}
        imf_values = single_power_law_raw(mass_out_of_range, **params)
        assert bool(jnp.allclose(imf_values, 0.0)), "IMF should be zero outside mass range"
