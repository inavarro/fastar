"""
Tests for Kroupa IMF implementation.
"""
import pytest
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
from fastar.imf.named_imf.kroupa import kroupa_imf_raw, kroupa


@pytest.mark.unit
@pytest.mark.imf
class TestKroupaIMF:
    """Test suite for Kroupa IMF."""

    def test_kroupa_returns_finite_values(self, mass_array):
        """Test that Kroupa IMF returns finite values."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = kroupa_imf_raw(mass_array, **params)
        assert jnp.all(jnp.isfinite(imf_values)), "IMF contains non-finite values"

    def test_kroupa_normalization(self, mass_array):
        """Test that Kroupa IMF is properly normalized."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = kroupa_imf_raw(mass_array, **params)
        integral = trapezoid(imf_values * mass_array, x=mass_array)
        assert bool(jnp.isclose(integral, 1.0, rtol=1e-2)), f"IMF not normalized: integral = {integral}"

    def test_kroupa_positive_values(self, mass_array):
        """Test that Kroupa IMF returns non-negative values."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = kroupa_imf_raw(mass_array, **params)
        assert jnp.all(imf_values >= 0), "IMF contains negative values"

    def test_kroupa_single_mass(self):
        """Test that Kroupa IMF works with a single mass value."""
        mass = 1.0
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_value = kroupa_imf_raw(mass, **params)
        assert isinstance(float(imf_value), float), "Single mass should return scalar"
        assert jnp.isfinite(imf_value), "Single mass value is not finite"

    def test_kroupa_wrapper_function(self, mass_array):
        """Test that kroupa wrapper function works correctly."""
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = kroupa(mass_array, params)
        assert jnp.all(jnp.isfinite(imf_values)), "Wrapper function failed"
        assert imf_values.shape == mass_array.shape, "Output shape mismatch"

    def test_kroupa_outside_mass_range(self):
        """Test that Kroupa IMF returns zero outside mass range."""
        mass_out_of_range = jnp.array([0.01, 200.0])
        params = {'m_min': 0.1, 'm_max': 100.0}
        imf_values = kroupa_imf_raw(mass_out_of_range, **params)
        assert bool(jnp.allclose(imf_values, 0.0)), "IMF should be zero outside mass range"
