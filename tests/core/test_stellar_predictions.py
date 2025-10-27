"""
Tests for stellar synthesizer core functionality.
"""
import pytest
import jax.numpy as jnp
from fastar.core.stellar_predictions import StellarSynthesizer


@pytest.mark.unit
class TestStellarSynthesizer:
    """Test suite for StellarSynthesizer."""

    @pytest.fixture
    def synthesizer(self):
        """Create a StellarSynthesizer instance."""
        return StellarSynthesizer()

    def test_synthesizer_initialization(self, synthesizer):
        """Test that StellarSynthesizer initializes correctly."""
        assert synthesizer is not None, "Synthesizer should initialize"
        assert hasattr(synthesizer, 'predict_spectrum'), "Should have predict_spectrum method"

    def test_synthesizer_has_model(self, synthesizer):
        """Test that synthesizer has loaded model."""
        assert hasattr(synthesizer, 'model'), "Should have model attribute"
        assert hasattr(synthesizer, 'params'), "Should have params attribute"

    def test_synthesizer_has_wavelength_grid(self, synthesizer):
        """Test that synthesizer has wavelength grid."""
        assert hasattr(synthesizer, 'wave'), "Should have wave attribute"
        assert len(synthesizer.wave) > 0, "Wavelength grid should not be empty"

    def test_predict_spectrum_returns_array(self, synthesizer):
        """Test that predict_spectrum returns an array."""
        logg = jnp.array([4.0, 4.5])
        teff = jnp.array([5000.0, 5500.0])
        fmet = jnp.array([0.0, -0.5])

        spectra = synthesizer.predict_spectrum(logg, teff, fmet)

        assert isinstance(spectra, jnp.ndarray), "Should return JAX array"

    def test_predict_spectrum_finite_values(self, synthesizer):
        """Test that predict_spectrum returns finite values."""
        logg = jnp.array([4.0])
        teff = jnp.array([5777.0])  # Solar
        fmet = jnp.array([0.0])

        spectra = synthesizer.predict_spectrum(logg, teff, fmet)

        assert jnp.all(jnp.isfinite(spectra)), "Spectra should be finite"

    def test_predict_spectrum_positive_flux(self, synthesizer):
        """Test that predict_spectrum returns non-negative flux."""
        logg = jnp.array([4.0, 4.5])
        teff = jnp.array([5000.0, 6000.0])
        fmet = jnp.array([0.0, 0.3])

        spectra = synthesizer.predict_spectrum(logg, teff, fmet)

        assert jnp.all(spectra >= 0), "Flux should be non-negative"

    def test_softplus_activation(self, synthesizer):
        """Test that softplus activation works correctly."""
        test_input = jnp.array([0.0, 1.0, -1.0])
        result = synthesizer._softplus(test_input)

        assert jnp.all(jnp.isfinite(result)), "Softplus output should be finite"
        assert jnp.all(result >= 0), "Softplus output should be non-negative"
