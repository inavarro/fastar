# -*- coding: utf-8 -*-
import jax.numpy as jnp
import pytest

from fastar import IntegratedSSPSynthesizer


@pytest.mark.unit
@pytest.mark.synthesis
class TestIntegratedSSPSynthesizer:
    """Test suite for IntegratedSynthesizer."""

    @pytest.fixture
    def synthesizer(self):
        """Create an IntegratedSynthesizer instance."""
        return IntegratedSSPSynthesizer()

    def test_synthesizer_initialization(self, synthesizer):
        """Test that IntegratedSynthesizer initializes correctly."""
        assert synthesizer is not None, 'Synthesizer should initialize'
        assert hasattr(synthesizer, 'synthesize'), (
            'Synthesizer should have synthesize method'
        )

    def test_synthesize_returns_tuple(self, synthesizer):
        """Test that synthesize returns a tuple."""
        age = 1.0  # Gyr
        met = 0.0  # Solar metallicity
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        result = synthesizer.synthesize(age, met, imf_params)

        assert isinstance(result, tuple), 'Synthesize should return a tuple'
        assert len(result) == 2, 'Synthesize should return (wavelength, spectrum)'

    def test_synthesize_finite_output(self, synthesizer):
        """Test that synthesize returns finite values."""
        age = 5.0
        met = -0.5
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        wave, spectrum = synthesizer.synthesize(age, met, imf_params)

        assert jnp.all(jnp.isfinite(wave)), 'Wavelength should be finite'
        assert jnp.all(jnp.isfinite(spectrum)), 'Spectrum should be finite'

    def test_synthesize_positive_flux(self, synthesizer):
        """Test that synthesize returns non-negative flux values."""
        age = 1.0
        met = 0.0
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        wave, spectrum = synthesizer.synthesize(age, met, imf_params)

        assert jnp.all(spectrum >= 0), 'Spectrum should be non-negative'

    def test_synthesize_wavelength_increasing(self, synthesizer):
        """Test that wavelength array is monotonically increasing."""
        age = 1.0
        met = 0.0
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        wave, spectrum = synthesizer.synthesize(age, met, imf_params)

        assert jnp.all(jnp.diff(wave) > 0), (
            'Wavelength should be monotonically increasing'
        )

    def test_synthesize_different_ages(self, synthesizer):
        """Test that different ages produce different spectra."""
        met = 0.0
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        wave1, spectrum1 = synthesizer.synthesize(0.1, met, imf_params)
        wave2, spectrum2 = synthesizer.synthesize(8.0, met, imf_params)
        print('allclose:', jnp.allclose(spectrum1, spectrum2))
        print(spectrum1)
        print(spectrum2)
        # Spectra should be different for very different ages
        # Convert JAX boolean to Python bool for assertion
        assert not jnp.allclose(spectrum1, spectrum2, rtol=1e-2, atol=1e-12), (
            'Different ages should produce different spectra'
        )
