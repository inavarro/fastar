"""
Tests for semi-resolved SSP synthesizer.
"""
import pytest
import jax.numpy as jnp
import jax.random as jr
from fastar.semiresolved_ssp import SemiresolvedSynthesizer


@pytest.mark.unit
@pytest.mark.synthesis
class TestSemiresolvedSynthesizer:
    """Test suite for SemiresolvedSynthesizer."""

    @pytest.fixture
    def synthesizer(self):
        """Create a SemiresolvedSynthesizer instance."""
        return SemiresolvedSynthesizer()

    @pytest.fixture
    def prng_key(self):
        """Create a PRNG key for stochastic sampling."""
        return jr.PRNGKey(42)

    def test_synthesizer_initialization(self, synthesizer):
        """Test that SemiresolvedSynthesizer initializes correctly."""
        assert synthesizer is not None, "Synthesizer should initialize"
        assert hasattr(synthesizer, 'synthesize'), "Synthesizer should have synthesize method"

    def test_synthesize_returns_tuple(self, synthesizer, prng_key, small_num_stars):
        """Test that synthesize returns a tuple."""
        age = 1.0  # Gyr
        met = 0.0  # Solar metallicity
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        result = synthesizer.synthesize(age, met, small_num_stars, prng_key, imf_params)

        assert isinstance(result, tuple), "Synthesize should return a tuple"
        assert len(result) == 3, "Synthesize should return (wavelength, spectrum, mass)"

    def test_synthesize_finite_output(self, synthesizer, prng_key, small_num_stars):
        """Test that synthesize returns finite values."""
        age = 5.0
        met = -0.5
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        wave, spectrum, mass = synthesizer.synthesize(age, met, small_num_stars, prng_key, imf_params)

        assert jnp.all(jnp.isfinite(wave)), "Wavelength should be finite"
        assert jnp.all(jnp.isfinite(spectrum)), "Spectrum should be finite"
        assert jnp.isfinite(mass), "Total mass should be finite"

    def test_synthesize_positive_flux(self, synthesizer, prng_key, small_num_stars):
        """Test that synthesize returns non-negative flux values."""
        age = 1.0
        met = 0.0
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        wave, spectrum, mass = synthesizer.synthesize(age, met, small_num_stars, prng_key, imf_params)

        assert jnp.all(spectrum >= 0), "Spectrum should be non-negative"

    def test_synthesize_positive_mass(self, synthesizer, prng_key, small_num_stars):
        """Test that total mass is positive."""
        age = 1.0
        met = 0.0
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        wave, spectrum, mass = synthesizer.synthesize(age, met, small_num_stars, prng_key, imf_params)

        assert mass > 0, "Total mass should be positive"

    def test_synthesize_stochastic_variance(self, synthesizer, small_num_stars):
        """Test that different random keys produce different results."""
        age = 1.0
        met = 0.0
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        key1 = jr.PRNGKey(42)
        key2 = jr.PRNGKey(123)

        wave1, spectrum1, mass1 = synthesizer.synthesize(age, met, small_num_stars, key1, imf_params)
        wave2, spectrum2, mass2 = synthesizer.synthesize(age, met, small_num_stars, key2, imf_params)

        # With stochastic sampling, different keys should give different results
        # Convert JAX boolean to Python bool for assertion
        assert not jnp.allclose(spectrum1, spectrum2, rtol=1e-2, atol=1e-12),\
            "Different random keys should produce different spectra"

    def test_synthesize_out_masses_option(self, synthesizer, prng_key, small_num_stars):
        """Test that out_masses=True returns array of stellar masses."""
        age = 1.0
        met = 0.0
        imf_params = {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35}

        wave, spectrum, masses = synthesizer.synthesize(
            age, met, small_num_stars, prng_key, imf_params, out_masses=True
        )

        assert isinstance(masses, jnp.ndarray), "Should return array of masses when out_masses=True"
        assert masses.shape[0] == small_num_stars, f"Should have {small_num_stars} stellar masses"
