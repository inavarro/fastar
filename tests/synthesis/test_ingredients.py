# -*- coding: utf-8 -*-
import jax.numpy as jnp
import pytest

from fastar.synthesis.ssp.base import BaseSSPSynthesizer


@pytest.mark.unit
class TestBaseSSPSynthesizer:
    """Test suite for PopulationIngredients."""

    @pytest.fixture
    def pop_ingredients(self):
        """Create a PopulationIngredients instance."""
        return BaseSSPSynthesizer()

    def test_initialization(self, pop_ingredients):
        """Test that PopulationIngredients initializes correctly."""
        assert pop_ingredients is not None, 'Should initialize'
        assert hasattr(pop_ingredients, 'imf_function'), 'Should have IMF function'

    def test_has_isochrone_data(self, pop_ingredients):
        """Test that isochrone data is loaded."""
        assert hasattr(pop_ingredients, 'ages'), 'Should have ages'
        assert hasattr(pop_ingredients, 'mets'), 'Should have metallicities'
        assert hasattr(pop_ingredients, 'mass_ini_data'), 'Should have mass data'

    def test_has_solar_reference(self, pop_ingredients):
        """Test that solar reference is loaded."""
        assert hasattr(pop_ingredients, 'sun_spec'), 'Should have solar spectrum'
        assert hasattr(pop_ingredients, 'sun_vmag'), 'Should have solar V magnitude'

    def test_get_isochrone_returns_four_values(self, pop_ingredients):
        """Test that _get_isochrone returns four arrays."""
        age = 1.0  # Gyr
        met = 0.0

        result = pop_ingredients._get_isochrone(age, met)

        assert isinstance(result, tuple), 'Should return tuple'
        assert len(result) == 4, 'Should return 4 arrays (mass, teff, logg, lumi)'

    def test_get_isochrone_finite_values(self, pop_ingredients):
        """Test that _get_isochrone returns finite values."""
        age = 5.0
        met = -0.5

        imass, iteff, ilogg, ilum = pop_ingredients._get_isochrone(age, met)

        assert jnp.all(jnp.isfinite(imass)), 'Mass should be finite'
        assert jnp.all(jnp.isfinite(iteff)), 'Teff should be finite'
        assert jnp.all(jnp.isfinite(ilogg)), 'Logg should be finite'
        assert jnp.all(jnp.isfinite(ilum)), 'Luminosity should be finite'

    def test_filter_response_loaded(self, pop_ingredients):
        """Test that filter response is loaded."""
        assert hasattr(pop_ingredients, 'filter_response'), (
            'Should have filter response'
        )
        assert len(pop_ingredients.filter_response) > 0, (
            'Filter response should not be empty'
        )
