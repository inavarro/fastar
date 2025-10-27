"""
Tests for isochrone interpolation functions.
"""
import pytest
import jax.numpy as jnp
from fastar.interpolate.isochrone import isochrone_interpolation


@pytest.mark.unit
class TestIsochroneInterpolation:
    """Test suite for isochrone interpolation."""

    def test_isochrone_interpolation_returns_four_arrays(self):
        """Test that isochrone_interpolation returns four arrays."""
        # Create simple test data
        ages = jnp.array([1000.0, 5000.0, 10000.0])  # Myr
        mets = jnp.array([-1.0, 0.0, 0.5])
        n_tracks = 10

        # Create mock isochrone data
        mass_ini_data = jnp.ones((n_tracks, len(ages), len(mets)))
        teff_out_data = jnp.ones((n_tracks, len(ages), len(mets))) * 5000
        logg_out_data = jnp.ones((n_tracks, len(ages), len(mets))) * 4.0
        lumi_out_data = jnp.ones((n_tracks, len(ages), len(mets)))

        age = 1.0  # Gyr
        met = 0.0

        result = isochrone_interpolation(
            age, met, ages, mets,
            mass_ini_data, teff_out_data, logg_out_data, lumi_out_data
        )

        assert len(result) == 4, "Should return 4 arrays"

    def test_isochrone_interpolation_finite_values(self):
        """Test that isochrone_interpolation returns finite values."""
        ages = jnp.array([1000.0, 5000.0, 10000.0])
        mets = jnp.array([-1.0, 0.0, 0.5])
        n_tracks = 10

        mass_ini_data = jnp.linspace(0.1, 10, n_tracks)[:, None, None] * jnp.ones((n_tracks, len(ages), len(mets)))
        teff_out_data = jnp.ones((n_tracks, len(ages), len(mets))) * 5000
        logg_out_data = jnp.ones((n_tracks, len(ages), len(mets))) * 4.0
        lumi_out_data = jnp.ones((n_tracks, len(ages), len(mets)))

        age = 5.0  # Gyr
        met = 0.0

        mass_ini, teff_out, logg_out, lumi_out = isochrone_interpolation(
            age, met, ages, mets,
            mass_ini_data, teff_out_data, logg_out_data, lumi_out_data
        )

        assert jnp.all(jnp.isfinite(mass_ini)), "Mass values should be finite"
        assert jnp.all(jnp.isfinite(teff_out)), "Teff values should be finite"
        assert jnp.all(jnp.isfinite(logg_out)), "Logg values should be finite"
        assert jnp.all(jnp.isfinite(lumi_out)), "Lumi values should be finite"

    def test_isochrone_interpolation_output_shape(self):
        """Test that output arrays have correct shape."""
        ages = jnp.array([1000.0, 5000.0, 10000.0])
        mets = jnp.array([-1.0, 0.0, 0.5])
        n_tracks = 15

        mass_ini_data = jnp.ones((n_tracks, len(ages), len(mets)))
        teff_out_data = jnp.ones((n_tracks, len(ages), len(mets))) * 5000
        logg_out_data = jnp.ones((n_tracks, len(ages), len(mets))) * 4.0
        lumi_out_data = jnp.ones((n_tracks, len(ages), len(mets)))

        age = 1.0
        met = 0.0

        mass_ini, teff_out, logg_out, lumi_out = isochrone_interpolation(
            age, met, ages, mets,
            mass_ini_data, teff_out_data, logg_out_data, lumi_out_data
        )

        assert mass_ini.shape == (n_tracks,), f"Expected shape ({n_tracks},), got {mass_ini.shape}"
        assert teff_out.shape == (n_tracks,), f"Expected shape ({n_tracks},), got {teff_out.shape}"
        assert logg_out.shape == (n_tracks,), f"Expected shape ({n_tracks},), got {logg_out.shape}"
        assert lumi_out.shape == (n_tracks,), f"Expected shape ({n_tracks},), got {lumi_out.shape}"
