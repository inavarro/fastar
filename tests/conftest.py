"""
Shared pytest fixtures for FASTAR test suite.

This module provides common fixtures used across all test modules.
"""

import pytest
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.integrate import trapezoid
import numpy as np
from pathlib import Path


@pytest.fixture(scope='session')
def jax_config():
    """Configure JAX for testing."""
    # Enable 64-bit precision if needed
    # jax.config.update("jax_enable_x64", True)
    return jax


@pytest.fixture
def random_key():
    """Provide a JAX random key for stochastic tests."""
    return jr.PRNGKey(42)


@pytest.fixture
def mass_array():
    """Standard mass array for IMF testing (0.01 to 150 solar masses)."""
    return jnp.logspace(-2, 2.2, 1000)


@pytest.fixture
def linear_mass_array():
    """Linear mass array for basic IMF testing."""
    return jnp.linspace(0.01, 100, 500)


@pytest.fixture
def valid_age_range():
    """Valid age range for SSP synthesis (in Gyr)."""
    return (0.01, 14.0)  # 10 Myr to 14 Gyr


@pytest.fixture
def valid_met_range():
    """Valid metallicity range for SSP synthesis."""
    return (-2.5, 0.5)  # [Fe/H] range


@pytest.fixture
def test_ages():
    """Array of test ages for synthesis (in Gyr)."""
    return jnp.array([0.1, 1.0, 5.0, 10.0, 13.0])


@pytest.fixture
def test_metallicities():
    """Array of test metallicities for synthesis."""
    return jnp.array([-2.0, -1.0, 0.0, 0.3])


@pytest.fixture
def test_stellar_params():
    """Dictionary of test stellar parameters."""
    return {
        'logg': jnp.array([3.5, 4.0, 4.5]),
        'teff': jnp.array([4500, 5500, 6500]),
        'fmet': jnp.array([-0.5, 0.0, 0.3]),
    }


@pytest.fixture
def imf_test_params():
    """Dictionary of test IMF parameters for various IMF types."""
    return {
        'single_power_law': {'m_min': 0.1, 'm_max': 100.0, 'alpha': 2.35},
        'broken_power_law': {
            'm_min': 0.1,
            'm_max': 100.0,
            'm_break': 0.5,
            'alpha1': 1.3,
            'alpha2': 2.3,
        },
        'kroupa': {'m_min': 0.1, 'm_max': 100.0},
        'chabrier': {'m_min': 0.1, 'm_max': 100.0},
        'flexi': {
            'm_min': 0.1,
            'm_max': 100.0,
            'm_peak': 0.5,
            'alpha': 2.3,
            'beta': 2.3,
        },
    }


@pytest.fixture
def asset_dir():
    """Path to the fastar assets directory."""
    fastar_dir = Path(__file__).parent.parent / 'fastar'
    return fastar_dir / 'assets'


@pytest.fixture
def wavelength_grid():
    """Standard wavelength grid for spectroscopy tests (in Angstroms)."""
    return jnp.linspace(3500, 7400, 1000)


@pytest.fixture
def small_num_stars():
    """Small number of stars for semi-resolved testing."""
    return 100


@pytest.fixture
def large_num_stars():
    """Large number of stars for semi-resolved batch testing."""
    return 50000


# Helper functions for assertions


def assert_normalized_imf(imf_values, mass_array, tolerance=1e-3):
    """Assert that IMF integrates to approximately 1."""
    integral = trapezoid(imf_values, x=mass_array)
    np.testing.assert_allclose(
        integral, 1.0, rtol=tolerance, err_msg='IMF is not properly normalized'
    )


def assert_positive_flux(flux):
    """Assert that all flux values are non-negative."""
    assert jnp.all(flux >= 0), 'Flux contains negative values'


def assert_finite_values(array, name='Array'):
    """Assert that all values in array are finite."""
    assert jnp.all(jnp.isfinite(array)), (
        f'{name} contains non-finite values (NaN or Inf)'
    )


def assert_monotonic_increasing(array, name='Array'):
    """Assert that array is monotonically increasing."""
    assert jnp.all(jnp.diff(array) > 0), (
        f'{name} is not monotonically increasing'
    )


# Export helper functions
__all__ = [
    'assert_normalized_imf',
    'assert_positive_flux',
    'assert_finite_values',
    'assert_monotonic_increasing',
]
