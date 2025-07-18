#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import jax.scipy.integrate as jsp_integrate


# =============================================================================
# 1. Single Power Law (honoring Salpeter's visionary work
# https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S)
# =============================================================================
def single_powerlaw_raw(mass, m_min=0.1, m_max=100.0, alpha=2.35):
    """
    Returns the normalized Salpeter IMF evaluated at `mass` over the range
    [m_min, m_max], fully JAX-compatible with numerical normalization.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    m_min : float, optional
        Lower mass limit of the IMF. Default is 0.1.
    m_max : float, optional
        Upper mass limit of the IMF. Default is 100.0.
    alpha : float, optional
        Power-law slope. Default is 2.35.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values for the input mass(es).
    """
    mass = jnp.atleast_1d(mass)

    def imf_unnormalized(mass_value):
        return mass_value ** (-alpha)

    m_vals = jnp.linspace(m_min, m_max, 5000)
    norm = jsp_integrate.trapezoid(imf_unnormalized(m_vals) * m_vals, x=m_vals)

    imf_vals = imf_unnormalized(mass) / norm
    imf_vals = jnp.where((mass >= m_min) & (mass <= m_max), imf_vals, 0.0)

    return imf_vals if imf_vals.shape[0] > 1 else imf_vals[0]


def single_powerlaw(mass, params):
    """
    Wrapper for the Salpeter IMF using a parameter dictionary.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    params : dict
        Dictionary of parameters to pass to `single_powerlaw_raw`.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    return single_powerlaw_raw(mass, **params)
