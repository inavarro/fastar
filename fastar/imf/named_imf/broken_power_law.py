#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=duplicate-code
# *** Duplicate code will be addressed in future IMF refactoring ***

"""
Broken power-law
"""

import jax.numpy as jnp
import jax.scipy.integrate as jsp_integrate


def broken_power_law_raw(
    mass, m_min=0.1, m_max=100.0, m_break=0.5, alpha1=1.3, alpha2=2.3
):
    """
    Returns the normalized broken power-law IMF evaluated at `mass`,
    fully JAX-compatible with numerical normalization.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    m_min : float, optional
        Lower mass limit. Default is 0.1.
    m_max : float, optional
        Upper mass limit. Default is 100.0.
    m_break : float, optional
        Break point mass. Default is 0.5.
    alpha1 : float, optional
        Slope for m < m_break. Default is 1.3.
    alpha2 : float, optional
        Slope for m >= m_break. Default is 2.3.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    mass = jnp.atleast_1d(mass)

    def imf_piecewise(mass_value):
        return jnp.where(
            mass_value < m_break,
            mass_value ** (-alpha1),
            (m_break ** (alpha2 - alpha1)) * mass_value ** (-alpha2),
        )

    m_vals = jnp.linspace(m_min, m_max, 5000)
    norm = jsp_integrate.trapezoid(imf_piecewise(m_vals) * m_vals, x=m_vals)

    imf_vals = imf_piecewise(mass) / norm
    imf_vals = jnp.where((mass >= m_min) & (mass <= m_max), imf_vals, 0.0)

    return imf_vals if imf_vals.shape[0] > 1 else imf_vals[0]


def broken_power_law(mass, params):
    """
    Wrapper for the broken power-law IMF using a parameter dictionary.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    params : dict
        Dictionary of parameters to pass to `broken_power_law_raw`.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    return broken_power_law_raw(mass, **params)
