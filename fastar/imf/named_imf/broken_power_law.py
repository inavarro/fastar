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
    mass,
    m_min=0.1,
    m_max=100.0,
    m_break1=0.5,
    m_break2=1.0,
    alpha1=1.3,
    alpha2=1.8,
    alpha3=2.3,
):
    """
    Returns the normalized broken power-law IMF evaluated at `mass`,
    fully JAX-compatible with numerical normalization.

    Behavior:
        m < m_break1            -> m^alpha1
        m_break1 ≤ m < m_break2 -> m^alpha2
        m ≥ m_break2            -> m^alpha3

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    m_min : float, optional
        Lower mass limit. Default is 0.1.
    m_max : float, optional
        Upper mass limit. Default is 100.0.
    m_break1, m_break2 : float
        Break masses (must satisfy m_min < m_break1 < m_break2 < m_max).
    alpha1, alpha2, alpha3 : float
        Slopes of the three segments.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    mass = jnp.atleast_1d(mass)

    def imf_piecewise(m):
        # Coefficients to ensure continuity
        seg1 = m ** (-alpha1)

        # Middle segment: A * m_break1^(alpha2-alpha1) * m^(-alpha2)
        coeff2 = m_break1 ** (alpha2 - alpha1)
        seg2 = coeff2 * m ** (-alpha2)

        # High-mass segment:
        # A * m_break1^(alpha2-alpha1) * m_break2^(alpha3-alpha2) * m^(-alpha3)
        coeff3 = coeff2 * (m_break2 ** (alpha3 - alpha2))
        seg3 = coeff3 * m ** (-alpha3)

        return jnp.where(
            m < m_break1,
            seg1,
            jnp.where(m < m_break2, seg2, seg3),
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
