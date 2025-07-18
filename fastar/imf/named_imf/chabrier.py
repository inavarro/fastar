#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=duplicate-code
# *** Duplicate code will be addressed in future IMF refactoring ***

"""
Chabrier 2003 IMF (https://arxiv.org/abs/astro-ph/0304382)
"""

import jax.numpy as jnp
import jax.scipy.integrate as jsp_integrate


def chabrier_imf_raw(mass, m_min=0.1, m_max=100.0):
    """
    Returns the normalized Chabrier IMF evaluated at `mass`,
    combining a log-normal and power-law form, fully JAX-compatible.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    m_min : float, optional
        Lower mass limit. Default is 0.1.
    m_max : float, optional
        Upper mass limit. Default is 100.0.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    mass = jnp.atleast_1d(mass)

    def log_normal(mass_value):
        return mass_value ** (-1) * jnp.exp(
            -((jnp.log10(mass_value) - jnp.log10(0.08)) ** 2) / 0.9522
        )

    def imf_unnormalized(mass_value):
        return jnp.where(
            mass_value <= 1,
            log_normal(mass_value),
            log_normal(1) * mass_value ** (-2.3),
        )

    m_vals = jnp.linspace(m_min, m_max, 5000)
    norm = jsp_integrate.trapezoid(imf_unnormalized(m_vals) * m_vals, x=m_vals)

    imf_vals = imf_unnormalized(mass) / norm
    imf_vals = jnp.where((mass >= m_min) & (mass <= m_max), imf_vals, 0.0)

    return imf_vals if imf_vals.shape[0] > 1 else imf_vals[0]


def chabrier(mass, params):
    """
    Wrapper for the Chabrier IMF using a parameter dictionary.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    params : dict
        Dictionary of parameters to pass to `chabrier_imf_raw`.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    return chabrier_imf_raw(mass, **params)
