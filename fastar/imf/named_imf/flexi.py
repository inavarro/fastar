#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import jax.scipy.integrate as jsp_integrate


# =============================================================================
# 3. Tapered power-law (https://arxiv.org/abs/astro-ph/0409601)
# =============================================================================
def flexi_imf_raw(
    mass, m_min=0.1, m_max=100.0, m_peak=0.5, alpha=2.3, beta=2.3
):
    """
    Returns the normalized tapered power-law IMF as described in de Marchi,
    Paresce & Portegies Zwart (2005), evaluated at `mass` and fully
    JAX-compatible with numerical normalization.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    m_min : float, optional
        Lower mass limit. Default is 0.1.
    m_max : float, optional
        Upper mass limit. Default is 100.0.
    m_peak : float, optional
        Peak mass for tapering. Default is 0.5.
    alpha : float, optional
        Power-law slope. Default is 2.3.
    beta : float, optional
        Sharpness of the exponential taper. Default is 2.3.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    mass = jnp.atleast_1d(mass)

    def imf_unnormalized(m):
        return m ** (-alpha) * (1 - jnp.exp(-((m / m_peak) ** beta)))

    m_vals = jnp.linspace(m_min, m_max, 5000)
    norm = jsp_integrate.trapezoid(imf_unnormalized(m_vals) * m_vals, x=m_vals)

    imf_vals = imf_unnormalized(mass) / norm
    imf_vals = jnp.where((mass >= m_min) & (mass <= m_max), imf_vals, 0.0)

    return imf_vals if imf_vals.shape[0] > 1 else imf_vals[0]


def flexi(mass, params):
    """
    Wrapper for the tapered power-law IMF using a parameter dictionary.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    params : dict
        Dictionary of parameters to pass to `flexi_imf_raw`.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    return flexi_imf_raw(mass, **params)
