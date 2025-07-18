#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import jax.scipy.integrate as jsp_integrate


# =============================================================================
# 5. Kroupa 2001 (https://arxiv.org/abs/astro-ph/0009005)
# =============================================================================
def kroupa_imf_raw(mass, m_min=0.1, m_max=100.0):
    """
    Returns the normalized Kroupa IMF evaluated at `mass`,
    using three power-law segments with continuity, fully JAX-compatible.

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

    m1 = 0.08
    m2 = 0.5

    a1 = 0.3
    a2 = 1.3
    a3 = 2.3

    A1 = 1.0
    A2 = A1 * m1 ** (a2 - a1)
    A3 = A2 * m2 ** (a3 - a2)

    def imf_unnormalized(m):
        return jnp.where(
            m < m1,
            A1 * m ** (-a1),
            jnp.where(m < m2, A2 * m ** (-a2), A3 * m ** (-a3)),
        )

    m_vals = jnp.linspace(m_min, m_max, 5000)
    norm = jsp_integrate.trapezoid(imf_unnormalized(m_vals) * m_vals, x=m_vals)

    imf_vals = imf_unnormalized(mass) / norm
    imf_vals = jnp.where((mass >= m_min) & (mass <= m_max), imf_vals, 0.0)

    return imf_vals if imf_vals.shape[0] > 1 else imf_vals[0]


def kroupa(mass, params):
    """
    Wrapper for the Kroupa IMF using a parameter dictionary.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    params : dict
        Dictionary of parameters to pass to `kroupa_imf_raw`.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    return kroupa_imf_raw(mass, **params)
