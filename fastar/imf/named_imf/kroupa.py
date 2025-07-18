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

    m_1 = 0.08
    m_2 = 0.5

    a_1 = 0.3
    a_2 = 1.3
    a_3 = 2.3

    upper_a_1 = 1.0
    upper_a_2 = upper_a_1 * m_1 ** (a_2 - a_1)
    upper_a_3 = upper_a_2 * m_2 ** (a_3 - a_2)

    def imf_unnormalized(mass_value):
        return jnp.where(
            mass_value < m_1,
            upper_a_1 * mass_value ** (-a_1),
            jnp.where(
                mass_value < m_2,
                upper_a_2 * mass_value ** (-a_2),
                upper_a_3 * mass_value ** (-a_3),
            ),
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
