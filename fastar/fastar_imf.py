#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
import jax.scipy.integrate as jsp_integrate

# =============================================================================
# Pre-defined IMF parametrizations
# =============================================================================


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

    def imf_unnormalized(m):
        return m ** (-alpha)

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


# =============================================================================
# 2. Broken power-law
# =============================================================================
def broken_powerlaw_raw(
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

    def imf_piecewise(m):
        return jnp.where(
            m < m_break,
            m ** (-alpha1),
            (m_break ** (alpha2 - alpha1)) * m ** (-alpha2),
        )

    m_vals = jnp.linspace(m_min, m_max, 5000)
    norm = jsp_integrate.trapezoid(imf_piecewise(m_vals) * m_vals, x=m_vals)

    imf_vals = imf_piecewise(mass) / norm
    imf_vals = jnp.where((mass >= m_min) & (mass <= m_max), imf_vals, 0.0)

    return imf_vals if imf_vals.shape[0] > 1 else imf_vals[0]


def broken_powerlaw(mass, params):
    """
    Wrapper for the broken power-law IMF using a parameter dictionary.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    params : dict
        Dictionary of parameters to pass to `broken_powerlaw_raw`.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    return broken_powerlaw_raw(mass, **params)


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


# =============================================================================
# 4. Chabrier 2003 IMF (https://arxiv.org/abs/astro-ph/0304382)
# =============================================================================
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

    def log_normal(m):
        return m ** (-1) * jnp.exp(
            -((jnp.log10(m) - jnp.log10(0.08)) ** 2) / 0.9522
        )

    def imf_unnormalized(m):
        return jnp.where(m <= 1, log_normal(m), log_normal(1) * m ** (-2.3))

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


# =============================================================================
# Template for adding a new IMF
# =============================================================================
# def new_imf_raw(mass, ...):
#     \"\"\"Docstring here.\"\"\"
#     ...
#
# def new_imf(mass, params):
#     return new_imf_raw(mass, **params)
