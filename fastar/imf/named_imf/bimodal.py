#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=duplicate-code
# *** Duplicate code will be addressed in future IMF refactoring ***
"""
Bimodal IMF shape as defined in Vazdekis et al. 1996
https://ui.adsabs.harvard.edu/abs/1996ApJS..106..307V

This shapes aims to generalize a Kroupa-like IMF, with
two segments connected through a smooth spline.

The version coded in FASTAR generalizes the bimodal IMF
with the possibility of a variable low-mass end slope.
"""

import jax.numpy as jnp
import jax.scipy.integrate as jsp_integrate


def _compute_spline(alpha_B, beta_B, bicl, bicp, bich):
    """
    Solve for the spline coefficients so the IMF is continuous
    and differentiable with the following behavior:
        xi_low ~ M^{-beta_B}
        xi_high ~ M^{-alpha_high}.
    """

    # Coefficient matrix
    A = jnp.zeros((4, 4))

    A = A.at[0, 0].set(bicl**3)
    A = A.at[1, 0].set(3.0 * bicl**2)
    A = A.at[2, 0].set(bich**3)
    A = A.at[3, 0].set(3.0 * bich**2)

    A = A.at[0, 1].set(bicl**2)
    A = A.at[1, 1].set(2.0 * bicl)
    A = A.at[2, 1].set(bich**2)
    A = A.at[3, 1].set(2.0 * bich)

    A = A.at[0, 2].set(bicl)
    A = A.at[1, 2].set(1.0)
    A = A.at[2, 2].set(bich)
    A = A.at[3, 2].set(1.0)

    A = A.at[0, 3].set(1.0)
    A = A.at[1, 3].set(0.0)
    A = A.at[2, 3].set(1.0)
    A = A.at[3, 3].set(0.0)

    # Boundaries
    bb0 = bicp ** (beta_B - alpha_B) * bicl ** (1.0 - beta_B)
    bb1 = bicp ** (beta_B - alpha_B) * (1.0 - beta_B) * bicl ** (-beta_B)
    bb2 = bich ** (1.0 - alpha_B)
    bb3 = (1.0 - alpha_B) * bich ** (-alpha_B)

    bb = jnp.array([bb0, bb1, bb2, bb3])

    coeff = jnp.linalg.solve(A, bb)

    return coeff


def _bimodal_unnormalized(mass, alpha_B, beta_B, bicl, bicp, bich):
    """
    Un-normalized (in mass) bimodal functional form
    """
    mass = jnp.atleast_1d(mass)

    s1, s2, s3, s4 = _compute_spline(alpha_B, beta_B, bicl, bicp, bich)

    # Low-mass: xi_low = bicp^{beta_B - alpha} * M^{-beta_B}
    f_low = bicp ** (beta_B - alpha_B) * mass ** (-beta_B)

    # Mid-range: spline
    f_mid = s1 * mass**2 + s2 * mass + s3 + s4 / mass

    # High-mass: xi_high = M^{-alpha_high}
    f_high = mass ** (-alpha_B)

    f = jnp.zeros_like(mass)
    f = jnp.where(mass <= bicl, f_low, f)
    f = jnp.where((mass > bicl) & (mass < bich), f_mid, f)
    f = jnp.where(mass >= bich, f_high, f)

    return f


def bimodal_raw(
    mass, m_min=0.1, m_max=100.0, alpha_B=2.3, beta_B=1.0, bicl=0.2, bicp=0.4, bich=0.6
):
    """
    Returns the normalized bimodal IMF evaluated at `mass`, with the low-mass
    end scaling as M^{-beta_B} and the high-mass as M^{-alpha_B}.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    m_min : float, optional
        Lower mass limit. Default is 0.1.
    m_max : float, optional
        Upper mass limit. Default is 100.0.
    alpha_B : float, optional
        High-mass end slope (Milky Way-like = 2.3)
    beta_B : float, optional
        Low-mass end slope (Milky Way-like = 1.0)
    bicl : float, optional
        End of the low-mass end regime. Default is 0.2
    bicp : float, optional
        Turning point. Default is 0.4
    bich : float, optional
        End of the high-mass end regime. Default is 0.6

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """
    mass = jnp.atleast_1d(mass)

    def imf_unnorm(m):
        return _bimodal_unnormalized(m, alpha_B, beta_B, bicl, bicp, bich)

    m_vals = jnp.linspace(m_min, m_max, 5000)
    f_vals = imf_unnorm(m_vals)

    norm = jsp_integrate.trapezoid(f_vals * m_vals, x=m_vals)

    imf_vals = imf_unnorm(mass) / norm
    in_range = (mass >= m_min) & (mass <= m_max)
    imf_vals = jnp.where(in_range, imf_vals, 0.0)

    return imf_vals if imf_vals.shape[0] > 1 else imf_vals[0]


def bimodal(mass, params):
    """
    Wrapper for the bimodal IMF using a parameter dictionary.

    Parameters
    ----------
    mass : array-like
        Stellar mass or array of masses.
    params : dict
        Dictionary of parameters to pass to `bimodal_raw`.

    Returns
    -------
    jnp.ndarray or float
        Normalized IMF values.
    """

    return bimodal_raw(mass, **params)
