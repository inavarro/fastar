#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


# =============================================================================
# Color Interpolation over (logg, log10(Teff), [Fe/H])
# =============================================================================


def color_interpolation(
    logg_val, logteff_val, fmet_val, ulogg, uteff_log10, ufmet, grid
):
    """
    Trilinear interpolation of a single color grid over stellar parameters.

    These stellar parameters are (logg, log10(Teff), [Fe/H]).

    Parameters
    ----------
    logg_val : float
        Target surface gravity (logg).
    logteff_val : float
        Target effective temperature (log10 Teff).
    fmet_val : float
        Target [Fe/H].
    ulogg, uteff_log10, ufmet : array-like
        1D grid points along each axis.
    grid : array-like, shape (len(ulogg), len(uteff_log10), len(ufmet))
        The color grid to interpolate.

    Returns
    -------
    float
        Interpolated color value.
    """
    # Convert to fractional indices
    g_idx = jnp.interp(logg_val, ulogg, jnp.arange(len(ulogg)))
    t_idx = jnp.interp(logteff_val, uteff_log10, jnp.arange(len(uteff_log10)))
    m_idx = jnp.interp(fmet_val, ufmet, jnp.arange(len(ufmet)))

    # Stack into a coordinate array with shape (3, 1)
    coords = jnp.array([[g_idx], [t_idx], [m_idx]])

    # Trilinear interpolation
    return map_coordinates(grid, coords, order=1, mode='nearest')[0]
