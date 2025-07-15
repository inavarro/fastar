#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

# =============================================================================
# Isochrone Interpolation over (age, [Fe/H])
# =============================================================================


def isochrone_interpolation(
    age,
    met,
    ages,
    mets,
    mass_ini_data,
    teff_out_data,
    logg_out_data,
    lumi_out_data,
):
    """
    Perform bilinear interpolation on isochrone data over a (age, [Fe/H]) grid
    using JAX's map_coordinates for efficient vectorized sampling.

    Parameters
    ----------
    age : float
        Target age in the same units as isochrone grid (e.g., Myr).
    met : float
        Target metallicity [Fe/H] value.
    ages : array-like
        1D array of age grid points.
    mets : array-like
        1D array of metallicity grid points.
    mass_ini_data : array-like, shape (n_tracks, len(ages), len(mets))
        Grid of initial stellar masses.
    teff_out_data : array-like, shape (n_tracks, len(ages), len(mets))
        Grid of effective temperatures.
    logg_out_data : array-like, shape (n_tracks, len(ages), len(mets))
        Grid of surface gravities.
    lumi_out_data : array-like, shape (n_tracks, len(ages), len(mets))
        Grid of luminosities.

    Returns
    -------
    tuple of ndarray
        Interpolated 1D arrays: (mass_ini, teff_out, logg_out, lumi_out).
    """

    # Convert age units (Gyr -> Myr)
    age = age * 1000

    # Compute fractional index positions along age and metallicity axes
    aidx = jnp.interp(age, ages, jnp.arange(len(ages)))
    midx = jnp.interp(met, mets, jnp.arange(len(mets)))

    # Prepare coordinates for sampling:
    # We sample each 'track' (first axis) exactly at integer positions,
    # and bilinearly along the last two axes via fractional aidx, midx.
    n_tracks = mass_ini_data.shape[0]

    # track indices as floats (to pick exact tracks)
    t_idx = jnp.arange(n_tracks, dtype=float)
    age_idx = jnp.full(n_tracks, aidx)
    met_idx = jnp.full(n_tracks, midx)

    # Stack into coords of shape (3, n_tracks)
    coords = jnp.stack([t_idx, age_idx, met_idx], axis=0)

    # Perform interpolation
    mass_ini = map_coordinates(mass_ini_data, coords, order=1, mode='nearest')
    teff_out = map_coordinates(teff_out_data, coords, order=1, mode='nearest')
    logg_out = map_coordinates(logg_out_data, coords, order=1, mode='nearest')
    lumi_out = map_coordinates(lumi_out_data, coords, order=1, mode='nearest')

    return mass_ini, teff_out, logg_out, lumi_out
