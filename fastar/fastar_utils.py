#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

# =============================================================================
# Photometric and Spectroscopic Utilities for FaStar
# =============================================================================


def compute_ab_magnitudes(wave, spectra, fresp):
    """
    Compute AB magnitudes from synthetic spectra using a filter response.

    Parameters
    ----------
    wave : array-like
        Wavelength grid (in Angstroms).
    spectra : array-like
        Spectral fluxes sampled over `wave`.
    fresp : array-like
        Filter transmission curve sampled over `wave`.

    Returns
    -------
    float
        AB magnitude.
    """
    denominator = trapezoid(fresp / wave, x=wave)
    numerators = trapezoid(spectra * fresp * wave, x=wave)
    flux_density = numerators / denominator
    return -2.5 * jnp.log10(flux_density) - 2.406


def flux_sum(wave, flux, bend, rend):
    """
    Integrate spectral flux within a specified wavelength band,
    accounting for partial pixel coverage at band edges.

    Parameters
    ----------
    wave : array-like
        Wavelength grid.
    flux : array-like
        Flux values corresponding to `wave`.
    bend : float
        Blue (lower) limit of the band.
    rend : float
        Red (upper) limit of the band.

    Returns
    -------
    float
        Total flux within the band.
    """
    step = wave[1] - wave[0]
    total = sum(flux[(wave >= bend + step / 2) & (wave <= rend - step / 2)])

    # Handle partial pixels on blue edge
    nfrac = wave[(wave > bend - step / 2) & (wave < bend + step / 2)]
    if nfrac.size != 0:
        wfrac = ((nfrac + step / 2) - bend) / step
        total += (
            flux[(wave > bend - step / 2) & (wave < bend + step / 2)] * wfrac
        )

    # Handle partial pixels on red edge
    nfrac = wave[(wave < rend + step / 2) & (wave > rend - step / 2)]
    if nfrac.size != 0:
        wfrac = (rend - (nfrac - step / 2)) / step
        total += (
            flux[(wave > rend - step / 2) & (wave < rend + step / 2)] * wfrac
        )

    return total


def compute_linetrength(wave, flux, index, dat):
    """
    Compute the strength of a spectral feature defined by a Lick-style index.

    Parameters
    ----------
    wave : array-like
        Wavelength grid.
    flux : array-like
        Spectral flux.
    index : str
        Name of the index to compute (must match `dat['NAME']`).
    dat : Table
        Table with index definitions (Blue_1, Blue_2, Red_1, Red_2, Line_1, Line_2, T).

    Returns
    -------
    float
        Line strength in Angstroms (T=2) or magnitudes (T=1).
    """
    # Continuum regions
    blue_c = flux_sum(
        wave,
        flux,
        dat[dat['NAME'] == index]['Blue_1'][0],
        dat[dat['NAME'] == index]['Blue_2'][0],
    )
    red_c = flux_sum(
        wave,
        flux,
        dat[dat['NAME'] == index]['Red_1'][0],
        dat[dat['NAME'] == index]['Red_2'][0],
    )

    blue_width = (
        dat[dat['NAME'] == index]['Blue_1'][0]
        + dat[dat['NAME'] == index]['Blue_2'][0]
    ) / 2.0
    red_width = (
        dat[dat['NAME'] == index]['Red_1'][0]
        + dat[dat['NAME'] == index]['Red_2'][0]
    ) / 2.0

    blue_c /= (
        dat[dat['NAME'] == index]['Blue_2'][0]
        - dat[dat['NAME'] == index]['Blue_1'][0]
    )
    red_c /= (
        dat[dat['NAME'] == index]['Red_2'][0]
        - dat[dat['NAME'] == index]['Red_1'][0]
    )

    mval = (red_c - blue_c) / (red_width - blue_width)
    cval_1 = (
        mval * (dat[dat['NAME'] == index]['Line_1'][0] - blue_width) + blue_c
    )
    cval_2 = (
        mval * (dat[dat['NAME'] == index]['Line_2'][0] - blue_width) + blue_c
    )

    cont = (
        0.5
        * (cval_1 + cval_2)
        * (
            dat[dat['NAME'] == index]['Line_2'][0]
            - dat[dat['NAME'] == index]['Line_1'][0]
        )
    )

    band_c = flux_sum(
        wave,
        flux,
        dat[dat['NAME'] == index]['Line_1'][0],
        dat[dat['NAME'] == index]['Line_2'][0],
    )

    if dat[dat['NAME'] == index]['T'] == 2:
        index_val = (1.0 - (band_c / cont)) * (
            dat[dat['NAME'] == index]['Line_2'][0]
            - dat[dat['NAME'] == index]['Line_1'][0]
        )
    elif dat[dat['NAME'] == index]['T'] == 1:
        index_val = -2.5 * jnp.log10(band_c / cont)

    return index_val
