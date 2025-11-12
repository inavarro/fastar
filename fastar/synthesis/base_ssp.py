#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from functools import partial

import h5py
import jax
import jax.numpy as jnp
import numpy
from astropy.io.ascii import read as ascii_read
from jax.scipy.integrate import trapezoid

from .stellar_predictions import StellarSynthesizer
from fastar.imf import single_power_law as unimodal
from fastar.interpolate.isochrone import isochrone_interpolation
from fastar.tools.assets import get_asset_path


class PopulationIngredients(StellarSynthesizer):
    """
    Synthesizer for Simple Stellar Populations (SSPs).
    """

    def __init__(self, model_label=None, imf_function=None):
        if model_label is None:
            with h5py.File(get_asset_path('sun_ref.hdf5'), 'r') as sun:
                self.sun_spec = sun['sun_spec'][:]

        if model_label == 'phot':
            with h5py.File(get_asset_path('sun_ref.hdf5'), 'r') as sun:
                self.sun_spec = sun['sun_phot'][:]

        self.imf_function = imf_function if imf_function is not None else unimodal

        super().__init__(model_label=model_label)

        self._load_auxiliary_data()

        # Solar constants
        sun_mbol = 4.70
        sun_bvc = -0.12
        self.sun_vmag = sun_mbol - sun_bvc

    def _load_auxiliary_data(self):
        """
        Load isochrones, V-band filter response, bolometric corrections,
        and optimized age and metallicity samplings.
        """

        with h5py.File(get_asset_path('BASTI-IAC_isochrones.hdf5'), 'r') as iso:
            self.mets = iso['mets'][:]
            self.ages = iso['ages'][:]
            self.mass_ini_data = iso['mass_ini'][:]
            self.teff_out_data = iso['teff_out'][:]
            self.logg_out_data = iso['logg_out'][:]
            self.lumi_out_data = iso['lumi_out'][:]

        tab = ascii_read(get_asset_path('filters_default.res'))
        fwave = numpy.array(tab['col1'])
        fresp = numpy.array(tab['col2'])
        self.filter_response = jnp.interp(self.wave, fwave, fresp, left=0, right=0)

        with h5py.File(get_asset_path('WORTHEY11_colors.hdf5'), 'r') as color:
            self.bcv_grid = color['bcv'][:]
            self.fmet_array = color['ufmet'][:]
            self.logg_array = color['ulogg'][:]
            self.teff_log10_array = color['uteff'][:]

        with h5py.File(get_asset_path('pop_iso.hdf5'), 'r') as color:
            self.iso_ages = color['grid_ages'][:]
            self.iso_mets = color['grid_mets'][:]

    @partial(jax.jit, static_argnames=['self'])
    def _get_isochrone(self, age, met):
        """
        Retrieve interpolated isochrone for given age and metallicity.
        """
        imass, iteff, ilogg, ilum = isochrone_interpolation(
            age,
            met,
            self.ages,
            self.mets,
            self.mass_ini_data,
            self.teff_out_data,
            self.logg_out_data,
            self.lumi_out_data,
        )
        return imass, iteff, ilogg, ilum

    @partial(jax.jit, static_argnames=['self'])
    def _compute_ab_magnitudes(self, spectra, filter_response=None):
        """
        Compute AB magnitudes from synthetic spectra using a filter response.

        Parameters
        ----------
        spectra : array
            Array of synthetic spectra (shape: N x WAVE or 1 x WAVE).
        filter_response : array, optional
            Response function sampled over wavelength grid.

        Returns
        -------
        array
            AB magnitudes per spectrum.
        """

        response = (
            filter_response if filter_response is not None else self.filter_response
        )

        # Compute AB magnitudes from synthetic spectra
        denominator = trapezoid(response / self.wave, x=self.wave)
        numerators = trapezoid(spectra * response * self.wave, x=self.wave, axis=1)
        flux_density = numerators / denominator
        return -2.5 * jnp.log10(flux_density) - 2.406
