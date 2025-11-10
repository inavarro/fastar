#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import os
import h5py
from functools import partial

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.integrate import trapezoid

from fastar.core.ingredients import PopulationIngredients
from fastar.interpolate.color import color_interpolation


class IntegratedSynthesizer(PopulationIngredients):
    """
    Class for generating synthetic integrated SSP spectroscopic and photometric
    predictions with a PCA-based stellar spectral model.
    """

    @partial(jax.jit, static_argnames=['self'])
    def synthesize(self, age, met, imf_params=None):
        """
        Generate an SSP spectrum for a given age and metallicity.

        Parameters
        ----------
        age : float
            Age of the population (in Gyr).
        met : float
            Metallicity [M/H].
        imf_params : dict, optional
            Parameters for the IMF. Default is empty dict.

        Returns
        -------
        tuple
            Wavelength grid and synthesized spectrum.
        """
        # ensure we always pass a dict to IMF **params
        imf_params = imf_params or {}

        # Interpolate the isochrones at the desired age and metallicity
        imass, iteff, ilogg, ilum = self._get_isochrone(age, met)

        # Mock metallicity array
        imet = jnp.full_like(iteff, met)

        # Evaluate IMF value at the isochrone stellar masses
        imf_val = self.imf_function(imass, imf_params)

        # Evaluate the SSP
        spec = self.wrapper(imass, iteff, ilogg, ilum, imet, imf_val)

        return self.wave, spec

    @partial(jax.jit, static_argnames=['self'])
    def wrapper(self, imass, iteff, ilogg, ilum, imet, imf_val):

        # Calculate the stellar spectra
        spectra = self.predict_spectrum(ilogg, iteff, imet)

        # Calculate the bolometric corrections
        bcv_val = color_interpolation(
            ilogg,
            iteff,
            imet,
            self.logg_array,
            self.teff_log10_array,
            self.fmet_array,
            self.bcv_grid,
        )

        # Get the V-band magnitudes of the predicted stellar spectra (they are
        # normalized to have a mean flux of 1)
        magnitudes = self._compute_ab_magnitudes(spectra)

        # Scale the predicted stellar spectra so they math their theoretical
        # luminosities
        vmags = -2.5 * ilum - bcv_val
        mtarg = vmags + self.sun_vmag
        corr = 1 / jnp.power(10.0, (magnitudes - mtarg) / -2.5)

        # Integrate corrected spectra over IMF-weighted stars
        spec = self._population_synthesis_integrate(
            spectra, corr, imf_val, imass
        )

        return spec

    @partial(jax.jit, static_argnames=['self','nsim'])
    def synthesize_nsim(
        self,
        age,
        met,
        imf_params=None,
        dmet=0.1,
        dteff=0.005,
        dlogg=0.2,
        nsim=50,
        key=jr.PRNGKey(0),
    ):
        """
        Estimate SSP spectral uncertainties via Monte Carlo perturbation
        of the stellar parameters. Returns (wave, std_spectrum).
        """

        imf_params = imf_params or {}

        # Base isochrone & arrays
        imass, iteff, ilogg, ilum = self._get_isochrone(age, met)
        imf_val = self.imf_function(imass, imf_params)
        
        imet = jnp.full_like(iteff, met)

        # Prepare per-sim RNG keys
        k1, k2, k3 = jr.split(key, 3)
        ilogg_p = ilogg + jr.normal(k1, (nsim,)+ilogg.shape) * dlogg
        iteff_p = iteff + jr.normal(k2, (nsim,)+ iteff.shape) * dteff
        imet_p  = imet  + jr.normal(k3, (nsim,)+ imet.shape) * dmet

        specs = jax.vmap(
            self.wrapper, 
            in_axes=(None,0,0,None,0,None)
            )(imass, iteff_p, ilogg_p, ilum, imet_p, imf_val)
        ssp_std = jnp.std(specs, axis=0)
        return self.wave, ssp_std

    @partial(jax.jit, static_argnames=['self','nsim'])
    def synthesize_nsim_systematic(
        self,
        age,
        met,
        imf_params=None,
        dmet=0.1,
        dteff=0.005,
        dlogg=0.2,
        nsim=50,
        key=jr.PRNGKey(0),
    ):
        """
        Estimate SSP spectral uncertainties via Monte Carlo perturbation
        of the stellar parameters. In contrast to `synthesize_nsim`, this
        systematically shifts parameters of all stars in the SSP by the same
        amount. Returns (wave, std_spectrum).
        """

        imf_params = imf_params or {}

        # Base isochrone & arrays
        imass, iteff, ilogg, ilum = self._get_isochrone(age, met)
        imf_val = self.imf_function(imass, imf_params)

        imet = jnp.full_like(iteff, met)

        # Prepare per-sim RNG keys
        k1, k2, k3 = jr.split(key, 3)
        ilogg_p = jr.normal(k1, (nsim,)) * dlogg
        iteff_p = jr.normal(k2, (nsim,)) * dteff
        imet_p  = jr.normal(k3, (nsim,)) * dmet

        ilogg_p = ilogg + ilogg_p[:,jnp.newaxis]
        iteff_p = iteff + iteff_p[:,jnp.newaxis]
        imet_p = imet + imet_p[:,jnp.newaxis]

        spec0 = self.wrapper(imass, iteff, ilogg, ilum, imet, imf_val)
        specs = jax.vmap(self.wrapper, in_axes=(None,0,0,None,0,None))(imass, iteff_p, ilogg_p, ilum, imet_p, imf_val)
        ssp_std = jnp.std(specs, axis=0)
        return self.wave, ssp_std

    @partial(jax.jit, static_argnames=['self'])
    def _population_synthesis_integrate(
        self, spectra, corr, imf_val, imass
    ):
        """
        Integrate IMF-weighted, corrected spectra over initial mass grid.
        """
        weights = corr * imf_val
        integrand = spectra * weights[:, None]
        return trapezoid(integrand, x=imass, axis=0)

    def stellar_mass(self, age, met, imf_params=None):
        """
        Compute the stellar mass still contributing to the
        flux in the SSP.

        Returns
        -------
        float
            Total stellar mass (M_sun).
        """
        # ensure we always pass a dict to IMF **params
        imf_params = imf_params or {}

        # Interpolate isochrone at given age and metallicity
        imass, _, _, _ = self._get_isochrone(age, met)

        # Evaluate IMF (can be overridden per call)
        imf_val = self.imf_function(imass, imf_params)

        return trapezoid(imf_val * imass, x=imass)

    @partial(jax.jit, static_argnames=['self'])
    def mass_to_light_ratio(
        self, age, met, imf_params=None, filter_response=None, solar_mag=None
    ):
        """
        Compute the mass-to-light ratio of an SSP in a any photometric filter.

        This function synthesizes an SSP spectrum for the given `age` and
        `met`, integrates the total stellar mass from the IMF, and computes the
        AB magnitude of the integrated spectrum using the specified filter
        response.

        If no solar magnitude is provided (`solar_mag=None`), the magnitude of
        the Sun in the same filter is computed from the stored reference solar
        spectrum. This allows the M/L ratio to be returned in solar units.

        Parameters
        ----------
        age : float
            Age of the stellar population in Gyr.
        met : float
            Metallicity [M/H] of the population.
        imf_params : dict, optional
            Dictionary of parameters for the initial mass function. Default is
            empty dict.
        filter_response : array-like or None, optional
            Response curve sampled over the wavelength grid. If None, the
            default V-band filter response is used.
        solar_mag : float or None, optional
            AB magnitude of the Sun in the same filter. If None, computed from
            solar spectrum.
        Returns
        -------
        dict
            Dictionary containing:
               - "ml_stars" : float
                 Stellar mass-to-light ratio (M*/L) in solar units.
               - "ml_total" : float
                 Total mass-to-light ratio (M_total/L), assuming total mass = 1
                 solar mass.
        """

        # ensure we always pass a dict to IMF **params
        imf_params = imf_params or {}

        response = (
            filter_response
            if filter_response is not None
            else self.filter_response
        )

        stellar_mass = self.stellar_mass(age, met, imf_params)
        _, spectrum = self.synthesize(age, met, imf_params)

        if solar_mag is None:
            m_sun = self._compute_ab_magnitudes(
                self.sun_spec[None, :], filter_response=response
            )[0]
        else:
            m_sun = solar_mag

        # AB magnitude of integrated spectrum
        ab_mag = self._compute_ab_magnitudes(
            spectrum[None, :], filter_response=response
        )[0]
        luminosity = 10 ** (-0.4 * (ab_mag - m_sun))

        return {
            'ml_stars': stellar_mass / luminosity,  # M*/L
            'ml_total': 1.0 / luminosity,  # M_total/L
        }

    def load_precomputed_models(
        self,
        age_range=None,
        met_range=None,
        imf_range=None,
        cache_dir="ssp_cache",
        user_label=''
    ):
        """
        Compute or load a grid of precomputed SSP spectra and save them to disk.

        This function evaluates the SSP model on a grid of age, metallicity, and IMF
        parameters. If a cached file with matching parameters exists, it loads the data
        from disk. Otherwise, it computes the SSP grid, saves it to an HDF5 file, and 
        returns the resulting data arrays.

        Parameters
        ----------
        age_range : array-like or None, optional
            Array of SSP ages in Gyr. If None, uses the default isochrone age grid.
        met_range : array-like or None, optional
            Array of SSP metallicities [M/H]. If None, uses the default isochrone metallicity grid.
        imf_range : list of dicts or None, optional
            List of parameter dictionaries for the IMF function. Each dictionary defines one IMF configuration.
            If None, defaults to a single empty dictionary (default IMF).
        cache_dir : str, optional
            Directory where the SSP grids are stored or will be saved. Default is "ssp_cache".
        user_label : str, optional
            Optional string appended to the output filename for custom identification.

        Returns
        -------
        wave : ndarray
            Wavelength grid of the synthesized SSP spectra.
        spec_grid : ndarray
            SSP spectra on the specified grid, with shape (n_ages, n_mets, n_imfs, n_wave).
        age_range : ndarray
            Array of ages used in the grid.
        met_range : ndarray
            Array of metallicities used in the grid.
        imf_range : list of dict
            List of IMF parameter dictionaries used in the grid.
        """

        age_range = jnp.array(age_range if age_range is not None else self.iso_ages)
        met_range = jnp.array(met_range if met_range is not None else self.iso_mets)
        imf_range = imf_range if imf_range is not None else [{}]

        # Validate ranges
        if (jnp.min(age_range) < jnp.min(self.ages/1000.)) or (jnp.max(age_range) > jnp.max(self.ages/1000.)):
            raise ValueError("Age range outside isochrone limits.")
        if (jnp.min(met_range) < jnp.min(self.mets)) or (jnp.max(met_range) > jnp.max(self.mets)):
            raise ValueError("Metallicity range outside isochrone limits.")

        # Format helpers for the output file name
        def _format_range(arr):
            """Return formatted string like '0.1-13.0' from an array."""
            return f"{np.min(arr):.2f}-{np.max(arr):.2f}"

        def _format_imf_range(imf_range):
            """Create a descriptive string for the IMF range."""
            if len(imf_range) == 1 and imf_range[0] == {}:
                # Use the name of the function as a fallback label
                imf_func_name = getattr(self.imf_function, "__name__", "imf")
                return imf_func_name.lower()
            # Otherwise, build a param-based label
            keys = sorted(imf_range[0].keys())
            key_strs = []
            for key in keys:
                vals = [params.get(key, None) for params in imf_range]
                key_str = f"{key}{min(vals):.1f}-{max(vals):.1f}"
                key_strs.append(key_str)
            return "_".join(key_strs)
    
        # Create filename from parameter ranges
        age_str = _format_range(age_range)
        met_str = _format_range(met_range)
        imf_str = _format_imf_range(imf_range)
        fname = f"sspgrid_age{age_str}_met{met_str}_imf{imf_str}"+self.rlabel+user_label+".hdf5"
        cache_path = os.path.join(cache_dir, fname)

        # Load if exists
        if os.path.exists(cache_path):
            print(f"Loading SSP grid from {cache_path}")
            with h5py.File(cache_path, "r") as f:
                wave = f["wavelength"][()]
                spec_grid = f["spectra"][()]
            return wave, spec_grid

        print(f"Calculating SSP predictions")
        # Compute SSP grid
        os.makedirs(cache_dir, exist_ok=True)
        na, nm, ni, nw = len(age_range), len(met_range), len(imf_range), len(self.wave)
        spec_grid = jnp.zeros((na, nm, ni, nw))

        for ia, age in enumerate(age_range):
            for im, met in enumerate(met_range):
                for ii, imf_params in enumerate(imf_range):
                    _, spec = self.synthesize(age, met, imf_params)
                    spec_grid = spec_grid.at[ia, im, ii, :].set(spec)

        # Save
        with h5py.File(cache_path, "w") as f:
            f.create_dataset("wavelength", data=np.array(self.wave))
            f.create_dataset("spectra", data=np.array(spec_grid))

        return self.wave, spec_grid
