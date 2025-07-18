#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.integrate import trapezoid

from fastar.core.ssp import SspSynthesizer
from fastar.interpolate.color import color_interpolation


class IntegratedSspSynthesizer(SspSynthesizer):
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

        # Calculate the stellar spectra
        spectra = self._predict_spectrum(ilogg, iteff, imet)

        # Evaluate IMF value at the isochrone stellar masses
        imf_val = self.imf_function(imass, imf_params)

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

        return self.wave, spec

    @partial(jax.jit, static_argnames=['self'])
    def synthesize_nsim(
        self,
        age,
        met,
        imf_params=None,
        dmet=0.1,
        dteff=0.01,
        dlogg=0.1,
        nsim=10,
    ):
        """
        Estimate SSP spectral uncertainties via Monte Carlo perturbation
        of the stellar parameters.

        Returns standard deviation of the perturbed SSP spectrum.
        """

        key = jr.PRNGKey(random.randint(0, 2**32 - 1))

        # ensure we always pass a dict to IMF **params
        imf_params = imf_params or {}

        # Interpolate isochrone at given age and metallicity
        imass, iteff, ilogg, ilum = self._get_isochrone(age, met)
        imet = jnp.full_like(iteff, met)

        all_specs = []
        for _ in range(nsim):
            # Perturb isochrone parameters
            key, subkey1 = jr.split(key)
            ilogg_perturbed = (
                ilogg + jr.normal(subkey1, shape=ilogg.shape) * dlogg
            )
            key, subkey2 = jr.split(key)
            iteff_perturbed = (
                iteff + jr.normal(subkey2, shape=iteff.shape) * dteff
            )
            key, subkey3 = jr.split(key)
            imet_perturbed = imet + jr.normal(subkey3, shape=imet.shape) * dmet

            # Predict spectra for the isochrone points
            spectra = self._predict_spectrum(
                ilogg_perturbed, iteff_perturbed, imet_perturbed
            )

            # Evaluate IMF (can be overridden per call)
            imf_val = self.imf_function(imass, imf_params)

            # Apply bolometric corrections
            bcv_val = color_interpolation(
                ilogg_perturbed,
                iteff_perturbed,
                imet_perturbed,
                self.logg_array,
                self.teff_log10_array,
                self.fmet_array,
                self.bcv_grid,
            )

            # Compute AB magnitudes for each point
            magnitudes = self._compute_ab_magnitudes(spectra)

            vmags = -2.5 * ilum - bcv_val
            mtarg = vmags + self.sun_vmag
            corr = 1 / jnp.power(10.0, (magnitudes - mtarg) / -2.5)

            # Integrate corrected spectra over IMF-weighted stars
            spec = self._population_synthesis_integrate(
                spectra, corr, imf_val, imass
            )
            all_specs.append(spec)  # Collect each perturbed spectrum

        ssp_std = jnp.std(jnp.stack(all_specs), axis=0)

        return self.wave, ssp_std

    # *** Review the following method: it should be a function, not? ***
    @partial(jax.jit, static_argnames=['self'])
    def _population_synthesis_integrate(  # pylint: disable=no-self-use
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
