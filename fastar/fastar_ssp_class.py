#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import h5py
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.serialization as flax_ser
import random

from functools import partial
from astropy.io import ascii
from flax import linen as nn
from jax.scipy.integrate import trapezoid

from fastar.path_utils import get_data_path
from fastar.fastar_imf import single_powerlaw as unimodal
from fastar.fastar_interpolate_isochrones import isochrone_interpolation
from fastar.fastar_interpolate_colors import color_interpolation

# =============================================================================
# Solar constants
# =============================================================================
sun_mbol = 4.70
sun_bvc = -0.12
sun_vmag = sun_mbol - sun_bvc


# =============================================================================
# PCA-based Neural Network Model Definition
# =============================================================================
class PCARegressor(nn.Module):
    """
    Simple feed-forward neural network for predicting PCA coefficients
    from stellar parameters.

    Attributes
    ----------
    output_dim : int
        Number of PCA components to output.
    activation_type : str
        Type of activation function ('relu', 'tanh', 'gelu').
    """

    output_dim: int = 16
    activation_type: str = 'gelu'

    @nn.compact
    def __call__(self, x):
        act = {'relu': nn.relu, 'tanh': nn.tanh, 'gelu': nn.gelu}[
            self.activation_type
        ]
        x = nn.Dense(64)(x)
        x = act(x)
        x = nn.Dense(128)(x)
        x = act(x)
        x = nn.Dense(128)(x)
        x = act(x)
        x = nn.Dense(64)(x)
        x = act(x)
        x = nn.Dense(self.output_dim)(x)
        return x


# =============================================================================
# SSP Population Synthesizer Class
# =============================================================================
class PopulationSynthesizer:
    """
    Class for generating synthetic integrated SSP spectroscopic and photometric
    predictions with a PCA-based stellar spectral model.
    """

    def __init__(self, model_label=None, imf_function=None):
        # Set model parameters and labels
        self.npc = 16
        self.activation_type = 'gelu'

        if model_label is None:
            self.rlabel = f'_spec'

            with h5py.File(
                get_data_path('sun_ref.hdf5', subdir='aux'), 'r'
            ) as sun:
                self.sun_spec = sun['sun_spec'][:]

        if model_label == 'phot':
            self.rlabel = f'_phot'

            with h5py.File(
                get_data_path('sun_ref.hdf5', subdir='aux'), 'r'
            ) as sun:
                self.sun_spec = sun['sun_phot'][:]

        self.imf_function = (
            imf_function if imf_function is not None else unimodal
        )

        # Load model weights and auxiliary data
        self._load_model()
        self._load_auxiliary_data()

    def _load_model(self):
        """
        Load trained PCA regressor, scalers, and PCA components.
        """
        model = PCARegressor(
            output_dim=self.npc, activation_type=self.activation_type
        )

        with open(
            get_data_path(f'pca_regressor{self.rlabel}.flax', subdir='aux'),
            'rb',
        ) as f:
            self.params = flax_ser.from_bytes(
                model.init(jax.random.PRNGKey(0), jnp.ones((1, 3))), f.read()
            )
        self.model = model

        with h5py.File(
            get_data_path(f'training_artifacts{self.rlabel}.h5', subdir='aux'),
            'r',
        ) as f:
            self.scaler_X_mean = f['scaler_X/mean_'][:]
            self.scaler_X_scale = f['scaler_X/scale_'][:]
            self.scaler_Y_mean = f['scaler_Y/mean_'][:]
            self.scaler_Y_scale = f['scaler_Y/scale_'][:]
            self.pca_components = f['pca/components_'][:]
            self.pca_mean = f['pca/mean_'][:]
            self.mean_spectrum = f['mean_spectrum'][:]
            self.wave = f['wave'][:]

    def _load_auxiliary_data(self):
        """
        Load isochrones, V-band filter response, bolometric corrections,
        and optimized age and metallicity samplings.
        """

        with h5py.File(
            get_data_path('BASTI-IAC_isochrones.hdf5', subdir='isochrones'),
            'r',
        ) as iso:
            self.mets = iso['mets'][:]
            self.ages = iso['ages'][:]
            self.mass_ini_data = iso['mass_ini'][:]
            self.teff_out_data = iso['teff_out'][:]
            self.logg_out_data = iso['logg_out'][:]
            self.lumi_out_data = iso['lumi_out'][:]

        tab = ascii.read(get_data_path('filters_default.res', subdir='aux'))
        fwave = tab['col1']
        fresp = tab['col2']
        self.filter_response = jnp.interp(
            self.wave, fwave, fresp, left=0, right=0
        )

        with h5py.File(
            get_data_path('WORTHEY11_colors.hdf5', subdir='aux'), 'r'
        ) as color:
            self.bcv_grid = color['bcv'][:]
            self.fmet_array = color['ufmet'][:]
            self.logg_array = color['ulogg'][:]
            self.teff_log10_array = color['uteff'][:]

        with h5py.File(
            get_data_path('pop_iso.hdf5', subdir='aux'), 'r'
        ) as color:
            self.iso_ages = color['grid_ages'][:]
            self.iso_mets = color['grid_mets'][:]

    @partial(jax.jit, static_argnames=['self'])
    def _predict_spectrum(self, logg, teff, fmet):
        """
        Predict stellar spectra given logg, Teff, and [Fe/H] using the
        PCA regressor.
        """
        inputs = jnp.stack([logg, teff, fmet], axis=-1)
        input_scaled = (inputs - self.scaler_X_mean) / self.scaler_X_scale
        pca_scaled = self.model.apply(self.params, input_scaled)
        pca_coeffs = pca_scaled * self.scaler_Y_scale + self.scaler_Y_mean
        spectra = (
            jnp.dot(pca_coeffs, self.pca_components)
            + self.pca_mean
            + self.mean_spectrum
        )
        return self._softplus(spectra)

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
        mtarg = vmags + sun_vmag
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
            mtarg = vmags + sun_vmag
            corr = 1 / jnp.power(10.0, (magnitudes - mtarg) / -2.5)

            # Integrate corrected spectra over IMF-weighted stars
            spec = self._population_synthesis_integrate(
                spectra, corr, imf_val, imass
            )
            all_specs.append(spec)  # Collect each perturbed spectrum

        ssp_std = jnp.std(jnp.stack(all_specs), axis=0)

        return self.wave, ssp_std

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
        imass, _, _, _ = isochrone_interpolation(
            age,
            met,
            self.ages,
            self.mets,
            self.mass_ini_data,
            self.teff_out_data,
            self.logg_out_data,
            self.lumi_out_data,
        )

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
        L = 10 ** (-0.4 * (ab_mag - m_sun))

        return {
            'ml_stars': stellar_mass / L,  # M*/L
            'ml_total': 1.0 / L,  # M_total/L
        }

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
            filter_response
            if filter_response is not None
            else self.filter_response
        )

        # Compute AB magnitudes from synthetic spectra
        denominator = trapezoid(response / self.wave, x=self.wave)
        numerators = trapezoid(
            spectra * response * self.wave, x=self.wave, axis=1
        )
        flux_density = numerators / denominator
        return -2.5 * jnp.log10(flux_density) - 2.406

    @partial(jax.jit, static_argnames=['self'])
    def _softplus(self, x, beta=100.0):
        """
        Smooth activation function with soft floor to prevent
        any negative flux in the spectrum of extreme stars
        """
        return (1.0 / beta) * jnp.logaddexp(0.0, beta * x)

    @partial(jax.jit, static_argnames=['self'])
    def _population_synthesis_integrate(self, spectra, corr, imf_val, imass):
        """
        Integrate IMF-weighted, corrected spectra over initial mass grid.
        """
        weights = corr * imf_val
        integrand = spectra * weights[:, None]
        return trapezoid(integrand, x=imass, axis=0)
