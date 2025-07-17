#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import h5py
import flax.serialization as flax_ser
import jax
import jax.numpy as jnp
from astropy.io.ascii import read as ascii_read
from flax import linen as nn
from jax.scipy.integrate import trapezoid

from fastar.imf.named_imf.single_power_law import single_powerlaw as unimodal
from fastar.interpolate.color import color_interpolation
from fastar.interpolate.isochrone import isochrone_interpolation
from fastar.tools.assets import get_asset_path


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
# Semi-Resolved Population Synthesizer Class
# =============================================================================
class SemiResolvedSynthesizer:
    """
    Class for generating synthetic semi-resolved stellar populations
    spectroscopic and photometric predictions with a PCA-based stellar spectral
    model and a stochastic IMF sampling.
    """

    def __init__(self, model_label=None, imf_function=None):
        self.npc = 16
        self.activation_type = 'gelu'

        if model_label is None:
            self.rlabel = '_spec'

            with h5py.File(get_asset_path('sun_ref.hdf5'), 'r') as sun:
                self.sun_spec = sun['sun_spec'][:]

        if model_label == 'phot':
            self.rlabel = '_phot'

            with h5py.File(get_asset_path('sun_ref.hdf5'), 'r') as sun:
                self.sun_spec = sun['sun_phot'][:]

        self.imf_function = (
            imf_function if imf_function is not None else unimodal
        )

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
            get_asset_path(f'pca_regressor{self.rlabel}.flax'), 'rb'
        ) as f:
            self.params = flax_ser.from_bytes(
                model.init(jax.random.PRNGKey(0), jnp.ones((1, 3))), f.read()
            )
        self.model = model

        with h5py.File(
            get_asset_path(f'training_artifacts{self.rlabel}.h5'), 'r'
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
            get_asset_path('BASTI-IAC_isochrones.hdf5'), 'r'
        ) as iso:
            self.mets = iso['mets'][:]
            self.ages = iso['ages'][:]
            self.mass_ini_data = iso['mass_ini'][:]
            self.teff_out_data = iso['teff_out'][:]
            self.logg_out_data = iso['logg_out'][:]
            self.lumi_out_data = iso['lumi_out'][:]

        tab = ascii_read(get_asset_path('filters_default.res'))
        fwave = tab['col1']
        fresp = tab['col2']
        self.filter_response = jnp.interp(
            self.wave, fwave, fresp, left=0, right=0
        )

        with h5py.File(get_asset_path('WORTHEY11_colors.hdf5'), 'r') as color:
            self.bcv_grid = color['bcv'][:]
            self.fmet_array = color['ufmet'][:]
            self.logg_array = color['ulogg'][:]
            self.teff_log10_array = color['uteff'][:]

        with h5py.File(get_asset_path('pop_iso.hdf5'), 'r') as color:
            self.iso_ages = color['grid_ages'][:]
            self.iso_mets = color['grid_ages'][:]

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
            jnp.dot(pca_coeffs, self.pca_components) + self.pca_mean
        ) + self.mean_spectrum

        return self._softplus(spectra)

    @partial(jax.jit, static_argnames=['self', 'Nstars'])
    def synthesize(self, age, met, imf_params, Nstars, key, out_masses=False):
        """
        Generate synthetic semi-resolved population spectrum for a given
        age and metallicity.

        Parameters
        ----------
        age : float
            Stellar population age (in Gyr).
        met : float
            Metallicity [M/H].
        imf_params : dict
            Parameters for the initial mass function.
        Nstars : int
            Number of stars to sample.
        key : PRNGKey
            Random key for JAX sampling.
        out_masses : bool, optional
            If True, return array of sampled stellar masses instead of the
            total mass. Default is False.

        Returns
        -------
        tuple
            (wavelengths, spectrum, total stellar mass) or (wavelengths,
            spectrum, sampled stellar masses) depending on `out_masses`.
        """
        # Interpolate the isochrones at the desired age and metallicity
        imass, iteff, ilogg, ilum = self._get_isochrone(age, met)

        # Stochastically sample the IMF
        sampled_masses = self._stochastic_IMF_sampling(
            imass, imf_params, Nstars, key
        )

        # Evaluate the isochrone at the interpolated masses
        iteff_interp = jnp.interp(sampled_masses, imass, iteff)
        ilogg_interp = jnp.interp(sampled_masses, imass, ilogg)
        ilum_interp = jnp.interp(sampled_masses, imass, ilum)

        # Calculate the stellar spectra
        spectra = self._predict_spectrum(
            ilogg_interp, iteff_interp, jnp.full_like(iteff_interp, met)
        )

        # Get the V-band magnitudes of the predicted stellar spectra (they are
        # normalized to have a mean flux of 1)
        magnitudes = self._compute_ab_magnitudes(spectra)

        # Calculate the bolometric corrections
        bcv_val = color_interpolation(
            ilogg_interp,
            iteff_interp,
            jnp.full_like(iteff_interp, met),
            self.logg_array,
            self.teff_log10_array,
            self.fmet_array,
            self.bcv_grid,
        )

        # Scale the predicted stellar spectra so they math their theoretical
        # luminosities
        vmags = -2.5 * ilum_interp - bcv_val
        mtarg = vmags + sun_vmag
        corr = 1 / (10 ** ((magnitudes - mtarg) / -2.5))

        # Add the flux of all the spectra. There is no IMF weighting here
        # since it naturally comes from the stochastic sampling
        spec = jnp.sum(spectra * corr[:, None], axis=0)

        # The function returns wavelength, spectrum and either the total
        # stellar mass of the population or the sampled
        if out_masses:
            result = (self.wave, spec, sampled_masses)
        else:
            result = (self.wave, spec, jnp.sum(sampled_masses))

        return result

    def synthesize_large(
        self, age, met, imf_params, Nstars, key, batch_size=10000
    ):
        """
        Generate synthetic population spectrum using chunked IMF sampling
        for large numbers of stars. This allows calculating predictions
        for a large number of stars without using too much memory
        """
        n_batches = int(Nstars) // int(batch_size)
        remainder = int(Nstars) % int(batch_size)
        n_total_chunks = n_batches + int(remainder > 0)

        keys = jax.random.split(key, n_total_chunks)

        # First batch
        wave, spec_total, mass_total = self.synthesize(
            age, met, imf_params, batch_size, keys[0]
        )

        # Loop over full-size batches
        for i in range(1, n_batches):
            _, spec_chunk, mass_chunk = self.synthesize(
                age, met, imf_params, batch_size, keys[i]
            )
            spec_total += spec_chunk
            mass_total += mass_chunk

        # Final chunk (remainder)
        if remainder > 0:
            _, spec_chunk, mass_chunk = self.synthesize(
                age, met, imf_params, remainder, keys[-1]
            )
            spec_total += spec_chunk
            mass_total += mass_chunk

        return wave, spec_total, mass_total

    @partial(jax.jit, static_argnames=['self', 'Nstars'])
    def _stochastic_IMF_sampling(self, imass, imf_params, Nstars, key):
        """
        Stochastically sample stellar masses from an IMF assuming
        it emerges from a probability distribution function.
        """

        # Normalize the IMF to a total number of stars equal to 1
        # Note this normalization is different than the one used
        # for the population synthesis as there the normalizing
        # quantity is the total mass (not the number of stars)
        mass_grid = jnp.linspace(imass.min(), imass.max(), 5000)
        imf_vals = self.imf_function(mass_grid, imf_params)

        # IMF to PDF (via normalization)
        pdf = imf_vals / trapezoid(imf_vals, x=mass_grid)
        cdf = jnp.cumsum(pdf)
        cdf = cdf / cdf[-1]

        # Uniform sampling f the CDF
        uniform_samples = jax.random.uniform(key, shape=(int(Nstars),))

        return jnp.interp(uniform_samples, cdf, mass_grid)

    @partial(jax.jit, static_argnames=['self'])
    def _compute_ab_magnitudes(self, spectra):
        """
        Compute V-band AB magnitudes of a given spectrum (in Flambda)
        """
        denominator = trapezoid(self.filter_response / self.wave, x=self.wave)
        numerators = trapezoid(
            spectra * self.filter_response * self.wave, axis=1, x=self.wave
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
