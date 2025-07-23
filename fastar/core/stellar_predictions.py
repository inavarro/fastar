#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import h5py
import flax.serialization as flax_ser
import jax
import jax.numpy as jnp

from fastar.nn.pca_regressor import PCARegressor
from fastar.tools.assets import get_asset_path


class StellarSynthesizer:
    """
    Base class. Synthesizer of stellar spectra.
    """

    def __init__(self, model_label=None):
        self.npc = 16
        self.activation_type = 'gelu'

        if model_label is None:
            self.rlabel = '_spec'

        if model_label == 'phot':
            self.rlabel = '_phot'

        self._load_model()

    def _load_model(self):
        """
        Load trained PCA regressor, scalers, and PCA components.
        """
        model = PCARegressor(
            output_dim=self.npc, activation_type=self.activation_type
        )

        with open(
            get_asset_path(f'pca_regressor{self.rlabel}.flax'),
            'rb',
        ) as pca_regressor_file:
            self.params = flax_ser.from_bytes(
                model.init(jax.random.PRNGKey(0), jnp.ones((1, 3))),
                pca_regressor_file.read(),
            )
        self.model = model

        with h5py.File(
            get_asset_path(f'training_artifacts{self.rlabel}.h5'), 'r'
        ) as training_artifacts_file:
            self.scaler_x_mean = training_artifacts_file['scaler_X/mean_'][:]
            self.scaler_x_scale = training_artifacts_file['scaler_X/scale_'][:]
            self.scaler_y_mean = training_artifacts_file['scaler_Y/mean_'][:]
            self.scaler_y_scale = training_artifacts_file['scaler_Y/scale_'][:]
            self.pca_components = training_artifacts_file['pca/components_'][:]
            self.pca_mean = training_artifacts_file['pca/mean_'][:]
            self.mean_spectrum = training_artifacts_file['mean_spectrum'][:]
            self.wave = training_artifacts_file['wave'][:]

    # *** Review the following method: it should be a function, not? ***
    @partial(jax.jit, static_argnames=['self'])
    def _softplus(self, input_flux, beta=100.0):  # pylint: disable=no-self-use
        """
        Smooth activation function with soft floor to prevent
        any negative flux in the spectrum of extreme stars
        """
        return (1.0 / beta) * jnp.logaddexp(0.0, beta * input_flux)

    @partial(jax.jit, static_argnames=['self'])
    def _predict_spectrum(self, logg, teff, fmet):
        """
        Predict stellar spectra given logg, Teff, and [Fe/H] using the
        PCA regressor.
        """
        inputs = jnp.stack([logg, teff, fmet], axis=-1)
        input_scaled = (inputs - self.scaler_x_mean) / self.scaler_x_scale
        pca_scaled = self.model.apply(self.params, input_scaled)
        pca_coeffs = pca_scaled * self.scaler_y_scale + self.scaler_y_mean
        spectra = (
            jnp.dot(pca_coeffs, self.pca_components) + self.pca_mean
        ) + self.mean_spectrum

        return self._softplus(spectra)
