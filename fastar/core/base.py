#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import h5py
import flax.serialization as flax_ser
import jax
import jax.numpy as jnp
from fastar.nn.pca_regressor import PCARegressor

from fastar.tools.assets import get_asset_path


class BaseSynthesizer:
    """
    Base class for synthesizers.
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

    @partial(jax.jit, static_argnames=['self'])
    def _softplus(self, x, beta=100.0):
        """
        Smooth activation function with soft floor to prevent
        any negative flux in the spectrum of extreme stars
        """
        return (1.0 / beta) * jnp.logaddexp(0.0, beta * x)
