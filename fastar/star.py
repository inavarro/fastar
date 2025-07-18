#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import jax
import jax.numpy as jnp

from fastar.core.base import BaseSynthesizer


class StellarSynthesizer(BaseSynthesizer):
    """
    Class for generating stellar SED predictions with a PCA-based model
    """

    @partial(jax.jit, static_argnames=['self'])
    def stellar_spectrum(self, logg, teff, fmet):
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
