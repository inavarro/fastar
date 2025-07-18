#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import jax

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
        return self._predict_spectrum(logg, teff, fmet)
