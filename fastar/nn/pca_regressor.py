#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from flax import linen as nn


# =============================================================================
# PCA-based Neural Network Model Definition
# =============================================================================
class PCARegressor(nn.Module):
    """
    Simple feed-forward neural network for predicting PCA coefficients
    from stellar parameters.

    Notes
    -----
    .. todo::
      Fix the warning related with the Sphinx when the attributes are added to
      the class PCARegressor.
    """

    # Attributes
    # ----------
    # output_dim : int
    #     Number of PCA components to output.
    # activation_type : str
    #     Type of activation function ('relu', 'tanh', 'gelu').

    output_dim: int = 16
    activation_type: str = 'gelu'

    # *** Review the following method, although it could be probably OK ***
    # W0221: Variadics removed in overridden 'PCARegressor.__call__' method
    @nn.compact
    def __call__(self, x):  # pylint: disable=arguments-differ
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
