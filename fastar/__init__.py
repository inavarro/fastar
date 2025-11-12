#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ._version import *  # noqa: F403
from .synthesis.integrated_ssp import IntegratedSynthesizer
from .synthesis.semiresolved_ssp import SemiresolvedSynthesizer
from .synthesis.stellar_predictions import StellarSynthesizer

__all__ = [
    'IntegratedSynthesizer',
    'SemiresolvedSynthesizer',
    'StellarSynthesizer',
]
