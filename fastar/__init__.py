#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ._version import *  # noqa: F403
from .synthesis.ssp.integrated import IntegratedSynthesizer
from .synthesis.ssp.semiresolved import SemiresolvedSynthesizer
from .synthesis.stellar_predictions import StellarSynthesizer

__all__ = [
    'IntegratedSynthesizer',
    'SemiresolvedSynthesizer',
    'StellarSynthesizer',
]
