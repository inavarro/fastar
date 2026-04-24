#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initial Mass Functions (IMFs) module.

This module provides various IMF parametrizations and a registry for
easy access. Users can:
- Import IMF functions directly: from fastar.imf import kroupa
- Use the registry: from fastar.imf import IMFRegistry
"""

from fastar.imf.named_imf.single_power_law import single_power_law
from fastar.imf.named_imf.broken_power_law import broken_power_law
from fastar.imf.named_imf.bimodal import bimodal
from fastar.imf.named_imf.kroupa import kroupa
from fastar.imf.named_imf.chabrier import chabrier
from fastar.imf.named_imf.flexi import flexi
from fastar.imf.registry import IMFRegistry

# Create a default registry instance for convenience
imf_registry = IMFRegistry()

# Define what gets exported when doing "from fastar.imf import *"
__all__ = [
    # IMF functions
    'single_power_law',
    'broken_power_law',
    'bimodal',
    'kroupa',
    'chabrier',
    'flexi',
    # Registry
    'IMFRegistry',
    'imf_registry',
]
