#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random

import jax.random as jr
import numpy as np
from astroquery.svo_fps import SvoFps

from fastar.imf.named_imf.single_power_law import single_powerlaw as unimodal
from fastar.semi import SemiResolvedSynthesizer
from fastar.tools.utils import compute_ab_magnitudes


# ---------------------------------------------------
# Spectroscopic predictions
# ---------------------------------------------------
# Load the synthesis code
semi_spec = SemiResolvedSynthesizer(imf_function=unimodal, model_label='phot')

# Let's focus now on a single SSP
age = 10
met = 0.0
imf_slope = 2.3

nstars = 1e3
key = jr.PRNGKey(random.randint(0, 2**32 - 1))

# Standard SSP call
wave, spec, mstar = semi_spec.synthesize(
    age=age, met=met, imf_params={'alpha': imf_slope}, key=key, Nstars=nstars
)

# # Magnitude and color predictions
data = SvoFps.get_transmission_data('SLOAN/SDSS.g')
gtrans = np.interp(
    wave, data['Wavelength'], data['Transmission'], left=0, right=0
)

data = SvoFps.get_transmission_data('SLOAN/SDSS.r')
rtrans = np.interp(
    wave, data['Wavelength'], data['Transmission'], left=0, right=0
)

gmag = compute_ab_magnitudes(wave, spec, gtrans)
rmag = compute_ab_magnitudes(wave, spec, rtrans)
gr_color = gmag - rmag
