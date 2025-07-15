#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time

import numpy as np
from astropy.io.ascii import read as ascii_read
from astroquery.svo_fps import SvoFps

from fastar.fastar_imf import single_powerlaw as unimodal
from fastar.fastar_ssp_class import PopulationSynthesizer
from fastar.fastar_utils import compute_ab_magnitudes, compute_linetrength


# ---------------------------------------------------
# Spectroscopic predictions
# ---------------------------------------------------
# Load the synthesis code
ssp_spec = PopulationSynthesizer(imf_function=unimodal)

# Let's focus now on a single SSP
age = 0.1
met = 0.0
imf_slope = 2.3

# Standard SSP call
wave, spec = ssp_spec.synthesize(age=age, met=met)

ages = ssp_spec.iso_ages()
mets = ssp_spec.iso_mets()

# Get the remaining stellar mass of the SSP (if 1 Msun formed), aka, the
# fraction of mass in stars compared to the initial stellar mass of the
# population
fmass = ssp_spec.stellar_mass(
    age=age, met=met, imf_params={'alpha': imf_slope}
)

# Get the M/L ratio in any given filter (default = V-band)
# Two ML are returned, one is the stellar ML and the other is the total
# (stars+ejected gas+remnants)
data = SvoFps.get_transmission_data('SLOAN/SDSS.g')
gtrans = np.interp(
    wave, data['Wavelength'], data['Transmission'], left=0, right=0
)

ml_g = ssp_spec.mass_to_light_ratio(
    age=age, met=met, imf_params={'alpha': imf_slope}, filter_response=gtrans
)

# Magnitude and color predictions
data = SvoFps.get_transmission_data('SLOAN/SDSS.r')
rtrans = np.interp(
    wave, data['Wavelength'], data['Transmission'], left=0, right=0
)

gmag = compute_ab_magnitudes(wave, spec, gtrans)
rmag = compute_ab_magnitudes(wave, spec, rtrans)
gr_color = gmag - rmag

# Line-strengths
index_tab = ascii_read('../../aux/indices.def')
hbeta = compute_linetrength(wave, spec, 'Hbeta_o', index_tab)

# Finally, let's get uncertainties for the SSP prediction
wave, error = ssp_spec.synthesize_nsim(
    age=age, met=met, imf_params={'alpha': imf_slope}
)

# Show of the speed
t0 = time.time()
for _ in range(50):
    age = np.random.uniform(low=1, high=13.5)
    met = np.random.uniform(low=-2, high=0.5)
    imf_slope = np.random.uniform(low=1.3, high=2.5)
    wave, spec = ssp_spec.synthesize(
        age=age, met=met, imf_params={'alpha': imf_slope}
    )
print(
    'It takes ' + str(np.round(time.time() - t0, 2)) + 's to calculate 50 SSPs'
)


# ---------------------------------------------------
# Photometric predictions (based on the XSL library)
# **SHOULD NOT BE USED FOR SPECTROSCOPIC MEASUREMENTS**
# ---------------------------------------------------
# Load the synthesis code
ssp_phot = PopulationSynthesizer(model_label='phot', imf_function=unimodal)

# Let's focus now on a single SSP
age = 0.1
met = 0.0

# Standard SSP call
phot_wave, phot_spec = ssp_phot.synthesize(age=age, met=met)

# Get predictions out of the MILES range
data = SvoFps.get_transmission_data('SLOAN/SDSS.i')
itrans = np.interp(
    phot_wave, data['Wavelength'], data['Transmission'], left=0, right=0
)
imag = compute_ab_magnitudes(phot_wave, phot_spec, itrans)

ml_i = ssp_phot.mass_to_light_ratio(
    age=age, met=met, imf_params={'alpha': imf_slope}, filter_response=itrans
)
