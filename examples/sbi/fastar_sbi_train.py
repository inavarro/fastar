#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# from tqdm import tqdm
# import h5py
# import jax
import jax.random as jr
import random

from astroquery.svo_fps import SvoFps


import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastar_imf import single_powerlaw as unimodal
from fastar_semi_class import SemiResolvedSynthesizer
# from fastar_utils import compute_ab_magnitudes


# --------------------------
# Configuration
# --------------------------
log_Nstars_min = 1.0
log_Nstars_max = 7.0
Nstellar_bins = 20

out_prefix = 'sbi_model'
rng = np.random.default_rng(seed=42)

# --------------------------
# Instantiate synthesizer
# --------------------------
semi_synth = SemiResolvedSynthesizer(imf_function=unimodal, model_label='phot')
wave, spec, Mstars = semi_synth.synthesize_large(
    age=10,
    met=0,
    Nstars=100,
    imf_params={'alpha': 2.3},
    key=jr.PRNGKey(random.randint(0, 2**32 - 1)),
)

# --------------------------
# Get HiPERCAM filters
# --------------------------
data = SvoFps.get_transmission_data('GTC/HIPERCAM.g')
gtrans = np.interp(
    wave, data['Wavelength'], data['Transmission'], left=0, right=0
)

data = SvoFps.get_transmission_data('GTC/HIPERCAM.r')
rtrans = np.interp(
    wave, data['Wavelength'], data['Transmission'], left=0, right=0
)

data = SvoFps.get_transmission_data('GTC/HIPERCAM.i')
itrans = np.interp(
    wave, data['Wavelength'], data['Transmission'], left=0, right=0
)

data = SvoFps.get_transmission_data('GTC/HIPERCAM.z')
ztrans = np.interp(
    wave, data['Wavelength'], data['Transmission'], left=0, right=0
)

# --------------------------
# Generate training set
# --------------------------
theta_list = []
spec_list = []

# Config
Nsamples = 50000
age_bounds = (0.1, 14)
met_bounds = (-2.0, 0.5)
Nstars_choices = (
    (10 ** np.linspace(log_Nstars_min, log_Nstars_max, Nstellar_bins))
    .round()
    .astype(int)
)  # quantized Nstars

# # Random keys for JAX
# master_key = jax.random.PRNGKey(42)
# keys = jax.random.split(master_key, Nsamples)

# # Simulation loop
# for k in range(Nstellar_bins):

#     Nstars = int(Nstars_choices[k])
#     for i in tqdm(range(Nsamples), desc="Simulating spectra"):

#         # Sample parameters
#         age = np.random.uniform(*age_bounds)
#         met = np.random.uniform(*met_bounds)
#         imf_slope = 2.3

#         key = keys[i]

#         # Generate spectrum
#         wave, spec, Mstars = semi_synth.synthesize_large(
#             age=age, met=met, Nstars=Nstars, imf_params={"alpha": imf_slope},
#             key=key)

#         # Measure magnitudes
#         gband = compute_ab_magnitudes(wave=wave, spectra=spec, fresp=gtrans)
#         rband = compute_ab_magnitudes(wave=wave, spectra=spec, fresp=rtrans)
#         iband = compute_ab_magnitudes(wave=wave, spectra=spec, fresp=itrans)
#         zband = compute_ab_magnitudes(wave=wave, spectra=spec, fresp=ztrans)

#         # spec /= np.mean(spec)  # Normalize

#         # Only include age and met
#         theta_list.append([age, met, np.log10(Nstars), np.log10(Mstars)])
#         spec_list.append(np.array([gband,rband,iband,zband]))

# theta_array = np.array(theta_list)
# spec_array = np.array(spec_list)

# f = h5py.File("semiphot_N1_N7.hdf5", "w")
# f.create_dataset(
#     'spectra', data=spec_array.astype(np.float32), compression="gzip",
#     compression_opts=9)
# f.create_dataset(
#     'param', data=theta_array.astype(np.float32), compression="gzip",
#     compression_opts=9)
# f.close()
