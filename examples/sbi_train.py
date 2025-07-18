#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=duplicate-code
# *** Duplicate code could be allowed for examples ***

import random

# import h5py
# import jax
import jax.random as jr
import numpy as np
from astroquery.svo_fps import SvoFps
# from tqdm import tqdm

from fastar.imf.named_imf.single_power_law import single_powerlaw as unimodal
from fastar.semi_resolved_ssp import SemiResolvedSspSynthesizer
# from fastar.tools.utils import compute_ab_magnitudes


def main():
    # --------------------------
    # Configuration
    # --------------------------
    log_num_stars_min = 1.0
    log_num_stars_max = 7.0
    num_stellar_bins = 20

    out_prefix = 'sbi_model'
    rng = np.random.default_rng(seed=42)

    # --------------------------
    # Instantiate synthesizer
    # --------------------------
    semi_synth = SemiResolvedSspSynthesizer(
        imf_function=unimodal, model_label='phot'
    )
    wave, spec, m_stars = semi_synth.synthesize_large(
        age=10,
        met=0,
        num_stars=100,
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
    num_samples = 50000
    age_bounds = (0.1, 14)
    met_bounds = (-2.0, 0.5)
    num_stars_choices = (
        (
            10
            ** np.linspace(
                log_num_stars_min, log_num_stars_max, num_stellar_bins
            )
        )
        .round()
        .astype(int)
    )  # quantized num_stars

    # Print results
    print(f'{out_prefix=}')
    print(f'{rng=}')
    print(f'{spec=}')
    print(f'{m_stars=}')
    print(f'{gtrans=}')
    print(f'{rtrans=}')
    print(f'{itrans=}')
    print(f'{ztrans=}')
    print(f'{theta_list=}')
    print(f'{spec_list=}')
    print(f'{num_samples=}')
    print(f'{age_bounds=}')
    print(f'{met_bounds=}')
    print(f'{num_stars_choices=}')

    # # Random keys for JAX
    # master_key = jax.random.PRNGKey(42)
    # keys = jax.random.split(master_key, num_samples)

    # # Simulation loop
    # for k in range(num_stellar_bins):

    #     num_stars = int(num_stars_choices[k])
    #     for i in tqdm(range(num_samples), desc="Simulating spectra"):

    #         # Sample parameters
    #         age = np.random.uniform(*age_bounds)
    #         met = np.random.uniform(*met_bounds)
    #         imf_slope = 2.3

    #         key = keys[i]

    #         # Generate spectrum
    #         wave, spec, m_stars = semi_synth.synthesize_large(
    #             age=age, met=met, num_stars=num_stars,
    #             imf_params={"alpha": imf_slope}, key=key)

    #         # Measure magnitudes
    #         gband = compute_ab_magnitudes(
    #               wave=wave, spectra=spec, fresp=gtrans
    #         )
    #         rband = compute_ab_magnitudes(
    #               wave=wave, spectra=spec, fresp=rtrans
    #         )
    #         iband = compute_ab_magnitudes(
    #               wave=wave, spectra=spec, fresp=itrans
    #         )
    #         zband = compute_ab_magnitudes(
    #               wave=wave, spectra=spec, fresp=ztrans
    #         )

    #         # spec /= np.mean(spec)  # Normalize

    #         # Only include age and met
    #         theta_list.append(
    #               [age, met, np.log10(num_stars), np.log10(m_stars)]
    #         )
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


if __name__ == '__main__':
    main()
