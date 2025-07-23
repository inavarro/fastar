#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# pylint: disable=duplicate-code
# *** Duplicate code could be allowed for examples ***

import random

import jax.random as jr
import numpy as np
from astroquery.svo_fps import SvoFps

from fastar.imf.named_imf.single_power_law import single_powerlaw as unimodal
from fastar.semiresolved_ssp import SemiresolvedSynthesizer
from fastar.tools.utils import compute_ab_magnitudes


def main():
    # ---------------------------------------------------
    # Spectroscopic predictions
    # ---------------------------------------------------
    # Load the synthesis code
    semi_spec = SemiresolvedSynthesizer(
        imf_function=unimodal, model_label='phot'
    )

    # Let's focus now on a single SSP
    age = 10
    met = 0.0
    imf_slope = 2.3

    num_stars = 1e3
    key = jr.PRNGKey(random.randint(0, 2**32 - 1))

    # Standard SSP call
    wave, spec, mstar = semi_spec.synthesize(
        age=age,
        met=met,
        imf_params={'alpha': imf_slope},
        key=key,
        num_stars=num_stars,
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

    # Print results
    print(f'{mstar=}')
    print(f'{gr_color=}')


if __name__ == '__main__':
    main()
