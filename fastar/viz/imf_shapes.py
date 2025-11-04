#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from fastar.imf.named_imf.broken_power_law import broken_power_law
from fastar.imf.named_imf.chabrier import chabrier
from fastar.imf.named_imf.flexi import flexi
from fastar.imf.named_imf.kroupa import kroupa
from fastar.imf.named_imf.single_power_law import single_power_law


def main():
    mass_arr = np.linspace(0.1, 5)

    fig, axis = plt.subplots()

    # Set log-log scale
    axis.set_xscale('log')
    axis.set_yscale('log')

    # Axis labels
    axis.set_xlabel(r'm [M$_\odot$]')
    axis.set_ylabel(r'$\chi (m) \\, \\left( \\frac{dN}{dm} \\right)$')

    # Custom x-ticks (e.g., linear-style values on log axis)
    custom_ticks = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5]
    axis.set_xticks(custom_ticks)
    axis.set_xticklabels([str(t) for t in custom_ticks])

    # Enable grid only at those ticks
    for tick in custom_ticks:
        axis.axvline(
            tick, color='gray', linestyle='--', linewidth=0.5, zorder=0
        )

    # Let's plot some IMFs
    imf = single_power_law(mass_arr, {})
    axis.plot(mass_arr, imf, label='Single power law', linewidth=1.5)

    imf = broken_power_law(mass_arr, {})
    axis.plot(mass_arr, imf, label='Broken power law', linewidth=1.5)

    imf = flexi(mass_arr, {'beta': 2})
    axis.plot(mass_arr, imf, label='Flexi IMF', linewidth=1.5)

    imf = chabrier(mass_arr, {})
    axis.plot(mass_arr, imf, label='Chabrier IMF', linewidth=1.5)

    imf = kroupa(mass_arr, {})
    axis.plot(mass_arr, imf, label='Kroupa IMF', linewidth=1.5)

    axis.legend()

    fig.tight_layout()
    fig.savefig('imf_shapes.pdf')
    plt.close('all')
