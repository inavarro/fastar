#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

# User can also import IMF functions directly: from fastar.imf import kroupa
# Here, we are showcasing the usage of the imf_registry
from fastar.imf import imf_registry


def main():
    mass_arr = np.linspace(0.1, 5)

    fig, axis = plt.subplots()

    # Set log-log scale
    axis.set_xscale('log')
    axis.set_yscale('log')

    # Axis labels
    axis.set_xlabel(r'm [M$_\odot$]')
    axis.set_ylabel(r'$\chi(m)\,\left(\frac{dN}{dm}\right)$')
    # Custom x-ticks (e.g., linear-style values on log axis)
    custom_ticks = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5]
    axis.set_xticks(custom_ticks)
    axis.set_xticklabels([str(t) for t in custom_ticks])

    # Enable grid only at those ticks
    for tick in custom_ticks:
        axis.axvline(
            tick, color='gray', linestyle='--', linewidth=0.5, zorder=0
        )

    print('Available IMFs: ', imf_registry.list_available())

    # Let's plot some IMFs
    single_power_law = imf_registry.load_by_name('single_power_law')
    imf = single_power_law(mass_arr, {})
    axis.plot(mass_arr, imf, label='Single power law', linewidth=1.5)

    broken_power_law = imf_registry.load_by_name('broken_power_law')
    imf = broken_power_law(mass_arr, {})
    axis.plot(mass_arr, imf, label='Broken power law', linewidth=1.5)

    flexi = imf_registry.load_by_name('flexi')
    imf = flexi(mass_arr, {'beta': 2})
    axis.plot(mass_arr, imf, label='Flexi IMF', linewidth=1.5)

    chabrier = imf_registry.load_by_name('chabrier')
    imf = chabrier(mass_arr, {})
    axis.plot(mass_arr, imf, label='Chabrier IMF', linewidth=1.5)

    kroupa = imf_registry.load_by_name('kroupa')
    imf = kroupa(mass_arr, {})
    axis.plot(mass_arr, imf, label='Kroupa IMF', linewidth=1.5)

    axis.legend()

    fig.tight_layout()
    fig.savefig('imf_shapes.pdf')
    plt.close('all')
