from matplotlib import pyplot as plt

import sys
import os
import numpy as np

# Add FaStar to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import fastar_imf

mass_arr = np.linspace(0.1,5)

fig, ax = plt.subplots()

# Set log-log scale
ax.set_xscale('log')
ax.set_yscale('log')

# Axis labels
ax.set_xlabel('m [M$_\odot$]')
ax.set_ylabel('$\chi (m) \\, \\left( \\frac{dN}{dm} \\right)$')

# Custom x-ticks (e.g., linear-style values on log axis)
custom_ticks = [0.1, 0.25, 0.5, 1, 2, 3, 4, 5]
ax.set_xticks(custom_ticks)
ax.set_xticklabels([str(t) for t in custom_ticks])

# Enable grid only at those ticks
for tick in custom_ticks:
    ax.axvline(tick, color='gray', linestyle='--', linewidth=0.5, zorder=0)

# Let's plot some IMFs
imf = fastar_imf.single_powerlaw(mass_arr,{})
ax.plot(mass_arr,imf, label='Single power law',linewidth=1.5)

imf = fastar_imf.broken_powerlaw(mass_arr,{})
ax.plot(mass_arr,imf, label='Broken power law',linewidth=1.5)

imf = fastar_imf.flexi(mass_arr,{'beta':2})
ax.plot(mass_arr,imf, label='Flexi IMF',linewidth=1.5)

imf = fastar_imf.chabrier(mass_arr,{})
ax.plot(mass_arr,imf, label='Chabrier IMF',linewidth=1.5)

imf = fastar_imf.kroupa(mass_arr,{})
ax.plot(mass_arr,imf, label='Kroupa IMF',linewidth=1.5)

ax.legend()

fig.tight_layout()
fig.savefig('../../figures/IMF_shapes.pdf')
plt.close('all')

