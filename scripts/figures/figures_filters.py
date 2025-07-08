import numpy as np
from astroquery.svo_fps import SvoFps
from matplotlib import pyplot as plt
from astropy.io import fits

from fastar_ssp_class import PopulationSynthesizer
from fastar_imf import single_powerlaw as unimodal

# ---------------------------------------------------
# Spectroscopic predictions
# ---------------------------------------------------
# Load the synthesis code
ssp_spec = PopulationSynthesizer(imf_function=unimodal)

# Let's focus now on a single SSP
age = 12
met = 0.

# Standard SSP call
wave, spec = ssp_spec.synthesize(age=age, met=met)

# # ---------------------------------------------------
# # Photometric predictions (based on the XSL library)
# # **SHOULD NOT BE USED FOR SPECTROSCOPIC MEASUREMENTS**
# # ---------------------------------------------------
# # Load the synthesis code
ssp_phot = PopulationSynthesizer(model_label='phot', imf_function=unimodal)

phot_wave, phot_spec = ssp_phot.synthesize(age=age, met=met)

# E-MILES
hdu = fits.open('/home/imartin/Basti/UN/EMILES_BASTI_BASE_UN/Eun1.30Zp0.06T12.0000_iTp0.00_baseFe.fits')

mspec = hdu[0].data
mwave =  1680 + np.arange(53689) * 0.9


# Load HiPERCAM filters
data = SvoFps.get_transmission_data('GTC/HIPERCAM.u')
utrans = np.interp(mwave, data['Wavelength'], data['Transmission'],left=0, right=0)

data = SvoFps.get_transmission_data('GTC/HIPERCAM.g')
gtrans = np.interp(mwave, data['Wavelength'], data['Transmission'],left=0, right=0)

data = SvoFps.get_transmission_data('GTC/HIPERCAM.r')
rtrans = np.interp(mwave, data['Wavelength'], data['Transmission'],left=0, right=0)

data = SvoFps.get_transmission_data('GTC/HIPERCAM.i')
itrans = np.interp(mwave, data['Wavelength'], data['Transmission'],left=0, right=0)

data = SvoFps.get_transmission_data('GTC/HIPERCAM.z')
ztrans = np.interp(mwave, data['Wavelength'], data['Transmission'],left=0, right=0)



plt.figure(figsize=(12, 6))
# Normalize and plot filter response curves as shaded regions
def plot_filter(wave, response, color, label):
    norm_response = response / response.max() * 0.5 * np.max(mspec)
    plt.fill_between(wave, 0, norm_response*50000, color=color, alpha=0.2, label=label)

plot_filter(mwave, utrans, 'violet', 'HiPERCAM-u')
plot_filter(mwave, gtrans, 'green', 'HiPERCAM-g')
plot_filter(mwave, rtrans, 'red', 'HiPERCAM-r')
plot_filter(mwave, itrans, 'darkorange', 'HiPERCAM-i')
plot_filter(mwave, ztrans, 'brown', 'HiPERCAM-z')

# Plot the spectra
plt.plot(mwave, mspec/np.mean(mspec[(mwave > 5000) & (mwave < 5100)]), label='E-MILES (12 Gyr, Z=+0.06)', color='grey', lw=1.4, alpha=0.8)
plt.plot(phot_wave, phot_spec/np.mean(phot_spec[(phot_wave > 5000) & (phot_wave < 5100)]), label='Photometric SSP', color='blue', lw=1.2)
plt.plot(wave, spec/np.mean(spec[(wave > 5000) & (wave < 5100)]), label='Spectroscopic SSP', color='xkcd:coral', lw=1.2)

# Labels and legend
plt.xlabel('Wavelength (Å)')
plt.ylabel('Flux (arbitrary units)')
plt.title('SSP Spectra with HiPERCAM Filter Responses')
plt.legend()
plt.tight_layout()
plt.xlim(2000, 11500)

plt.tight_layout()

# Save to file (high resolution)
plt.savefig('figures/ssp_with_filters.pdf', dpi=300)

plt.close('all')