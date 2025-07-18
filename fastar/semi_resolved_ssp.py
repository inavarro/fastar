#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.integrate import trapezoid

from fastar.core.ssp import SspSynthesizer
from fastar.interpolate.color import color_interpolation


class SemiResolvedSspSynthesizer(SspSynthesizer):
    """
    Class for generating synthetic semi-resolved stellar populations
    spectroscopic and photometric predictions with a PCA-based stellar spectral
    model and a stochastic IMF sampling.
    """

    @partial(jax.jit, static_argnames=['self', 'Nstars'])
    def synthesize(self, age, met, imf_params, Nstars, key, out_masses=False):
        """
        Generate synthetic semi-resolved population spectrum for a given
        age and metallicity.

        Parameters
        ----------
        age : float
            Stellar population age (in Gyr).
        met : float
            Metallicity [M/H].
        imf_params : dict
            Parameters for the initial mass function.
        Nstars : int
            Number of stars to sample.
        key : PRNGKey
            Random key for JAX sampling.
        out_masses : bool, optional
            If True, return array of sampled stellar masses instead of the
            total mass. Default is False.

        Returns
        -------
        tuple
            (wavelengths, spectrum, total stellar mass) or (wavelengths,
            spectrum, sampled stellar masses) depending on `out_masses`.
        """
        # Interpolate the isochrones at the desired age and metallicity
        imass, iteff, ilogg, ilum = self._get_isochrone(age, met)

        # Stochastically sample the IMF
        sampled_masses = self._stochastic_IMF_sampling(
            imass, imf_params, Nstars, key
        )

        # Evaluate the isochrone at the interpolated masses
        iteff_interp = jnp.interp(sampled_masses, imass, iteff)
        ilogg_interp = jnp.interp(sampled_masses, imass, ilogg)
        ilum_interp = jnp.interp(sampled_masses, imass, ilum)

        # Calculate the stellar spectra
        spectra = self._predict_spectrum(
            ilogg_interp, iteff_interp, jnp.full_like(iteff_interp, met)
        )

        # Get the V-band magnitudes of the predicted stellar spectra (they are
        # normalized to have a mean flux of 1)
        magnitudes = self._compute_ab_magnitudes(spectra)

        # Calculate the bolometric corrections
        bcv_val = color_interpolation(
            ilogg_interp,
            iteff_interp,
            jnp.full_like(iteff_interp, met),
            self.logg_array,
            self.teff_log10_array,
            self.fmet_array,
            self.bcv_grid,
        )

        # Scale the predicted stellar spectra so they math their theoretical
        # luminosities
        vmags = -2.5 * ilum_interp - bcv_val
        mtarg = vmags + self.sun_vmag
        corr = 1 / (10 ** ((magnitudes - mtarg) / -2.5))

        # Add the flux of all the spectra. There is no IMF weighting here
        # since it naturally comes from the stochastic sampling
        spec = jnp.sum(spectra * corr[:, None], axis=0)

        # The function returns wavelength, spectrum and either the total
        # stellar mass of the population or the sampled
        if out_masses:
            result = (self.wave, spec, sampled_masses)
        else:
            result = (self.wave, spec, jnp.sum(sampled_masses))

        return result

    def synthesize_large(
        self, age, met, imf_params, Nstars, key, batch_size=10000
    ):
        """
        Generate synthetic population spectrum using chunked IMF sampling
        for large numbers of stars. This allows calculating predictions
        for a large number of stars without using too much memory
        """
        n_batches = int(Nstars) // int(batch_size)
        remainder = int(Nstars) % int(batch_size)
        n_total_chunks = n_batches + int(remainder > 0)

        keys = jax.random.split(key, n_total_chunks)

        # First batch
        wave, spec_total, mass_total = self.synthesize(
            age, met, imf_params, batch_size, keys[0]
        )

        # Loop over full-size batches
        for i in range(1, n_batches):
            _, spec_chunk, mass_chunk = self.synthesize(
                age, met, imf_params, batch_size, keys[i]
            )
            spec_total += spec_chunk
            mass_total += mass_chunk

        # Final chunk (remainder)
        if remainder > 0:
            _, spec_chunk, mass_chunk = self.synthesize(
                age, met, imf_params, remainder, keys[-1]
            )
            spec_total += spec_chunk
            mass_total += mass_chunk

        return wave, spec_total, mass_total

    @partial(jax.jit, static_argnames=['self', 'Nstars'])
    def _stochastic_IMF_sampling(self, imass, imf_params, Nstars, key):
        """
        Stochastically sample stellar masses from an IMF assuming
        it emerges from a probability distribution function.
        """

        # Normalize the IMF to a total number of stars equal to 1
        # Note this normalization is different than the one used
        # for the population synthesis as there the normalizing
        # quantity is the total mass (not the number of stars)
        mass_grid = jnp.linspace(imass.min(), imass.max(), 5000)
        imf_vals = self.imf_function(mass_grid, imf_params)

        # IMF to PDF (via normalization)
        pdf = imf_vals / trapezoid(imf_vals, x=mass_grid)
        cdf = jnp.cumsum(pdf)
        cdf = cdf / cdf[-1]

        # Uniform sampling f the CDF
        uniform_samples = jax.random.uniform(key, shape=(int(Nstars),))

        return jnp.interp(uniform_samples, cdf, mass_grid)
