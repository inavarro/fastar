#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import partial

import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.integrate import trapezoid
from flax.core import FrozenDict

from fastar.core.ingredients import PopulationIngredients
from fastar.interpolate.color import color_interpolation

class SemiresolvedSynthesizer(PopulationIngredients):
    """
    Class for generating synthetic semi-resolved stellar populations
    spectroscopic and photometric predictions with a PCA-based stellar spectral
    model and a stochastic IMF sampling.
    """

    @partial(jax.jit, static_argnames=['self','num_stars','out_masses'])
    def synthesize(
        self, age, met, num_stars, key, imf_params=None, out_masses=False
    ):
        """
        Generate synthetic semi-resolved population spectrum for a given
        age and metallicity.

        Parameters
        ----------
        age : float
            Stellar population age (in Gyr).
        met : float
            Metallicity [M/H].
        num_stars : int
            Number of stars to sample.
        key : PRNGKey
            Random key for JAX sampling.
        imf_params : dict, optional
            Parameters for the initial mass function.
        out_masses : bool, optional
            If True, return array of sampled stellar masses instead of the
            total mass. Default is False.

        Returns
        -------
        tuple
            (wavelengths, spectrum, total stellar mass) or (wavelengths,
            spectrum, sampled stellar masses) depending on `out_masses`.
        """
        # ensure we always pass a dict to IMF **params
        imf_params = imf_params or {}

        # Interpolate the isochrones at the desired age and metallicity
        imass, iteff, ilogg, ilum = self._get_isochrone(age, met)

        # Stochastically sample the IMF
        sampled_masses = self._stochastic_imf_sampling(
            imass, imf_params, num_stars, key
        )

        # Evaluate the isochrone at the interpolated masses
        iteff_interp = jnp.interp(sampled_masses, imass, iteff)
        ilogg_interp = jnp.interp(sampled_masses, imass, ilogg)
        ilum_interp = jnp.interp(sampled_masses, imass, ilum)

        # Calculate the stellar spectra
        spectra = self.predict_spectrum(
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

    @partial(jax.jit, static_argnames=['self','num_stars'])
    def _stochastic_imf_sampling(self, imass, imf_params, num_stars, key):
        """
        Stochastically sample stellar masses from an IMF assuming
        it emerges from a probability distribution function.
        """
        imf_params = imf_params or {}

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
        uniform_samples = jax.random.uniform(key, shape=(num_stars,))

        return jnp.interp(uniform_samples, cdf, mass_grid)


    @partial(jax.jit, static_argnames=['self'])
    def _synthesize_massgiven(
        self, met, imass, iteff, ilogg, ilum, sampled_masses
    ):
        """
        Generate the integrated spectrum of a stellar population using a
        pre-sampled set of stellar masses and isochrone quantities.
        """
        # Evaluate the isochrone at the interpolated masses
        iteff_interp = jnp.interp(sampled_masses, imass, iteff)
        ilogg_interp = jnp.interp(sampled_masses, imass, ilogg)
        ilum_interp = jnp.interp(sampled_masses, imass, ilum)

        # Calculate the stellar spectra
        spectra = self.predict_spectrum(
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

        return spec
    
    @partial(jax.jit, static_argnames=['self','num_stars','batch_size', 'out_masses'])
    def synthesize_large(
        self, age, met, num_stars, key, batch_size=10000,
        out_masses=False, imf_params=None
    ):
        imf_params = imf_params or {}

        # Isochrone interpolation (same for all batches)
        imass, iteff, ilogg, ilum = self._get_isochrone(age, met)

        # Sample IMF once
        sampled_masses = self._stochastic_imf_sampling(imass, imf_params, num_stars, key)

        # Split samples into batches
        n_batches = num_stars // batch_size
        remainder = num_stars % batch_size

        full_samples = sampled_masses[:n_batches * batch_size]
        batches = full_samples.reshape((n_batches, batch_size))

        # Scan-compatible batch function
        def batch_fn(batch_masses):
            return self._synthesize_massgiven(
                met=met,
                imass=imass, iteff=iteff, ilogg=ilogg, ilum=ilum,
                sampled_masses=batch_masses,
            )

        # Initial spectrum accumulator
        init_spec = jnp.zeros_like(self.wave)

        # Use lax.scan for batches
        def scan_fn(spec_accum, batch_masses):
            return spec_accum + batch_fn(batch_masses), None

        spec_total, _ = lax.scan(scan_fn, init_spec, batches)

        # Handle remainder using lax.cond (fully JAX-compatible)
        def add_remainder(spec_accum):
            rem_masses = sampled_masses[-remainder:]
            rem_spec = self._synthesize_massgiven(
                met=met, imass=imass, iteff=iteff,
                ilogg=ilogg, ilum=ilum, sampled_masses=rem_masses
            )
            return spec_accum + rem_spec

        spec_total = lax.cond(remainder > 0, add_remainder, lambda x: x, spec_total)

        if out_masses:
            return self.wave, spec_total, sampled_masses
        else:
            return self.wave, spec_total, jnp.sum(sampled_masses)