What's FASTAR
===============

In a nutshell
----------------------

FASTAR is an evolutionary stellar population synthesis code to generate simple stellar population (SSP) model predictions. In short, FASTAR shares the same principles and advantages of standard SSP models but offers some unique features:

- **Continuity.** FASTAR predictions can be generated for any age, chemical composition, and IMF.
- **Differentiability.** The synthesis of SSP models relies on the `JAX <https://docs.jax.dev/en/latest/index.html>`_ Python library, allowing for native numerical autodifferentiation and optimized CPU/GPU computation.
- **Speed.** FASTAR makes use of a PCA-based, neural network regressor to generate the underlying stellar spectra. This approach is significantly faster than alternative, more standard approaches.
- **Flexibility.** FASTAR does not provide a grid of pre-computed models but the tools to quickly generate them. In addition, model choices can be easily tweaked (e.g. seamlessly changing between different IMFs).
- **Reproducibility.** The synthesis of FASTAR models is openly accessible to the astronomical community.

Different FASTARs for different applications
----------------------------------------------

Currently, FASTAR comes in two different flavors:

.. topic:: Spectroscopic predictions

   **Wavelength coverage:** 3,540 - 7,400 Å; **Resolution (FWHM)**: 2.51 Å; **Scale**: 0.9 Å / pixel


FASTAR can synthesize models over the optical wavelength range at a moderate resolution (FWHM=2.51 Å), meant to be used for detailed spectroscopic measurements. These predictions rely on both the empirical `MILES stellar library <https://ui.adsabs.harvard.edu/abs/2006MNRAS.371..703S>`_ and the `BOSZ <https://ui.adsabs.harvard.edu/abs/2024A%2526A...688A.197M>`_ set of theoretical stellar templates.


.. topic:: Photometric predictions

   **Wavelength coverage:** 2,000 - 12,000 Å; **Resolution (FWHM)**: 2.51 Å; **Scale**: 4 Å / pixel


For photometric purposes it might be useful to have a long wavelength baseline. To fill that niche, we also offer the possibility of synthesizing FASTAR models, solely based on the the `BOSZ <https://ui.adsabs.harvard.edu/abs/2024A%2526A...688A.197M>`_ stellar library but covering a broader wavelength range. These predictions have a coarse spectral sampling (4 Å per pixel) but can be convolved with any set of photometric filters.


Semi-resolved vs fully-sampled models
---------------------------------------------

Traditional SSP models are calculated assuming an *infinitely* large collection of stars. Under this assumption, the flux emitted by an SSP can be modelled as:

.. math::
        F_\lambda \bigl(\text{age}, [\mathrm{M}/\mathrm{H}] \bigr)
    = \int_{m_{\mathrm{ini}}}^{m_{\mathrm{end}}}
      S_\lambda \!\left(\log g, T_{\mathrm{eff}}, [\mathrm{M}/\mathrm{H}] \right)
      \,\chi(m) \, dm

This is the basis of evolutionary stellar population modelling, where the IMF defines the weight of each stellar type (/mass). In some circumstances, however, the number of stars per resolution element is not large enough and the integral limit does not hold. In this so-called semi-resolved regime, the flux of an SSP can be instead represented as:

.. math::
        f_\lambda\bigl(\text{age}, [\mathrm{M}/\mathrm{H}], N_\mathrm{stars}\bigr)
    = \sum_{i=1}^{N_\mathrm{stars}} S_{\lambda, i} \, \!\left(\log g_i, T_{\mathrm{eff},i}, [\mathrm{M}/\mathrm{H}]_i \right)


The semi-resolved regime is stochastic by construction. Therefore, even after fixing the age, chemical composition, IMF and the number of stars contributing to the observed flux, there is a range of possible model predictions depending on the discrete sampling of the IMF.

**Thanks to the efficient JAX implementation, FASTAR offers the possibility of computing both integral and semi-resolved SSP models.**

Ingredients
-----------------------

Isochrones
^^^^^^^^^^^^

We make use of the `BaSTI-IAC <http://basti-iac.oa-abruzzo.inaf.it/>`_ set of isochrones. Specifically we use their solar-scaled isochrones, with overshooting and atomic diffusion treatment, mass loss :math:`\eta=0.3` and He:math:`=0.247`.

.. important::

   Our choice if isochrones sets a hard limit on the validity range of the FASTAR models.

   **Ages must be between 20 Myr and 14 Gyr, while metallicities between -2.5 and +0.3**


Stellar libraries
^^^^^^^^^^^^^^^^^^

Currently, FASTAR is based on two main stellar libraries. `MILES <https://ui.adsabs.harvard.edu/abs/2006MNRAS.371..703S>`_ is a library of approximately 1,000 stars observed at an intermediate spectral resolution of 2.51 Å (FWHM). These stellar templates along with the estimated atmospheric parameters are accessible through the website of the `MILES collaboration <https://research.iac.es/proyecto/miles/pages/stellar-libraries/miles-library.php>`_.

Complementary, we also use the `BOSZ <https://ui.adsabs.harvard.edu/abs/2024A%2526A...688A.197M>`_ theoretical templates, convolved to match the spectral resolution of MILES.


Bolometric corrections
^^^^^^^^^^^^^^^^^^^^^^^

The synthesis of FASTAR models does not directly use the theoretical luminosities predicted by the isochrones but transforms them into V-band magnitudes that are then directly measured from the stellar spectra. This transformation is done using the bolometric corrections presented in `Worthey & Lee (2011) <https://ui.adsabs.harvard.edu/abs/2011ApJS..193....1W>`_.

Initial mass functions
^^^^^^^^^^^^^^^^^^^^^^^

The synthesis of new FASTAR models can be done assuming a variety of functional forms for the IMF. Currently there are six different parametrizations immediately available for the user

**Milky Way-like IMFs**
    Both `Kroupa <https://ui.adsabs.harvard.edu/abs/2001MNRAS.322..231K>`_ and `Chabrier <https://ui.adsabs.harvard.edu/abs/2003PASP..115..763C>`_ can be used to create models based on the Milky Way standard.

**Single power-law**
    A scale-free IMF defined by a single slope, generalizing pioneering `work <https://ui.adsabs.harvard.edu/abs/1955ApJ...121..161S>`_  of Edwin Salpeter.

**Broken power-law**
    Two-segment IMF definition, similar to that implemented in `Conroy & van Dokkum (2012) <https://ui.adsabs.harvard.edu/abs/2012ApJ...747...69C>`_.

**Bimodal IMF**
    The same functional form of the MILES models as defined in `Vazdekis et al. (1996) <https://ui.adsabs.harvard.edu/abs/1996ApJS..106..307V>`_.

**Tapered power-law**
    Following the flexible definition of `De Marchi et al. (2005) <https://ui.adsabs.harvard.edu/abs/2005ASSL..327...77D>`_, naturally including a characteristic stellar mass.

.. tip::

    **FASTAR can be used to synthesize SSP models assuming any functional form for the IMF**. Instructions and examples are given to easily implement new IMFs if you want to explore with alternative definitions.


Units and conventions
-----------------------

On the IMF
^^^^^^^^^^^^^^^^^^^^^^^

The synthesis of FASTAR models adopts a **linear definition for the IMF**. That is, the IMF is defined as

.. math::
    \chi(M) = \frac{dN}{dM}

This means, for example, that a Salpeter-like IMF would be characterized by a slope :math:`\alpha=2.35`. Note that this linear definition is not the same as the logarithmic description assumed in our MILES models.

In addition, we adopted :math:`m_\star=0.1` and :math:`m_\star=100` as the low- and high-mass end cutoffs for the IMF. Although the performance of the FASTAR predictions has been tested assuming these limits, new models can be generated with variable mass cutoffs.

Finally, **the integral version of the FASTAR models is normalized to 1 solar mass at birth**. In practice, this implies that integral FASTAR models correspond to the flux that a population with a certain age, metallicity and IMF would emit if it originally weighted 1 :math:`M_\odot`

Solar reference
^^^^^^^^^^^^^^^^^^^^^^^

FASTAR models are anchored to the V-band absolute magnitude of the Sun. We assume an absolute bolometric magnitude for the sun of 4.70 and a V-band bolometric correction of -0.12.

Moreover, in order to calculate generic mass-to-light ratio predictions for any photometric filter within the FASTAR wavelength range(s) one must assume a model spectrum for the Sun. In FASTAR, we adopted the `CALSPEC <https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/astronomical-catalogs/calspec>`_ Kurucz model of the Sun available `here <https://archive.stsci.edu/hlsps/reference-atlases/cdbs/current_calspec/sun_mod_001.fits>`_.


Absolute magnitudes
^^^^^^^^^^^^^^^^^^^^^^^

Because of the solar normalization described above, FASTAR predictions have units of

.. math::

    [F_\lambda] = [\mathrm{erg}\,\mathrm{s}^{-1}\,\mathrm{cm}^{-2}\,\text{Å}^{-1}]


Because the models are scaled to the absolute magnitude of the Sun, the FASTAR predictions follow the same scaling. Therefore, FASTAR SSP models can be directly translated into :func:`absolute magnitudes <fastar.tools.utils.compute_ab_magnitudes>`.

.. important::

    **Note that the interpretation of the predicted absolute magnitudes differs between the integral version of FASTAR and the semi-resolved version.**

    In the integral case, as described above, the model is normalized to a total initial mass of one solar mass. Therefore, the predicted absolute magnitude corresponds to a stellar population with that mass at birth.

    In contrast, the absolute magnitudes derived from a semi-resolved model represent the expectation value for a population containing a specified number of stars (e.g. 1e3 stars), rather than a fixed total stellar mass.

Publications
-------------

If you use FASTAR in your research, please consider citing the following papers:

- **FASTAR I — Differentiable synthesis of evolutionary stellar population models**

  *Martín-Navarro I. et al.*
  `MNRAS, 447, 1033 (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.447.1033M>`_

- **FASTAR II — Semi-resolved evolutionary stellar population models**

  *Martín-Navarro I. et al.*
  `MNRAS, 447, 1033 (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.447.1033M>`_
