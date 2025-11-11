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

Ingredients and units
-----------------------

Publications
-------------

If you use FASTAR in your research, please consider citing the following papers:

- **FASTAR I — Differentiable synthesis of evolutionary stellar population models** 

  *Martín-Navarro I. et al.*
  `MNRAS, 447, 1033 (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.447.1033M>`_

- **FASTAR II — Semi-resolved evolutionary stellar population models** 

  *Martín-Navarro I. et al.*
  `MNRAS, 447, 1033 (2015) <https://ui.adsabs.harvard.edu/abs/2015MNRAS.447.1033M>`_