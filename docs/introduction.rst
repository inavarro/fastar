What's FASTAR
===============

In a nutshell
----------------------

FASTAR is an evolutionary stellar population synthesis code to generate simple stellar population (SSP) model predictions. In short, FASTAR shares the same principles and advantages of standard SSP models but offers some unique features:

- **Continuity.** FASTAR predictions can be generated for any age, chemical composition, and IMF.
- **Differentiability.** The synthesis of SSP models relies on the JAX Python library, allowing for native numerical autodifferentiation and optimized CPU/GPU computation.
- **Speed.** FASTAR makes use of a PCA-based, neural network regressor to generate the underlying stellar spectra. This approach is significantly faster than alternative, more standard approaches.
- **Flexibility.** FASTAR does not provide a grid of pre-computed models but the tools to quickly generate them. In addition, model choices can be easily tweaked (e.g. seamlessly changing between different IMFs).
- **Reproducibility.** The synthesis of FASTAR models is openly accessible to the astronomical community.

FASTAR flavors
---------------------------

Semi-resolved vs fully-sampled observations
---------------------------------------------

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