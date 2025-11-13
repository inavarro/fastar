Predictions and functionalities
================================

FASTAR can be used to generate a wide range of spectro-photometric predictions. Listed below are some of the most relevant quantities that can be computed using FASTAR.


FASTAR flavors
----------------------------------------

The spectroscopic synthesis is the default FASTAR synthesis. To select the FASTAR **photometric** predictions, you must pass ``model_label="phot"`` when initializing the synthesizer.

.. code-block:: python

    from fastar.integrated_ssp import IntegratedSynthesizer

    # Spectroscopic predictions (default)
    ssp_spec = IntegratedSynthesizer()
    wave_spec, model_spec = ssp_spec.synthesize(age=10, met=0)

    # Photometric predictions
    ssp_phot = IntegratedSynthesizer(model_label="phot")
    wave_phot, model_phot = ssp_phot.synthesize(age=10, met=0)


Integral FASTAR models (standard SSP)
----------------------------------------

**SSP models**
^^^^^^^^^^^^^^^^^^
The easiest way of using FASTAR is to generate a single SSP model. This is done by loading the synthesizer and specifying the IMF functional form

.. code-block:: python

    from fastar.integrated_ssp import IntegratedSynthesizer
    from fastar.imf import kroupa

    ssp = IntegratedSynthesizer(imf_function=kroupa)
    wave, model = ssp.synthesize(age=10, met=0)


**Mass-to-light ratios**
^^^^^^^^^^^^^^^^^^^^^^^^^
FASTAR can predict mass-to-light in any photometric band. Note that the filter response has to be evaluated at the same wavelengths as the FASTAR models.

.. code-block:: python

    from fastar.integrated_ssp import IntegratedSynthesizer
    from fastar.imf import kroupa

    ssp = IntegratedSynthesizer(imf_function=kroupa)
    mass_to_light = ssp.mass_to_light_ratio(age=10, met=0, filter_response=filter_response)

.. important::

   The retrieved variable ``mass_to_light`` is a dictionary containing two
   mass-to-light ratios:

   * ``mass_to_light["ml_stars"]``: pure stellar mass-to-light ratio (stellar light divided by stellar mass).
   * ``mass_to_light["ml_total"]``: total mass-to-light ratio including  th mass in stars, remnants, and gas ejected during stellar evolution.

**Model grids**
^^^^^^^^^^^^^^^^^^^^^^^^^

Although one of the biggest advantages of FASTAR is that it is able to generate continuous SSP predictions for any age, metallicity and IMF value, certain applications might require the use of a rigid grid of SSP models.

.. code-block:: python

    from fastar.integrated_ssp import IntegratedSynthesizer
    from fastar.imf import kroupa

    ssp = IntegratedSynthesizer(imf_function=kroupa)
    model_grid = load_precomputed_models(age_range, met_range, imf_range)

.. tip::

    **Once calculated the first time, model grids are cached.** For convenience, subsequent calls using the same stellar population ranges will be loaded from disk.


Semi-resolved models
----------------------------------------------

**SSP models**
^^^^^^^^^^^^^^^^^^
The synthesis of semi-resolved models follows the same principles and philosophy as the integral version but now depending on an additional parameter: the number of stars contributing to the emitted flux. The stochastic nature of the IMF sampling is introduced through a randomizable key. Note that the predicted spectrum still corresponds to a coeval stellar population and therefore semi-resolved models are still labeled as SSP.

.. code-block:: python

    from fastar.semiresolved_ssp import SemiresolvedSynthesizer
    from fastar.imf import kroupa

    ssp = SemiresolvedSynthesizer(imf_function=kroupa)
    wave, model = ssp.synthesize(age=10, met=0, num_stars=1e3, key=key)


Line-strengths, magnitudes and colors
---------------------------------------

In addition to the synthesis of evolutionary stellar population models and related quantities, FASTAR also implements basic analysis tools to measure colors, magnitudes and line-strength indices from the model predictions.

**Line-strength indices**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Line-strength indices can be directly computed from the predicted FASTAR spectra at the 2.51 Å resolution of the models

.. code-block:: python

    from fastar.tools.utils import compute_linetrength

    line_strength = compute_linetrength(wave, model, index_name, index_table)


**Magnitudes and colors**
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Given a filter response evaluated at the same wavelength as the FASTAR models, the absolute AB magnitude of that population can be directly retrieved

.. code-block:: python

    from fastar.tools.utils import compute_ab_magnitudes

    ab_mag = compute_ab_magnitudes(wave, model, filter_response)


with a color being simply the difference between two filters

.. code-block:: python

    from fastar.tools.utils import compute_ab_magnitudes

    ab_mag_1 = compute_ab_magnitudes(wave, model, filter_response_1)
    ab_mag_2 = compute_ab_magnitudes(wave, model, filter_response_2)

    color = ab_mag_1 - ab_mag_2
