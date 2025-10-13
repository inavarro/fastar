<br>
<p align="center">
  <img src="docs/_static/fastar_logo-text-black_and_white.png" width="300"
   alt="FaStar logo">
</p>
<br>

# FASTAR

**FASTAR** is a fully differentiable, evolutionary stellar population synthesis
code. It generates single stellar population (SSP) model predictions for both
spectroscopic —from ~3,500 Å to ~7,400 Å— and spectral energy distribution
(SED) —from ~2,000 Å to ~11,500 Å— measurements.

Unlike traditional SSP models, **FASTAR** produces continuous predictions for
any age, metallicity and initial mass function (IMF) value. It also supports
SSP model uncertainty estimation through Monte Carlo sampling.

Critically, **FASTAR** can also generate on the fly semi-resolved SSP models
assuming the IMF is not fully sampled, i.e., when the number of stars per
resolution element is not large enough.

---

## Ingredients

- **[MILES](https://ui.adsabs.harvard.edu/abs/2006MNRAS.371..703S)** —
  empirical stellar library
- **[BOSZ](https://ui.adsabs.harvard.edu/abs/2024A%26A...688A.197M)** —
  theoretical stellar library
- **[Worthey & Lee (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJS..193....1W)** —
  bolometric correction tables
- **[BaSTI-IAC](http://basti-iac.oa-abruzzo.inaf.it/)** — isochrones

---

## Features

- NN-based predictions of stellar spectra.
- SSP synthesis for photometric and spectroscopic data.
- Integrated and semi-resolved SSP predictions.

---

## Installation

```bash
git clone git@github.com:inavarro/fastar.git
cd fastar/
pip install .
```

---

## Quick Start

```python
from fastar.imf.named_imf.kroupa import kroupa
from fastar.integrated_ssp import IntegratedSspSynthesizer

ssp = IntegratedSspSynthesizer(imf_function=kroupa)
wave, flux = ssp.synthesize(age=10.0, met=0.0)
```
