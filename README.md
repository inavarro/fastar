<br>
<p align="center">
  <img src="https://raw.githubusercontent.com/inavarro/fastar/main/docs/_static/fastar_logo.png" width="300">
   alt="FaStar logo">
</p>
<br>

# FASTAR

**FASTAR** is a fully differentiable, evolutionary stellar population synthesis
code. It generates single stellar population (SSP) model predictions for both
spectroscopic (from ~3,500 Å to ~7,400 Å, at a 2.51 Å FWHM resolution) and spectral
energy distribution (SED, from ~2,000 Å to ~11,500 Å) measurements.

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
python3 -m pip install fastar-astro
```

This will install it in the current environment, but it is recommended to use
a virtual environment for the installation (using `venv`, `conda`, `uv` or the
tool of your choosing).

### Development installation

If you wish to modify the `fastar` code, we recommend to use the provided
`Makefile` to setup your environment. It uses [uv](https://docs.astral.sh/uv/)
to setup a virtual environment and it should be transparent to the developer.

```
git clone https://github.com/inavarro/fastar.git
cd fastar/
make install-dev
# ... do your development and test them
make tests
# ... add your changes and do a commit
# git add ...
# git commit ...
# This will trigger some automatic checks (format, linting,...) that you must
# address before committing (some of them are fixed automatically).
git push
```

Changes are welcome! But remember that it is always safer to work on your local
fork and only when your development is finished you can proceed to do a PR.


---

## Quick Start

```python
from fastar import IntegratedSSPSynthesizer
from fastar.imf import kroupa

ssp = IntegratedSSPSynthesizer(imf_function=kroupa)
flux = ssp.synthesize(age=10.0, met=0.0)
wave = ssp.wave
```
