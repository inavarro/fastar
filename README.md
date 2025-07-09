<br>
<p align="center">
  <img src="logo/logo.png" width="300" alt="FaStar logo">
</p>
<br>

# FaStar


**FaStar** is a fully differentiable, evolutionary stellar population synthesis code. It generates single stellar population (SSP) model predictions for both spectroscopic (from ~3,540 Å to ~7,410 Å) and photometric (from ~2,000 Å to ~11,500 Å) observations.

Unlike traditional SSP models, **FaStar** produces continuous predictions for any  age, metallicity, and initial mass function (IMF). It also supports SSP model uncertainty estimation through Monte Carlo sampling.

Critically, **FaStar** can also generate on the fly semi-resolved SSP models assuming the IMF is not fully sampled, i.e., when the number of stars per resolution element is not large enough. 

---
## ⚙️ Core ingredients

- **[MILES](https://ui.adsabs.harvard.edu/abs/2006MNRAS.371..703S)** — empirical stellar library  
- **[BOSZ](https://ui.adsabs.harvard.edu/abs/2024A%2526A...688A.197M)** — theoretical stellar library  
- **[Worthey & Lee (2011)](https://ui.adsabs.harvard.edu/abs/2011ApJS..193....1W)** — bolometric correction tables  
- **[BaSTI-IAC](http://basti-iac.oa-abruzzo.inaf.it/)** — isochrones
---

## 🚀 Features

- NN-based predictions of stellar spectra.
- SSP synthesis for photometric and spectroscopic data.
- Integrated and semi-resolved SSP predictions.

---

## 📦 Installation

To install in development mode:

```bash
git clone https://github.com/inavarro/fastar.git
cd fastar
pip install -e .
