# Simulation-trained measurement of patient positioning errors in panoramic radiography

Code accompanying the article:

> Halloum K, et al. *Do visual grades reflect geometry? Simulation-trained measurement of patient positioning errors in panoramic radiography: metrological validation and comparison with expert grading.* (submitted)

A ConvNeXt-Tiny regressor is trained **only on physics-based simulations** to measure three positioning errors on panoramic radiographs:

| Symbol | Error | Unit |
|---|---|---|
| *θ* | chin inclination (positive = chin too low) | degrees |
| *φ* | head tilt | degrees |
| *s* | antero-posterior scale of the incisor region | dimensionless (≈ mm via Eq. 2) |

It is then validated metrologically on clinical images by **paired injection** of known errors: repeatability *σ*₀, dose–response linearity and Currie limit of detection (LoD = 3.29 *σ*₀ / *b*).

## Repository layout

```
src/
  pano_artifacts.py      physics-based artefact simulator (from our previous work)
  pano_calibrated.py     simulator expressed in physical units (θ, φ in degrees)
  pano_geometry.py       panoramic projection geometry (Eqs. 1–3)
  a7_core.py             targets, photometric augmentation, statistics helpers
  a7_core2.py            domain-randomised simulation (v2) and histogram matching
  lire_tags_geometrie.py reads geometry tags (pixel spacing, kV, mA) from DICOM headers
scripts/
  01_train_and_validate.py  trains the measurer (main + no-blur ablation) and validates it
  02_metrology.py           repeatability, dose–response, LoD, cohort analysis, agreement
equations/               LaTeX source and renderer of Eqs. (1)–(8)
figures/                 Figures 1–4 of the article (+ data of Figure 3)
```

## How to run

The scripts were run as Kaggle notebooks (NVIDIA T4 GPU, Python 3, PyTorch, timm).

1. Add as notebook inputs: this repository (folder `src/`), the public **STS-2D-Tooth** dataset, and your own anonymised panoramic images with a metadata CSV.
2. Run `scripts/01_train_and_validate.py`. It saves `convnext_tiny__v2.best.pt` and `convnext_tiny__v2_sans_flou.best.pt`.
3. Add the output of step 2 as input and run `scripts/02_metrology.py`.

Install locally with `pip install -r requirements.txt`.

## Data

- **STS-2D-Tooth** (training): public, CC-BY 4.0 — Wang Y, et al. *Sci Data* 2025;12:117.
- **Clinical images** (validation): 151 anonymised panoramic radiographs from one Owandy I-Max unit. They **cannot be shared** for privacy reasons and are not included in this repository.

## Licence

MIT (see `LICENSE`).
