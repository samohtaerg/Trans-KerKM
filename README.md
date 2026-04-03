# Trans-KerKM: Kernel-Weighted Survival Estimation via Cross-Domain Transfer

Implementation details for the experiments in the paper **"Kernel-Weighted Survival Estimation via Cross-Domain Transfer"**.

> **Abstract.** [To be added.]

---

## Requirements

The HPC experiments were run on Python 3.9.21 with the following dependencies:

```
numpy==2.0.2
scikit-learn==1.6.1
scipy==1.13.1
lifelines==0.30.0
pandas==2.2.2
```

---

## Repository Structure

```
Trans-KerKM/
├── README.md
├── Model: Trans-KerKM        # Core estimator: Trans-KerKM with CV parameter selection
├── Data Generation           # Simulation data generation (HPC version)
├── Benchmarks/               # Baseline models: Target-Cox, Pool-Cox, Target-KerKM
├── Simulation/               # Full HPC simulation scripts (Settings 1, 2a, 2b, 3a, 3b)
├── Real Data Result/         # TCGA real data experiment scripts
└── demo/
    └── trans_kerkm_demo.py   # Self-contained toy example (see Quick Start)
```

---

## Quick Start (Toy Example)

To help readers understand the method's inputs, outputs, and structure without
running the full HPC pipeline, we provide a self-contained demo script that
reproduces a single run of **Setting 2b** (signal level sweep, imbalanced target
mixture) from Section 4 of the paper on simulated data.

### What the demo contains

The demo script `trans_kerkm_demo.py` includes the following components, each
corresponding directly to the paper:

- **`gaussian_kernel`**: Gaussian (RBF) kernel function used to measure covariate similarity between patients.
- **`compute_individualized_hazard`**: implements the kernel-weighted event count $D^{(-i)}(t_\ell \mid x_i)$ and risk set $R^{(-i)}(t_{\ell-1} \mid x_i)$ (Equations 3.1–3.3), with leave-one-out (LOO) correction.
- **`compute_survival_function`**: converts discrete hazard estimates to an individualized survival curve via the product-limit update (Equations 3.4–3.5).
- **`compute_c_index_from_survival_curves`**: summarizes each survival curve by its expected survival time $M_i$ (Equation 3.6) and computes Harrell's C-index (Equation 3.8).
- **`grid_search_cv`**: K-fold cross-validated grid search over $(\sigma, \lambda)$, maximizing the held-out C-index (Section 3.4).
- **`kernel_weighted_transfer_km`**: full Trans-KerKM pipeline — standardization, CV parameter selection, and test-set evaluation.
- **`fit_cox_model`**: target-only Cox proportional hazards baseline.
- **`fit_transfer_cox_model`**: naive pooling Cox baseline (source + target combined).
- **`generate_multigroup_data`**: latent subgroup survival data generator following the simulation framework in Section 4.1.

### Differences from the full HPC experiment

The demo uses the same functions and data-generating process as the HPC scripts,
with the following simplifications for runtime:

| | HPC experiment | Demo |
|---|---|---|
| Parameter grid | 5×5 log-scale | 2×2 (`σ ∈ {0.1, 1.0}`, `λ ∈ {1.0, 2.0}`) |
| CV folds | 5 | 3 |
| `n_source` | 2000 | 1000 |
| `n_test_target` | 1000 | 500 |
| Monte Carlo replicates | 100 | 1 (fixed seed = 2026) |
| `np.repeat` augmentation | present in HPC script | removed (violates LOO guarantee; not part of the estimator as defined in the paper) |

### Fixed setup

```
distribution_type : "separate"
data_type         : "unbalanced"  (target_mix = [0.5, 0.2, 0.3, 0, 0])
signal_level      : 4
n_source          : 1000
n_train_target    : 50
n_test_target     : 500
CV folds          : 3
sigma_grid        : [0.1, 1.0]
lambda_grid       : [1.0, 2.0]
seed              : 2026
```

### Running the demo

**Option 1 — Google Colab (recommended):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iD5HMgU08i5X3aI9z2pNCohy_skvXmQP#scrollTo=d6wk-ykGeE7L)

**Option 2 — Local:**

```bash
pip install numpy scipy scikit-learn lifelines
python demo/trans_kerkm_demo.py
```

Expected runtime: approximately 2–3 minutes on a standard CPU.

### Expected output

```
============================================================
Results Summary
============================================================
Method                       C-index
------------------------------------
Trans-KerKM                   0.7557 <-- proposed
Target-KerKM                  0.7298
Pool-Cox                      0.6058
Target-Cox                    0.6108
============================================================
```

---

## Reproducing the TCGA Experiments

[To be added.]

---

## Citation

```bibtex
@article{chen2026transkerkm,
  title     = {Kernel-Weighted Survival Estimation via Cross-Domain Transfer},
  author    = {Chen, Aoran and Feng, Yang},
  journal   = {[To be added]},
  year      = {2026}
}
```
