# Trans-KerKM

Implementation for the paper **"Kernel-Weighted Survival Estimation via Cross-Domain Transfer"**. [[Paper]([URL])]

---

## Requirements

```
numpy==2.0.2
scikit-learn==1.6.1
scipy==1.13.1
lifelines==0.30.0
pandas==2.2.2
```

---

## Demo

We provide a self-contained demo that runs a single replicate of **Setting 2b** (signal level sweep, imbalanced target mixture; Section 4 of the paper) on simulated data with a reduced parameter grid and fixed seed. The demo illustrates the method's inputs, core functions, and output format. Full details on the simplifications relative to the HPC experiments are documented in the script header.

**Google Colab** (no setup required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iD5HMgU08i5X3aI9z2pNCohy_skvXmQP#scrollTo=d6wk-ykGeE7L)

**Local:**

```bash
pip install numpy scipy scikit-learn lifelines
python demo/trans_kerkm_demo.py
```

Expected runtime: under 60 seconds. Expected output:

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

## TCGA Experiments

[To be added.]
