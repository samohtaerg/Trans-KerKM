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

We provide a self-contained demo that runs **one replicate** (single random seed, no Monte Carlo averaging) of Setting 2b (signal level sweep, imbalanced target mixture; Section 4 of the paper) on simulated data with a reduced parameter grid and fixed seed. The demo illustrates the method's inputs, core functions, and output format. Full details on the simplifications relative to the HPC experiments are documented in the script header.

**Google Colab** (no setup required):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1iD5HMgU08i5X3aI9z2pNCohy_skvXmQP#scrollTo=d6wk-ykGeE7L)

**Local:**

```bash
pip install numpy scipy scikit-learn lifelines
python demo/trans_kerkm_demo.py
```

**Expected runtime:** under 60 seconds. 


**Expected output:**
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

### Data

Download the clinical JSON files for the five cancer types (BRCA, OV, LUAD, GBM, UCEC) from the [GDC Data Portal](https://portal.gdc.cancer.gov/analysis_page?app=Projects) and place them under `Real Data Result/TCGA Dataset/Raw Data/`. The filenames include the download date (e.g., `clinical.project-tcga-brca.2025-07-25.json`); update the `FILE_MAPPING` dictionary in the `USER CONFIGURATION` block of `tcga_data_cleaning.py` to match your downloaded filenames before running.

### Preprocessing

Run `Real Data Result/TCGA Dataset/tcga_data_cleaning.py` on your local machine or HPC, which produces `combined_survival_final.json`. Edit the `USER CONFIGURATION` block at the top of the script (paths `RAW_DATA_PATH` and `SAVE_PATH`) before running.

### Experiment

Edit the `USER CONFIGURATION` block at the top of `Real Data Result/run_tcga.py.py`: set `OUTPUT_DIR` and `TCGA_FILE` to your local or HPC paths before running.
