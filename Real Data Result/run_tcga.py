# -*- coding: utf-8 -*-
"""
run_tcga.py
-----------
Trans-KerKM experiment on TCGA real data.

The seed is read from $SLURM_ARRAY_TASK_ID (defaults to 1 for local runs).
Run with seeds 0-99 via SLURM array job.

Edit the USER CONFIGURATION block below before running.
"""

import numpy as np
import pandas as pd
import time
import os
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from Trans_KerKM import kernel_weighted_transfer_km
from Benchmarks.baselines import fit_cox_model

# ============================================================
# USER CONFIGURATION — edit these paths before running
# ============================================================
OUTPUT_DIR    = ""  # directory where result JSONs will be saved
TCGA_FILE     = "/combined_survival_final.json"  # path to preprocessed TCGA data
# ============================================================


# =============================================================================
# Data loading and splitting
# =============================================================================

def load_and_preprocess_tcga(df_tcga):
    feature_cols = [
        'age_at_index', 'gender_numeric', 'num_treatments',
        'chemotherapy', 'radiation', 'surgery', 'hormone_therapy'
    ]
    X = df_tcga[feature_cols].values
    Y = df_tcga['T'].values
    delta = df_tcga['E'].values
    print(f"Loaded TCGA: {X.shape[0]} samples, event rate {delta.mean():.3f}")
    return X, Y, delta, feature_cols


def create_imbalanced_tcga_splits(df_tcga,
                                  source_size=2000, target_size=50, test_size=1000,
                                  random_state=20):
    np.random.seed(random_state)

    source_data = df_tcga.sample(n=source_size, random_state=random_state)

    target_cancers = ['brca', 'luad']
    target_test_pool = df_tcga[df_tcga['cancer_type'].isin(target_cancers)]
    needed = target_size + test_size
    if len(target_test_pool) < needed:
        needed = len(target_test_pool)
        target_size = min(target_size, needed // 2)
        test_size = needed - target_size

    target_test_subset = target_test_pool.sample(n=needed, random_state=random_state)
    target_data = target_test_subset.iloc[:target_size]
    test_data   = target_test_subset.iloc[target_size:target_size + test_size]

    feature_cols = [
        'age_at_index', 'gender_numeric', 'num_treatments',
        'chemotherapy', 'radiation', 'surgery', 'hormone_therapy'
    ]

    def _split(df):
        return df[feature_cols].values, df['T'].values, df['E'].values

    X_source, Y_source, d_source = _split(source_data)
    X_target, Y_target, d_target = _split(target_data)
    X_test,   Y_test,   d_test   = _split(test_data)

    print(f"Source: {len(X_source)} samples (event rate {d_source.mean():.3f})")
    print(f"Target: {len(X_target)} samples (event rate {d_target.mean():.3f})")
    print(f"Test  : {len(X_test)} samples (event rate {d_test.mean():.3f})")

    return {
        'source': {'X': X_source, 'Y': Y_source, 'delta': d_source},
        'target': {'X': X_target, 'Y': Y_target, 'delta': d_target},
        'test':   {'X': X_test,   'Y': Y_test,   'delta': d_test},
        'feature_names': feature_cols,
    }


# =============================================================================
# Experiment runner
# =============================================================================

def run_experiment_with_models_tcga(X_source, Y_source, delta_source,
                                    X_target, Y_target, delta_target,
                                    X_test, Y_test, delta_test,
                                    sigma_grid, lambda_grid,
                                    n_folds=3, feature_names=None,
                                    apply_loo=True, random_state=None):
    start_time = time.time()
    results = {}

    # Target-Cox
    print("\nFitting Cox (Target Only)...")
    _, c = fit_cox_model(X_target, Y_target, delta_target,
                         X_test, Y_test, delta_test, feature_names)
    results["Cox (Target Only)"] = c
    print(f"C-index = {c:.4f}")

    # Pool-Cox
    print("\nFitting Cox (Naive Pooling)...")
    X_combined = np.vstack([X_source, X_target])
    Y_combined = np.concatenate([Y_source, Y_target])
    d_combined = np.concatenate([delta_source, delta_target])
    _, c = fit_cox_model(X_combined, Y_combined, d_combined,
                         X_test, Y_test, delta_test, feature_names)
    results["Cox (Naive Pooling)"] = c
    print(f"C-index = {c:.4f}")

    # Target-KerKM
    print("\nFitting Kernel-weighted KM (Target Only)...")
    X_empty = np.empty((0, X_target.shape[1]))
    X_target_test = np.vstack([X_target, X_test])
    Y_target_test = np.concatenate([Y_target, Y_test])
    d_target_test = np.concatenate([delta_target, delta_test])
    test_ratio = len(X_test) / len(X_target_test)

    _, _, c = kernel_weighted_transfer_km(
        X_empty, np.array([]), np.array([]),
        X_target_test, Y_target_test, d_target_test,
        sigma_grid=sigma_grid, lambda_grid=[1.0],
        n_folds=n_folds, test_size=test_ratio,
        apply_loo=apply_loo, random_state=random_state)
    results["Kernel-weighted KM (Target Only)"] = c
    print(f"C-index = {c:.4f}")

    # Trans-KerKM
    print("\nFitting Trans-KerKM (proposed)...")
    _, _, c = kernel_weighted_transfer_km(
        X_source, Y_source, delta_source,
        X_target_test, Y_target_test, d_target_test,
        sigma_grid=sigma_grid, lambda_grid=lambda_grid,
        n_folds=n_folds, test_size=test_ratio,
        apply_loo=apply_loo, random_state=random_state)
    results["Feature-based KM (Transfer)"] = c
    print(f"C-index = {c:.4f}")

    print(f"\nTotal runtime: {time.time() - start_time:.2f}s")
    return {'results': results}


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    task_id      = os.environ.get("SLURM_ARRAY_TASK_ID", "1")
    random_state = int(task_id)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, f'tcga_result_{task_id}.json')

    if os.path.exists(output_file):
        print(f"File {output_file} exists. Stopping.", flush=True)
        sys.exit(0)

    # Load data
    try:
        df_tcga = pd.read_json(TCGA_FILE)
        X_all, Y_all, delta_all, feature_names = load_and_preprocess_tcga(df_tcga)
    except FileNotFoundError:
        print(f"ERROR: TCGA data file not found at {TCGA_FILE}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR loading TCGA data: {e}")
        sys.exit(1)

    # Parameter grid (5x5 log-scale, matching paper)
    sigma_grid  = np.logspace(np.log10(0.1), np.log10(1.0),  5).tolist()
    lambda_grid = np.logspace(np.log10(1.0), np.log10(10.0), 5).tolist()

    # Data splits
    data_splits = create_imbalanced_tcga_splits(
        df_tcga, source_size=2000, target_size=50, test_size=1000,
        random_state=random_state)

    X_source = data_splits['source']['X']
    Y_source = data_splits['source']['Y']
    d_source = data_splits['source']['delta']
    X_target = data_splits['target']['X']
    Y_target = data_splits['target']['Y']
    d_target = data_splits['target']['delta']
    X_test   = data_splits['test']['X']
    Y_test   = data_splits['test']['Y']
    d_test   = data_splits['test']['delta']

    # Run experiment
    experiment_results = run_experiment_with_models_tcga(
        X_source, Y_source, d_source,
        X_target, Y_target, d_target,
        X_test,   Y_test,   d_test,
        sigma_grid=sigma_grid, lambda_grid=lambda_grid,
        n_folds=3, feature_names=feature_names,
        apply_loo=True, random_state=random_state)

    final_results = {k: float(v) for k, v in experiment_results['results'].items()
                     if v is not None}

    with open(output_file, 'w') as f:
        json.dump(final_results, f, indent=4)

    print(f"\nResults saved to {output_file}")
    print("Experiment completed!")
