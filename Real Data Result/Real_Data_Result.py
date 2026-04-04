# -*- coding: utf-8 -*-
"""
Real Data Result.py
-------------------
Trans-KerKM experiment on TCGA real data.

The seed is read from $SLURM_ARRAY_TASK_ID (defaults to 1 for local runs).
Run with seeds 0-99 via SLURM array job.

Edit the USER CONFIGURATION block below before running.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from lifelines.utils import concordance_index
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import time
import os
import json
import sys

# ============================================================
# USER CONFIGURATION — edit these paths before running
# ============================================================
OUTPUT_DIR    = "/scratch/<your_netid>/tcgaI/"              # directory where result JSONs will be saved
TCGA_FILE     = "/scratch/<your_netid>/combined_survival_final.json"  # path to preprocessed TCGA data
# ============================================================


# =============================================================================
# Core functions
# =============================================================================

def gaussian_kernel(X1, X2, sigma):
    dists = cdist(X1, X2, metric='sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))


def compute_individualized_hazard(x_i, X_train, Y_train, delta_train,
                                  X_source, Y_source, delta_source,
                                  sigma, lambda_value, apply_loo=True):
    x_i = np.atleast_2d(x_i)
    X_train = np.atleast_2d(X_train)
    X_source = np.atleast_2d(X_source)

    K_target = gaussian_kernel(x_i, X_train, sigma).flatten()
    K_source = gaussian_kernel(x_i, X_source, sigma).flatten()

    if apply_loo:
        self_idx = np.where(K_target > 0.999)[0]
        if len(self_idx) > 0:
            K_target[self_idx] = 0

    unique_times = np.sort(np.unique(np.concatenate([Y_train, Y_source])))
    hazard_values = []

    for l, t_l in enumerate(unique_times):
        t_prev = 0 if l == 0 else unique_times[l - 1]

        target_events_sum = np.sum(K_target * (delta_train == 1) * (Y_train == t_l))
        source_events_sum = np.sum(K_source * (delta_source == 1) * (Y_source == t_l))
        numerator = target_events_sum + source_events_sum / lambda_value

        target_at_risk_sum = np.sum(K_target * (Y_train > t_prev))
        source_at_risk_sum = np.sum(K_source * (Y_source > t_prev))
        denominator = target_at_risk_sum + source_at_risk_sum / lambda_value

        hazard_values.append(numerator / denominator if denominator > 0 else 0)

    return unique_times, np.array(hazard_values)


def compute_survival_function(hazard_times, hazard_values):
    n_times = len(hazard_times)
    survival_probs = np.ones(n_times + 1)
    times = np.concatenate(([0], hazard_times))
    for i in range(n_times):
        survival_probs[i + 1] = survival_probs[i] * (1 - hazard_values[i])
    survival_probs = np.clip(survival_probs, 0, 1)
    return times, survival_probs


def compute_c_index_from_survival_curves(survival_curves, Y_true, delta):
    n_samples = len(Y_true)
    risk_scores = np.zeros(n_samples)
    for i in range(n_samples):
        if i not in survival_curves:
            continue
        times, probs = survival_curves[i]
        if len(times) < 2 or len(probs) < 2 or np.any(np.isnan(probs)):
            risk_scores[i] = 0
            continue
        time_diffs = np.diff(times)
        avg_probs = (probs[:-1] + probs[1:]) / 2
        risk_scores[i] = np.sum(avg_probs * time_diffs)
    try:
        c_index = concordance_index(Y_true, risk_scores, delta)
    except ZeroDivisionError:
        print("Warning: No admissible pairs for C-index. Returning NaN.")
        c_index = np.nan
    return c_index


def grid_search_cv(X_train, Y_train, delta_train,
                   X_source, Y_source, delta_source,
                   sigma_grid, lambda_grid,
                   n_folds=5, apply_loo=True, random_state=None):
    best_sigma, best_lambda, best_c_index = None, None, 0.0
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for sigma in sigma_grid:
        for lambda_value in lambda_grid:
            fold_scores = []
            for train_idx, val_idx in kf.split(X_train):
                X_tr, Y_tr, d_tr = X_train[train_idx], Y_train[train_idx], delta_train[train_idx]
                X_val, Y_val, d_val = X_train[val_idx], Y_train[val_idx], delta_train[val_idx]
                curves = {}
                for i, x_i in enumerate(X_val):
                    ht, hv = compute_individualized_hazard(
                        x_i, X_tr, Y_tr, d_tr,
                        X_source, Y_source, delta_source,
                        sigma, lambda_value, apply_loo=apply_loo)
                    times, probs = compute_survival_function(ht, hv)
                    curves[i] = (times, probs)
                fold_scores.append(
                    compute_c_index_from_survival_curves(curves, Y_val, d_val))
            mean_score = np.mean(fold_scores)
            print(f"sigma={sigma:.4f}, lambda={lambda_value:.4f}: CV C-index={mean_score:.4f}")
            if mean_score > best_c_index:
                best_c_index = mean_score
                best_sigma = sigma
                best_lambda = lambda_value

    print(f"Best: sigma={best_sigma}, lambda={best_lambda}, C-index={best_c_index:.4f}")
    return best_sigma, best_lambda, best_c_index


def kernel_weighted_transfer_km(X_source, Y_source, delta_source,
                                X_target, Y_target, delta_target,
                                sigma_grid=None, lambda_grid=None,
                                n_folds=5, test_size=0.2,
                                apply_loo=True, random_state=None):
    np.random.seed(random_state)

    scaler = StandardScaler()
    X_all = np.vstack([X_source, X_target]) if X_source.shape[0] > 0 else X_target
    scaler.fit(X_all)
    X_source_s = scaler.transform(X_source) if X_source.shape[0] > 0 else X_source
    X_target_s = scaler.transform(X_target)

    n_target = X_target_s.shape[0]
    n_test = int(n_target * test_size)
    all_indices = np.arange(n_target)
    np.random.shuffle(all_indices)
    test_indices  = all_indices[:n_test]
    train_indices = all_indices[n_test:]

    X_test,  Y_test,  d_test  = X_target_s[test_indices],  Y_target[test_indices],  delta_target[test_indices]
    X_train, Y_train, d_train = X_target_s[train_indices], Y_target[train_indices], delta_target[train_indices]

    if sigma_grid is None:
        sigma_grid = np.logspace(np.log10(0.1), np.log10(1.0), 5).tolist()
    if lambda_grid is None:
        lambda_grid = np.logspace(np.log10(1.0), np.log10(10.0), 5).tolist()

    best_sigma, best_lambda, best_cv = grid_search_cv(
        X_train, Y_train, d_train,
        X_source_s, Y_source, delta_source,
        sigma_grid, lambda_grid,
        n_folds=n_folds, apply_loo=apply_loo, random_state=random_state)

    if best_sigma is None or best_lambda is None or np.isnan(best_cv):
        print("All parameter combinations failed. Returning null result.")
        return {'sigma': None, 'lambda': None}, None, None

    test_curves = {}
    for i, x_i in enumerate(X_test):
        ht, hv = compute_individualized_hazard(
            x_i, X_train, Y_train, d_train,
            X_source_s, Y_source, delta_source,
            best_sigma, best_lambda, apply_loo=apply_loo)
        times, probs = compute_survival_function(ht, hv)
        test_curves[i] = (times, probs)

    test_c_index = compute_c_index_from_survival_curves(test_curves, Y_test, d_test)
    print(f"Test set C-index: {test_c_index:.4f}")

    return {'sigma': best_sigma, 'lambda': best_lambda}, None, test_c_index


# =============================================================================
# Baseline models
# =============================================================================

def fit_cox_model(X_train, Y_train, delta_train,
                  X_test, Y_test, delta_test, col_names=None):
    if col_names is None:
        col_names = [f"X{i+1}" for i in range(X_train.shape[1])]
    train_df = pd.DataFrame(X_train, columns=col_names)
    train_df['time'] = Y_train
    train_df['event'] = delta_train
    cph = CoxPHFitter()
    cph.fit(train_df, duration_col='time', event_col='event')
    test_df = pd.DataFrame(X_test, columns=col_names)
    risk_scores = cph.predict_partial_hazard(test_df)
    c_index = concordance_index(Y_test, -risk_scores, delta_test)
    return cph, c_index


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
