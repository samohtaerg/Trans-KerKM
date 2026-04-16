# -*- coding: utf-8 -*-
"""
Trans_KerKM.py
--------------
Trans-KerKM: kernel primitives, cross-validation, and main estimator.
"""

import numpy as np
from scipy.spatial.distance import cdist
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


# --- Kernel & hazard primitives ---

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


# --- Cross-validation ---

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


# --- Main estimator ---

def kernel_weighted_transfer_km(X_source, Y_source, delta_source,
                                X_target, Y_target, delta_target,
                                sigma_grid=None, lambda_grid=None,
                                n_folds=5, test_size=0.2,
                                apply_loo=True, random_state=None):
    np.random.seed(random_state)

    # Step 1: split BEFORE fitting the scaler (no leakage)
    n_target = X_target.shape[0]
    n_test = int(n_target * test_size)
    all_indices = np.arange(n_target)
    np.random.shuffle(all_indices)
    test_indices  = all_indices[:n_test]
    train_indices = all_indices[n_test:]

    X_train_raw, Y_train, d_train = (X_target[train_indices],
                                     Y_target[train_indices],
                                     delta_target[train_indices])
    X_test_raw,  Y_test,  d_test  = (X_target[test_indices],
                                     Y_target[test_indices],
                                     delta_target[test_indices])

    # Step 2: fit scaler on source + train only
    scaler = StandardScaler()
    X_all = np.vstack([X_source, X_train_raw]) if X_source.shape[0] > 0 else X_train_raw
    scaler.fit(X_all)

    X_source_s = scaler.transform(X_source) if X_source.shape[0] > 0 else X_source
    X_train_s  = scaler.transform(X_train_raw)
    X_test_s   = scaler.transform(X_test_raw)

    if sigma_grid is None:
        sigma_grid = np.logspace(np.log10(0.1), np.log10(1.0), 5).tolist()
    if lambda_grid is None:
        lambda_grid = np.logspace(np.log10(1.0), np.log10(10.0), 5).tolist()

    best_sigma, best_lambda, best_cv = grid_search_cv(
        X_train_s, Y_train, d_train,
        X_source_s, Y_source, delta_source,
        sigma_grid, lambda_grid,
        n_folds=n_folds, apply_loo=apply_loo, random_state=random_state)

    if best_sigma is None or best_lambda is None or np.isnan(best_cv):
        print("All parameter combinations failed. Returning null result.")
        return {'sigma': None, 'lambda': None}, None, None

    test_curves = {}
    for i, x_i in enumerate(X_test_s):
        ht, hv = compute_individualized_hazard(
            x_i, X_train_s, Y_train, d_train,
            X_source_s, Y_source, delta_source,
            best_sigma, best_lambda, apply_loo=apply_loo)
        times, probs = compute_survival_function(ht, hv)
        test_curves[i] = (times, probs)

    test_c_index = compute_c_index_from_survival_curves(test_curves, Y_test, d_test)
    print(f"Test set C-index: {test_c_index:.4f}")

    return {'sigma': best_sigma, 'lambda': best_lambda}, None, test_c_index
