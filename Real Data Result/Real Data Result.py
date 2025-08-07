# -*- coding: utf-8 -*-
"""Final: Kernel KM (TCGA) 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from scipy.stats import median_abs_deviation
from scipy.spatial.distance import cdist
from scipy.optimize import minimize_scalar
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold
import time
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
import os
import json
import sys
from sklearn.preprocessing import StandardScaler


# Trans-Kernel-KM helper functions
### RBF kernel
"""

def gaussian_kernel(X1, X2, sigma):
    """Compute Gaussian kernel matrix"""
    dists = cdist(X1, X2, metric='sqeuclidean')
    return np.exp(-dists / (2 * sigma ** 2))

"""## Individualized Hazard Function"""

def compute_individualized_hazard(x_i, X_train, Y_train, delta_train,
                                 X_source, Y_source, delta_source,
                                 sigma, lambda_value, apply_loo=True):
    """
    Compute individualized hazard function using kernel-weighted target and source data.

    Parameters:
    x_i: ndarray - feature vector of a single sample
    X_train: ndarray - training feature matrix
    Y_train: ndarray - training survival times
    delta_train: ndarray - training event indicators
    X_source: ndarray - source domain feature matrix
    Y_source: ndarray - source domain survival times
    delta_source: ndarray - source domain event indicators
    sigma: float - kernel bandwidth parameter
    lambda_value: float - source weight parameter
    apply_loo: bool - whether to apply leave-one-out when test sample is in training set

    Returns:
    tuple: (unique_times, hazard_values) - time points and corresponding hazard values
    """
    # Ensure inputs are properly shaped
    x_i = np.atleast_2d(x_i)
    X_train = np.atleast_2d(X_train)
    X_source = np.atleast_2d(X_source)

    # Calculate kernel weights
    K_target = gaussian_kernel(x_i, X_train, sigma).flatten()
    K_source = gaussian_kernel(x_i, X_source, sigma).flatten()

    # Apply Leave-One-Out if requested
    if apply_loo:
        # Find samples very similar to the test sample (likely the same sample)
        # Using a high threshold to identify nearly identical samples
        self_idx = np.where(K_target > 0.999)[0]
        if len(self_idx) > 0:
            # Zero out kernel weights for self (equivalent to removing from training set)
            K_target[self_idx] = 0

    # Get all possible event times
    unique_times = np.sort(np.unique(np.concatenate([Y_train, Y_source])))
    hazard_values = []

    for l, t_l in enumerate(unique_times):
        # Previous time point (t_{ℓ-1})
        t_prev = 0 if l == 0 else unique_times[l-1]

        # Weighted event count in target data at time t_l
        target_events = K_target * (delta_train == 1) * (Y_train == t_l)
        target_events_sum = np.sum(target_events)

        # Weighted event count in source data at time t_l
        source_events = K_source * (delta_source == 1) * (Y_source == t_l)
        source_events_sum = np.sum(source_events) / lambda_value

        # Numerator: weighted event sum
        numerator = target_events_sum + source_events_sum

        # Weighted risk set count in target data at t_prev
        target_at_risk = K_target * (Y_train > t_prev)
        target_at_risk_sum = np.sum(target_at_risk)

        # Weighted risk set count in source data at t_prev
        source_at_risk = K_source * (Y_source > t_prev)
        source_at_risk_sum = np.sum(source_at_risk) / lambda_value

        # Denominator: weighted risk set size
        denominator = target_at_risk_sum + source_at_risk_sum

        # Compute hazard at t_l
        if denominator > 0:
            hazard = numerator / denominator
        else:
            hazard = 0

        hazard_values.append(hazard)

    return unique_times, np.array(hazard_values)

"""## Survival Function


"""

def compute_survival_function(hazard_times, hazard_values):
    """
    Calculate survival function from hazard function

    Args:
        hazard_times: array of time points
        hazard_values: array of hazard values at each time pointI

    Returns:
        tuple: (times, survival_probs) - time points and survival probabilities
    """
    n_times = len(hazard_times)
    # Start with survival prob = 1 at time 0
    survival_probs = np.ones(n_times + 1)
    # Add time 0 to our time points
    times = np.concatenate(([0], hazard_times))

    # Calculate survival probabilities at each time point
    for i in range(n_times):
        # Multiply previous survival prob by (1 - current hazard)
        survival_probs[i+1] = survival_probs[i] * (1 - hazard_values[i])

    # Make sure probabilities stay between 0 and 1
    survival_probs = np.clip(survival_probs, 0, 1)

    return times, survival_probs

"""## C-index Function"""

def compute_c_index_from_survival_curves(survival_curves, Y_true, delta):
    """
    Calculate C-index from predicted survival curves

    Args:
        survival_curves: dict where keys are sample indices and values are (times, probs) tuples
        Y_true: array of observed survival times
        delta: array of event indicators (1=event occurred, 0=censored)

    Returns:
        float: C-index value
    """
    n_samples = len(Y_true)

    # Calculate expected survival time as risk score
    risk_scores = np.zeros(n_samples)

    for i in range(n_samples):
        if i not in survival_curves:
            continue

        times, probs = survival_curves[i]

        # Calculate expected survival time using the area under the curve
        time_diffs = np.diff(times)
        avg_probs = (probs[:-1] + probs[1:]) / 2
        expected_time = np.sum(avg_probs * time_diffs)

        # FIXED: Use expected survival time directly as the risk score
        # Longer expected survival = lower risk
        risk_scores[i] = expected_time

    # Calculate C-index using lifelines or our custom function
    try:
        from lifelines.utils import concordance_index
        c_index = concordance_index(Y_true, risk_scores, delta)
    except ImportError:
        c_index = compute_c_index(Y_true, risk_scores, delta)

    return c_index

def compute_c_index_from_survival_curves(survival_curves, Y_true, delta):
    """
    Calculate C-index from predicted survival curves

    Args:
        survival_curves: dict where keys are sample indices and values are (times, probs) tuples
        Y_true: array of observed survival times
        delta: array of event indicators (1=event occurred, 0=censored)

    Returns:
        float: C-index value
    """
    n_samples = len(Y_true)

    # Calculate expected survival time as risk score
    risk_scores = np.zeros(n_samples)

    for i in range(n_samples):
        if i not in survival_curves:
            continue

        times, probs = survival_curves[i]

        if len(times) < 2 or len(probs) < 2 or np.any(np.isnan(probs)):
            risk_scores[i] = 0  # or np.nan
            continue

        # Expected survival time = area under the survival curve
        time_diffs = np.diff(times)
        avg_probs = (probs[:-1] + probs[1:]) / 2
        expected_time = np.sum(avg_probs * time_diffs)

        risk_scores[i] = expected_time

    try:
        from lifelines.utils import concordance_index
        c_index = concordance_index(Y_true, risk_scores, delta)
    except ZeroDivisionError:
        print("Warning: No admissible pairs for C-index. Returning NaN.")
        c_index = np.nan
    except ImportError:
        c_index = compute_c_index(Y_true, risk_scores, delta)

    return c_index

"""## K-fold Cross Validation"""

def grid_search_cv(X_train, Y_train, delta_train, X_source, Y_source, delta_source,
                  sigma_grid, lambda_grid, n_folds=5, apply_loo=True, random_state=None):
    """
    Perform grid search with cross-validation to find optimal parameters

    Args:
        X_train: feature matrix of target training data
        Y_train: survival times of target training data
        delta_train: event indicators of target training data
        X_source: feature matrix of source data
        Y_source: survival times of source data
        delta_source: event indicators of source data
        sigma_grid: list of bandwidth parameters to try
        lambda_grid: list of source weight parameters to try
        n_folds: number of cross-validation folds
        apply_loo: whether to apply leave-one-out when computing hazard
        random_state: random seed for reproducibility

    Returns:
        tuple: (best_sigma, best_lambda, best_c_index) - optimal parameters and score
    """
    # Initialize best parameters and score
    best_sigma = None
    best_lambda = None
    best_c_index = 0

    # Create k-fold cross-validation splits
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Store results for each parameter combination
    results = []

    # Loop through all parameter combinations
    for sigma in sigma_grid:
        for lambda_value in lambda_grid:
            fold_c_indices = []

            # Perform k-fold cross-validation
            for train_idx, val_idx in kf.split(X_train):
                # Split data into training and validation sets
                X_train_fold = X_train[train_idx]
                Y_train_fold = Y_train[train_idx]
                delta_train_fold = delta_train[train_idx]

                X_val_fold = X_train[val_idx]
                Y_val_fold = Y_train[val_idx]
                delta_val_fold = delta_train[val_idx]

                # Generate survival curves for validation samples
                survival_curves = {}

                for i, x_i in enumerate(X_val_fold):
                    # Compute hazard function with LOO option
                    hazard_times, hazard_values = compute_individualized_hazard(
                        x_i, X_train_fold, Y_train_fold, delta_train_fold,
                        X_source, Y_source, delta_source,
                        sigma, lambda_value, apply_loo=apply_loo
                    )

                    # Convert to survival function
                    times, survival_probs = compute_survival_function(hazard_times, hazard_values)

                    # Store survival curve
                    survival_curves[i] = (times, survival_probs)

                # Compute C-index for this fold
                fold_c_index = compute_c_index_from_survival_curves(
                    survival_curves, Y_val_fold, delta_val_fold
                )

                fold_c_indices.append(fold_c_index)

            # Average C-index across folds
            mean_c_index = np.mean(fold_c_indices)

            # Store results
            results.append({
                'sigma': sigma,
                'lambda': lambda_value,
                'c_index': mean_c_index,
                'fold_scores': fold_c_indices
            })

            print(f"σ={sigma}, λ={lambda_value}: C-index = {mean_c_index:.4f}")

            # Update best parameters if better performance found
            if mean_c_index > best_c_index:
                best_c_index = mean_c_index
                best_sigma = sigma
                best_lambda = lambda_value

    print(f"\nBest parameters: σ={best_sigma}, λ={best_lambda}, C-index={best_c_index:.4f}")

    return best_sigma, best_lambda, best_c_index

"""# Trans-Kernel-KM Model"""

def kernel_weighted_transfer_km(X_source, Y_source, delta_source,
                               X_target, Y_target, delta_target,
                               sigma_grid=None, lambda_grid=None,
                               n_folds=5, test_size=0.2, apply_loo=True, random_state=None):
    """
    Kernel-Weighted Transfer Kaplan-Meier Estimation with Cross-Validated Source Weighting

    Args:
        X_source: feature matrix of source data
        Y_source: survival times of source data
        delta_source: event indicators of source data
        X_target: feature matrix of target data
        Y_target: survival times of target data
        delta_target: event indicators of target data
        sigma_grid: list of bandwidth parameters to search (default: auto-generated)
        lambda_grid: list of source weight parameters to search (default: auto-generated)
        n_folds: number of cross-validation folds
        test_size: proportion of target data to use as test set
        apply_loo: whether to apply leave-one-out when computing hazard (default: True)
        random_state: random seed for reproducibility

    Returns:
        tuple: (best_params, final_model, test_c_index) - optimal parameters, final model and test performance
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)

    # Apply standardization to features
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_all = np.vstack([X_source, X_target])
    scaler.fit(X_all)
    if X_source.shape[0] > 0:
        X_source = scaler.transform(X_source)
    else:
        X_source = np.empty((0, X_target.shape[1]))  # optional fallback
    X_target = scaler.transform(X_target)

    # Split target data into training and test sets
    n_target = X_target.shape[0]
    n_test = int(n_target * test_size)

    # Randomly select indices for test set
    all_indices = np.arange(n_target)
    np.random.shuffle(all_indices)
    test_indices = all_indices[:n_test]
    train_indices = all_indices[n_test:]

    X_test = X_target[test_indices]
    Y_test = Y_target[test_indices]
    delta_test = delta_target[test_indices]

    X_train = X_target[train_indices]
    Y_train = Y_target[train_indices]
    delta_train = delta_target[train_indices]

    # Generate default parameter grids if not provided
    if sigma_grid is None:
        # Generate sigma grid based on median distance in data
        median_dist = np.median(cdist(X_train, X_train, metric='euclidean')[np.triu_indices(X_train.shape[0], k=1)])
        sigma_grid = [median_dist * factor for factor in [0.1, 0.5, 1.0, 2.0, 5.0]]

    if lambda_grid is None:
        # Generate lambda grid with different source weighting factors
        lambda_grid = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]

    print("Starting parameter search with grid:")
    print(f"Sigma grid: {sigma_grid}")
    print(f"Lambda grid: {lambda_grid}")
    print(f"Cross-validation: {n_folds} folds")
    print(f"Apply LOO: {apply_loo}")

    # Find optimal parameters through grid search
    best_sigma, best_lambda, best_cv_score = grid_search_cv(
        X_train, Y_train, delta_train,
        X_source, Y_source, delta_source,
        sigma_grid, lambda_grid, n_folds=n_folds, apply_loo=apply_loo, random_state=random_state
    )

    # ===== PATCH START: fallback if no valid parameters found =====
    if best_sigma is None or best_lambda is None or np.isnan(best_cv_score):
        print("All parameter combinations failed. Returning null result.")
        return {'sigma': None, 'lambda': None}, None, None
    # ===== PATCH END =====

    # Train final model using best parameters on full training data
    print("\nTraining final model with best parameters...")

    # Generate survival curves for test samples
    test_survival_curves = {}

    for i, x_i in enumerate(X_test):
        # Compute hazard function with LOO option
        hazard_times, hazard_values = compute_individualized_hazard(
            x_i, X_train, Y_train, delta_train,
            X_source, Y_source, delta_source,
            best_sigma, best_lambda, apply_loo=apply_loo
        )

        # Convert to survival function
        times, survival_probs = compute_survival_function(hazard_times, hazard_values)

        # Store survival curve
        test_survival_curves[i] = (times, survival_probs)

    # Evaluate final model on test set
    test_c_index = compute_c_index_from_survival_curves(
        test_survival_curves, Y_test, delta_test
    )

    print(f"Test set C-index: {test_c_index:.4f}")

    # Create final model object (parameters and predict function)
    best_params = {'sigma': best_sigma, 'lambda': best_lambda}

    # Define prediction function for new data
    def predict_survival(X_new, apply_loo_pred=apply_loo):
        """
        Predict survival curves for new samples

        Args:
            X_new: feature matrix for new samples
            apply_loo_pred: whether to apply leave-one-out (default: same as training)

        Returns:
            dict: keys are sample indices, values are (times, probs) tuples
        """
        # Apply the same standardization to new data
        X_new_scaled = predict_survival.scaler.transform(X_new)

        survival_curves = {}

        for i, x_i in enumerate(X_new_scaled):
            # Compute hazard function with LOO option
            hazard_times, hazard_values = compute_individualized_hazard(
                x_i, X_train, Y_train, delta_train,
                X_source, Y_source, delta_source,
                best_sigma, best_lambda, apply_loo=apply_loo_pred
            )

            # Convert to survival function
            times, survival_probs = compute_survival_function(hazard_times, hazard_values)

            # Store survival curve
            survival_curves[i] = (times, survival_probs)

        return survival_curves

    # Create final model as callable function with attached parameters
    final_model = predict_survival
    final_model.params = best_params
    # Store LOO setting in the model
    final_model.apply_loo = apply_loo
    # Store the scaler in the model for future predictions
    final_model.scaler = scaler

    return best_params, final_model, test_c_index

"""# Baseline Model

## Cox Model
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

def fit_cox_model(X_train, Y_train, delta_train, X_test, Y_test, delta_test, col_names=None):
   """
   Fit a Cox Proportional Hazards model and evaluate its performance

   Args:
       X_train: training feature matrix
       Y_train: training survival times
       delta_train: training event indicators
       X_test: test feature matrix
       Y_test: test survival times
       delta_test: test event indicators
       col_names: feature column names

   Returns:
       tuple: (model, c_index) - fitted model and test set C-index
   """
   # Create a DataFrame for the training data
   if col_names is None:
       col_names = [f"X{i+1}" for i in range(X_train.shape[1])]

   train_df = pd.DataFrame(X_train, columns=col_names)
   train_df['time'] = Y_train
   train_df['event'] = delta_train

   # Fit the Cox model
   cph = CoxPHFitter()
   cph.fit(train_df, duration_col='time', event_col='event')

   # Create a DataFrame for the test data
   test_df = pd.DataFrame(X_test, columns=col_names)

   # Predict risk scores
   risk_scores = cph.predict_partial_hazard(test_df)

   # Compute C-index (negative risk scores because higher risk = shorter survival time)
   c_index = concordance_index(Y_test, -risk_scores, delta_test)

   return cph, c_index

"""## Naive Pooling-Cox"""

def fit_transfer_cox_model(X_source, Y_source, delta_source,
                          X_target, Y_target, delta_target,
                          test_size=0.3, random_state=None, col_names=None):
   """
   Fit Cox model with transfer learning (simply combining source and target data)

   Args:
       X_source: source feature matrix
       Y_source: source survival times
       delta_source: source event indicators
       X_target: target feature matrix
       Y_target: target survival times
       delta_target: target event indicators
       test_size: proportion of target data to use as test set
       random_state: random seed
       col_names: feature column names

   Returns:
       tuple: (model, c_index) - fitted model and test set C-index
   """
   # Split target data into train and test sets
   X_train_target, X_test, Y_train_target, Y_test, delta_train_target, delta_test = train_test_split(
       X_target, Y_target, delta_target, test_size=test_size, random_state=random_state
   )

   # Create feature column names if not provided
   if col_names is None:
       col_names = [f"X{i+1}" for i in range(X_source.shape[1])]

   # Create DataFrames
   source_df = pd.DataFrame(X_source, columns=col_names)
   source_df['time'] = Y_source
   source_df['event'] = delta_source

   target_train_df = pd.DataFrame(X_train_target, columns=col_names)
   target_train_df['time'] = Y_train_target
   target_train_df['event'] = delta_train_target

   # Combine source and target training data
   combined_df = pd.concat([source_df, target_train_df])

   # Fit the Cox model on combined data
   cph = CoxPHFitter()
   cph.fit(combined_df, duration_col='time', event_col='event')

   # Create test DataFrame
   test_df = pd.DataFrame(X_test, columns=col_names)

   # Predict risk scores
   risk_scores = cph.predict_partial_hazard(test_df)

   # Compute C-index
   c_index = concordance_index(Y_test, -risk_scores, delta_test)

   return cph, c_index, X_test, Y_test, delta_test

"""# TCGA Data Loading"""

def load_and_preprocess_tcga(df_tcga):
   """
   Load and preprocess TCGA dataset from cleaned dataframe

   Args:
       df_tcga: pre-cleaned TCGA dataframe

   Returns:
       tuple: (X, Y, delta, feature_names) - processed data
   """
   print(f"Loaded TCGA dataset with {len(df_tcga)} samples")
   print("Original data info:")
   print(df_tcga.info())

   # Extract features, survival times, and event indicators
   feature_cols = [
    'age_at_index', 'gender_numeric', 'num_treatments',
    'chemotherapy', 'radiation', 'surgery', 'hormone_therapy'
]

   # No need for dummy variables - all features are numeric
   X = df_tcga[feature_cols].values
   feature_names = feature_cols

   Y = df_tcga['T'].values
   delta = df_tcga['E'].values

   print(f"Features shape: {X.shape}")
   print(f"Survival times shape: {Y.shape}")
   print(f"Event rate: {np.mean(delta):.3f}")

   print(f"Feature ranges:")
   for i, col in enumerate(feature_names):
       print(f"  {col}: {X[:, i].min():.1f} - {X[:, i].max():.1f}")

   return X, Y, delta, feature_names

"""### Split"""

def create_tcga_splits(df_tcga,
                    source_size=200, target_size=50, test_size=100,
                    random_state=20):
  """
  Create source/target/test splits from TCGA data for transfer learning experiments
  All splits contain same cancer type distribution (no domain shift)

  Args:
      df_tcga: cleaned TCGA dataframe
      source_size: number of samples for source domain (default: 2000)
      target_size: number of samples for target domain (default: 50)
      test_size: number of samples for test set (default: 1000)
      random_state: random seed

  Returns:
      dict: containing source, target, and test data
  """
  np.random.seed(random_state)

  total_size = source_size + target_size + test_size
  print(f"Creating TCGA splits with {total_size} total samples")
  print(f"  Source: {source_size}, Target: {target_size}, Test: {test_size}")

  # Check if we have enough samples
  if total_size > len(df_tcga):
      print(f"Warning: Need {total_size} samples but only {len(df_tcga)} available")
      ratio = len(df_tcga) / total_size
      source_size = int(source_size * ratio)
      target_size = int(target_size * ratio)
      test_size = len(df_tcga) - source_size - target_size
      total_size = len(df_tcga)
      print(f"Adjusted sizes - Source: {source_size}, Target: {target_size}, Test: {test_size}")

  # Random sampling from all data (no stratification by cancer type)
  final_subset = df_tcga.sample(n=total_size, random_state=random_state).reset_index(drop=True)

  print(f"Final subset: {len(final_subset):,} samples, event rate: {final_subset['E'].mean():.3f}")

  # Extract features and outcomes
  feature_cols = [
    'age_at_index', 'gender_numeric', 'num_treatments',
    'chemotherapy', 'radiation', 'surgery', 'hormone_therapy'
]
  X = final_subset[feature_cols].values
  feature_names = feature_cols
  Y = final_subset['T'].values
  delta = final_subset['E'].values

  # Split indices
  source_indices = np.arange(source_size)
  target_indices = np.arange(source_size, source_size + target_size)
  test_indices = np.arange(source_size + target_size, source_size + target_size + test_size)

  # Create splits
  X_source = X[source_indices]
  Y_source = Y[source_indices]
  delta_source = delta[source_indices]

  X_target = X[target_indices]
  Y_target = Y[target_indices]
  delta_target = delta[target_indices]

  X_test = X[test_indices]
  Y_test = Y[test_indices]
  delta_test = delta[test_indices]

  print(f"Source domain: {len(X_source)} samples (event rate: {np.mean(delta_source):.3f})")
  print(f"Target domain: {len(X_target)} samples (event rate: {np.mean(delta_target):.3f})")
  print(f"Test domain: {len(X_test)} samples (event rate: {np.mean(delta_test):.3f})")

  return {
      'source': {
          'X': X_source,
          'Y': Y_source,
          'delta': delta_source,
          'n_samples': len(X_source),
          'event_rate': np.mean(delta_source)
      },
      'target': {
          'X': X_target,
          'Y': Y_target,
          'delta': delta_target,
          'n_samples': len(X_target),
          'event_rate': np.mean(delta_target)
      },
      'test': {
          'X': X_test,
          'Y': Y_test,
          'delta': delta_test,
          'n_samples': len(X_test),
          'event_rate': np.mean(delta_test)
      },
      'feature_names': feature_names,
      'split_info': {
          'total_samples': total_size,
          'actual_event_rate': np.mean(delta),
          'random_state': random_state
      }
  }

"""## Imbalanced Split"""

def create_imbalanced_tcga_splits(df_tcga,
                                 source_size=2000, target_size=50, test_size=1000,
                                 random_state=20):
    """
    Create imbalanced source/target splits to demonstrate domain mismatch
    Source: All cancer types (diverse)
    Target: Only BRCA + GBM (treatment-pattern mismatch)
    """
    np.random.seed(random_state)

    print(f"Creating IMBALANCED TCGA splits:")
    print(f"  Source: {source_size} samples from ALL cancer types")
    print(f"  Target: {target_size} samples from BRCA + GBM only")
    print(f"  Test: {test_size} samples from BRCA + GBM only")

    # Source: Sample from ALL cancer types
    source_data = df_tcga.sample(n=source_size, random_state=random_state)

    # Target+Test: Only BRCA and GBM
    target_cancers = ['brca', 'luad']
    target_test_pool = df_tcga[df_tcga['cancer_type'].isin(target_cancers)]

    # Sample target + test from the restricted pool
    needed_samples = target_size + test_size
    if len(target_test_pool) < needed_samples:
        print(f"Warning: Only {len(target_test_pool)} BRCA+GBM samples available, need {needed_samples}")
        needed_samples = len(target_test_pool)
        target_size = min(target_size, needed_samples // 2)
        test_size = needed_samples - target_size

    target_test_subset = target_test_pool.sample(n=needed_samples, random_state=random_state)

    # Split into target and test
    target_data = target_test_subset.iloc[:target_size]
    test_data = target_test_subset.iloc[target_size:target_size+test_size]

    print(f"Cancer distribution in source: {source_data['cancer_type'].value_counts().to_dict()}")
    print(f"Cancer distribution in target: {target_data['cancer_type'].value_counts().to_dict()}")
    print(f"Cancer distribution in test: {test_data['cancer_type'].value_counts().to_dict()}")

    # Extract features using the same logic as original function
    feature_cols = [
        'age_at_index', 'gender_numeric', 'num_treatments',
        'chemotherapy', 'radiation', 'surgery', 'hormone_therapy'
    ]

    # Process each split
    X_source = source_data[feature_cols].values
    Y_source = source_data['T'].values
    delta_source = source_data['E'].values

    X_target = target_data[feature_cols].values
    Y_target = target_data['T'].values
    delta_target = target_data['E'].values

    X_test = test_data[feature_cols].values
    Y_test = test_data['T'].values
    delta_test = test_data['E'].values

    print(f"Source domain: {len(X_source)} samples (event rate: {np.mean(delta_source):.3f})")
    print(f"Target domain: {len(X_target)} samples (event rate: {np.mean(delta_target):.3f})")
    print(f"Test domain: {len(X_test)} samples (event rate: {np.mean(delta_test):.3f})")

    return {
        'source': {
            'X': X_source, 'Y': Y_source, 'delta': delta_source,
            'n_samples': len(X_source), 'event_rate': np.mean(delta_source)
        },
        'target': {
            'X': X_target, 'Y': Y_target, 'delta': delta_target,
            'n_samples': len(X_target), 'event_rate': np.mean(delta_target)
        },
        'test': {
            'X': X_test, 'Y': Y_test, 'delta': delta_test,
            'n_samples': len(X_test), 'event_rate': np.mean(delta_test)
        },
        'feature_names': feature_cols,
        'split_info': {
            'total_samples': len(X_source) + len(X_target) + len(X_test),
            'imbalance_type': 'cancer_type_mismatch',
            'source_cancers': 'all', 'target_cancers': target_cancers,
            'random_state': random_state
        }
    }

"""# Simulation Function"""

def run_experiment_with_models_tcga(X_source, Y_source, delta_source,
                                  X_target, Y_target, delta_target,
                                  X_test, Y_test, delta_test,
                                  sigma_grid, lambda_grid,
                                  n_folds=3, feature_names=None,
                                  apply_loo=True, random_state=None):
   """
   Run model experiments on TCGA data comparing different approaches

   Args:
       X_source, Y_source, delta_source: source domain data
       X_target, Y_target, delta_target: target domain data
       X_test, Y_test, delta_test: independent test data
       sigma_grid, lambda_grid: hyperparameter grids
       n_folds: CV folds for hyperparameter tuning
       feature_names: feature column names
       apply_loo: whether to apply leave-one-out
       random_state: random seed
   """
   start_time = time.time()

   print(f"Running TCGA experiments:")
   print(f"  Source: {len(X_source)} samples (event rate: {np.mean(delta_source):.3f})")
   print(f"  Target: {len(X_target)} samples (event rate: {np.mean(delta_target):.3f})")
   print(f"  Test: {len(X_test)} samples (event rate: {np.mean(delta_test):.3f})")

   results = {}

   # Baseline 1: Cox model on target data only
   print("\nFitting Cox model on target data only...")
   cox_model, cox_c_index = fit_cox_model(
       X_target, Y_target, delta_target, X_test, Y_test, delta_test, feature_names
   )
   results["Cox (Target Only)"] = cox_c_index
   print(f"Cox model on target data: C-index = {cox_c_index:.4f}")

   # Baseline 2: Cox model with naive transfer (combined data)
   print("\nFitting Cox model with naive transfer...")
   # Combine source and target for training
   X_combined = np.vstack([X_source, X_target])
   Y_combined = np.concatenate([Y_source, Y_target])
   delta_combined = np.concatenate([delta_source, delta_target])

   transfer_cox_model, transfer_cox_c_index = fit_cox_model(
       X_combined, Y_combined, delta_combined, X_test, Y_test, delta_test, feature_names
   )
   results["Cox (Naive Transfer)"] = transfer_cox_c_index
   print(f"Cox model with naive transfer: C-index = {transfer_cox_c_index:.4f}")

   # Baseline 3: Kernel-weighted KM using only target data (no source)
   print("\nFitting Kernel-weighted KM (Target Only)...")
   X_source_empty = np.empty((0, X_target.shape[1]))
   Y_source_empty = np.array([])
   delta_source_empty = np.array([])

   # Create a temporary combined target+test for the KM function's internal split
   X_target_test = np.vstack([X_target, X_test])
   Y_target_test = np.concatenate([Y_target, Y_test])
   delta_target_test = np.concatenate([delta_target, delta_test])

   # Calculate test_size ratio for the internal split
   test_size_ratio = len(X_test) / len(X_target_test)

   best_params_target_only, kw_model_target_only, kw_c_index_target_only = kernel_weighted_transfer_km(
       X_source_empty, Y_source_empty, delta_source_empty,
       X_target_test, Y_target_test, delta_target_test,
       sigma_grid=sigma_grid, lambda_grid=[1.0],
       n_folds=n_folds, test_size=test_size_ratio, apply_loo=apply_loo, random_state=random_state
   )

   results["Kernel-weighted KM (Target Only)"] = kw_c_index_target_only
   print(f"Kernel-weighted KM (Target Only): C-index = {kw_c_index_target_only:.4f}")

   # Method 4: Feature-based Kernel KM with transfer learning
   print("\nFitting Feature-based Kernel KM with transfer learning...")
   feature_params, feature_model, feature_c_index = kernel_weighted_transfer_km(
       X_source, Y_source, delta_source,
       X_target_test, Y_target_test, delta_target_test,
       sigma_grid=sigma_grid, lambda_grid=lambda_grid,
       n_folds=n_folds, test_size=test_size_ratio, apply_loo=apply_loo, random_state=random_state
   )

   results["Feature-based KM (Transfer)"] = feature_c_index
   print(f"Feature-based Kernel KM with transfer: C-index = {feature_c_index:.4f}")
   if feature_params['sigma'] is not None:
       print(f"Best parameters: sigma={feature_params['sigma']:.4f}, lambda={feature_params['lambda']:.4f}")

   # Print total runtime
   end_time = time.time()
   print(f"\nTotal experiment runtime: {end_time - start_time:.2f} seconds")

   return {
       'results': results,
       'models': {
           'cox': cox_model,
           'transfer_cox': transfer_cox_model,
           'feature_km': feature_model
       },
       'best_params': {
           'feature_km': feature_params
       },
       'data': {
           'X_source': X_source, 'Y_source': Y_source, 'delta_source': delta_source,
           'X_target': X_target, 'Y_target': Y_target, 'delta_target': delta_target,
           'X_test': X_test, 'Y_test': Y_test, 'delta_test': delta_test
       }
   }

"""# Main()"""

if __name__ == "__main__":
   print("===== Running Transfer Learning Experiment with TCGA Real Data =====")

   # Get SLURM job ID as additional identifier (optional)
   task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "1")
   random_state = int(task_id)

   # Define output directory
   output_dir_name = "tcgaI"  # @param {type:"string"}
   output_dir = f'/scratch/ac10374/{output_dir_name}/'
   print(f"Output directory set to: {output_dir}")
   os.makedirs(output_dir, exist_ok=True)

   output_file = os.path.join(output_dir, f'tcga_result_{task_id}.json')

   # Check if output file already exists
   if os.path.exists(output_file):
       print(f"File {output_file} exists. Stopping the script.", flush=True)
       sys.exit(0)
   else:
       print(f"File does not exist. Continuing the script.", flush=True)

   # ==================== LOAD REAL TCGA DATA ====================
   filename = "combined_survival_final.json"  # @param {type:"string"}
  #  tcga_file_path = f'/content/{filename}'
   tcga_file_path = f'/scratch/ac10374/{filename}'

   try:
       # Load and preprocess TCGA data (only once)
       df_tcga = pd.read_json(tcga_file_path)
       X_all, Y_all, delta_all, feature_names = load_and_preprocess_tcga(df_tcga)
       print(f"Successfully loaded TCGA data with {X_all.shape[0]} samples and {X_all.shape[1]} features")

   except FileNotFoundError:
       print(f"ERROR: TCGA data file not found at {tcga_file_path}")
       print("Please update the tcga_file_path variable with the correct path to your TCGA JSON file")
       sys.exit(1)
   except Exception as e:
       print(f"ERROR loading TCGA data: {e}")
       sys.exit(1)

   # Fixed experiment parameters
   n_folds = 3
   apply_loo = True

   # Model parameter grid
   param_grid_type = "HPC"  # @param ["comprehensive", "testing","HPC"]

   if param_grid_type == "comprehensive":
       sigma_grid = [0.05, 0.1, 0.2, 0.5]
       lambda_grid = [0.5, 1.0, 2.0]
       print("Using comprehensive parameter grid for thorough model evaluation")
   elif param_grid_type == "testing":
       sigma_grid = [0.1]
       lambda_grid = [1.0]
   elif param_grid_type == "HPC":
       # Log-scale grid: 5x5 evenly spaced in log space
       sigma_grid = np.logspace(np.log10(0.1), np.log10(1.0), 5).tolist()
       lambda_grid = np.logspace(np.log10(1.0), np.log10(10.0), 5).tolist()
       print("Using fixed 5x5 log-scale grid for HPC evaluation")
   else:
       raise NotImplementedError(f"Parameter grid type '{param_grid_type}' not implemented")

   try:
       # Create source-target-test splits for transfer learning with current random state
       data_splits = create_imbalanced_tcga_splits(
           df_tcga,
           source_size=2000, target_size=50, test_size=1000,
           random_state=random_state
       )

       X_source = data_splits['source']['X']
       Y_source = data_splits['source']['Y']
       delta_source = data_splits['source']['delta']

       X_target = data_splits['target']['X']
       Y_target = data_splits['target']['Y']
       delta_target = data_splits['target']['delta']

       X_test = data_splits['test']['X']
       Y_test = data_splits['test']['Y']
       delta_test = data_splits['test']['delta']

       print(f"\n{'-'*20} TCGA TRANSFER LEARNING EXPERIMENT {'-'*20}")
       print(f"Source data: {X_source.shape[0]} samples")
       print(f"Target data: {X_target.shape[0]} samples")
       print(f"Test data: {X_test.shape[0]} samples")

       # Run model experiments
       experiment_results = run_experiment_with_models_tcga(
           X_source, Y_source, delta_source,
           X_target, Y_target, delta_target,
           X_test, Y_test, delta_test,
           sigma_grid=sigma_grid, lambda_grid=lambda_grid,
           n_folds=n_folds, feature_names=feature_names,
           apply_loo=apply_loo, random_state=random_state
       )

       results = experiment_results['results']

       # Add simplified results
       final_results = {}
       for key in ['Cox (Target Only)', 'Cox (Naive Transfer)', 'Feature-based KM (Transfer)', 'Kernel-weighted KM (Target Only)']:
           if key in results:
               # Convert float64 to Python float
               final_results[key] = float(results[key])

       # Save results to file
       with open(output_file, 'w') as f:
           json.dump(final_results, f, indent=4)

       print(f"\nResults saved to {output_file}")

   except Exception as e:
       print(f"ERROR: {e}")
       print(f"Skipping and continuing...")

   print(f"\nExperiment completed!")
