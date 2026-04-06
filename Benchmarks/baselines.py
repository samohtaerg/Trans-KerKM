# -*- coding: utf-8 -*-
"""
baselines.py
------------
Baseline models for comparison in Trans-KerKM experiments.
"""

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


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
