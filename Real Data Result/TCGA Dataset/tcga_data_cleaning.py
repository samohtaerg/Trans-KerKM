# -*- coding: utf-8 -*-
"""
tcga_data_cleaning.py
---------------------
Preprocesses raw TCGA clinical JSON files into a single combined survival
dataset saved as combined_survival_final.json.

Usage
-----
    python tcga_data_cleaning.py

Edit the USER CONFIGURATION block below before running.
"""

import json
import pandas as pd

# ============================================================
# USER CONFIGURATION — edit these paths before running
# ============================================================
RAW_DATA_PATH = "Real Data Result/TCGA Dataset/Raw Data/"  # directory containing the raw clinical JSON files
SAVE_PATH     = "combined_survival_final.json"              # output path for the combined survival dataset

# File mapping: update the filenames to match your downloaded files
# (filenames include the download date, e.g. clinical.project-tcga-brca.2025-07-25.json)
FILE_MAPPING = {
    'brca': 'clinical.project-tcga-brca.2025-07-25.json',
    'ov':   'clinical.project-tcga-ov.2025-07-25.json',
    'luad': 'clinical.project-tcga-luad.2025-07-25.json',
    'gbm':  'clinical.project-tcga-gbm.2025-07-25.json',
    'ucec': 'clinical.project-tcga-ucec.2025-07-25.json',
}
# ============================================================


def json_to_dataframe_final(json_data):
    """Convert raw TCGA clinical JSON to a DataFrame with survival outcomes and treatment features."""
    rows = []
    for patient in json_data:
        row = {}
        row['submitter_id'] = patient.get('submitter_id')

        demo = patient.get('demographic', {})
        row['age_at_index'] = demo.get('age_at_index')
        row['vital_status'] = demo.get('vital_status')
        row['gender'] = demo.get('gender')

        diag = patient.get('diagnoses', [{}])[0]
        fu = next((f for f in patient.get('follow_ups', [])
                   if f.get('timepoint_category') == 'Last Contact'), {})
        row['survival_time'] = fu.get('days_to_follow_up') or diag.get('days_to_last_follow_up')

        treatments = patient.get('diagnoses', [{}])[0].get('treatments', [])
        treatment_types = [t.get('treatment_type', '') for t in treatments]
        row['num_treatments']  = len(treatments)
        row['chemotherapy']    = sum(1 for t in treatment_types if 'Chemotherapy' in t)
        row['radiation']       = sum(1 for t in treatment_types if 'Radiation' in t)
        row['surgery']         = sum(1 for t in treatment_types if 'Surgery' in t)
        row['hormone_therapy'] = sum(1 for t in treatment_types if 'Hormone' in t)

        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":

    # Load raw data
    data = {}
    for cancer_type, filename in FILE_MAPPING.items():
        filepath = f'{RAW_DATA_PATH}{filename}'
        with open(filepath, 'r') as f:
            data[cancer_type] = json.load(f)
        print(f"Loaded {cancer_type}: {len(data[cancer_type])} records")

    # Process and combine
    dfs = []
    for name, dataset in data.items():
        df = json_to_dataframe_final(dataset)
        df['cancer_type'] = name
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)

    # Filter and prepare survival outcomes
    survival_df = combined_df[
        (combined_df['survival_time'] > 0) &
        (combined_df['vital_status'].isin(['Alive', 'Dead'])) &
        (combined_df['age_at_index'].notna())
    ].copy()

    survival_df['T'] = survival_df['survival_time']
    survival_df['E'] = (survival_df['vital_status'] == 'Dead').astype(int)
    survival_df['gender_numeric'] = (survival_df['gender'] == 'male').astype(int)

    final_features = [
        'age_at_index', 'gender_numeric', 'num_treatments',
        'chemotherapy', 'radiation', 'surgery', 'hormone_therapy'
    ]
    final_df = survival_df[['T', 'E'] + final_features + ['cancer_type', 'gender']].dropna()

    # Save
    final_df.to_json(SAVE_PATH, orient='records', indent=2)
    print(f"\nSaved: {SAVE_PATH}")
    print(f"Records: {len(final_df)}")
