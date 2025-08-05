import json
import pandas as pd

def json_to_dataframe_final(json_data):
    """Convert JSON to DataFrame with final extended features"""
    rows = []
    for patient in json_data:
        row = {}

        # Basic info
        row['submitter_id'] = patient.get('submitter_id')

        # Demographics
        demo = patient.get('demographic', {})
        row['age_at_index'] = demo.get('age_at_index')
        row['vital_status'] = demo.get('vital_status')
        row['gender'] = demo.get('gender')

        # Follow-up time
        diag = patient.get('diagnoses', [{}])[0]
        fu = next((f for f in patient.get('follow_ups', []) if f.get('timepoint_category') == 'Last Contact'), {})
        row['survival_time'] = fu.get('days_to_follow_up') or diag.get('days_to_last_follow_up')

        # Treatments - extended set including num_treatments
        treatments = patient.get('diagnoses', [{}])[0].get('treatments', [])
        treatment_types = [t.get('treatment_type', '') for t in treatments]

        row['num_treatments'] = len(treatments)
        row['chemotherapy'] = sum(1 for t in treatment_types if 'Chemotherapy' in t)
        row['radiation'] = sum(1 for t in treatment_types if 'Radiation' in t)
        row['surgery'] = sum(1 for t in treatment_types if 'Surgery' in t)
        row['hormone_therapy'] = sum(1 for t in treatment_types if 'Hormone' in t)

        rows.append(row)

    return pd.DataFrame(rows)

# Read data
path = 'TCGA/'
file_mapping = {
    'brca': 'clinical.project-tcga-brca.2025-07-25.json',
    'ov': 'clinical.project-tcga-ov.2025-07-25.json', 
    'luad': 'clinical.project-tcga-luad.2025-07-25.json',
    'gbm': 'clinical.project-tcga-gbm.2025-07-25.json',
    'ucec': 'clinical.project-tcga-ucec.2025-07-25.json'
}

data = {}
for cancer_type, filename in file_mapping.items():
    with open(f'{path}{filename}', 'r') as file:
        data[cancer_type] = json.load(file)

# Process data
dfs = []
for name, dataset in data.items():
    df = json_to_dataframe_final(dataset)
    df['cancer_type'] = name
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# Prepare survival data with extended features
survival_df = combined_df[
    (combined_df['survival_time'] > 0) &
    (combined_df['vital_status'].isin(['Alive', 'Dead'])) &
    (combined_df['age_at_index'].notna())
].copy()

survival_df['T'] = survival_df['survival_time']
survival_df['E'] = (survival_df['vital_status'] == 'Dead').astype(int)

# Convert gender to numeric for modeling
survival_df['gender_numeric'] = (survival_df['gender'] == 'male').astype(int)

# Final feature set based on our analysis
final_features = [
    'age_at_index',
    'gender_numeric',
    'num_treatments',
    'chemotherapy',
    'radiation',
    'surgery',
    'hormone_therapy'
]

# Create final dataset
final_df = survival_df[['T', 'E'] + final_features + ['cancer_type', 'gender']].dropna()

# Save DataFrame to JSON
save_path = 'combined_survival_final.json'
final_df.to_json(save_path, orient='records', indent=2)
print(f"Saved: {save_path}")
print(f"Records: {len(final_df)}")