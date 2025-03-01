import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Set pandas to show more columns and rows if needed
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)

# Function to load data with progress bar
def load_data(filepath):
    print(f"Loading {os.path.basename(filepath)}...")
    return pd.read_csv(filepath)

# Load datasets
base_path = os.path.expanduser("~/Fluid-Solutions-ML/data/raw/")
items_df = load_data(os.path.join(base_path, "d_items.csv"))
chart_df = load_data(os.path.join(base_path, "chartevents.csv"))
fluid_df = load_data(os.path.join(base_path, "inputevents.csv"))

# Check which version of MIMIC is being used (based on column names)
if 'endtime' in fluid_df.columns:
    # MIMIC-III
    print("Detected MIMIC-III format")
    time_cols_chart = ['charttime', 'storetime']
    time_cols_fluid = ['starttime', 'endtime']
    hadm_id_col = 'hadm_id'
    icustay_id_col = 'icustay_id'
else:
    # MIMIC-IV
    print("Detected MIMIC-IV format")
    time_cols_chart = ['charttime', 'storetime'] 
    time_cols_fluid = ['starttime', 'endtime']
    hadm_id_col = 'hadm_id'
    icustay_id_col = 'stay_id'

# Convert timestamp columns to datetime
def convert_timestamps(df, time_cols):
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')
    return df

chart_df = convert_timestamps(chart_df, time_cols_chart)
fluid_df = convert_timestamps(fluid_df, time_cols_fluid)

# Define vital signs and interventions of interest with MIMIC-specific itemids if known
# These are based on common MIMIC-III item IDs - adjust if you're using MIMIC-IV
vital_sign_items = {
    'central venous pressure': [220074, 113, 220059],  # CVP
    'mean arterial pressure': [220052, 220181, 225312],  # MAP
    'spo2': [220277, 220739],  # SpO2
    'ppv': [220059],  # PPV (Pulse Pressure Variation)
    'systolic blood pressure': [220050, 220179, 225309],  # Systolic BP
    'diastolic blood pressure': [220051, 220180, 225310],  # Diastolic BP
    'heart rate': [220045, 211, 220047]  # Heart Rate
}

# Dialysis and fluid related items
# These would need to be refined based on your specific MIMIC version and research question
dialysis_items = [
    226118,  # Dialysis: Hemodialysis Output
    226499,  # Dialysis Access Site
    225952,  # Dialysate Flow
    225953,  # Dialysate Temperature
    226457   # Dialysis Net
]

# Additional filter for fluid items (typically in inputevents, not d_items)
fluid_keywords = [
    'bolus', 'infusion', 'iv', 'fluid', 'saline', 'lactated', 'dextrose',
    'albumin', 'plasma', 'blood', 'packed', 'prbc', 'ffp'
]

# Find all vital sign item IDs in d_items
print("Mapping item IDs to clinical concepts...")
all_vital_ids = []
vital_id_to_feature = {}

# First use the known item IDs
for feature, ids in vital_sign_items.items():
    for item_id in ids:
        vital_id_to_feature[item_id] = feature
        all_vital_ids.append(item_id)

# Then search through item labels for additional matches
for feature in vital_sign_items.keys():
    mask = items_df['label'].str.lower().str.contains(feature, na=False)
    matching_items = items_df[mask]
    for _, item in matching_items.iterrows():
        if item['itemid'] not in vital_id_to_feature:
            vital_id_to_feature[item['itemid']] = feature
            all_vital_ids.append(item['itemid'])

print(f"Found {len(all_vital_ids)} vital sign item IDs")

# Find dialysis items
dialysis_ids = dialysis_items.copy()
mask = items_df['label'].str.lower().str.contains('dialysis', na=False)
dialysis_items_df = items_df[mask]
for _, item in dialysis_items_df.iterrows():
    if item['itemid'] not in dialysis_ids:
        dialysis_ids.append(item['itemid'])

print(f"Found {len(dialysis_ids)} dialysis-related item IDs")

# Get vital sign measurements
print("Filtering relevant chart events...")
vitals_events = chart_df[
    (chart_df['itemid'].isin(all_vital_ids)) &
    (~chart_df['valuenum'].isna())
].copy()

# Match each vital sign event to its feature category
vitals_events['feature'] = vitals_events['itemid'].map(vital_id_to_feature)

# Filter fluid intake events
# In MIMIC, fluid events are typically in inputevents table
print("Filtering fluid administration events...")
fluid_events = fluid_df.copy()

# Add a flag for dialysis events - these can be in both chartevents or inputevents depending on MIMIC version
dialysis_chart_events = chart_df[chart_df['itemid'].isin(dialysis_ids)].copy()
dialysis_chart_events['is_dialysis'] = True
dialysis_chart_events['amount'] = 0  # Placeholder for dialysis events without specific fluid amounts
dialysis_chart_events['intervention_type'] = 'dialysis'

# Create combined intervention events dataset
if 'ordercategoryname' in fluid_df.columns:  # MIMIC-III structure
    fluid_events['is_dialysis'] = fluid_events['ordercategoryname'].str.lower().str.contains('dialysis', na=False)
    fluid_events['intervention_type'] = np.where(
        fluid_events['is_dialysis'], 
        'dialysis',
        np.where(fluid_events['amount'] > 0, 'fluid_administration', 'fluid_removal')
    )
else:  # MIMIC-IV structure or simplified approach
    # For MIMIC-IV, you may need to adjust based on the actual column names
    fluid_events['is_dialysis'] = fluid_events.apply(
        lambda row: any(word in str(row).lower() for word in ['dialysis', 'dialysate', 'dialyzer']), 
        axis=1
    )
    fluid_events['intervention_type'] = np.where(
        fluid_events['is_dialysis'], 
        'dialysis',
        np.where(fluid_events['amount'] > 0, 'fluid_administration', 'fluid_removal')
    )

# Ensure all necessary columns are present
chart_columns = set(vitals_events.columns)
fluid_columns = set(fluid_events.columns)
dialysis_columns = set(dialysis_chart_events.columns)

# Get common columns between the datasets
common_columns = (chart_columns & fluid_columns & dialysis_columns)
key_columns = ['subject_id', hadm_id_col]
if icustay_id_col in common_columns:
    key_columns.append(icustay_id_col)

# Prepare dialysis events for concatenation
for col in fluid_columns:
    if col not in dialysis_chart_events.columns:
        dialysis_chart_events[col] = np.nan

# Combine fluid events with dialysis events
all_fluid_events = pd.concat([
    fluid_events[list(fluid_columns)], 
    dialysis_chart_events[list(fluid_columns)]
], ignore_index=True)

# Find common patients
print("Finding patients with both vital signs and interventions...")
vitals_patients = set(vitals_events['subject_id'].unique())
fluid_patients = set(all_fluid_events['subject_id'].unique())
common_patients = vitals_patients.intersection(fluid_patients)
print(f"Found {len(common_patients)} patients with both vitals and fluid/dialysis events")

# Filter events for common patients only
vitals_events = vitals_events[vitals_events['subject_id'].isin(common_patients)]
all_fluid_events = all_fluid_events[all_fluid_events['subject_id'].isin(common_patients)]

# Time-based grouping function
def create_time_windows(patient_id, time_window=60):
    """
    Creates time windows for a patient to group vitals with interventions.
    
    Args:
        patient_id: The subject_id of the patient
        time_window: Window size in minutes (before and after intervention)
        
    Returns:
        DataFrame with features and labels for this patient
    """
    patient_vitals = vitals_events[vitals_events['subject_id'] == patient_id].copy()
    patient_fluids = all_fluid_events[all_fluid_events['subject_id'] == patient_id].copy()
    
    if patient_vitals.empty or patient_fluids.empty:
        return None
    
    # Get all hospital admissions for this patient
    hadm_ids = patient_fluids[hadm_id_col].unique()
    
    # Create a list to hold the datapoints
    datapoints = []
    
    # Process each hospital admission separately
    for hadm_id in hadm_ids:
        hadm_vitals = patient_vitals[patient_vitals[hadm_id_col] == hadm_id].copy()
        hadm_fluids = patient_fluids[patient_fluids[hadm_id_col] == hadm_id].copy()
        
        if hadm_vitals.empty or hadm_fluids.empty:
            continue
        
        # Sort by time
        hadm_vitals = hadm_vitals.sort_values('charttime')
        
        # Determine which time column to use for fluid events
        time_col = 'starttime' if 'starttime' in hadm_fluids.columns else 'charttime'
        hadm_fluids = hadm_fluids.sort_values(time_col)
        
        # For each fluid intervention, find the vitals within the time window
        for _, fluid_row in hadm_fluids.iterrows():
            intervention_time = fluid_row[time_col]
            if pd.isnull(intervention_time):
                continue
                
            intervention_type = fluid_row['intervention_type']
            
            # Calculate time window boundaries
            window_start = intervention_time - timedelta(minutes=time_window)
            window_end = intervention_time + timedelta(minutes=time_window)
            
            # Find vitals within the time window
            window_vitals = hadm_vitals[
                (hadm_vitals['charttime'] >= window_start) & 
                (hadm_vitals['charttime'] <= window_end)
            ]
            
            if window_vitals.empty:
                continue
            
            # Group vitals by feature and get the average or latest value
            feature_values = {}
            for feature in vital_sign_items.keys():
                feature_data = window_vitals[window_vitals['feature'] == feature]
                
                if not feature_data.empty:
                    # Take the average
                    feature_values[feature] = feature_data['valuenum'].mean()
                    
                    # Also add the closest measurement to the intervention
                    feature_data['time_diff'] = abs(feature_data['charttime'] - intervention_time)
                    closest_idx = feature_data['time_diff'].idxmin()
                    feature_values[f"{feature}_closest"] = feature_data.loc[closest_idx, 'valuenum']
            
            # If we have at least some vital signs, create a datapoint
            if feature_values:
                datapoint = {
                    'subject_id': patient_id,
                    hadm_id_col: hadm_id,
                    'intervention_time': intervention_time,
                    'intervention_type': intervention_type,
                }
                
                # Add ICU stay ID if available
                if icustay_id_col in fluid_row:
                    datapoint[icustay_id_col] = fluid_row[icustay_id_col]
                
                # Add fluid amount if available
                if 'amount' in fluid_row and not pd.isnull(fluid_row['amount']):
                    datapoint['amount'] = fluid_row['amount']
                else:
                    datapoint['amount'] = 0
                
                datapoint.update(feature_values)
                datapoints.append(datapoint)
    
    if not datapoints:
        return None
    
    return pd.DataFrame(datapoints)

# Process all patients and combine the results
print("Creating time-based feature windows for each patient...")
all_patient_data = []

for patient_id in tqdm(common_patients):
    patient_data = create_time_windows(patient_id)
    if patient_data is not None:
        all_patient_data.append(patient_data)

if all_patient_data:
    final_dataset = pd.concat(all_patient_data, ignore_index=True)
    print(f"Created dataset with {len(final_dataset)} data points")
    
    # Handle missing values - fill with medians per feature
    for feature in vital_sign_items.keys():
        for col_suffix in ['', '_closest']:
            col = feature + col_suffix
            if col in final_dataset.columns and final_dataset[col].isna().any():
                median_value = final_dataset[col].median()
                final_dataset[col] = final_dataset[col].fillna(median_value)
    
    # Save the preprocessed dataset
    output_path = os.path.expanduser("~/Fluid-Solutions-ML/data/processed/")
    os.makedirs(output_path, exist_ok=True)
    final_dataset.to_csv(os.path.join(output_path, "preprocessed_mimic_data.csv"), index=False)
    print(f"Saved preprocessed dataset to {os.path.join(output_path, 'preprocessed_mimic_data.csv')}")
    
    # Basic data exploration
    print("\nData overview:")
    print(final_dataset.head())
    print("\nDistribution of intervention types:")
    print(final_dataset['intervention_type'].value_counts())
    
    # Check for remaining missing values
    missing_data = final_dataset.isnull().sum()
    if missing_data.sum() > 0:
        print("\nRemaining missing values:")
        print(missing_data[missing_data > 0])
    
    # Example visualization
    plt.figure(figsize=(12, 6))
    intervention_counts = final_dataset['intervention_type'].value_counts()
    plt.bar(intervention_counts.index, intervention_counts.values)
    plt.title('Distribution of Intervention Types')
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, "intervention_distribution.png"))
    plt.close()
    
    # Visualize vital signs by intervention
    vital_features = list(vital_sign_items.keys())
    for feature in vital_features:
        if feature in final_dataset.columns:
            plt.figure(figsize=(10, 5))
            sns.boxplot(x='intervention_type', y=feature, data=final_dataset)
            plt.title(f'Distribution of {feature} by Intervention Type')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_path, f"{feature}_by_intervention.png"))
            plt.close()
    
    print("\nGenerated exploratory visualizations in the processed data directory")
    
    # Prepare data for ML model
    print("\nPreparing final ML-ready dataset...")
    
    # Create binary labels for classification
    final_dataset['label_dialysis'] = (final_dataset['intervention_type'] == 'dialysis').astype(int)
    final_dataset['label_fluid_administration'] = (final_dataset['intervention_type'] == 'fluid_administration').astype(int)
    
    # Select features for the model
    feature_cols = []
    for feature in vital_features:
        if feature in final_dataset.columns:
            feature_cols.append(feature)
        closest_feature = f"{feature}_closest"
        if closest_feature in final_dataset.columns:
            feature_cols.append(closest_feature)
    
    # Create train/test split by patient to prevent data leakage
    patient_ids = final_dataset['subject_id'].unique()
    np.random.seed(42)
    np.random.shuffle(patient_ids)
    
    train_size = int(0.8 * len(patient_ids))
    train_patients = patient_ids[:train_size]
    test_patients = patient_ids[train_size:]
    
    train_data = final_dataset[final_dataset['subject_id'].isin(train_patients)]
    test_data = final_dataset[final_dataset['subject_id'].isin(test_patients)]
    
    print(f"Training set: {len(train_data)} samples from {len(train_patients)} patients")
    print(f"Test set: {len(test_data)} samples from {len(test_patients)} patients")
    
    # Scale features
    scaler = StandardScaler()
    train_data[feature_cols] = scaler.fit_transform(train_data[feature_cols])
    test_data[feature_cols] = scaler.transform(test_data[feature_cols])
    
    # Save ML-ready datasets
    train_data.to_csv(os.path.join(output_path, "train_mimic_data.csv"), index=False)
    test_data.to_csv(os.path.join(output_path, "test_mimic_data.csv"), index=False)
    
    # Save the scaler for future use
    import joblib
    joblib.dump(scaler, os.path.join(output_path, "feature_scaler.pkl"))
    
    print(f"Saved ML-ready datasets to {output_path}")
    print("Done!")
    
else:
    print("No valid data points were created. Check your data and grouping logic.")