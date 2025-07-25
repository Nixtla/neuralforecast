import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from neuralforecast.core import NeuralForecast
from neuralforecast.models import MLP, NHITS
from utilsforecast.feature_engineering import time_features, fourier
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
import back
import importlib

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Data setup
Y_train_df = AirPassengersPanel[AirPassengersPanel['ds'] < AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)
Y_test_df = AirPassengersPanel[AirPassengersPanel['ds'] >= AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)

transformed_df, _ = time_features(AirPassengersPanel, freq='ME', features=['month', 'week'], h=1)
Y_train_df = Y_train_df.merge(transformed_df[["unique_id", "ds", "month", "week"]], how="left")
Y_test_df = Y_test_df.merge(transformed_df[["unique_id", "ds", "month", "week"]], how="left")

transformed_df, _ = fourier(AirPassengersPanel, freq='ME', season_length=12, k=4)
Y_train_df = Y_train_df.merge(transformed_df[["unique_id", "ds", 'sin1_12', 'sin2_12', 'sin3_12', 'sin4_12', 'cos1_12', 'cos2_12', 'cos3_12', 'cos4_12']], how="left")
Y_test_df = Y_test_df.merge(transformed_df[["unique_id", "ds", 'sin1_12', 'sin2_12', 'sin3_12', 'sin4_12', 'cos1_12', 'cos2_12', 'cos3_12', 'cos4_12']], how="left")

hist_exog_cols = ["y_[lag12]"]
futr_exog_cols = ["trend", "month", "week", 'sin1_12', 'sin2_12', 'sin3_12', 'sin4_12', 'cos1_12', 'cos2_12', 'cos3_12', 'cos4_12']
stat_exog_cols = ['airline1']

# Model setup
models = [
    MLP(
        h=12,
        input_size=24,
        hist_exog_list=hist_exog_cols,
        futr_exog_list=futr_exog_cols,
        stat_exog_list=stat_exog_cols,
        max_steps=200,
        alias="MLP"
    ),
    NHITS(
        h=12,
        input_size=24,
        hist_exog_list=hist_exog_cols,
        futr_exog_list=futr_exog_cols,
        stat_exog_list=stat_exog_cols,
        max_steps=200,
        alias="NHITS"
    )
]

nf = NeuralForecast(models=models, freq="ME")
nf.fit(
    df=Y_train_df[Y_train_df["unique_id"] == "Airline1"],
    static_df=AirPassengersStatic
)

# Prepare test data
futr_exog_df = Y_test_df.drop(["y", "y_[lag12]"], axis=1)
futr_exog_df = futr_exog_df[futr_exog_df['unique_id'] == 'Airline1'].reset_index(drop=True)
# ==================== SHAP ANALYSIS ====================
importlib.reload(back)

model = nf.models[0]
# Initialize DeepExplainer with wrapper model
explainer = back.create_explainer(
    model = model,
    df=Y_train_df[Y_train_df["unique_id"] == "Airline1"],
    static_df=AirPassengersStatic,
    futr_exog_df=futr_exog_df
)

# Define horizons to explain
horizons_to_explain = {
    'Forecast for 1st Month': 0, 
    'Forecast for 6th Month': 5, 
    'Forecast for 12th Month': 11
}

# Calculate SHAP values for each horizon
all_shap_values = {}

for name, h_idx in horizons_to_explain.items():
    # Get SHAP values for all horizons - disable internal additivity check
    shap_values = explainer.shap_values(X_explain_tensor, check_additivity=False)
    
    # SHAP values shape: (1, 132, 12) = [batch_size, n_features, n_outputs]
    # Extract SHAP values for specific horizon
    horizon_shap = shap_values[0, :, h_idx]  # All features for this specific horizon
    all_shap_values[name] = horizon_shap

# ==================== VISUALIZATION ====================

# Create comprehensive feature names for all varying features
def create_complete_feature_names():
    """Create feature names for all varying features"""
    feature_names = []
    
    # Future exogenous features (if they exist)
    if futr_exog_cols:
        for i in range(model.h):
            for col in futr_exog_cols:
                feature_names.append(f'{col}_h{i+1}')
    
    # Historical exogenous features (if they exist)
    if hist_exog_cols:
        for i in range(model.input_size):
            for col in hist_exog_cols:
                feature_names.append(f'{col}_lag{i+1}')
    
    # Historical target values (always present)
    for i in range(model.input_size):
        feature_names.append(f'y_lag{i+1}')
    
    return feature_names

feature_names = create_complete_feature_names()

print(f"Total features explained: {len(feature_names)}")

# Create flattened version of X_explain for visualization
X_explain_flat = X_explain_tensor.numpy().flatten()

# Plot results
for name, shap_vals in all_shap_values.items():
    plt.figure(figsize=(12, 8))
    
    # Get the correct baseline prediction for this specific horizon
    h_idx = horizons_to_explain[name]
    baseline_input = torch.mean(background_tensor, dim=0, keepdim=True)
    with torch.no_grad():
        baseline_prediction_tensor = wrapper_model(baseline_input)
        baseline_prediction = baseline_prediction_tensor[0, h_idx, 0].item()
    
    # Create SHAP explanation object with correct baseline
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=baseline_prediction,  # Use actual baseline prediction
        data=X_explain_flat,
        feature_names=feature_names
    )
    
    plt.title(f"{name} - Complete Feature Analysis (Baseline: {baseline_prediction:.2f})", fontsize=16)
    shap.plots.bar(explanation, max_display=25)  # Show more features
    plt.tight_layout()
    plt.show()

# Additional plot: Show breakdown by feature type
for name, shap_vals in all_shap_values.items():
    # Split SHAP values by feature type
    n_futr = len(futr_exog_cols) * model.h if futr_exog_cols else 0
    n_hist_exog = len(hist_exog_cols) * model.input_size if hist_exog_cols else 0
    n_hist_target = model.input_size
    
    # Only split if the feature type exists
    if n_futr > 0:
        futr_shap = shap_vals[:n_futr]
    else:
        futr_shap = np.array([])
    
    if n_hist_exog > 0:
        hist_exog_shap = shap_vals[n_futr:n_futr + n_hist_exog]
    else:
        hist_exog_shap = np.array([])
    
    hist_target_shap = shap_vals[n_futr + n_hist_exog:]
    
    # Create summary plot (only for existing feature types)
    feature_groups = []
    group_contributions = []
    
    if n_futr > 0:
        feature_groups.append('Future Exogenous')
        group_contributions.append(np.sum(futr_shap))
    
    if n_hist_exog > 0:
        feature_groups.append('Historical Exogenous')
        group_contributions.append(np.sum(hist_exog_shap))
    
    feature_groups.append('Historical Target')
    group_contributions.append(np.sum(hist_target_shap))
    
    plt.figure(figsize=(10, 6))
    plt.bar(feature_groups, group_contributions)
    plt.title(f"{name} - Feature Group Contributions (VERIFIED ORDER)")
    plt.ylabel("SHAP Value Contribution")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==================== ADDITIVITY CHECK ====================

# Verify SHAP additivity for DeepExplainer
name_to_verify = 'Forecast for 1st Month'
shap_values_to_verify = all_shap_values[name_to_verify]
h_idx_to_verify = horizons_to_explain[name_to_verify]

# Get the model's actual prediction using the wrapper model (for consistency)
with torch.no_grad():
    actual_prediction_tensor = wrapper_model(X_explain_tensor)
    actual_prediction = actual_prediction_tensor[0, h_idx_to_verify, 0].item()

# Get the baseline prediction using the wrapper model
baseline_input = torch.mean(background_tensor, dim=0, keepdim=True)
with torch.no_grad():
    baseline_prediction_tensor = wrapper_model(baseline_input)
    baseline_prediction = baseline_prediction_tensor[0, h_idx_to_verify, 0].item()

# Sum the SHAP values for the prediction
sum_of_shap_values = np.sum(shap_values_to_verify)

# Perform the verification check
# DeepExplainer additivity: f(x) = f(baseline) + sum(SHAP values)
verified_prediction = baseline_prediction + sum_of_shap_values

print(f"--- Additivity Check for: {name_to_verify} ---")
print(f"Baseline Prediction: {baseline_prediction:.4f}")
print(f"Sum of all SHAP values: {sum_of_shap_values:.4f}")
print(f"Verified Prediction (Baseline + SHAP Values): {verified_prediction:.4f}")
print(f"Actual Model Prediction: {actual_prediction:.4f}")
print(f"Difference: {abs(verified_prediction - actual_prediction):.4f}")

# Check if additivity holds with relative tolerance
relative_tolerance = 0.01
absolute_tolerance = abs(actual_prediction) * relative_tolerance
relative_error = abs(verified_prediction - actual_prediction) / abs(actual_prediction) * 100

print(f"Relative error: {relative_error:.2f}%")
print(f"Tolerance: {relative_tolerance * 100:.1f}%")

if abs(verified_prediction - actual_prediction) < absolute_tolerance:
    print("✅ Additivity check PASSED - SHAP values sum correctly!")
else:
    print("❌ Additivity check FAILED - There may be an issue with the implementation")