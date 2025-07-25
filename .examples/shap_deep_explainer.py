import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from neuralforecast.core import NeuralForecast
from neuralforecast.models import MLP, NHITS
from neuralforecast import models as nf_model
from utilsforecast.feature_engineering import time_features, fourier
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic

import shap

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

# ==================== DEEPEXPLAINER IMPLEMENTATION ====================

class ModelWrapper(torch.nn.Module):
    """
    Wrapper model that converts flattened tensor input to dictionary format
    expected by NeuralForecast MLP model.
    """
    def __init__(
            self,
            model
        ):
        super().__init__()
        self.model = model
        self.futr_exog_cols = model.futr_exog_list
        self.hist_exog_cols = model.hist_exog_list
        self.stat_exog_cols = model.stat_exog_list
        
        # Calculate input dimensions (only for existing feature types)
        self.n_futr_features = len(futr_exog_cols) * self.model.h if futr_exog_cols else 0
        self.n_hist_exog_features = len(hist_exog_cols) * self.model.input_size if hist_exog_cols else 0
        self.n_hist_target_features = self.model.input_size
        
        # Get fixed static data (if exists)
        if stat_exog_cols:
            self.stat_exog_fixed = torch.tensor(
                AirPassengersStatic[AirPassengersStatic['unique_id'] == 'Airline1'][stat_exog_cols].values[0], 
                dtype=torch.float32
            )
        else:
            self.stat_exog_fixed = None
            
        # Get fixed historical part of future exogenous features (if exists)
        if futr_exog_cols:
            self.futr_hist_fixed = torch.tensor(
                Y_train_df[Y_train_df['unique_id'] == 'Airline1'][futr_exog_cols].values[-self.model.input_size:], 
                dtype=torch.float32
            )
        else:
            self.futr_hist_fixed = None
    
    def forward(self, X_flat):
        """
        Convert flattened tensor input to dictionary format and call original model
        X_flat: [batch_size, n_total_features] where n_total_features = n_futr + n_hist_exog + n_hist_target
        """
        batch_size = X_flat.shape[0]
        
        # Split the input tensor
        idx = 0
        
        # Future exogenous features (varying or None)
        if self.futr_exog_cols:
            futr_flat = X_flat[:, idx:idx + self.n_futr_features]
            idx += self.n_futr_features
            
            futr_pred = futr_flat.reshape(batch_size, self.model.h, len(self.futr_exog_cols))
            futr_hist = self.futr_hist_fixed.unsqueeze(0).repeat(batch_size, 1, 1)
            futr_exog = torch.cat([futr_hist, futr_pred], dim=1)
        else:
            futr_exog = None
        
        # Historical exogenous features (varying or None)
        if self.hist_exog_cols:
            hist_exog_flat = X_flat[:, idx:idx + self.n_hist_exog_features]
            hist_exog = hist_exog_flat.reshape(batch_size, self.model.input_size, len(self.hist_exog_cols))
            idx += self.n_hist_exog_features
        else:
            hist_exog = None
        
        # Historical target values (always present)
        hist_target_flat = X_flat[:, idx:idx + self.n_hist_target_features]
        insample_y = hist_target_flat.reshape(batch_size, self.model.input_size)
        
        # Static exogenous features (constant or None)
        if self.stat_exog_cols:
            stat_exog = self.stat_exog_fixed.unsqueeze(0).repeat(batch_size, 1)
        else:
            stat_exog = None
        
        # Create windows_batch dictionary
        windows_batch = {
            'insample_y': insample_y.unsqueeze(-1),
            'futr_exog': futr_exog,
            'hist_exog': hist_exog,
            'stat_exog': stat_exog
        }
        
        # Call original model
        return self.model(windows_batch)

# Create input for explanation - include all varying features
def create_complete_input_tensor():
    """Create input tensor with all varying features: future exog, hist exog, and hist target"""
    input_components = []
    
    # Future exogenous features (if they exist)
    if futr_exog_cols:
        futr_exog_data = futr_exog_df[futr_exog_cols].to_numpy().flatten()
        input_components.append(futr_exog_data)
    
    # Historical exogenous features (if they exist)
    if hist_exog_cols:
        hist_exog_data = Y_train_df[Y_train_df['unique_id'] == 'Airline1'][hist_exog_cols].values[-model.input_size:].flatten()
        input_components.append(hist_exog_data)
    
    # Historical target values (always present)
    hist_target_data = Y_train_df[Y_train_df['unique_id'] == 'Airline1']['y'].values[-model.input_size:]
    input_components.append(hist_target_data)
    
    # Combine all existing features
    if input_components:
        complete_input = np.concatenate(input_components)
    else:
        # Should not happen as hist_target_data is always present
        complete_input = hist_target_data
    
    return torch.tensor(complete_input, dtype=torch.float32).reshape(1, -1)

X_explain_tensor = create_complete_input_tensor()

# Create background data - include all varying features
def create_complete_background_data():
    """Create background data with all varying features"""
    train_df = Y_train_df[Y_train_df['unique_id'] == 'Airline1']
    background_samples = []
    
    # Determine the range of valid indices
    start_idx = self.model.input_size  # Need historical data
    if futr_exog_cols:
        end_idx = len(train_df) - model.h + 1  # Need future data
    else:
        end_idx = len(train_df)  # No future data needed
    
    for i in range(start_idx, end_idx, 3):  # Use stride for efficiency
        sample_components = []
        
        # Future exogenous features (if they exist)
        if futr_exog_cols:
            futr_window = train_df[futr_exog_cols].iloc[i:i + model.h].to_numpy().flatten()
            sample_components.append(futr_window)
        
        # Historical exogenous features (if they exist)
        if hist_exog_cols:
            hist_exog_window = train_df[hist_exog_cols].iloc[i-model.input_size:i].to_numpy().flatten()
            sample_components.append(hist_exog_window)
        
        # Historical target values (always present)
        hist_target_window = train_df['y'].iloc[i-model.input_size:i].to_numpy()
        sample_components.append(hist_target_window)
        
        # Combine all existing features
        if sample_components:
            complete_sample = np.concatenate(sample_components)
        else:
            # Should not happen as hist_target_window is always present
            complete_sample = hist_target_window
        
        background_samples.append(complete_sample)
    
    return np.array(background_samples)

background_array = create_complete_background_data()
background_summary = shap.kmeans(background_array, min(50, len(background_array)))

# Handle DenseData object from shap.kmeans
if hasattr(background_summary, 'data'):
    background_data = background_summary.data
else:
    background_data = background_summary

background_tensor = torch.tensor(background_data, dtype=torch.float32)



pytorch_model = nf.models[0]
wrapper_model = ModelWrapper(pytorch_model)

# ==================== SHAP ANALYSIS ====================

# Initialize DeepExplainer with wrapper model
explainer = shap.DeepExplainer(wrapper_model, background_tensor)

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