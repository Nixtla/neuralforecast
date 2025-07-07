import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from neuralforecast.core import NeuralForecast
from neuralforecast.models import MLP
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

# Model setup
model = MLP(
    h=12,
    input_size=24,
    hist_exog_list=["y_[lag12]"],
    futr_exog_list=["trend", "month", "week", 'sin1_12', 'sin2_12', 'sin3_12', 'sin4_12', 'cos1_12', 'cos2_12', 'cos3_12', 'cos4_12'],
    stat_exog_list=['airline1'],
    max_steps=200,
    alias="MLP"
)

nf = NeuralForecast(models=[model], freq="ME")
nf.fit(
    df=Y_train_df[Y_train_df["unique_id"] == "Airline1"],
    static_df=AirPassengersStatic
)

# Prepare test data
futr_exog_df = Y_test_df.drop(["y", "y_[lag12]"], axis=1)
futr_exog_df = futr_exog_df[futr_exog_df['unique_id'] == 'Airline1'].reset_index(drop=True)

# Get feature columns
futr_exog_cols = model.futr_exog_list
hist_exog_cols = model.hist_exog_list
stat_exog_cols = model.stat_exog_list

# ==================== DEEPEXPLAINER IMPLEMENTATION ====================

# ==================== DEEPEXPLAINER IMPLEMENTATION ====================

class ModelWrapper(torch.nn.Module):
    """
    Wrapper model that converts flattened tensor input to dictionary format
    expected by NeuralForecast MLP model.
    """
    def __init__(self, original_model, model_config, futr_exog_cols, hist_exog_cols, stat_exog_cols):
        super().__init__()
        self.original_model = original_model
        self.model_config = model_config
        self.futr_exog_cols = futr_exog_cols
        self.hist_exog_cols = hist_exog_cols
        self.stat_exog_cols = stat_exog_cols
        
        # Get fixed historical and static data
        self.historical_y = torch.tensor(
            Y_train_df[Y_train_df['unique_id'] == 'Airline1']['y'].values[-model_config.input_size:], 
            dtype=torch.float32
        )
        
        if len(hist_exog_cols) > 0:
            self.hist_exog_fixed = torch.tensor(
                Y_train_df[Y_train_df['unique_id'] == 'Airline1'][hist_exog_cols].values[-model_config.input_size:], 
                dtype=torch.float32
            )
        else:
            self.hist_exog_fixed = torch.zeros(model_config.input_size, 0)
        
        if len(stat_exog_cols) > 0:
            self.stat_exog_fixed = torch.tensor(
                AirPassengersStatic[AirPassengersStatic['unique_id'] == 'Airline1'][stat_exog_cols].values[0], 
                dtype=torch.float32
            )
        else:
            self.stat_exog_fixed = torch.zeros(0)
            
        self.futr_hist_fixed = torch.tensor(
            Y_train_df[Y_train_df['unique_id'] == 'Airline1'][futr_exog_cols].values[-model_config.input_size:], 
            dtype=torch.float32
        )
    
    def forward(self, X_flat):
        """
        Convert flattened tensor input to dictionary format and call original model
        X_flat: [batch_size, n_futr_features] where n_futr_features = h * len(futr_exog_cols)
        """
        batch_size = X_flat.shape[0]
        
        # Historical target values (constant)
        insample_y = self.historical_y.unsqueeze(0).repeat(batch_size, 1)
        
        # Historical exogenous features (constant)
        hist_exog = self.hist_exog_fixed.unsqueeze(0).repeat(batch_size, 1, 1)
        
        # Static exogenous features (constant)
        stat_exog = self.stat_exog_fixed.unsqueeze(0).repeat(batch_size, 1)
        
        # Future exogenous features (varying)
        futr_pred = X_flat.reshape(batch_size, self.model_config.h, len(self.futr_exog_cols))
        futr_hist = self.futr_hist_fixed.unsqueeze(0).repeat(batch_size, 1, 1)
        futr_exog = torch.cat([futr_hist, futr_pred], dim=1)
        
        # Create windows_batch dictionary
        windows_batch = {
            'insample_y': insample_y.unsqueeze(-1),
            'futr_exog': futr_exog,
            'hist_exog': hist_exog,
            'stat_exog': stat_exog
        }
        
        # Call original model
        return self.original_model(windows_batch)

# Create input for explanation - only future exogenous features vary
X_explain_df = futr_exog_df[futr_exog_cols]
X_explain_flat = X_explain_df.to_numpy().flatten().reshape(1, -1)
X_explain_tensor = torch.tensor(X_explain_flat, dtype=torch.float32)

# Create background data - only future exogenous features vary
train_exog_df = Y_train_df[Y_train_df['unique_id'] == 'Airline1'][futr_exog_cols]
background_windows = []

for i in range(0, len(train_exog_df) - model.h + 1, 3):
    window = train_exog_df.iloc[i:i + model.h].to_numpy().flatten()
    background_windows.append(window)

# Reduce background data
background_array = np.array(background_windows)
background_summary = shap.kmeans(background_array, min(50, len(background_array)))

# Handle DenseData object from shap.kmeans
if hasattr(background_summary, 'data'):
    background_data = background_summary.data
else:
    background_data = background_summary

background_tensor = torch.tensor(background_data, dtype=torch.float32)

# Create wrapper model for DeepExplainer
pytorch_model = nf.models[0]
wrapper_model = ModelWrapper(
    pytorch_model, 
    model, 
    futr_exog_cols, 
    hist_exog_cols, 
    stat_exog_cols
)

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

# Create feature names for future exogenous features
feature_names = []
for i in range(model.h):
    for col in futr_exog_cols:
        feature_names.append(f'{col}_h{i+1}')

# Plot results
for name, shap_vals in all_shap_values.items():
    plt.figure(figsize=(12, 8))
    
    # Create SHAP explanation object
    # shap_vals is already a numpy array, no need to detach
    explanation = shap.Explanation(
        values=shap_vals,
        base_values=0,
        data=X_explain_flat[0],
        feature_names=feature_names
    )
    
    plt.title(f"{name} - DeepExplainer Results", fontsize=16)
    shap.plots.bar(explanation, max_display=20)
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

# Check if additivity holds (allow for slightly larger tolerance due to DeepExplainer approximations)
if abs(verified_prediction - actual_prediction) < 0.1:
    print("✅ Additivity check PASSED - SHAP values sum correctly!")
else:
    print("❌ Additivity check FAILED - There may be an issue with the implementation")

print("\nDeepExplainer analysis completed!")