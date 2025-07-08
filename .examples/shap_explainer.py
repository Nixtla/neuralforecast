import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import MLP
from utilsforecast.feature_engineering import time_features, fourier
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
from typing import Tuple, List

import shap
import torch

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Provided data setup
Y_train_df = AirPassengersPanel[AirPassengersPanel['ds'] < AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)
Y_test_df = AirPassengersPanel[AirPassengersPanel['ds'] >= AirPassengersPanel['ds'].values[-12]].reset_index(drop=True)

transformed_df, _ = time_features(AirPassengersPanel,  freq='ME', features=['month', 'week'], h=1)
Y_train_df = Y_train_df.merge(transformed_df[["unique_id", "ds", "month", "week"]], how="left")
Y_test_df = Y_test_df.merge(transformed_df[["unique_id", "ds", "month", "week"]], how="left")

transformed_df, _ = fourier(AirPassengersPanel,  freq='ME', season_length=12, k=4)
Y_train_df = Y_train_df.merge(transformed_df[["unique_id", "ds", 'sin1_12', 'sin2_12', 'sin3_12', 'sin4_12', 'cos1_12', 'cos2_12', 'cos3_12', 'cos4_12']], how="left")
Y_test_df = Y_test_df.merge(transformed_df[["unique_id", "ds", 'sin1_12', 'sin2_12', 'sin3_12', 'sin4_12', 'cos1_12', 'cos2_12', 'cos3_12', 'cos4_12']], how="left")

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

futr_exog_df = Y_test_df.drop(["y", "y_[lag12]"], axis=1)
futr_exog_df = futr_exog_df[futr_exog_df['unique_id'] == 'Airline1'].reset_index(drop=True)

# The model's prediction is based on the full set of future exogenous variables.
# We need to treat this entire block of data as the "input" for one explanation.
futr_exog_cols = model.futr_exog_list
X_explain_df = futr_exog_df[futr_exog_cols]


# For KernelExplainer, the input must be a 1D array for a single prediction.
# We flatten the (12 horizons x 11 features) dataframe into a single row vector.
X_explain_flat = X_explain_df.to_numpy().flatten().reshape(1, -1)

# 1. Create Background Data
# SHAP needs a background dataset to represent "missing" features.
# We create this by taking sliding windows of the future exogenous features from the training data.
train_exog_df = Y_train_df[futr_exog_cols]
background_windows = []
for i in range(len(train_exog_df) - model.h + 1):
    window = train_exog_df.iloc[i:i + model.h].to_numpy()
    background_windows.append(window.flatten())

# Use shap.kmeans to summarize the background data into a smaller, representative set.
background_summary = shap.kmeans(np.array(background_windows), 25)

# 2. Defines helpfull functions to produce the explanations
def predict_fn(X_flat, horizon:int) -> np.ndarray:
    # This function adapts the NeuralForecast model for SHAP's KernelExplainer.
    predictions = []
    for i in range(X_flat.shape[0]):
        # Reshape the flat array back to the (h, n_features) format
        futr_array = X_flat[i].reshape(model.h, len(futr_exog_cols))
        
        # Create the future dataframe required by nf.predict
        temp_futr_df = pd.DataFrame(futr_array, columns=futr_exog_cols)
        temp_futr_df['ds'] = futr_exog_df['ds'].values
        temp_futr_df['unique_id'] = 'Airline1'
        
        # Make a prediction
        forecast = nf.predict(futr_df=temp_futr_df)
        
        # Extract the prediction for the specific horizon (horizon)
        prediction_for_h = forecast[model.alias].iloc[horizon]
        predictions.append(prediction_for_h)

    return np.array(predictions)

def explain_horizon(horizon: int, n_samples:int) -> Tuple[np.ndarray, shap.Explainer]:
    explainer = shap.KernelExplainer(lambda X_flat: predict_fn(X_flat, horizon), background_summary)
    shap_values = explainer.shap_values(X_explain_flat, nsamples=n_samples)
    return shap_values, explainer

def plot_explanation(shap_values: np.ndarray, explainer: shap.Explainer, X_explain_flat: np.ndarray, feature_names: List):
    explanation = shap.Explanation(
        values = shap_values,
        base_values = explainer.expected_value,
        data=X_explain_flat,
        feature_names=feature_names
    )
    return shap.plots.bar(explanation[0], max_display=15)

# 3. Creates explanations
n_samples = 2
feature_names = [f'{col}_h{i+1}' for i in range(model.h) for col in futr_exog_cols]
shaps = {}
for i in [0, 5, 11]:
    shap_values, explainer = explain_horizon(i, n_samples)
    shaps[i] = (shap_values, explainer)

plot_explanation(shaps[0][0], shaps[0][1], X_explain_flat, feature_names) # Explanation for first month
plot_explanation(shaps[5][0], shaps[5][1], X_explain_flat, feature_names) # Explanation for sixth month
plot_explanation(shaps[11][0], shaps[11][1], X_explain_flat, feature_names) # Explanation for twelve month
# TODO: find a better way to store shaps and create list explanation for multiple periods
# 4. Check consistency

def check_consistency(horizon: int, shap_values: np.ndarray, explainer: shap.Explainer, nf: NeuralForecast, future_exog_df:pd.DataFrame, model) -> None:
    actual_prediction_df = nf.predict(futr_df=future_exog_df)
    actual_prediction = actual_prediction_df[model.alias].iloc[horizon]
    print(actual_prediction)
    base_value = explainer.expected_value
    sum_of_shap_values = np.sum(shap_values)
    verified_prediction = base_value + sum_of_shap_values
    print(f"--- Verification for horizon: {horizon} ---")
    print(f"SHAP Base Value: {base_value:.4f}")
    print(f"Sum of all SHAP values: {sum_of_shap_values:.4f}")
    print(f"Verified Prediction (Base + SHAP Values): {verified_prediction:.4f}")
    print(f"Actual Model Prediction: {actual_prediction:.4f}\n")

check_consistency(0, shaps[0][0], shaps[0][1], nf, futr_exog_df, model)
check_consistency(5, shaps[5][0], shaps[5][1], nf, futr_exog_df, model)
check_consistency(11, shaps[11][0], shaps[11][1], nf, futr_exog_df, model)