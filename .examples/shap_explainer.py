import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import MLP
from utilsforecast.feature_engineering import time_features, fourier
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
import numpy as np
import pandas as pd

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

# 2. Loop Through Horizons and Explain
# Store calculated SHAP values for each horizon we want to analyze.
all_shap_values = {}
# Let's explain the 1st, 6th, and 12th month predictions.
# h=1 -> index 0, h=6 -> index 5, h=12 -> index 11
horizons_to_explain = {'Forecast for 1st Month': 0, 'Forecast for 6th Month': 5, 'Forecast for 12th Month': 11}

print("Calculating SHAP values for different forecast horizons... This may take a few minutes.")
n_samples = 2 # Samples to use by the KernelExplainer 
# I used low number because it's super slow. The reason is that is getting a lot of predict_fn calls.
# If this works and its ok we can think in implementing the shap.KernelExplainer specific for predicting a lot of the samples once and then just follow the logic.
for name, h_idx in horizons_to_explain.items():
    # 3. Define the Wrapper Prediction Function
    def predict_fn(X_flat):
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
            
            # Extract the prediction for the specific horizon (h_idx)
            prediction_for_h = forecast[model.alias].iloc[h_idx]
            predictions.append(prediction_for_h)
            
        return np.array(predictions)

    # 4. Initialize the Explainer and Calculate SHAP values
    explainer = shap.KernelExplainer(predict_fn, background_summary)
    shap_values = explainer.shap_values(X_explain_flat, nsamples=n_samples)
    all_shap_values[name] = shap_values
    print(f"Finished explaining: {name}")


# 5. Plot the Feature Importances
# Create meaningful names for our 132 flattened features (12 horizons * 11 features)
# Example: 'trend_h1', 'month_h1', ..., 'cos4_12_h12'
feature_names = [f'{col}_h{i+1}' for i in range(model.h) for col in futr_exog_cols]

for name, shaps in all_shap_values.items():
    # Create a SHAP Explanation object for easy plotting
    explanation = shap.Explanation(
        values=shaps,
        base_values=explainer.expected_value,
        data=X_explain_flat,
        feature_names=feature_names
    )

    print(f"\nDisplaying plot for: {name}")
    plt.title(name, fontsize=16)
    
    # Use a bar plot to show the top contributing features
    shap.plots.bar(explanation[0], max_display=15)

# 6. Check consistency
# We'll use the results from the 1st month forecast for this example.
# You must have already run the code that generates these variables.
name_to_verify = 'Forecast for 1st Month'
shap_values_to_verify = all_shap_values[name_to_verify]
h_idx_to_verify = horizons_to_explain[name_to_verify]

# 1. Get the model's actual prediction for the 1st month
# We can get this by running a clean prediction on the original test data
actual_prediction_df = nf.predict(futr_df=futr_exog_df)
actual_prediction = actual_prediction_df[model.alias].iloc[h_idx_to_verify]

# (Assuming you just ran the loop, the last explainer is for the 12th month, so we can use it directly)
base_value = explainer.expected_value

# 3. Sum the SHAP values for the prediction
sum_of_shap_values = np.sum(shap_values_to_verify)

# 4. Perform the verification check
verified_prediction = base_value + sum_of_shap_values

print_bool = True
while print_bool:
    print(f"--- Verification for: {name_to_verify} ---")
    print(f"SHAP Base Value: {base_value:.4f}")
    print(f"Sum of all SHAP values: {sum_of_shap_values:.4f}")
    print(f"Verified Prediction (Base + SHAP Values): {verified_prediction:.4f}")
    print(f"Actual Model Prediction: {actual_prediction:.4f}\n")
    print_bool=False
