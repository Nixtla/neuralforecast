import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from neuralforecast.core import NeuralForecast
from neuralforecast.models import MLP, NHITS
from utilsforecast.feature_engineering import time_features, fourier
from neuralforecast.utils import AirPassengersPanel, AirPassengersStatic
import back # In order for this to run Python needs to be opened in the .examples folder sorry :(
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
explainer = back.NeuralExplainer(
    model = model,
    df=Y_train_df[Y_train_df["unique_id"] == "Airline1"],
    static_df=AirPassengersStatic,
    futr_exog_df=futr_exog_df
)
# Can run in the fly
explainer.get_explanations([0,5,11], check_additivity=False)
# Can check additivity with relative error
explainer.get_explanations([0,5,11], check_additivity=True) # It stills fails in the 11th prediction (not by much though...)
#  Plot results for a specific horizon
explainer.plot(horizon=0).show()
#  Shows error if horizon isnt computed in get explanations...
explainer.plot(horizon=2).show()
# ==================== TODO ====================

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
