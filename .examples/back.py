import torch
from typing import Optional, Union, Sequence, List
from neuralforecast.compat import SparkDataFrame
from utilsforecast.compat import DataFrame
from neuralforecast.common._base_model import BaseModel
import numpy as np
import shap




class ModelWrapper(torch.nn.Module):
    """
    Wrapper model that converts flattened tensor input to dictionary format
    expected by NeuralForecast MLP model.
    """
    def __init__(
            self,
            model: BaseModel,
            df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
            static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        ):

        super().__init__()
        self.model = model
        self.futr_exog_cols = model.futr_exog_list
        self.hist_exog_cols = model.hist_exog_list
        self.stat_exog_cols = model.stat_exog_list
        
        # Calculate input dimensions (only for existing feature types)
        self.n_futr_features = len(self.futr_exog_cols) * self.model.h if self.futr_exog_cols else 0
        
        self.n_hist_target_features = self.model.input_size
        
        # Check if there are any static variables
        if self.stat_exog_cols:
            # Input dimensions
            self.n_hist_exog_features = len(self.hist_exog_cols) * self.model.input_size
            # Gets static data
            self.stat_exog_fixed = torch.tensor(
                static_df[self.stat_exog_cols].values[0], 
                dtype=torch.float32
            )
        else:
            self.stat_exog_fixed = None
            self.n_hist_exog_features = 0

        # Checks fixed historical part of future exogenous features   
        if self.futr_exog_cols:
            # Input dimentions
            self.n_futr_features = len(self.futr_exog_cols) * self.model.h
            # Gets fixed historical exogenous features
            self.futr_hist_fixed = torch.tensor(
                df[self.futr_exog_cols].values[-self.model.input_size:], 
                dtype=torch.float32
            )
        else:
            self.n_futr_features = 0
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
def create_complete_input_tensor(
    model: BaseModel,
    df: Union[DataFrame, SparkDataFrame, Sequence[str]],
    futr_exog_df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
):

    """Create input tensor with all varying features: future exog, hist exog, and hist target"""
    input_components = []
    futr_exog_cols = model.futr_exog_list
    hist_exog_cols = model.hist_exog_list

    # Future exogenous features (if they exist)
    if futr_exog_cols:
        futr_exog_data = futr_exog_df[futr_exog_cols].to_numpy().flatten()
        input_components.append(futr_exog_data)
    
    # Historical exogenous features (if they exist)
    if hist_exog_cols:
        hist_exog_data = df[hist_exog_cols].values[-model.input_size:].flatten()
        input_components.append(hist_exog_data)
    
    # Historical target values (always present)
    hist_target_data = df['y'].values[-model.input_size:]
    input_components.append(hist_target_data)
    
    # Combine all existing features
    if input_components:
        complete_input = np.concatenate(input_components)
    else:
        # Should not happen as hist_target_data is always present
        complete_input = hist_target_data
    
    return torch.tensor(complete_input, dtype=torch.float32).reshape(1, -1)

# Create background data - include all varying features
def create_complete_background_data(
    model: BaseModel,
    df: Union[DataFrame, SparkDataFrame, Sequence[str]],
):
    """Create background data with all varying features"""
    background_samples = []
    futr_exog_cols = model.futr_exog_list
    hist_exog_cols = model.hist_exog_list
    
    # Determine the range of valid indices
    start_idx = model.input_size  # Need historical data
    if futr_exog_cols:
        end_idx = len(df) - model.h + 1  # Need future data
    else:
        end_idx = len(df)  # No future data needed
    
    for i in range(start_idx, end_idx, 3):  # Use stride for efficiency
        sample_components = []
        
        # Future exogenous features (if they exist)
        if futr_exog_cols:
            futr_window = df[futr_exog_cols].iloc[i:i + model.h].to_numpy().flatten()
            sample_components.append(futr_window)
        
        # Historical exogenous features (if they exist)
        if hist_exog_cols:
            hist_exog_window = df[hist_exog_cols].iloc[i-model.input_size:i].to_numpy().flatten()
            sample_components.append(hist_exog_window)
        
        # Historical target values (always present)
        hist_target_window = df['y'].iloc[i-model.input_size:i].to_numpy()
        sample_components.append(hist_target_window)
        
        # Combine all existing features
        if sample_components:
            complete_sample = np.concatenate(sample_components)
        else:
            # Should not happen as hist_target_window is always present
            complete_sample = hist_target_window
        
        background_samples.append(complete_sample)
    
    return np.array(background_samples)


def create_explainer(
    model: BaseModel,
    df: Union[DataFrame, SparkDataFrame, Sequence[str]],
    static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
    futr_exog_df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
    kmeans_dim: int = 50
):
    
    wrapper_model = ModelWrapper(
        model = model,
        df=df,
        static_df=static_df
    )

    X_explain_tensor = create_complete_input_tensor(
        model = model,
        df=df,
        futr_exog_df=futr_exog_df
    )
    background_array = create_complete_background_data(
        model = model,
        df=df
    )
    background_summary = shap.kmeans(background_array, min(kmeans_dim, len(background_array)))

    # Handle DenseData object from shap.kmeans
    if hasattr(background_summary, 'data'):
        background_data = background_summary.data
    else:
        background_data = background_summary

    background_tensor = torch.tensor(background_data, dtype=torch.float32)
    explainer = shap.DeepExplainer(wrapper_model, background_tensor)

    return explainer