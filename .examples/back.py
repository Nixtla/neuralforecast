import torch
from typing import Optional, Union, Sequence, List
from neuralforecast.compat import SparkDataFrame
from utilsforecast.compat import DataFrame
from neuralforecast.common._base_model import BaseModel
import numpy as np
import matplotlib.pyplot as plt
import shap

class NeuralExplainer:
    """
    A class to explain NeuralForecast models using SHAP (SHapley Additive exPlanations).

    This class wraps a NeuralForecast model and provides methods to compute SHAP values
    for model predictions, allowing for the interpretation of feature contributions
    to the forecast at different horizons.

    Parameters
    ----------
    model : BaseModel
        The NeuralForecast model to be explained.
    df : Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]], default=None
        DataFrame containing historical data, including the target variable 'y' and
        any historical exogenous variables.
    static_df : Optional[Union[DataFrame, SparkDataFrame]], default=None
        DataFrame containing static exogenous features.
    futr_exog_df : Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]], default=None
        DataFrame containing future exogenous features for the forecast horizon.
    kmeans_dim : int, default=50
        The number of samples to use for K-Means summarization of the background dataset.
        This reduces the computational cost of the SHAP explanation.

    Attributes
    ----------
    model : BaseModel
        The NeuralForecast model being explained.
    explainer : shap.DeepExplainer
        The SHAP DeepExplainer object.
    X_explain_tensor : torch.Tensor
        The input tensor for which SHAP values are to be computed.
    background_tensor : torch.Tensor
        The background dataset tensor used by the SHAP explainer.
    wrapped_model : ModelWrapper
        An internal wrapper for the NeuralForecast model to be compatible with SHAP.
    futr_exog_cols : List[str]
        List of future exogenous column names.
    hist_exog_cols : List[str]
        List of historical exogenous column names.
    stat_exog_cols : List[str]
        List of static exogenous column names.
    feature_names : List[str]
        A comprehensive list of all feature names used in the explanation,
        including lagged historical features and future exogenous features.
    explanations : Dict[int, shap.Explanation]
        A dictionary to store SHAP Explanation objects for each explained horizon.
    all_shap_values : Dict[int, np.ndarray]
        A dictionary to store raw SHAP values for each explained horizon.
    """
    def __init__(
        self,
        model: BaseModel,
        df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        futr_exog_df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
        kmeans_dim: int = 50
    ):
        self.model = model

        self.explainer, self.X_explain_tensor, self.background_tensor, self.wrapped_model = create_explainer(
            model=model, df=df, static_df=static_df,
            futr_exog_df=futr_exog_df, kmeans_dim=kmeans_dim
        )
        self.futr_exog_cols: List[str] = model.futr_exog_list
        self.hist_exog_cols: List[str] = model.hist_exog_list
        self.stat_exog_cols: List[str] = model.stat_exog_list
        self.feature_names: List[str] = create_complete_feature_names(model, self.futr_exog_cols, self.hist_exog_cols)
        self.explanations: dict = {}
        self.all_shap_values: dict = {}

    def get_explanations(
        self,
        horizons_to_explain: List[int],
        check_additivity: bool = True,
        relative_tolerance: float = 0.01
    ) -> None:
        """
        Computes SHAP explanations for the specified horizons.

        Parameters
        ----------
        horizons_to_explain : List[int]
            A list of forecast horizons for which to compute SHAP values.
            Horizons are 0-indexed relative to the model's output.
        check_additivity : bool, default=True
            If True, performs an additivity check to ensure that the sum of SHAP values
            plus the baseline prediction approximates the actual model prediction.
        relative_tolerance : float, default=0.01
            The relative tolerance for the additivity check. Only used if `check_additivity` is True.
        """
        for horizon in horizons_to_explain:
            # Compute SHAP values for the given input tensor.
            # check_additivity is set to False here as we perform a custom check later.
            shap_values: np.ndarray = self.explainer.shap_values(self.X_explain_tensor, check_additivity=False)
            # Extract SHAP values for the specific horizon.
            horizon_shap: np.ndarray = shap_values[0, :, horizon]
            
            if check_additivity:
                test_additivity(
                    horizon=horizon,
                    shap_values=horizon_shap,
                    wrapped_model=self.wrapped_model,
                    X_explain_tensor=self.X_explain_tensor,
                    background_tensor=self.background_tensor,
                    relative_tolerance=relative_tolerance
                )

            self.all_shap_values[horizon] = horizon_shap

            # Calculate the baseline prediction from the mean of the background tensor.
            baseline_input: torch.Tensor = torch.mean(self.background_tensor, dim=0, keepdim=True)
            with torch.no_grad():
                baseline_prediction_tensor: torch.Tensor = self.wrapped_model(baseline_input)
                baseline_prediction: float = baseline_prediction_tensor[0, horizon, 0].item()

            # Create a shap.Explanation object for the current horizon.
            self.explanations[horizon] = shap.Explanation(
                values=horizon_shap,
                base_values=baseline_prediction,  # Use actual baseline prediction
                data=self.X_explain_tensor.numpy().flatten(),
                feature_names=self.feature_names
            )

    def plot(
        self,
        horizon: int,
        max_display: int = 10
    ) -> plt.Figure:
        """
        Generates a SHAP bar plot for a specific forecast horizon.

        Parameters
        ----------
        horizon : int
            The forecast horizon for which to generate the plot.
            This horizon must have been computed by `get_explanations` previously.
        max_display : int, default=10
            The maximum number of features to display in the plot.

        Returns
        -------
        plt.Figure
            The matplotlib Figure object containing the SHAP bar plot.

        Raises
        ------
        AssertionError
            If the SHAP explanation for the given horizon has not been computed yet.
        """
        assert horizon in self.explanations.keys(), f"Horizon: {horizon} is not computed. Call get_explanations with a list containing {horizon}"
        fig, ax = plt.subplots()
        shap.plots.bar(self.explanations[horizon], max_display=max_display, show=False)
        ax.set_title(f"Horizon: {horizon} explanation")
        fig.tight_layout()

        return fig


def test_additivity(
    horizon: int,
    shap_values: np.ndarray,
    wrapped_model: 'ModelWrapper',  # Forward reference for type hinting
    X_explain_tensor: torch.Tensor,
    background_tensor: torch.Tensor,
    relative_tolerance: float = 0.1
) -> None:
    """
    Performs an additivity check for SHAP values, verifying if f(x) approx f(baseline) + sum(SHAP values).

    Parameters
    ----------
    horizon : int
        The forecast horizon being checked.
    shap_values : np.ndarray
        The SHAP values for the specific horizon.
    wrapped_model : ModelWrapper
        The wrapped NeuralForecast model used for predictions.
    X_explain_tensor : torch.Tensor
        The input tensor for which the actual prediction was made.
    background_tensor : torch.Tensor
        The background dataset tensor used by the SHAP explainer.
    relative_tolerance : float, default=0.1
        The allowed relative tolerance for the additivity check. A smaller value
        means a stricter check.
    """
    # Get the model's actual prediction using the wrapper model (for consistency)
    with torch.no_grad():
        actual_prediction_tensor: torch.Tensor = wrapped_model(X_explain_tensor)
        actual_prediction: float = actual_prediction_tensor[0, horizon, 0].item()

    # Get the baseline prediction using the wrapper model
    baseline_input: torch.Tensor = torch.mean(background_tensor, dim=0, keepdim=True)
    with torch.no_grad():
        baseline_prediction_tensor: torch.Tensor = wrapped_model(baseline_input)
        baseline_prediction: float = baseline_prediction_tensor[0, horizon, 0].item()

    # Sum the SHAP values for the prediction
    sum_of_shap_values: float = np.sum(shap_values).item()

    # Perform the verification check
    # DeepExplainer additivity: f(x) = f(baseline) + sum(SHAP values)
    verified_prediction: float = baseline_prediction + sum_of_shap_values

    # Calculate relative error
    absolute_difference: float = abs(verified_prediction - actual_prediction)
    relative_error_percentage: float = (absolute_difference / abs(actual_prediction)) * 100 if actual_prediction != 0 else 0

    if absolute_difference < (abs(actual_prediction) * relative_tolerance):
        print(f"✅ Additivity check PASSED for horizon: {horizon} with relative error: {relative_error_percentage:.2f}%")
    else:
        print(f"❌ Additivity check FAILED for horizon: {horizon} with relative error: {relative_error_percentage:.2f}%")


def create_complete_feature_names(
    model: BaseModel,
    futr_exog_cols: List[str],
    hist_exog_cols: List[str]
) -> List[str]:
    """
    Create comprehensive feature names for all varying features used in the explanation.

    This includes future exogenous features (prefixed with their horizon),
    historical exogenous features (prefixed with their lag), and lagged historical
    target values (prefixed with 'y_lag').

    Parameters
    ----------
    model : BaseModel
        The NeuralForecast model instance, used to get `h` (forecast horizon)
        and `input_size` (historical input size).
    futr_exog_cols : List[str]
        A list of column names for future exogenous variables.
    hist_exog_cols : List[str]
        A list of column names for historical exogenous variables.

    Returns
    -------
    List[str]
        A list of strings, where each string is a descriptive name for a feature.
    """
    feature_names: List[str] = []
    
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


class ModelWrapper(torch.nn.Module):
    """
    Wrapper model that converts a flattened tensor input to the dictionary format
    expected by NeuralForecast models (e.g., MLP, NBEATS, NHITS).

    This wrapper is necessary because SHAP's DeepExplainer expects a model that
    takes a single tensor as input, while NeuralForecast models typically
    expect a dictionary of tensors representing different input components
    (e.g., `insample_y`, `futr_exog`, `hist_exog`, `stat_exog`).

    Parameters
    ----------
    model : BaseModel
        The NeuralForecast model to be wrapped.
    df : Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]], default=None
        DataFrame containing historical data. Used to extract fixed historical
        parts of future exogenous features.
    static_df : Optional[Union[DataFrame, SparkDataFrame]], default=None
        DataFrame containing static exogenous features. Used to extract
        fixed static exogenous features.

    Attributes
    ----------
    model : BaseModel
        The original NeuralForecast model.
    futr_exog_cols : List[str]
        List of future exogenous column names from the model.
    hist_exog_cols : List[str]
        List of historical exogenous column names from the model.
    stat_exog_cols : List[str]
        List of static exogenous column names from the model.
    n_futr_features : int
        Total number of future exogenous features (number of columns * horizon).
    n_hist_target_features : int
        Number of historical target features (model's input size).
    n_hist_exog_features : int
        Total number of historical exogenous features (number of columns * input size).
    stat_exog_fixed : Optional[torch.Tensor]
        Tensor containing fixed static exogenous data, if available.
    futr_hist_fixed : Optional[torch.Tensor]
        Tensor containing the fixed historical part of future exogenous data, if available.
    """
    def __init__(
            self,
            model: BaseModel,
            df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
            static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        ):

        super().__init__()
        self.model = model
        self.futr_exog_cols: List[str] = model.futr_exog_list
        self.hist_exog_cols: List[str] = model.hist_exog_list
        self.stat_exog_cols: List[str] = model.stat_exog_list
        
        # Calculate input dimensions (only for existing feature types)
        self.n_futr_features: int = len(self.futr_exog_cols) * self.model.h if self.futr_exog_cols else 0
        self.n_hist_target_features: int = self.model.input_size
        
        # Check if there are any static variables
        if self.stat_exog_cols and static_df is not None:
            # Gets static data
            self.stat_exog_fixed: Optional[torch.Tensor] = torch.tensor(
                static_df[self.stat_exog_cols].values[0], 
                dtype=torch.float32
            )
        else:
            self.stat_exog_fixed: Optional[torch.Tensor] = None
        
        # Input dimensions for historical exogenous features
        self.n_hist_exog_features: int = len(self.hist_exog_cols) * self.model.input_size if self.hist_exog_cols else 0

        # Checks fixed historical part of future exogenous features   
        if self.futr_exog_cols and df is not None:
            # Gets fixed historical exogenous features
            self.futr_hist_fixed: Optional[torch.Tensor] = torch.tensor(
                df[self.futr_exog_cols].values[-self.model.input_size:], 
                dtype=torch.float32
            )
        else:
            self.futr_hist_fixed: Optional[torch.Tensor] = None
    
    def forward(self, X_flat: torch.Tensor) -> torch.Tensor:
        """
        Converts a flattened tensor input into the dictionary format expected
        by the NeuralForecast model and calls the original model's forward pass.

        Parameters
        ----------
        X_flat : torch.Tensor
            A flattened tensor of shape `[batch_size, n_total_features]`,
            where `n_total_features` is the sum of `n_futr_features`,
            `n_hist_exog_features`, and `n_hist_target_features`.

        Returns
        -------
        torch.Tensor
            The output predictions from the wrapped NeuralForecast model.
        """
        # TODO: Adapt for NHITS and NBEATS - Their forward pass might expect slightly different window_batch keys or tensor shapes.
        batch_size: int = X_flat.shape[0]
        
        # Split the input tensor based on feature types
        idx: int = 0
        
        # Future exogenous features (varying part)
        futr_exog: Optional[torch.Tensor] = None
        if self.futr_exog_cols:
            futr_flat: torch.Tensor = X_flat[:, idx:idx + self.n_futr_features]
            idx += self.n_futr_features
            
            # Reshape the future exogenous features to (batch_size, h, num_futr_exog_cols)
            futr_pred: torch.Tensor = futr_flat.reshape(batch_size, self.model.h, len(self.futr_exog_cols))
            
            # Repeat the fixed historical part of future exogenous features for the batch
            if self.futr_hist_fixed is not None:
                futr_hist: torch.Tensor = self.futr_hist_fixed.unsqueeze(0).repeat(batch_size, 1, 1)
                # Concatenate historical and future parts of future exogenous features
                futr_exog = torch.cat([futr_hist, futr_pred], dim=1)
            else:
                # If no fixed historical part, futr_exog is just the predicted part
                futr_exog = futr_pred # This case might need review depending on model's expectation
        
        # Historical exogenous features (varying part)
        hist_exog: Optional[torch.Tensor] = None
        if self.hist_exog_cols:
            hist_exog_flat: torch.Tensor = X_flat[:, idx:idx + self.n_hist_exog_features]
            hist_exog = hist_exog_flat.reshape(batch_size, self.model.input_size, len(self.hist_exog_cols))
            idx += self.n_hist_exog_features
        
        # Historical target values (always present and varying)
        hist_target_flat: torch.Tensor = X_flat[:, idx:idx + self.n_hist_target_features]
        insample_y: torch.Tensor = hist_target_flat.reshape(batch_size, self.model.input_size)
        
        # Static exogenous features (constant part)
        stat_exog: Optional[torch.Tensor] = None
        if self.stat_exog_cols and self.stat_exog_fixed is not None:
            stat_exog = self.stat_exog_fixed.unsqueeze(0).repeat(batch_size, 1)
        
        # Create windows_batch dictionary, which is the expected input format for NeuralForecast models
        windows_batch: dict = {
            'insample_y': insample_y.unsqueeze(-1), # NeuralForecast models often expect 3D target (batch, seq_len, 1)
            'futr_exog': futr_exog,
            'hist_exog': hist_exog,
            'stat_exog': stat_exog
        }
        
        # Call original model's forward method
        return self.model(windows_batch)
    

def create_complete_input_tensor(
    model: BaseModel,
    df: Union[DataFrame, SparkDataFrame, Sequence[str]],
    futr_exog_df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
) -> torch.Tensor:
    """
    Creates a single flattened tensor containing all varying input features for explanation.

    This tensor combines future exogenous features, historical exogenous features,
    and historical target values, arranged in a way that can be reshaped by `ModelWrapper`.

    Parameters
    ----------
    model : BaseModel
        The NeuralForecast model instance.
    df : Union[DataFrame, SparkDataFrame, Sequence[str]]
        DataFrame containing historical data, used for historical exogenous features
        and historical target values.
    futr_exog_df : Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]], default=None
        DataFrame containing future exogenous data, if applicable.

    Returns
    -------
    torch.Tensor
        A 2D tensor of shape `[1, n_total_features]` containing the concatenated
        and flattened input features for a single sample.
    """
    input_components: List[np.ndarray] = []
    futr_exog_cols: List[str] = model.futr_exog_list
    hist_exog_cols: List[str] = model.hist_exog_list

    # Future exogenous features (if they exist)
    if futr_exog_cols and futr_exog_df is not None:
        # Assuming futr_exog_df contains only the future values for the horizon
        futr_exog_data: np.ndarray = futr_exog_df[futr_exog_cols].to_numpy().flatten()
        input_components.append(futr_exog_data)
    
    # Historical exogenous features (if they exist)
    if hist_exog_cols:
        # Get the last `model.input_size` historical exogenous values
        hist_exog_data: np.ndarray = df[hist_exog_cols].values[-model.input_size:].flatten()
        input_components.append(hist_exog_data)
    
    # Historical target values (always present)
    # Get the last `model.input_size` historical target values
    hist_target_data: np.ndarray = df['y'].values[-model.input_size:]
    input_components.append(hist_target_data)
    
    # Combine all existing features into a single flattened numpy array
    complete_input: np.ndarray
    if input_components:
        complete_input = np.concatenate(input_components)
    else:
        # This case should ideally not be reached if 'y' is always present
        complete_input = np.array([]) # Fallback, though hist_target_data should always be there
    
    # Reshape to [1, -1] for a single sample and convert to torch.Tensor
    return torch.tensor(complete_input, dtype=torch.float32).reshape(1, -1)

def create_complete_background_data(
    model: BaseModel,
    df: Union[DataFrame, SparkDataFrame, Sequence[str]],
) -> np.ndarray:
    """
    Creates a background dataset for SHAP explanation by sampling windows
    of historical, future exogenous, and target data from the provided DataFrame.

    This background dataset is used by DeepExplainer to estimate the expected value
    of the model's output.

    Parameters
    ----------
    model : BaseModel
        The NeuralForecast model instance, used for `input_size` and `h`.
    df : Union[DataFrame, SparkDataFrame, Sequence[str]]
        The DataFrame containing historical data, including target and all
        exogenous features.

    Returns
    -------
    np.ndarray
        A 2D NumPy array where each row represents a flattened sample from
        the background dataset, containing all relevant input features.
    """
    background_samples: List[np.ndarray] = []
    futr_exog_cols: List[str] = model.futr_exog_list
    hist_exog_cols: List[str] = model.hist_exog_list
    
    # Determine the range of valid indices for creating windows
    # `start_idx`: We need at least `model.input_size` historical points.
    start_idx: int = model.input_size  
    # `end_idx`: If future exogenous features exist, we need `model.h` future points
    # beyond the current window's end. So, the last valid `i` is `len(df) - model.h`.
    # If no future exog, we can go up to `len(df)`.
    end_idx: int = len(df) - model.h + 1 if futr_exog_cols else len(df)
    
    # Iterate through the DataFrame to create samples for the background dataset
    # Using a stride of 3 for efficiency to reduce the size of the background dataset.
    for i in range(start_idx, end_idx, 3):  
        sample_components: List[np.ndarray] = []
        
        # Future exogenous features (if they exist)
        if futr_exog_cols:
            # Extract the future exogenous window for the current sample
            futr_window: np.ndarray = df[futr_exog_cols].iloc[i:i + model.h].to_numpy().flatten()
            sample_components.append(futr_window)
        
        # Historical exogenous features (if they exist)
        if hist_exog_cols:
            # Extract the historical exogenous window (lagged features)
            hist_exog_window: np.ndarray = df[hist_exog_cols].iloc[i-model.input_size:i].to_numpy().flatten()
            sample_components.append(hist_exog_window)
        
        # Historical target values (always present)
        # Extract the historical target window (lagged 'y' values)
        hist_target_window: np.ndarray = df['y'].iloc[i-model.input_size:i].to_numpy()
        sample_components.append(hist_target_window)
        
        # Combine all existing features for the current sample
        complete_sample: np.ndarray
        if sample_components:
            complete_sample = np.concatenate(sample_components)
        else:
            # This case should theoretically not be hit as 'y' is always present.
            complete_sample = hist_target_window
        
        background_samples.append(complete_sample)
    
    # Convert the list of samples into a single NumPy array
    return np.array(background_samples)


def create_explainer(
    model: BaseModel,
    df: Union[DataFrame, SparkDataFrame, Sequence[str]],
    static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
    futr_exog_df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
    kmeans_dim: int = 50
) -> tuple[shap.DeepExplainer, torch.Tensor, torch.Tensor, 'ModelWrapper']:
    """
    Initializes and returns the SHAP DeepExplainer, the input tensor to explain,
    the background tensor, and the wrapped model.

    This function orchestrates the preparation of all necessary components
    for SHAP explanation.

    Parameters
    ----------
    model : BaseModel
        The NeuralForecast model to be explained.
    df : Union[DataFrame, SparkDataFrame, Sequence[str]]
        DataFrame containing historical data.
    static_df : Optional[Union[DataFrame, SparkDataFrame]], default=None
        DataFrame containing static exogenous features.
    futr_exog_df : Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]], default=None
        DataFrame containing future exogenous features for the forecast horizon.
    kmeans_dim : int, default=50
        The number of samples to use for K-Means summarization of the background dataset.

    Returns
    -------
    tuple[shap.DeepExplainer, torch.Tensor, torch.Tensor, ModelWrapper]
        A tuple containing:
        - `explainer`: The initialized `shap.DeepExplainer` object.
        - `X_explain_tensor`: The `torch.Tensor` representing the input instance to be explained.
        - `background_tensor`: The `torch.Tensor` representing the summarized background dataset.
        - `wrapped_model`: The `ModelWrapper` instance wrapping the original NeuralForecast model.
    """
    # Wrap the original NeuralForecast model to make it compatible with SHAP DeepExplainer.
    wrapped_model: ModelWrapper = ModelWrapper(
        model = model,
        df=df,
        static_df=static_df
    )

    # Create the input tensor for which SHAP values will be computed.
    X_explain_tensor: torch.Tensor = create_complete_input_tensor(
        model = model,
        df=df,
        futr_exog_df=futr_exog_df
    )
    
    # Create the raw background data array.
    background_array: np.ndarray = create_complete_background_data(
        model = model,
        df=df
    )
    
    # Summarize the background data using K-Means to reduce computational load for SHAP.
    # Use min(kmeans_dim, len(background_array)) to avoid errors if background_array is smaller than kmeans_dim.
    background_summary: Union[np.ndarray, shap.DenseData] = shap.kmeans(background_array, min(kmeans_dim, len(background_array)))

    # Extract the data from the DenseData object if shap.kmeans returns one.
    background_data: np.ndarray
    if hasattr(background_summary, 'data'):
        background_data = background_summary.data
    else:
        background_data = background_summary

    # Convert the background data to a torch.Tensor.
    background_tensor: torch.Tensor = torch.tensor(background_data, dtype=torch.float32)
    
    # Initialize the SHAP DeepExplainer with the wrapped model and the background tensor.
    explainer: shap.DeepExplainer = shap.DeepExplainer(wrapped_model, background_tensor)

    return explainer, X_explain_tensor, background_tensor, wrapped_model