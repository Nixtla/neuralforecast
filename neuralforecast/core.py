# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/core.ipynb.

# %% auto 0
__all__ = ['NeuralForecast']

# %% ../nbs/core.ipynb 4
import os
import pickle
import warnings
from copy import deepcopy
from itertools import chain
from os.path import isfile, join
from typing import Any, List, Optional

import numpy as np
import pandas as pd

from .tsdataset import TimeSeriesDataset
from neuralforecast.models import (
    GRU,
    LSTM,
    RNN,
    TCN,
    DeepAR,
    DilatedRNN,
    MLP,
    NHITS,
    NBEATS,
    NBEATSx,
    TFT,
    VanillaTransformer,
    Informer,
    Autoformer,
    FEDformer,
    StemGNN,
    PatchTST,
    TimesNet,
)

# %% ../nbs/core.ipynb 5
def _cv_dates(last_dates, freq, h, test_size, step_size=1):
    # assuming step_size = 1
    if (test_size - h) % step_size:
        raise Exception("`test_size - h` should be module `step_size`")
    n_windows = int((test_size - h) / step_size) + 1
    if len(np.unique(last_dates)) == 1:
        if issubclass(last_dates.dtype.type, np.integer):
            total_dates = np.arange(last_dates[0] - test_size + 1, last_dates[0] + 1)
            out = np.empty((h * n_windows, 2), dtype=last_dates.dtype)
            freq = 1
        else:
            total_dates = pd.date_range(end=last_dates[0], periods=test_size, freq=freq)
            out = np.empty((h * n_windows, 2), dtype="datetime64[s]")
        for i_window, cutoff in enumerate(
            range(-test_size, -h + 1, step_size), start=0
        ):
            end_cutoff = cutoff + h
            out[h * i_window : h * (i_window + 1), 0] = (
                total_dates[cutoff:]
                if end_cutoff == 0
                else total_dates[cutoff:end_cutoff]
            )
            out[h * i_window : h * (i_window + 1), 1] = np.tile(
                total_dates[cutoff] - freq, h
            )
        dates = pd.DataFrame(
            np.tile(out, (len(last_dates), 1)), columns=["ds", "cutoff"]
        )
    else:
        dates = pd.concat(
            [
                _cv_dates(np.array([ld]), freq, h, test_size, step_size)
                for ld in last_dates
            ]
        )
        dates = dates.reset_index(drop=True)
    return dates

# %% ../nbs/core.ipynb 6
def _insample_dates(uids, last_dates, freq, h, len_series, step_size=1):
    """
    Generate insample dates for `predict_insample` function. Uses `_cv_dates`
    method with separate sizes and last dates for each series.
    """
    if (len(np.unique(last_dates)) == 1) and (len(np.unique(len_series)) == 1):
        # Dates can be generated simulatenously if ld and ls are the same for all series
        dates = _cv_dates(last_dates, freq, h, len_series[0], step_size)
        dates["unique_id"] = np.repeat(uids, len(dates) // len(uids))
    else:
        dates = []
        for ui, ld, ls in zip(uids, last_dates, len_series):
            # Dates have to be generated for each series separately, considering its own ld and ls
            dates_series = _cv_dates(np.array([ld]), freq, h, ls, step_size)
            dates_series["unique_id"] = ui
            dates.append(dates_series)
        dates = pd.concat(dates)
    dates = dates.reset_index(drop=True)
    dates = dates[["unique_id", "ds", "cutoff"]]
    return dates

# %% ../nbs/core.ipynb 7
def _future_dates(dataset, uids, last_dates, freq, h):
    """
    Generate future dates for `predict` function.
    """
    if issubclass(last_dates.dtype.type, np.integer):
        last_date_f = lambda x: np.arange(x + 1, x + 1 + h, dtype=last_dates.dtype)
    else:
        last_date_f = lambda x: pd.date_range(x + freq, periods=h, freq=freq)
    if len(np.unique(last_dates)) == 1:
        dates = np.tile(last_date_f(last_dates[0]), len(dataset))
    else:
        dates = np.hstack([last_date_f(last_date) for last_date in last_dates])
    idx = pd.Index(np.repeat(uids, h), name="unique_id")
    df = pd.DataFrame({"ds": dates}, index=idx)
    return df

# %% ../nbs/core.ipynb 11
MODEL_FILENAME_DICT = {
    "gru": GRU,
    "lstm": LSTM,
    "rnn": RNN,
    "tcn": TCN,
    "deepar": DeepAR,
    "dilatedrnn": DilatedRNN,
    "mlp": MLP,
    "nbeats": NBEATS,
    "nbeatsx": NBEATSx,
    "nhits": NHITS,
    "tft": TFT,
    "vanillatransformer": VanillaTransformer,
    "informer": Informer,
    "autoformer": Autoformer,
    "patchtst": PatchTST,
    "stemgnn": StemGNN,
    "autogru": GRU,
    "autolstm": LSTM,
    "autornn": RNN,
    "autotcn": TCN,
    "autodeepar": DeepAR,
    "autodilatedrnn": DilatedRNN,
    "automlp": MLP,
    "autonbeats": NBEATS,
    "autonbeatsx": NBEATSx,
    "autonhits": NHITS,
    "autotft": TFT,
    "autovanillatransformer": VanillaTransformer,
    "autoinformer": Informer,
    "autoautoformer": Autoformer,
    "autopatchtst": PatchTST,
    "autofedformer": FEDformer,
    "autostemgnn": StemGNN,
    "autotimesnet": TimesNet,
}

# %% ../nbs/core.ipynb 12
class NeuralForecast:
    def __init__(
        self, models: List[Any], freq: str, local_scaler_type: Optional[str] = None
    ):
        """
        The `core.StatsForecast` class allows you to efficiently fit multiple `NeuralForecast` models
        for large sets of time series. It operates with pandas DataFrame `df` that identifies series
        and datestamps with the `unique_id` and `ds` columns. The `y` column denotes the target
        time series variable.

        Parameters
        ----------
        models : List[typing.Any]
            Instantiated `neuralforecast.models`
            see [collection here](https://nixtla.github.io/neuralforecast/models.html).
        freq : str
            Frequency of the data,
            see [panda's available frequencies](https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases).
        local_scaler_type : str, optional (default=None)
            Scaler to apply per-serie to all features before fitting, which is inverted after predicting.
            Can be 'standard', 'robust', 'robust-iqr', 'minmax' or 'boxcox'

        Returns
        -------
        self : NeuralForecast
            Returns instantiated `NeuralForecast` class.
        """
        assert all(
            model.h == models[0].h for model in models
        ), "All models should have the same horizon"

        self.h = models[0].h
        self.models_init = models
        self.models = [deepcopy(model) for model in self.models_init]
        self.freq = pd.tseries.frequencies.to_offset(freq)
        self.local_scaler_type = local_scaler_type

        # Flags and attributes
        self._fitted = False

    def _prepare_fit(self, df, static_df, sort_df, scaler_type):
        # TODO: uids, last_dates and ds should be properties of the dataset class. See github issue.
        dataset, uids, last_dates, ds = TimeSeriesDataset.from_df(
            df=df, static_df=static_df, sort_df=sort_df, scaler_type=scaler_type
        )
        return dataset, uids, last_dates, ds

    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        static_df: Optional[pd.DataFrame] = None,
        val_size: Optional[int] = 0,
        sort_df: bool = True,
        use_init_models: bool = False,
        verbose: bool = False,
    ):
        """Fit the core.NeuralForecast.

        Fit `models` to a large set of time series from DataFrame `df`.
        and store fitted models for later inspection.

        Parameters
        ----------
        df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
            If None, a previously stored dataset is required.
        static_df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`] and static exogenous.
        val_size : int, optional (default=0)
            Size of validation set.
        sort_df : bool, optional (default=False)
            Sort `df` before fitting.
        use_init_models : bool, optional (default=False)
            Use initial model passed when NeuralForecast object was instantiated.
        verbose : bool (default=False)
            Print processing steps.

        Returns
        -------
        self : NeuralForecast
            Returns `NeuralForecast` class with fitted `models`.
        """
        if (df is None) and not (hasattr(self, "dataset")):
            raise Exception("You must pass a DataFrame or have one stored.")

        # Model and datasets interactions protections
        if (any(model.early_stop_patience_steps > 0 for model in self.models)) and (
            val_size == 0
        ):
            raise Exception("Set val_size>0 if early stopping is enabled.")

        # Process and save new dataset (in self)
        if df is not None:
            self.dataset, self.uids, self.last_dates, self.ds = self._prepare_fit(
                df=df,
                static_df=static_df,
                sort_df=sort_df,
                scaler_type=self.local_scaler_type,
            )
            self.sort_df = sort_df
        else:
            if verbose:
                print("Using stored dataset.")

        if val_size is not None:
            if self.dataset.min_size < val_size:
                warnings.warn(
                    "Validation set size is larger than the shorter time-series."
                )

        # Recover initial model if use_init_models
        if use_init_models:
            self.models = [deepcopy(model) for model in self.models_init]
            if self._fitted:
                print("WARNING: Deleting previously fitted models.")

        for model in self.models:
            model.fit(self.dataset, val_size=val_size)

        self._fitted = True

    def predict(
        self,
        df: Optional[pd.DataFrame] = None,
        static_df: Optional[pd.DataFrame] = None,
        futr_df: Optional[pd.DataFrame] = None,
        sort_df: bool = True,
        verbose: bool = False,
        **data_kwargs,
    ):
        """Predict with core.NeuralForecast.

        Use stored fitted `models` to predict large set of time series from DataFrame `df`.

        Parameters
        ----------
        df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
            If a DataFrame is passed, it is used to generate forecasts.
        static_df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`] and static exogenous.
        futr_df : pandas.DataFrame, optional (default=None)
            DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous.
        sort_df : bool (default=True)
            Sort `df` before fitting.
        verbose : bool (default=False)
            Print processing steps.
        data_kwargs : kwargs
            Extra arguments to be passed to the dataset within each model.

        Returns
        -------
        fcsts_df : pandas.DataFrame
            DataFrame with insample `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
        """
        if (df is None) and not (hasattr(self, "dataset")):
            raise Exception("You must pass a DataFrame or have one stored.")

        if not self._fitted:
            raise Exception("You must fit the model before predicting.")

        needed_futr_exog = set(
            chain.from_iterable(getattr(m, "futr_exog_list", []) for m in self.models)
        )
        if needed_futr_exog:
            if futr_df is None:
                raise ValueError(
                    f"Models require the following future exogenous features: {needed_futr_exog}. "
                    "Please provide them through the `futr_df` argument."
                )
            else:
                missing = needed_futr_exog - set(futr_df.columns)
                if missing:
                    raise ValueError(
                        f"The following features are missing from `futr_df`: {missing}"
                    )

        # Process new dataset but does not store it.
        if df is not None:
            dataset, uids, last_dates, _ = self._prepare_fit(
                df=df, static_df=static_df, sort_df=sort_df, scaler_type=None
            )
            dataset.scalers_ = self.dataset.scalers_
            dataset._transform_temporal()
        else:
            dataset = self.dataset
            uids = self.uids
            last_dates = self.last_dates
            if verbose:
                print("Using stored dataset.")

        cols = []
        count_names = {"model": 0}
        for model in self.models:
            model_name = repr(model)
            count_names[model_name] = count_names.get(model_name, -1) + 1
            if count_names[model_name] > 0:
                model_name += str(count_names[model_name])
            cols += [model_name + n for n in model.loss.output_names]

        # Placeholder dataframe for predictions with unique_id and ds
        fcsts_df = _future_dates(
            dataset=dataset, uids=uids, last_dates=last_dates, freq=self.freq, h=self.h
        )

        # Update and define new forecasting dataset
        if futr_df is not None:
            futr_orig_rows = futr_df.shape[0]
            futr_df = futr_df.merge(fcsts_df, on=["unique_id", "ds"])
            base_err_msg = f"`futr_df` must have one row per id and ds in the forecasting horizon ({self.h})."
            if futr_df.shape[0] < fcsts_df.shape[0]:
                raise ValueError(base_err_msg)
            if futr_orig_rows > futr_df.shape[0]:
                dropped_rows = futr_orig_rows - futr_df.shape[0]
                warnings.warn(
                    f"Dropped {dropped_rows:,} unused rows from `futr_df`. "
                    + base_err_msg
                )
            if any(futr_df[col].isnull().any() for col in needed_futr_exog):
                raise ValueError("Found null values in `futr_df`")
            dataset = TimeSeriesDataset.update_dataset(
                dataset=dataset, future_df=futr_df
            )
        else:
            dataset = TimeSeriesDataset.update_dataset(
                dataset=dataset, future_df=fcsts_df.reset_index()
            )

        col_idx = 0
        fcsts = np.full((self.h * len(uids), len(cols)), fill_value=np.nan)
        for model in self.models:
            old_test_size = model.get_test_size()
            model.set_test_size(self.h)  # To predict h steps ahead
            model_fcsts = model.predict(dataset=dataset, **data_kwargs)
            # Append predictions in memory placeholder
            output_length = len(model.loss.output_names)
            fcsts[:, col_idx : col_idx + output_length] = model_fcsts
            col_idx += output_length
            model.set_test_size(old_test_size)  # Set back to original value
        if self.dataset.scalers_ is not None:
            indptr = np.append(0, np.full(len(uids), self.h).cumsum())
            fcsts = self.dataset._invert_target_transform(fcsts, indptr)

        # Declare predictions pd.DataFrame
        fcsts = pd.DataFrame.from_records(fcsts, columns=cols, index=fcsts_df.index)
        fcsts_df = pd.concat([fcsts_df, fcsts], axis=1)

        return fcsts_df

    def cross_validation(
        self,
        df: Optional[pd.DataFrame] = None,
        static_df: Optional[pd.DataFrame] = None,
        n_windows: int = 1,
        step_size: int = 1,
        val_size: Optional[int] = 0,
        test_size: Optional[int] = None,
        sort_df: bool = True,
        use_init_models: bool = False,
        verbose: bool = False,
        **data_kwargs,
    ):
        """Temporal Cross-Validation with core.NeuralForecast.

        `core.NeuralForecast`'s cross-validation efficiently fits a list of NeuralForecast
        models through multiple windows, in either chained or rolled manner.

        Parameters
        ----------
        df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
            If None, a previously stored dataset is required.
        static_df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`] and static exogenous.
        n_windows : int (default=1)
            Number of windows used for cross validation.
        step_size : int (default=1)
            Step size between each window.
        val_size : int, optional (default=None)
            Length of validation size. If passed, set `n_windows=None`.
        test_size : int, optional (default=None)
            Length of test size. If passed, set `n_windows=None`.
        sort_df : bool (default=True)
            Sort `df` before fitting.
        use_init_models : bool, option (default=False)
            Use initial model passed when object was instantiated.
        verbose : bool (default=False)
            Print processing steps.
        data_kwargs : kwargs
            Extra arguments to be passed to the dataset within each model.

        Returns
        -------
        fcsts_df : pandas.DataFrame
            DataFrame with insample `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
        """
        if (df is None) and not (hasattr(self, "dataset")):
            raise Exception("You must pass a DataFrame or have one stored.")

        # Process and save new dataset (in self)
        if df is not None:
            self.dataset, self.uids, self.last_dates, self.ds = self._prepare_fit(
                df=df,
                static_df=static_df,
                sort_df=sort_df,
                scaler_type=self.local_scaler_type,
            )
            self.sort_df = sort_df
        else:
            if verbose:
                print("Using stored dataset.")

        # Recover initial model if use_init_models.
        if use_init_models:
            self.models = [deepcopy(model) for model in self.models_init]
            if self._fitted:
                print("WARNING: Deleting previously fitted models.")

        cols = []
        count_names = {"model": 0}
        for model in self.models:
            model_name = repr(model)
            count_names[model_name] = count_names.get(model_name, -1) + 1
            if count_names[model_name] > 0:
                model_name += str(count_names[model_name])
            cols += [model_name + n for n in model.loss.output_names]

        h = self.models[0].h
        if test_size is None:
            test_size = h + step_size * (n_windows - 1)
        elif n_windows is None:
            if (test_size - h) % step_size:
                raise Exception("`test_size - h` should be module `step_size`")
            n_windows = int((test_size - h) / step_size) + 1
        elif (n_windows is None) and (test_size is None):
            raise Exception("you must define `n_windows` or `test_size`")
        else:
            raise Exception("you must define `n_windows` or `test_size` but not both")

        if val_size is not None:
            if self.dataset.min_size < (val_size + test_size):
                warnings.warn(
                    "Validation and test sets are larger than the shorter time-series."
                )

        fcsts_df = _cv_dates(
            last_dates=self.last_dates,
            freq=self.freq,
            h=h,
            test_size=test_size,
            step_size=step_size,
        )
        idx = pd.Index(np.repeat(self.uids, h * n_windows), name="unique_id")
        fcsts_df.index = idx

        col_idx = 0
        fcsts = np.full(
            (self.dataset.n_groups * h * n_windows, len(cols)), np.nan, dtype=np.float32
        )

        for model in self.models:
            model.fit(dataset=self.dataset, val_size=val_size, test_size=test_size)
            model_fcsts = model.predict(
                self.dataset, step_size=step_size, **data_kwargs
            )

            # Append predictions in memory placeholder
            output_length = len(model.loss.output_names)
            fcsts[:, col_idx : (col_idx + output_length)] = model_fcsts
            col_idx += output_length
        if self.dataset.scalers_ is not None:
            indptr = np.append(
                0, np.full(self.dataset.n_groups, self.h * n_windows).cumsum()
            )
            fcsts = self.dataset._invert_target_transform(fcsts, indptr)

        self._fitted = True

        # Add predictions to forecasts DataFrame
        fcsts = pd.DataFrame.from_records(fcsts, columns=cols, index=fcsts_df.index)
        fcsts_df = pd.concat([fcsts_df, fcsts], axis=1)

        # Add original input df's y to forecasts DataFrame
        fcsts_df = fcsts_df.merge(df, how="left", on=["unique_id", "ds"])
        return fcsts_df

    def predict_insample(self, step_size: int = 1):
        """Predict insample with core.NeuralForecast.

        `core.NeuralForecast`'s `predict_insample` uses stored fitted `models`
        to predict historic values of a time series from the stored dataframe.

        Parameters
        ----------
        step_size : int (default=1)
            Step size between each window.

        Returns
        -------
        fcsts_df : pandas.DataFrame
            DataFrame with insample predictions for all fitted `models`.
        """
        if not self._fitted:
            raise Exception(
                "The models must be fitted first with `fit` or `cross_validation`."
            )

        for model in self.models:
            if model.SAMPLING_TYPE == "recurrent":
                warnings.warn(
                    f"Predict insample might not provide accurate predictions for \
                       recurrent model {repr(model)} class yet due to scaling."
                )
                print(
                    f"WARNING: Predict insample might not provide accurate predictions for \
                      recurrent model {repr(model)} class yet due to scaling."
                )

        cols = []
        count_names = {"model": 0}
        for model in self.models:
            model_name = repr(model)
            count_names[model_name] = count_names.get(model_name, -1) + 1
            if count_names[model_name] > 0:
                model_name += str(count_names[model_name])
            cols += [model_name + n for n in model.loss.output_names]

        # Remove test set from dataset and last dates
        test_size = self.models[0].get_test_size()
        if test_size > 0:
            trimmed_dataset = TimeSeriesDataset.trim_dataset(
                dataset=self.dataset, right_trim=test_size, left_trim=0
            )
            last_dates_train = self.last_dates.shift(-test_size, freq=self.freq)
        else:
            trimmed_dataset = self.dataset
            last_dates_train = self.last_dates

        # Generate dates
        len_series = np.diff(
            trimmed_dataset.indptr
        )  # Computes the length of each time series based on indptr
        fcsts_df = _insample_dates(
            uids=self.uids,
            last_dates=last_dates_train,
            freq=self.freq,
            h=self.h,
            len_series=len_series,
            step_size=step_size,
        )
        fcsts_df = fcsts_df.set_index("unique_id")

        col_idx = 0
        fcsts = np.full((len(fcsts_df), len(cols)), np.nan, dtype=np.float32)

        for model in self.models:
            # Test size is the number of periods to forecast (full size of trimmed dataset)
            model.set_test_size(test_size=trimmed_dataset.max_size)

            # Predict
            model_fcsts = model.predict(trimmed_dataset, step_size=step_size)
            # Append predictions in memory placeholder
            output_length = len(model.loss.output_names)
            fcsts[:, col_idx : (col_idx + output_length)] = model_fcsts
            col_idx += output_length
            model.set_test_size(test_size=test_size)  # Set original test_size

        # Add predictions to forecasts DataFrame
        fcsts = pd.DataFrame.from_records(fcsts, columns=cols, index=fcsts_df.index)
        fcsts_df = pd.concat([fcsts_df, fcsts], axis=1)

        # Add original input df's y to forecasts DataFrame
        Y_df = pd.DataFrame.from_records(
            self.dataset.temporal[:, [0]].numpy(), columns=["y"], index=self.ds
        )
        Y_df = Y_df.reset_index(drop=False)
        fcsts_df = fcsts_df.merge(Y_df, how="left", on=["unique_id", "ds"])
        if self.dataset.scalers_ is not None:
            sizes = fcsts_df.groupby("unique_id", observed=True).size().values
            indptr = np.append(0, sizes.cumsum())
            invert_cols = cols + ["y"]
            fcsts_df[invert_cols] = self.dataset._invert_target_transform(
                fcsts_df[invert_cols].values, indptr
            )

        return fcsts_df

    # Save list of models with pytorch lightning save_checkpoint function
    def save(
        self,
        path: str,
        model_index: Optional[List] = None,
        save_dataset: bool = True,
        overwrite: bool = False,
    ):
        """Save NeuralForecast core class.

        `core.NeuralForecast`'s method to save current status of models, dataset, and configuration.
        Note that by default the `models` are not saving training checkpoints to save disk memory,
        to get them change the individual model `**trainer_kwargs` to include `enable_checkpointing=True`.

        Parameters
        ----------
        path : str
            Directory to save current status.
        model_index : list, optional (default=None)
            List to specify which models from list of self.models to save.
        save_dataset : bool (default=True)
            Whether to save dataset or not.
        overwrite : bool (default=False)
            Whether to overwrite files or not.
        """
        # Standarize path without '/'
        if path[-1] == "/":
            path = path[:-1]

        # Model index list
        if model_index is None:
            model_index = list(range(len(self.models)))

        # Create directory if not exists
        os.makedirs(path, exist_ok=True)

        # Check if directory is empty to protect overwriting files
        dir = os.listdir(path)

        # Checking if the list is empty or not
        if (len(dir) > 0) and (not overwrite):
            raise Exception(
                "Directory is not empty. Set `overwrite=True` to overwrite files."
            )

        # Save models
        count_names = {"model": 0}
        for i, model in enumerate(self.models):
            # Skip model if not in list
            if i not in model_index:
                continue

            model_name = repr(model).lower().replace("_", "")
            count_names[model_name] = count_names.get(model_name, -1) + 1
            model.save(f"{path}/{model_name}_{count_names[model_name]}.ckpt")

        # Save dataset
        if (save_dataset) and (hasattr(self, "dataset")):
            with open(f"{path}/dataset.pkl", "wb") as f:
                pickle.dump(self.dataset, f)
        elif save_dataset:
            raise Exception(
                "You need to have a stored dataset to save it, \
                             set `save_dataset=False` to skip saving dataset."
            )

        # Save configuration and parameters
        config_dict = {
            "h": self.h,
            "freq": self.freq,
            "uids": self.uids,
            "last_dates": self.last_dates,
            "ds": self.ds,
            "sort_df": self.sort_df,
            "_fitted": self._fitted,
        }

        with open(f"{path}/configuration.pkl", "wb") as f:
            pickle.dump(config_dict, f)

    @staticmethod
    def load(path, verbose=False, **kwargs):
        """Load NeuralForecast

        `core.NeuralForecast`'s method to load checkpoint from path.

        Parameters
        -----------
        path : str
            Directory to save current status.
        kwargs
            Additional keyword arguments to be passed to the function
            `load_from_checkpoint`.

        Returns
        -------
        result : NeuralForecast
            Instantiated `NeuralForecast` class.
        """
        files = [f for f in os.listdir(path) if isfile(join(path, f))]

        # Load models
        models_ckpt = [f for f in files if f.endswith(".ckpt")]
        if len(models_ckpt) == 0:
            raise Exception("No model found in directory.")

        if verbose:
            print(10 * "-" + " Loading models " + 10 * "-")
        models = []
        for model in models_ckpt:
            model_name = model.split("_")[0]
            models.append(
                MODEL_FILENAME_DICT[model_name].load_from_checkpoint(
                    f"{path}/{model}", **kwargs
                )
            )
            if verbose:
                print(f"Model {model_name} loaded.")

        if verbose:
            print(10 * "-" + " Loading dataset " + 10 * "-")
        # Load dataset
        if "dataset.pkl" in files:
            with open(f"{path}/dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
            if verbose:
                print("Dataset loaded.")
        else:
            dataset = None
            if verbose:
                print("No dataset found in directory.")

        if verbose:
            print(10 * "-" + " Loading configuration " + 10 * "-")
        # Load configuration
        if "configuration.pkl" in files:
            with open(f"{path}/configuration.pkl", "rb") as f:
                config_dict = pickle.load(f)
            if verbose:
                print("Configuration loaded.")
        else:
            raise Exception("No configuration found in directory.")

        # Create NeuralForecast object
        neuralforecast = NeuralForecast(models=models, freq=config_dict["freq"])

        # Dataset
        if dataset is not None:
            neuralforecast.dataset = dataset
            neuralforecast.uids = config_dict["uids"]
            neuralforecast.last_dates = config_dict["last_dates"]
            neuralforecast.ds = config_dict["ds"]
            neuralforecast.sort_df = config_dict["sort_df"]

        # Fitted flag
        neuralforecast._fitted = config_dict["_fitted"]

        return neuralforecast
