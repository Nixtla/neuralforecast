# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/core.ipynb.

# %% auto 0
__all__ = ['NeuralForecast']

# %% ../nbs/core.ipynb 4
import os
import pickle
import warnings

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
    DilatedRNN,
    MLP,
    NHITS,
    NBEATS,
    NBEATSx,
    TFT,
    VanillaTransformer,
    Informer,
    Autoformer,
    StemGNN,
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

# %% ../nbs/core.ipynb 9
MODEL_FILENAME_DICT = {
    "gru": GRU,
    "lstm": LSTM,
    "rnn": RNN,
    "tcn": TCN,
    "dilatedrnn": DilatedRNN,
    "mlp": MLP,
    "nbeats": NBEATS,
    "nbeatsx": NBEATSx,
    "nhits": NHITS,
    "tft": TFT,
    "stemgnn": StemGNN,
    "informer": Informer,
    "autogru": GRU,
    "autolstm": LSTM,
    "autornn": RNN,
    "autotcn": TCN,
    "autodilatedrnn": DilatedRNN,
    "automlp": MLP,
    "autonbeats": NBEATS,
    "autonhits": NHITS,
    "autotft": TFT,
    "autovanillatransformer": VanillaTransformer,
    "autoinformer": Informer,
    "autoautoformer": Autoformer,
    "autostemgnn": StemGNN,
}

# %% ../nbs/core.ipynb 10
class NeuralForecast:
    def __init__(self, models: List[Any], freq: str):
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

        Returns
        -------
        self : NeuralForecast
            Returns instantiated `NeuralForecast` class.
        """
        assert all(
            model.h == models[0].h for model in models
        ), "All models should have the same horizon"

        self.h = models[0].h
        self.models = models
        self.freq = pd.tseries.frequencies.to_offset(freq)

        # Flags and attributes
        self._fitted = False

    def _prepare_fit(self, df, static_df, sort_df):
        # TODO: uids, last_dates and ds should be properties of the dataset class. See github issue.
        self.dataset, self.uids, self.last_dates, self.ds = TimeSeriesDataset.from_df(
            df=df, static_df=static_df, sort_df=sort_df
        )
        self.sort_df = sort_df

    def fit(
        self,
        df: Optional[pd.DataFrame] = None,
        static_df: Optional[pd.DataFrame] = None,
        val_size: Optional[int] = 0,
        sort_df: bool = True,
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
            DataFrame with columns [`unique_id`, `ds`] and static exogenous.
        val_size : int, optional (default=0)
            Size of validation set.
        sort_df : bool, optional (default=False)
            Sort `df` before fitting.
        verbose : bool (default=False)
            Print processing steps.

        Returns
        -------
        self : NeuralForecast
            Returns `NeuralForecast` class with fitted `models`.
        """
        if (df is None) and not (hasattr(self, "dataset")):
            raise Exception("You must pass a DataFrame or have one stored.")

        # Process and save new dataset (in self)
        if df is not None:
            self._prepare_fit(df=df, static_df=static_df, sort_df=sort_df)
        else:
            if verbose:
                print("Using stored dataset.")

        if val_size is not None:
            if self.dataset.min_size < val_size:
                warnings.warn(
                    "Validation set size is larger than the shorter time-series."
                )

        # train + validation
        for model in self.models:
            model.fit(self.dataset, val_size=val_size)
        # train with the full dataset

        self._fitted = True

    def _make_future_df(self, h: int):
        if issubclass(self.last_dates.dtype.type, np.integer):
            last_date_f = lambda x: np.arange(
                x + 1, x + 1 + h, dtype=self.last_dates.dtype
            )
        else:
            last_date_f = lambda x: pd.date_range(
                x + self.freq, periods=h, freq=self.freq
            )
        if len(np.unique(self.last_dates)) == 1:
            dates = np.tile(last_date_f(self.last_dates[0]), len(self.dataset))
        else:
            dates = np.hstack([last_date_f(last_date) for last_date in self.last_dates])
        idx = pd.Index(np.repeat(self.uids, h), name="unique_id")
        df = pd.DataFrame({"ds": dates}, index=idx)
        return df

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
            DataFrame with columns [`unique_id`, `ds`] and static exogenous.
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

        # Process and save new dataset (in self)
        if df is not None:
            self._prepare_fit(df=df, static_df=static_df, sort_df=sort_df)
        else:
            if verbose:
                print("Using stored dataset.")

        cols = []
        count_names = {"model": 0}
        for model in self.models:
            model_name = type(model).__name__
            count_names[model_name] = count_names.get(model_name, -1) + 1
            if count_names[model_name] > 0:
                model_name += str(count_names[model_name])
            cols += [model_name + n for n in model.loss.output_names]

        # Placeholder dataframe for predictions with unique_id and ds
        fcsts_df = self._make_future_df(h=self.h)

        # Update and define new forecasting dataset
        if futr_df is not None:
            dataset = TimeSeriesDataset.update_dataset(
                dataset=self.dataset, future_df=futr_df
            )
        else:
            dataset = TimeSeriesDataset.update_dataset(
                dataset=self.dataset, future_df=fcsts_df.reset_index()
            )

        col_idx = 0
        fcsts = np.full((self.h * len(self.uids), len(cols)), fill_value=np.nan)
        for model in self.models:
            model.set_test_size(self.h)  # To predict h steps ahead
            model_fcsts = model.predict(dataset=dataset, **data_kwargs)
            # Append predictions in memory placeholder
            output_length = len(model.loss.output_names)
            fcsts[:, col_idx : col_idx + output_length] = model_fcsts
            col_idx += output_length

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
        fit_models: bool = True,
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
            DataFrame with columns [`unique_id`, `ds`] and static exogenous.
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
        fit_models: bool (default=True)
            Fit models before cross-validation.
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

        # Declare predictions pd.DataFrame
        if df is not None:
            self._prepare_fit(df=df, static_df=static_df, sort_df=sort_df)
        else:
            if verbose:
                print("Using stored dataset.")

        cols = []
        count_names = {"model": 0}
        for model in self.models:
            model_name = type(model).__name__
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
            # Fit
            if fit_models:
                model.fit(dataset=self.dataset, val_size=val_size, test_size=test_size)
            else:
                model.set_test_size(test_size=test_size)

            # Predict
            model_fcsts = model.predict(
                self.dataset, step_size=step_size, **data_kwargs
            )

            # Append predictions in memory placeholder
            output_length = len(model.loss.output_names)
            fcsts[:, col_idx : (col_idx + output_length)] = model_fcsts
            col_idx += output_length

        # Add predictions to forecasts DataFrame
        fcsts = pd.DataFrame.from_records(fcsts, columns=cols, index=fcsts_df.index)
        fcsts_df = pd.concat([fcsts_df, fcsts], axis=1)

        # Add original input df's y to forecasts DataFrame
        fcsts_df = fcsts_df.merge(df, how="left", on=["unique_id", "ds"])
        return fcsts_df

    def predict_rolled(
        self,
        df: Optional[pd.DataFrame] = None,
        static_df: Optional[pd.DataFrame] = None,
        n_windows: int = 1,
        step_size: int = 1,
        insample_size: Optional[int] = None,
        sort_df: bool = True,
        verbose: bool = False,
        **data_kwargs,
    ):
        """Predict insample with core.NeuralForecast.

        Use stored fitted `models` to predict historic values of a time series from DataFrame `df`.

        Parameters
        ----------
        df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
            If None, a previously stored dataset is required.
        static_df : pandas.DataFrame, optional (default=None)
            DataFrame with columns [`unique_id`, `ds`] and static exogenous.
        n_windows : int (default=1)
            Number of windows used for cross validation.
        step_size : int (default=1)
            Step size between each window.
        insample_size : int, optional (default=None)
            Length of insample size to produce forecasts. If passed, set `n_windows=None`.
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
        fcsts_df = self.cross_validation(
            df=df,
            static_df=static_df,
            n_windows=n_windows,
            step_size=step_size,
            val_size=0,
            test_size=insample_size,
            sort_df=sort_df,
            fit_models=False,
            verbose=verbose,
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

            model_name = type(model).__name__.lower().replace("_", "")
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
    def load(path, verbose=False):
        """Load NeuralForecast

        `core.NeuralForecast`'s method to load checkpoint from path.

        Parameters
        -----------
        path : str
            Directory to save current status.

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
                MODEL_FILENAME_DICT[model_name].load_from_checkpoint(f"{path}/{model}")
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
