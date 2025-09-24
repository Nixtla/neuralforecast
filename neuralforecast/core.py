__all__ = ['NeuralForecast']


import pickle
import warnings
from copy import deepcopy
from itertools import chain
from typing import Any, Dict, List, Optional, Sequence, Union

import fsspec
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import utilsforecast.processing as ufp
from coreforecast.grouped_array import GroupedArray
from coreforecast.scalers import (
    LocalBoxCoxScaler,
    LocalMinMaxScaler,
    LocalRobustScaler,
    LocalStandardScaler,
)
from utilsforecast.compat import DataFrame, DFType, Series, pl_DataFrame, pl_Series
from utilsforecast.validation import validate_freq
from neuralforecast.common.enums import ExplainerEnum
from neuralforecast.models import (
    GRU,
    KAN,
    LSTM,
    MLP,
    NBEATS,
    NHITS,
    RNN,
    SOFTS,
    TCN,
    TFT,
    Autoformer,
    BiTCN,
    DeepAR,
    DeepNPTS,
    DilatedRNN,
    DLinear,
    FEDformer,
    Informer,
    MLPMultivariate,
    NBEATSx,
    NLinear,
    PatchTST,
    RMoK,
    StemGNN,
    TiDE,
    TimeLLM,
    TimeMixer,
    TimesNet,
    TimeXer,
    TSMixer,
    TSMixerx,
    VanillaTransformer,
    iTransformer,
    xLSTM,
)
from neuralforecast.tsdataset import (
    LocalFilesTimeSeriesDataset,
    TimeSeriesDataset,
    _FilesDataset,
)
from neuralforecast.utils import (
    PredictionIntervals,
    get_prediction_interval_method,
    level_to_quantiles,
    quantiles_to_level,
)

from .common._base_auto import BaseAuto, MockTrial
from .common._base_model import DistributedConfig
from .compat import SparkDataFrame
from .losses.pytorch import HuberIQLoss, IQLoss

# this disables warnings about the number of workers in the dataloaders
# which the user can't control
warnings.filterwarnings("ignore", category=pl.utilities.warnings.PossibleUserWarning)


def _insample_times(
    times: np.ndarray,
    uids: Series,
    indptr: np.ndarray,
    h: int,
    freq: Union[int, str, pd.offsets.BaseOffset],
    step_size: int = 1,
    id_col: str = "unique_id",
    time_col: str = "ds",
) -> DataFrame:
    sizes = np.diff(indptr)
    if (sizes < h).any():
        raise ValueError("`sizes` should be greater or equal to `h`.")
    # TODO: we can just truncate here instead of raising an error
    ns, resids = np.divmod(sizes - h, step_size)
    if (resids != 0).any():
        raise ValueError("`sizes - h` should be multiples of `step_size`")
    windows_per_serie = ns + 1
    # determine the offsets for the cutoffs, e.g. 2 means the 3rd training date is a cutoff
    cutoffs_offsets = step_size * np.hstack([np.arange(w) for w in windows_per_serie])
    # start index of each serie, e.g. [0, 17] means the the second serie starts on the 18th entry
    # we repeat each of these as many times as we have windows, e.g. windows_per_serie = [2, 3]
    # would yield [0, 0, 17, 17, 17]
    start_idxs = np.repeat(indptr[:-1], windows_per_serie)
    # determine the actual indices of the cutoffs, we repeat the cutoff for the complete horizon
    # e.g. if we have two series and h=2 this could be [0, 0, 1, 1, 17, 17, 18, 18]
    # which would have the first two training dates from each serie as the cutoffs
    cutoff_idxs = np.repeat(start_idxs + cutoffs_offsets, h)
    cutoffs = times[cutoff_idxs]
    total_windows = windows_per_serie.sum()
    # determine the offsets for the actual dates. this is going to be [0, ..., h] repeated
    ds_offsets = np.tile(np.arange(h), total_windows)
    # determine the actual indices of the times
    # e.g. if we have two series and h=2 this could be [0, 1, 1, 2, 17, 18, 18, 19]
    ds_idxs = cutoff_idxs + ds_offsets
    ds = times[ds_idxs]
    if isinstance(uids, pl_Series):
        df_constructor = pl_DataFrame
    else:
        df_constructor = pd.DataFrame
    out = df_constructor(
        {
            id_col: ufp.repeat(uids, h * windows_per_serie),
            time_col: ds,
            "cutoff": cutoffs,
        }
    )
    # the first cutoff is before the first train date
    actual_cutoffs = ufp.offset_times(out["cutoff"], freq, -1)
    out = ufp.assign_columns(out, "cutoff", actual_cutoffs)
    return out


MODEL_FILENAME_DICT = {
    "autoformer": Autoformer,
    "autoautoformer": Autoformer,
    "deepar": DeepAR,
    "autodeepar": DeepAR,
    "dlinear": DLinear,
    "autodlinear": DLinear,
    "nlinear": NLinear,
    "autonlinear": NLinear,
    "dilatedrnn": DilatedRNN,
    "autodilatedrnn": DilatedRNN,
    "fedformer": FEDformer,
    "autofedformer": FEDformer,
    "gru": GRU,
    "autogru": GRU,
    "informer": Informer,
    "autoinformer": Informer,
    "lstm": LSTM,
    "autolstm": LSTM,
    "mlp": MLP,
    "automlp": MLP,
    "nbeats": NBEATS,
    "autonbeats": NBEATS,
    "nbeatsx": NBEATSx,
    "autonbeatsx": NBEATSx,
    "nhits": NHITS,
    "autonhits": NHITS,
    "patchtst": PatchTST,
    "autopatchtst": PatchTST,
    "rnn": RNN,
    "autornn": RNN,
    "stemgnn": StemGNN,
    "autostemgnn": StemGNN,
    "tcn": TCN,
    "autotcn": TCN,
    "tft": TFT,
    "autotft": TFT,
    "timesnet": TimesNet,
    "autotimesnet": TimesNet,
    "vanillatransformer": VanillaTransformer,
    "autovanillatransformer": VanillaTransformer,
    "timellm": TimeLLM,
    "tsmixer": TSMixer,
    "autotsmixer": TSMixer,
    "tsmixerx": TSMixerx,
    "autotsmixerx": TSMixerx,
    "mlpmultivariate": MLPMultivariate,
    "automlpmultivariate": MLPMultivariate,
    "itransformer": iTransformer,
    "autoitransformer": iTransformer,
    "bitcn": BiTCN,
    "autobitcn": BiTCN,
    "tide": TiDE,
    "autotide": TiDE,
    "deepnpts": DeepNPTS,
    "autodeepnpts": DeepNPTS,
    "softs": SOFTS,
    "autosofts": SOFTS,
    "timemixer": TimeMixer,
    "autotimemixer": TimeMixer,
    "kan": KAN,
    "autokan": KAN,
    "rmok": RMoK,
    "autormok": RMoK,
    "timexer": TimeXer,
    "autotimexer": TimeXer,
    "xlstm": xLSTM,
    "autoxlstm": xLSTM,
}


_type2scaler = {
    "standard": LocalStandardScaler,
    "robust": lambda: LocalRobustScaler(scale="mad"),
    "robust-iqr": lambda: LocalRobustScaler(scale="iqr"),
    "minmax": LocalMinMaxScaler,
    "boxcox": lambda: LocalBoxCoxScaler(method="loglik", lower=0.0),
}


class NeuralForecast:
    models: List[Any]

    def __init__(
        self,
        models: List[Any],
        freq: Union[str, int],
        local_scaler_type: Optional[str] = None,
    ):
        """The `core.StatsForecast` class allows you to efficiently fit multiple `NeuralForecast` models
        for large sets of time series. It operates with pandas DataFrame `df` that identifies series
        and datestamps with the `unique_id` and `ds` columns. The `y` column denotes the target
        time series variable.

        Args:
            models (List[typing.Any]): Instantiated `neuralforecast.models`
                see [collection here](./models).
            freq (str or int): Frequency of the data. Must be a valid pandas or polars offset alias, or an integer.
            local_scaler_type (str, optional): Scaler to apply per-serie to all features before fitting, which is inverted after predicting.
                Can be 'standard', 'robust', 'robust-iqr', 'minmax' or 'boxcox'. Defaults to None.

        Returns:
            NeuralForecast: Returns instantiated `NeuralForecast` class.
        """
        assert all(
            model.h == models[0].h for model in models
        ), "All models should have the same horizon"

        self.h = models[0].h
        self.models_init = models
        self.freq = freq
        if local_scaler_type is not None and local_scaler_type not in _type2scaler:
            raise ValueError(f"scaler_type must be one of {_type2scaler.keys()}")
        self.local_scaler_type = local_scaler_type
        self.scalers_: Dict

        # Flags and attributes
        self._fitted = False
        self._reset_models()
        self._add_level = False

    def _scalers_fit_transform(self, dataset: TimeSeriesDataset) -> None:
        self.scalers_ = {}
        if self.local_scaler_type is None:
            return None
        for i, col in enumerate(dataset.temporal_cols):
            if col == "available_mask":
                continue
            ga = GroupedArray(dataset.temporal[:, i].numpy(), dataset.indptr)
            self.scalers_[col] = _type2scaler[self.local_scaler_type]().fit(ga)
            dataset.temporal[:, i] = torch.from_numpy(self.scalers_[col].transform(ga))

    def _scalers_transform(self, dataset: TimeSeriesDataset) -> None:
        if not self.scalers_:
            return None
        for i, col in enumerate(dataset.temporal_cols):
            scaler = self.scalers_.get(col, None)
            if scaler is None:
                continue
            ga = GroupedArray(dataset.temporal[:, i].numpy(), dataset.indptr)
            dataset.temporal[:, i] = torch.from_numpy(scaler.transform(ga))

    def _scalers_target_inverse_transform(
        self, data: np.ndarray, indptr: np.ndarray
    ) -> np.ndarray:
        if not self.scalers_:
            return data
        for i in range(data.shape[1]):
            ga = GroupedArray(data[:, i], indptr)
            data[:, i] = self.scalers_[self.target_col].inverse_transform(ga)
        return data

    def _prepare_fit(self, df, static_df, predict_only, id_col, time_col, target_col):
        # TODO: uids, last_dates and ds should be properties of the dataset class. See github issue.
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self._check_nan(df, static_df, id_col, time_col, target_col)

        dataset, uids, last_dates, ds = TimeSeriesDataset.from_df(
            df=df,
            static_df=static_df,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        if predict_only:
            self._scalers_transform(dataset)
        else:
            self._scalers_fit_transform(dataset)
        return dataset, uids, last_dates, ds

    def _check_nan(self, df, static_df, id_col, time_col, target_col):
        cols_with_nans = []

        temporal_cols = [target_col] + [
            c for c in df.columns if c not in (id_col, time_col, target_col)
        ]
        if "available_mask" in temporal_cols:
            available_mask = df["available_mask"].to_numpy().astype(bool)
        else:
            available_mask = np.full(df.shape[0], True)

        df_to_check = ufp.filter_with_mask(df, available_mask)
        for col in temporal_cols:
            if ufp.is_nan_or_none(df_to_check[col]).any():
                cols_with_nans.append(col)

        if static_df is not None:
            for col in [x for x in static_df.columns if x != id_col]:
                if ufp.is_nan_or_none(static_df[col]).any():
                    cols_with_nans.append(col)

        if cols_with_nans:
            raise ValueError(f"Found missing values in {cols_with_nans}.")

    def _prepare_fit_distributed(
        self,
        df: SparkDataFrame,
        static_df: Optional[SparkDataFrame],
        id_col: str,
        time_col: str,
        target_col: str,
        distributed_config: Optional[DistributedConfig],
    ):
        if distributed_config is None:
            raise ValueError(
                "Must set `distributed_config` when using a spark dataframe"
            )
        if self.local_scaler_type is not None:
            raise ValueError(
                "Historic scaling isn't supported in distributed. "
                "Please open an issue if this would be valuable to you."
            )
        temporal_cols = [c for c in df.columns if c not in (id_col, time_col)]
        if static_df is not None:
            static_cols = [c for c in static_df.columns if c != id_col]
            df = df.join(static_df, on=[id_col], how="left")
        else:
            static_cols = None
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.scalers_ = {}
        num_partitions = distributed_config.num_nodes * distributed_config.devices
        df = df.repartitionByRange(num_partitions, id_col)
        df.write.parquet(path=distributed_config.partitions_path, mode="overwrite")
        fs, _, _ = fsspec.get_fs_token_paths(distributed_config.partitions_path)
        protocol = fs.protocol
        if isinstance(protocol, tuple):
            protocol = protocol[0]
        files = [
            f"{protocol}://{file}"
            for file in fs.ls(distributed_config.partitions_path)
            if file.endswith("parquet")
        ]
        return _FilesDataset(
            files=files,
            temporal_cols=temporal_cols,
            static_cols=static_cols,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
            min_size=df.groupBy(id_col).count().agg({"count": "min"}).first()[0],
        )

    def _prepare_fit_for_local_files(
        self,
        files_list: Sequence[str],
        static_df: Optional[DataFrame],
        id_col: str,
        time_col: str,
        target_col: str,
    ):
        if self.local_scaler_type is not None:
            raise ValueError(
                "Historic scaling isn't supported when the dataset is split between files. "
                "Please open an issue if this would be valuable to you."
            )

        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.scalers_ = {}

        exogs = self._get_needed_exog()
        return LocalFilesTimeSeriesDataset.from_data_directories(
            directories=files_list,
            static_df=static_df,
            exogs=exogs,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )

    def fit(
        self,
        df: Optional[Union[DataFrame, SparkDataFrame, Sequence[str]]] = None,
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        val_size: Optional[int] = 0,
        use_init_models: bool = False,
        verbose: bool = False,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        distributed_config: Optional[DistributedConfig] = None,
        prediction_intervals: Optional[PredictionIntervals] = None,
    ) -> None:
        """Fit the core.NeuralForecast.

        Fit `models` to a large set of time series from DataFrame `df`.
        and store fitted models for later inspection.

        Args:
            df (pandas, polars or spark DataFrame, or a list of parquet files containing the series, optional): DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
                If None, a previously stored dataset is required. Defaults to None.
            static_df (pandas, polars or spark DataFrame, optional): DataFrame with columns [`unique_id`] and static exogenous. Defaults to None.
            val_size (int, optional): Size of validation set. Defaults to 0.
            use_init_models (bool, optional): Use initial model passed when NeuralForecast object was instantiated. Defaults to False.
            verbose (bool): Print processing steps. Defaults to False.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            distributed_config (neuralforecast.DistributedConfig): Configuration to use for DDP training. Currently only spark is supported.
            prediction_intervals (PredictionIntervals, optional): Configuration to calibrate prediction intervals (Conformal Prediction). Defaults to None.

        Returns:
            NeuralForecast: Returns `NeuralForecast` class with fitted `models`.
        """
        if (df is None) and not (hasattr(self, "dataset")):
            raise Exception("You must pass a DataFrame or have one stored.")

        # Model and datasets interactions protections
        if (
            any(model.early_stop_patience_steps > 0 for model in self.models)
            and val_size == 0
        ):
            raise Exception("Set val_size>0 if early stopping is enabled.")

        if (val_size is not None) and (0 < val_size < self.h):
            raise ValueError(
                f"val_size must be either 0 or greater than or equal to the horizon: {self.h}"
            )

        self._cs_df: Optional[DataFrame] = None
        self.prediction_intervals: Optional[PredictionIntervals] = None

        # Process and save new dataset (in self)
        if isinstance(df, (pd.DataFrame, pl_DataFrame)):
            validate_freq(df[time_col], self.freq)
            self.dataset, self.uids, self.last_dates, self.ds = self._prepare_fit(
                df=df,
                static_df=static_df,
                predict_only=False,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            if prediction_intervals is not None:
                self.prediction_intervals = prediction_intervals
                self._cs_df = self._conformity_scores(
                    df=df,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    static_df=static_df,
                )

        elif isinstance(df, SparkDataFrame):
            if static_df is not None and not isinstance(static_df, SparkDataFrame):
                raise ValueError(
                    "`static_df` must be a spark dataframe when `df` is a spark dataframe."
                )
            self.dataset = self._prepare_fit_distributed(
                df=df,
                static_df=static_df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                distributed_config=distributed_config,
            )

            if prediction_intervals is not None:
                raise NotImplementedError(
                    "Prediction intervals are not supported for distributed training."
                )

        elif isinstance(df, Sequence):
            if not all(isinstance(val, str) for val in df):
                raise ValueError(
                    "All entries in the list of files must be of type string"
                )
            self.dataset = self._prepare_fit_for_local_files(
                files_list=df,
                static_df=static_df,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
            self.uids = self.dataset.indices
            self.last_dates = self.dataset.last_times

            if prediction_intervals is not None:
                raise NotImplementedError(
                    "Prediction intervals are not supported for local files."
                )

        elif df is None:
            if verbose:
                print("Using stored dataset.")
        else:
            raise ValueError(
                f"`df` must be a pandas, polars or spark DataFrame, or a list of parquet files containing the series, or `None`, got: {type(df)}"
            )

        if val_size is not None:
            if self.dataset.min_size < val_size:
                warnings.warn(
                    "Validation set size is larger than the shorter time-series."
                )

        # Recover initial model if use_init_models
        if use_init_models:
            self._reset_models()

        for i, model in enumerate(self.models):
            self.models[i] = model.fit(
                self.dataset, val_size=val_size, distributed_config=distributed_config
            )

        self._fitted = True

    def make_future_dataframe(
        self, df: Optional[DFType] = None, h: Optional[int] = None
    ) -> DFType:
        """Create a dataframe with all ids and future times in the forecasting horizon.

        Args:
            df (pandas or polars DataFrame, optional): DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
                Only required if this is different than the one used in the fit step. Defaults to None.
        """
        if not self._fitted:
            raise Exception("You must fit the model first.")
        if df is not None:
            df = ufp.sort(df, by=[self.id_col, self.time_col])
            last_times_by_id = ufp.group_by_agg(
                df,
                by=self.id_col,
                aggs={self.time_col: "max"},
                maintain_order=True,
            )
            uids = last_times_by_id[self.id_col]
            last_times = last_times_by_id[self.time_col]
        else:
            uids = self.uids
            last_times = self.last_dates
        if h is None:
            h = self.h
        return ufp.make_future_dataframe(
            uids=uids,
            last_times=last_times,
            freq=self.freq,
            h=h,
            id_col=self.id_col,
            time_col=self.time_col,
        )

    def get_missing_future(
        self, futr_df: DFType, df: Optional[DFType] = None, h: Optional[int] = None
    ) -> DFType:
        """Get the missing ids and times combinations in `futr_df`.

        Args:
            futr_df (pandas or polars DataFrame): DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous.
            df (pandas or polars DataFrame, optional): DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
                Only required if this is different than the one used in the fit step. Defaults to None.
        """
        expected = self.make_future_dataframe(df, h=h)
        ids = [self.id_col, self.time_col]
        return ufp.anti_join(expected, futr_df[ids], on=ids)

    def _get_needed_futr_exog(self):
        futr_exogs = []
        for m in self.models:
            if isinstance(m, BaseAuto):
                if isinstance(m.config, dict):  # ray
                    exogs = m.config.get("futr_exog_list", [])
                    if hasattr(
                        exogs, "categories"
                    ):  # features are being tuned, get possible values
                        exogs = exogs.categories
                else:  # optuna
                    exogs = m.config(MockTrial()).get("futr_exog_list", [])
            else:  # regular model, extract them directly
                exogs = getattr(m, "futr_exog_list", [])

            for exog in exogs:
                if isinstance(exog, str):
                    futr_exogs.append(exog)
                else:
                    futr_exogs.extend(exog)

        return set(futr_exogs)

    def _get_needed_exog(self):
        futr_exog = self._get_needed_futr_exog()

        hist_exog = []
        for m in self.models:
            if isinstance(m, BaseAuto):
                if isinstance(m.config, dict):  # ray
                    exogs = m.config.get("hist_exog_list", [])
                    if hasattr(
                        exogs, "categories"
                    ):  # features are being tuned, get possible values
                        exogs = exogs.categories
                else:  # optuna
                    exogs = m.config(MockTrial()).get("hist_exog_list", [])
            else:  # regular model, extract them directly
                exogs = getattr(m, "hist_exog_list", [])

            for exog in exogs:
                if isinstance(exog, str):
                    hist_exog.append(exog)
                else:
                    hist_exog.extend(exog)

        return futr_exog | set(hist_exog)

    def _get_model_names(self, add_level=False) -> List[str]:
        names: List[str] = []
        count_names = {"model": 0}
        for model in self.models:
            model_name = repr(model)
            count_names[model_name] = count_names.get(model_name, -1) + 1
            if count_names[model_name] > 0:
                model_name += str(count_names[model_name])

            if add_level and (
                model.loss.outputsize_multiplier > 1
                or isinstance(model.loss, (IQLoss, HuberIQLoss))
            ):
                continue

            names.extend(model_name + n for n in model.loss.output_names)
        return names

    def _predict_distributed(
        self,
        df: Optional[SparkDataFrame],
        static_df: Optional[SparkDataFrame],
        futr_df: Optional[SparkDataFrame],
        engine,
        h: Optional[int] = None,
    ):
        import fugue.api as fa

        def _predict(
            df: pd.DataFrame,
            static_cols,
            futr_exog_cols,
            models,
            freq,
            id_col,
            time_col,
            target_col,
            h,
        ) -> pd.DataFrame:
            from neuralforecast import NeuralForecast

            nf = NeuralForecast(models=models, freq=freq)
            nf.id_col = id_col
            nf.time_col = time_col
            nf.target_col = target_col
            nf.scalers_ = {}
            nf._fitted = True
            if futr_exog_cols:
                # if we have futr_exog we'll have extra rows with the future values
                futr_rows = df[target_col].isnull()
                futr_df = df.loc[
                    futr_rows, [self.id_col, self.time_col] + futr_exog_cols
                ].copy()
                df = df[~futr_rows].copy()
            else:
                futr_df = None
            if static_cols:
                static_df = (
                    df[[self.id_col] + static_cols]
                    .groupby(self.id_col, observed=True)
                    .head(1)
                )
                df = df.drop(columns=static_cols)
            else:
                static_df = None
            return nf.predict(df=df, static_df=static_df, futr_df=futr_df, h=h)

        # df
        if isinstance(df, SparkDataFrame):
            repartition = True
        else:
            if engine is None:
                raise ValueError("engine is required for distributed inference")
            df = engine.read.parquet(*self.dataset.files)
            # we save the datataset with partitioning
            repartition = False

        # static
        static_cols = set(
            chain.from_iterable(getattr(m, "stat_exog_list", []) for m in self.models)
        )
        if static_df is not None:
            if not isinstance(static_df, SparkDataFrame):
                raise ValueError(
                    "`static_df` must be a spark dataframe when `df` is a spark dataframe "
                    "or the models were trained in a distributed setting.\n"
                    "You can also provide local dataframes (pandas or polars) as `df` and `static_df`."
                )
            missing_static = static_cols - set(static_df.columns)
            if missing_static:
                raise ValueError(
                    f"The following static columns are missing from the static_df: {missing_static}"
                )
            # join is supposed to preserve the partitioning
            df = df.join(static_df, on=[self.id_col], how="left")

        # exog
        if futr_df is not None:
            if not isinstance(futr_df, SparkDataFrame):
                raise ValueError(
                    "`futr_df` must be a spark dataframe when `df` is a spark dataframe "
                    "or the models were trained in a distributed setting.\n"
                    "You can also provide local dataframes (pandas or polars) as `df` and `futr_df`."
                )
            if self.target_col in futr_df.columns:
                raise ValueError("`futr_df` must not contain the target column.")
            # df has the statics, historic exog and target at this point, futr_df doesnt
            df = df.unionByName(futr_df, allowMissingColumns=True)
            # union doesn't guarantee preserving the partitioning
            repartition = True

        if repartition:
            df = df.repartitionByRange(df.rdd.getNumPartitions(), self.id_col)

        # predict
        base_schema = fa.get_schema(df).extract([self.id_col, self.time_col])
        models_schema = {model: "float" for model in self._get_model_names()}
        return fa.transform(
            df=df,
            using=_predict,
            schema=base_schema.append(models_schema),
            params=dict(
                static_cols=list(static_cols),
                futr_exog_cols=list(self._get_needed_futr_exog()),
                models=self.models,
                freq=self.freq,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
                h=h,
            ),
        )

    def predict(
        self,
        df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        futr_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        verbose: bool = False,
        engine=None,
        level: Optional[List[Union[int, float]]] = None,
        quantiles: Optional[List[float]] = None,
        h: Optional[int] = None,
        **data_kwargs,
    ):
        """Predict with core.NeuralForecast.

        Use stored fitted `models` to predict large set of time series from DataFrame `df`.

        Args:
            df (pandas, polars or spark DataFrame, optional): DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
                If a DataFrame is passed, it is used to generate forecasts. Defaults to None.
            static_df (pandas, polars or spark DataFrame, optional): DataFrame with columns [`unique_id`] and static exogenous. Defaults to None.
            futr_df (pandas, polars or spark DataFrame, optional): DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous. Defaults to None.
            verbose (bool): Print processing steps. Defaults to False.
            engine (spark session): Distributed engine for inference. Only used if df is a spark dataframe or if fit was called on a spark dataframe.
            level (list of ints or floats, optional): Confidence levels between 0 and 100. Defaults to None.
            quantiles (list of floats, optional): Alternative to level, target quantiles to predict. Defaults to None.
            h (int, optional): Forecasting horizon. If None, uses the horizon of the fitted models. Defaults to None.
            data_kwargs (kwargs): Extra arguments to be passed to the dataset within each model.

        Returns:
            fcsts_df (pandas or polars DataFrame): DataFrame with insample `models` columns for point predictions and probabilistic
                predictions for all fitted `models`.
        """
        if df is None and not hasattr(self, "dataset"):
            raise Exception("You must pass a DataFrame or have one stored.")

        if not self._fitted:
            raise Exception("You must fit the model before predicting.")

        if h is not None:
            if h > self.h:
                # if only cross_validation called without fit() called first, prediction_intervals
                # attribute is not defined
                if getattr(self, "prediction_intervals", None) is not None:
                    raise ValueError(
                        f"The specified horizon h={h} is larger than the horizon of the fitted models: {self.h}. "
                        "Forecast with prediction intervals is not supported."
                    )

                for model in self.models:
                    if model.hist_exog_list:
                        raise NotImplementedError(
                            f"Model {model} has historic exogenous features, "
                            "which is not compatible with setting a larger horizon during prediction."
                        )
            elif h < self.h:
                raise ValueError(
                    f"The specified horizon h={h} must be greater than the horizon of the fitted models: {self.h}."
                )
            else:
                h = self.h
        else:
            h = self.h

        quantiles_ = None
        level_ = None
        has_level = False
        if level is not None:
            has_level = True
            if quantiles is not None:
                raise ValueError("You can't set both level and quantiles.")
            level_ = sorted(list(set(level)))
            quantiles_ = level_to_quantiles(level_)

        if quantiles is not None:
            if level is not None:
                raise ValueError("You can't set both level and quantiles.")
            quantiles_ = sorted(list(set(quantiles)))
            level_ = quantiles_to_level(quantiles_)

        needed_futr_exog = self._get_needed_futr_exog()
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

        # distributed df or NeuralForecast instance was trained with a distributed input and no df is provided
        # we assume the user wants to perform distributed inference as well
        is_files_dataset = isinstance(getattr(self, "dataset", None), _FilesDataset)
        is_dataset_local_files = isinstance(
            getattr(self, "dataset", None), LocalFilesTimeSeriesDataset
        )
        if isinstance(df, SparkDataFrame) or (df is None and is_files_dataset):
            return self._predict_distributed(
                df=df,
                static_df=static_df,
                futr_df=futr_df,
                engine=engine,
                h=h,
            )

        if is_dataset_local_files and df is None:
            raise ValueError(
                "When the model has been trained on a dataset that is split between multiple files, you must pass in a specific dataframe for prediciton."
            )

        # Process new dataset but does not store it.
        if df is not None:
            validate_freq(df[self.time_col], self.freq)
            dataset, uids, last_dates, _ = self._prepare_fit(
                df=df,
                static_df=static_df,
                predict_only=True,
                id_col=self.id_col,
                time_col=self.time_col,
                target_col=self.target_col,
            )
        else:
            dataset = self.dataset
            uids = self.uids
            last_dates = self.last_dates
            if verbose:
                print("Using stored dataset.")

        # Placeholder dataframe for predictions with unique_id and ds
        fcsts_df = ufp.make_future_dataframe(
            uids=uids,
            last_times=last_dates,
            freq=self.freq,
            h=h,
            id_col=self.id_col,
            time_col=self.time_col,
        )

        # Update and define new forecasting dataset
        if futr_df is None:
            futr_df = fcsts_df
        else:
            futr_orig_rows = futr_df.shape[0]
            futr_df = ufp.join(futr_df, fcsts_df, on=[self.id_col, self.time_col])
            if futr_df.shape[0] < fcsts_df.shape[0]:
                if df is None:
                    if h != self.h:
                        expected_cmd = f"make_future_dataframe(h={h})"
                        missing_cmd = f"get_missing_future(futr_df, h={h})"
                    else:
                        expected_cmd = "make_future_dataframe()"
                        missing_cmd = "get_missing_future(futr_df)"
                else:
                    if h != self.h:
                        expected_cmd = f"make_future_dataframe(df, h={h})"
                        missing_cmd = f"get_missing_future(futr_df, df, h={h})"
                    else:
                        expected_cmd = "make_future_dataframe(df)"
                        missing_cmd = "get_missing_future(futr_df, df)"
                raise ValueError(
                    "There are missing combinations of ids and times in `futr_df`.\n"
                    f"You can run the `{expected_cmd}` method to get the expected combinations or "
                    f"the `{missing_cmd}` method to get the missing combinations."
                )
            if futr_orig_rows > futr_df.shape[0]:
                dropped_rows = futr_orig_rows - futr_df.shape[0]
                warnings.warn(f"Dropped {dropped_rows:,} unused rows from `futr_df`.")
            if any(ufp.is_none(futr_df[col]).any() for col in needed_futr_exog):
                raise ValueError("Found null values in `futr_df`")
        futr_dataset = dataset.align(
            futr_df,
            id_col=self.id_col,
            time_col=self.time_col,
            target_col=self.target_col,
        )
        self._scalers_transform(futr_dataset)
        dataset = dataset.append(futr_dataset)

        fcsts, cols = self._generate_forecasts(
            dataset=dataset,
            uids=uids,
            quantiles_=quantiles_,
            level_=level_,
            has_level=has_level,
            h=h,
            **data_kwargs,
        )

        if self.scalers_:
            indptr = np.append(0, np.full(len(uids), h).cumsum())
            fcsts = self._scalers_target_inverse_transform(fcsts, indptr)

        # Declare predictions pd.DataFrame
        if isinstance(fcsts_df, pl_DataFrame):
            fcsts = pl_DataFrame(dict(zip(cols, fcsts.T)))
        else:
            fcsts = pd.DataFrame(fcsts, columns=cols)
        fcsts_df = ufp.horizontal_concat([fcsts_df, fcsts])

        return fcsts_df

    def explain(
        self,
        horizons: Optional[list[int]] = None,
        outputs: list[int] = [0],
        explainer: str = ExplainerEnum.IntegratedGradients,
        df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        static_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        futr_df: Optional[Union[DataFrame, SparkDataFrame]] = None,
        h: Optional[int] = None,
        verbose: bool = True,
        engine=None,
        level: Optional[List[Union[int, float]]] = None,
        quantiles: Optional[List[float]] = None,
        **data_kwargs,
    ):
        """(BETA) - Explain with core.NeuralForecast.

        Use stored fitted `models` to explain large set of time series from DataFrame `df`.

        Args:
            horizons (list of int, optional): List of horizons to explain. If None, all horizons are explained. Defaults to None.
            outputs (list of int, optional): List of outputs to explain for models with multiple outputs. Defaults to [0] (first output).
            explainer (str): Name of the explainer to use. Options are 'IntegratedGradients', 'ShapleyValueSampling', 'InputXGradient'. Defaults to 'IntegratedGradients'.
            df (pandas, polars or spark DataFrame, optional): DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
            If a DataFrame is passed, it is used to generate forecasts. Defaults to None.
            static_df (pandas, polars or spark DataFrame, optional): DataFrame with columns [`unique_id`] and static exogenous. Defaults to None.
            futr_df (pandas, polars or spark DataFrame, optional): DataFrame with [`unique_id`, `ds`] columns and `df`'s future exogenous. Defaults to None.
            h (int): The forecast horizon. Can be larger than the horizon set during training.
            verbose (bool): Print processing steps. Defaults to False.
            engine (spark session): Distributed engine for inference. Only used if df is a spark dataframe or if fit was called on a spark dataframe.
            level (list of ints or floats, optional): Confidence levels between 0 and 100. Defaults to None.
            quantiles (list of floats, optional): Alternative to level, target quantiles to predict. Defaults to None.
            data_kwargs (kwargs): Extra arguments to be passed to the dataset within each model.

        Returns:
            fcsts_df (pandas or polars DataFrame): DataFrame with insample `models` columns for point predictions and probabilistic
            predictions for all fitted `models`.
            explanations (dict): Dictionary of explanations for the predictions.
        """
        warnings.warn("This function is beta and subject to change.")

        if h is None:
            h_explain = self.h  # Default to model's training horizon
        else:
            h_explain = h
        
        # Validate and set horizons
        if horizons is None:
            horizons = list(range(h_explain))
        elif not horizons or len(horizons) > h_explain or any(h < 0 or h >= h_explain for h in horizons):
            raise ValueError(
                f"Invalid indices. Make sure to select horizon steps within {list(range(h_explain))} or set it to None to explain all horizon steps"
            )

        try:
            import captum
        except ImportError:
            raise ImportError(
                "Captum is not installed. Please install it with `pip install captum`."
            )
        if not hasattr(captum.attr, explainer):
            raise ValueError(f"Explainer {explainer} is not available in captum.")
        if explainer not in ExplainerEnum.AllExplainers:
            all_explainers = ", ".join(ExplainerEnum.AllExplainers)
            raise ValueError(
                f"Explainer {explainer} is not supported. Supported explainers are: {all_explainers}."
            )

        models_to_explain = []
        skipped_models = []
        
        for model in self.models:
            model_name = model.hparams.alias if hasattr(model.hparams, 'alias') and model.hparams.alias else model.__class__.__name__
            
            # Check for multivariate models
            if model.MULTIVARIATE:
                skipped_models.append(model_name)
                if verbose:
                    warnings.warn(f"Skipping {model_name}: Explanations are not currently supported for multivariate models.")
                continue
                
            # Check for DistributionLoss
            if hasattr(model.loss, 'is_distribution_output') and model.loss.is_distribution_output:
                loss_name = model.loss.__class__.__name__
                skipped_models.append(model_name)
                if verbose:
                    warnings.warn(
                        f"Skipping {model_name}: Explanations are not currently supported for {model_name} with {loss_name}. "
                        f"Please use a point loss (MAE, MSE, etc.) or a non-parametric probabilistic loss (MQLoss, IQLoss, etc.). "
                        f"Point losses and non-parametric probabilistic losses are listed here: "
                        f"https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/objectives.html"
                    )
                continue
                
            # Check for recurrent models with incompatible configurations
            if model.RECURRENT:
                # Check for IntegratedGradients incompatibility
                if explainer == "IntegratedGradients":
                    skipped_models.append(model_name)
                    if verbose:
                        warnings.warn(
                            f"Skipping {model_name}: IntegratedGradients is not compatible with recurrent models. "
                            f"Either set recurrent=False when initializing the model, or use a different explainer."
                        )
                    continue
                
                # Check for InputXGradient + GPU incompatibility (cudnn error)
                if explainer == "InputXGradient":
                    using_gpu = False
                    if hasattr(model, 'trainer_kwargs'):
                        accelerator = model.trainer_kwargs.get('accelerator', 'auto')
                        using_gpu = (accelerator == 'gpu' or 
                                    (accelerator == 'auto' and torch.cuda.is_available()))
                    elif torch.cuda.is_available():
                        using_gpu = True
                    
                    if using_gpu:
                        skipped_models.append(model_name)
                        if verbose:
                            warnings.warn(
                                f"Skipping {model_name}: InputXGradient with recurrent models on GPU causes cudnn errors. "
                                f"To fix this, either: 1) Set recurrent=False when initializing the model, "
                                f"2) Use ShapleyValueSampling instead, or "
                                f"3) Set accelerator='cpu' and devices=1 when initializing the model."
                            )
                        continue
                
            models_to_explain.append(model)
        
        if not models_to_explain:
            # Build a more specific error message based on what was skipped
            error_msg = "No models support explanations with the current configuration. "
            if any(model.RECURRENT for model in self.models) and explainer == ExplainerEnum.IntegratedGradients:
                error_msg += (
                    f"{ExplainerEnum.IntegratedGradients} is not compatible with recurrent models. "
                    "Either set recurrent=False or use a different explainer. "
                )
            error_msg += (
                f"The following models were skipped: {', '.join(skipped_models)}. "
            )
            raise ValueError(error_msg)
        
        # Determine minimum outputs across all models
        min_outputs = min(
            model.loss.outputsize_multiplier if hasattr(model.loss, 'outputsize_multiplier')
            else len(model.loss.output_names) if hasattr(model.loss, 'output_names')
            else 1
            for model in models_to_explain
        )

        # Validate outputs
        if outputs is None:
            outputs = [0]  # Default to first output
        elif not outputs or any(o < 0 or o >= min_outputs for o in outputs):
            raise ValueError(
                f"Invalid output indices. Based on the models being explained, valid outputs are in {list(range(min_outputs))}. "
                f"You must set valid output indices for all models, which is the minimum number of ouputs amongst all models. "
                f"You can always set outputs=None to default to [0] (first output)."
            )
        
        # Temporarily replace self.models with only explainable models
        original_models = self.models
        self.models = models_to_explain

        explainer_config = {
            "explainer": captum.attr.__dict__[explainer],
            "horizons": horizons,
            "output_index": outputs,
        }

        try:
            fcsts_df = self.predict(
                df=df,
                static_df=static_df,
                futr_df=futr_df,
                h=h_explain,
                verbose=verbose,
                engine=engine,
                level=level,
                quantiles=quantiles,
                explainer_config=explainer_config,
                **data_kwargs,
            )
        finally:
            # Restore original models
            self.models = original_models

        if self.scalers_:
            warnings.warn(
                "You used a global scaler, so explanations will be scaled. Additivity may not hold, but the relative importance is still correct. "
                "To have explanations in the same scale as the original data, use window scaling by setting scaler_type when initializing a model instead of local_scaler_type in the NeuralForecast object. "
                "Read more on the two types of temporal scaling here: https://nixtlaverse.nixtla.io/neuralforecast/docs/capabilities/time_series_scaling.html "
            )

        # Collect explanations from models that were explained
        explanations = {}
        for model in models_to_explain:
            if hasattr(model, "explanations") and model.explanations is not None:
                model_name = model.hparams.alias if hasattr(model.hparams, 'alias') and model.hparams.alias else model.__class__.__name__
                explanations[model_name] = {
                    "insample": model.explanations["insample_explanations"],           # [batch_size, horizon, n_series, n_output, input_size, 2 (y attribution, mask attribution)]
                    "futr_exog": model.explanations["futr_exog_explanations"],         # [batch_size, horizon, n_series, n_output, input_size+horizon, n_futr_features]
                    "hist_exog": model.explanations["hist_exog_explanations"],         # [batch_size, horizon, n_series, n_output, input_size, n_hist_features]
                    "stat_exog": model.explanations["stat_exog_explanations"],         # [batch_size, horizon, n_series, n_output, n_static_features]
                    "baseline_predictions": model.explanations["baseline_predictions"] # [batch_size, horizon, n_series, n_output]
                }
                # Delete explanations attribute once extracted
                delattr(model, "explanations")

        return fcsts_df, explanations

    def _reset_models(self):
        self.models = [deepcopy(model) for model in self.models_init]
        if self._fitted:
            print("WARNING: Deleting previously fitted models.")

    def _no_refit_cross_validation(
        self,
        df: Optional[DataFrame],
        static_df: Optional[DataFrame],
        n_windows: int,
        step_size: int,
        val_size: Optional[int],
        test_size: int,
        verbose: bool,
        id_col: str,
        time_col: str,
        target_col: str,
        h: int,
        **data_kwargs,
    ) -> DataFrame:
        if (df is None) and not (hasattr(self, "dataset")):
            raise Exception("You must pass a DataFrame or have one stored.")

        # Process and save new dataset (in self)
        if df is not None:
            validate_freq(df[time_col], self.freq)
            self.dataset, self.uids, self.last_dates, self.ds = self._prepare_fit(
                df=df,
                static_df=static_df,
                predict_only=False,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
            )
        else:
            if verbose:
                print("Using stored dataset.")

        if val_size is not None:
            if self.dataset.min_size < (val_size + test_size):
                warnings.warn(
                    "Validation and test sets are larger than the shorter time-series."
                )

        fcsts_df = ufp.cv_times(
            times=self.ds,
            uids=self.uids,
            indptr=self.dataset.indptr,
            h=h,
            test_size=test_size,
            step_size=step_size,
            id_col=id_col,
            time_col=time_col,
        )
        # the cv_times is sorted by window and then id
        fcsts_df = ufp.sort(fcsts_df, [id_col, "cutoff", time_col])

        fcsts_list: List = []
        for model in self.models:
            if self._add_level and (
                model.loss.outputsize_multiplier > 1
                or isinstance(model.loss, (IQLoss, HuberIQLoss))
            ):
                continue

            model.fit(dataset=self.dataset, val_size=val_size, test_size=test_size)
            model_fcsts = model.predict(
                self.dataset, step_size=step_size, h=h, **data_kwargs
            )
            # Append predictions in memory placeholder
            fcsts_list.append(model_fcsts)

        fcsts = np.concatenate(fcsts_list, axis=-1)
        # we may have allocated more space than needed
        # each serie can produce at most (serie.size - 1) // self.h CV windows
        effective_sizes = ufp.counts_by_id(fcsts_df, id_col)["counts"].to_numpy()
        needs_trim = effective_sizes.sum() != fcsts.shape[0]
        if self.scalers_ or needs_trim:
            indptr = np.arange(
                0,
                n_windows * h * (self.dataset.n_groups + 1),
                n_windows * h,
                dtype=np.int32,
            )
            if self.scalers_:
                fcsts = self._scalers_target_inverse_transform(fcsts, indptr)
            if needs_trim:
                # we keep only the effective samples of each serie from the cv results
                trimmed = np.empty_like(
                    fcsts, shape=(effective_sizes.sum(), fcsts.shape[1])
                )
                cv_indptr = np.append(0, effective_sizes).cumsum(dtype=np.int32)
                for i in range(fcsts.shape[1]):
                    ga = GroupedArray(fcsts[:, i], indptr)
                    trimmed[:, i] = ga._tails(cv_indptr)
                fcsts = trimmed

        self._fitted = True

        # Add predictions to forecasts DataFrame
        cols = self._get_model_names(add_level=self._add_level)
        if isinstance(self.uids, pl_Series):
            fcsts = pl_DataFrame(dict(zip(cols, fcsts.T)))
        else:
            fcsts = pd.DataFrame(fcsts, columns=cols)
        fcsts_df = ufp.horizontal_concat([fcsts_df, fcsts])

        # Add original input df's y to forecasts DataFrame
        return ufp.join(
            fcsts_df,
            df[[id_col, time_col, target_col]],
            how="left",
            on=[id_col, time_col],
        )

    def cross_validation(
        self,
        df: Optional[DataFrame] = None,
        static_df: Optional[DataFrame] = None,
        n_windows: int = 1,
        step_size: int = 1,
        val_size: Optional[int] = 0,
        test_size: Optional[int] = None,
        use_init_models: bool = False,
        verbose: bool = False,
        refit: Union[bool, int] = False,
        id_col: str = "unique_id",
        time_col: str = "ds",
        target_col: str = "y",
        prediction_intervals: Optional[PredictionIntervals] = None,
        level: Optional[List[Union[int, float]]] = None,
        quantiles: Optional[List[float]] = None,
        h: Optional[int] = None,
        **data_kwargs,
    ) -> DataFrame:
        """Temporal Cross-Validation with core.NeuralForecast.

        `core.NeuralForecast`'s cross-validation efficiently fits a list of NeuralForecast
        models through multiple windows, in either chained or rolled manner.

        Args:
            df (pandas or polars DataFrame, optional): DataFrame with columns [`unique_id`, `ds`, `y`] and exogenous variables.
                If None, a previously stored dataset is required. Defaults to None.
            static_df (pandas or polars DataFrame, optional): DataFrame with columns [`unique_id`] and static exogenous. Defaults to None.
            n_windows (int): Number of windows used for cross validation. Defaults to 1.
            step_size (int): Step size between each window. Defaults to 1.
            val_size (int, optional): Length of validation size. If passed, set `n_windows=None`. Defaults to 0.
            test_size (int, optional): Length of test size. If passed, set `n_windows=None`. Defaults to None.
            use_init_models (bool, optional): Use initial model passed when object was instantiated. Defaults to False.
            verbose (bool): Print processing steps. Defaults to False.
            refit (bool or int): Retrain model for each cross validation window.
                If False, the models are trained at the beginning and then used to predict each window.
                If positive int, the models are retrained every `refit` windows. Defaults to False.
            id_col (str): Column that identifies each serie. Defaults to 'unique_id'.
            time_col (str): Column that identifies each timestep, its values can be timestamps or integers. Defaults to 'ds'.
            target_col (str): Column that contains the target. Defaults to 'y'.
            prediction_intervals (PredictionIntervals, optional): Configuration to calibrate prediction intervals (Conformal Prediction). Defaults to None.
            level (list of ints or floats, optional): Confidence levels between 0 and 100. Defaults to None.
            quantiles (list of floats, optional): Alternative to level, target quantiles to predict. Defaults to None.
            h (int, optional): Forecasting horizon. If None, uses the horizon of the fitted models. Defaults to None.
            data_kwargs (kwargs): Extra arguments to be passed to the dataset within each model.

        Returns:
            fcsts_df (pandas or polars DataFrame): DataFrame with insample `models` columns for point predictions and probabilistic
                predictions for all fitted `models`.
        """
        if h is not None:
            if h > self.h:
                # if only cross_validation called without fit() called first, prediction_intervals
                # attribute is not defined
                if getattr(self, "prediction_intervals", None) is not None:
                    raise ValueError(
                        f"The specified horizon h={h} is larger than the horizon of the fitted models: {self.h}. "
                        "Forecast with prediction intervals is not supported."
                    )

                for model in self.models:
                    if model.hist_exog_list:
                        raise NotImplementedError(
                            f"Model {model} has historic exogenous features, "
                            "which is not compatible with setting a larger horizon during cross-validation."
                        )
                # Refit is not supported with cross-validation on longer horizons than the trained horizon
                if not refit:
                    raise ValueError(
                        f"The specified horizon h={h} is larger than the horizon of the fitted models: {self.h}. "
                        "Set refit=True in this setting."
                    )
            elif h < self.h:
                raise ValueError(
                    f"The specified horizon h={h} must be greater than the horizon of the fitted models: {self.h}."
                )
            else:
                h = self.h
        else:
            h = self.h

        if n_windows is None and test_size is None:
            raise Exception("you must define `n_windows` or `test_size`.")
        if test_size is None and h is not None:
            test_size = h + step_size * (n_windows - 1)
        elif n_windows is None:
            if (test_size - h) % step_size:
                raise Exception("`test_size - h` should be module `step_size`")
            n_windows = int((test_size - h) / step_size) + 1
        else:
            raise Exception("you must define `n_windows` or `test_size` but not both")

        # Recover initial model if use_init_models.
        if use_init_models:
            self._reset_models()

        # Checks for prediction intervals
        if prediction_intervals is not None:
            if level is None and quantiles is None:
                raise Exception(
                    "When passing prediction_intervals you need to set the level or quantiles argument."
                )
            if not refit:
                raise Exception(
                    "Passing prediction_intervals is only supported with refit=True."
                )

        if level is not None and quantiles is not None:
            raise ValueError("You can't set both level and quantiles argument.")

        if not refit:

            return self._no_refit_cross_validation(
                df=df,
                static_df=static_df,
                n_windows=n_windows,
                step_size=step_size,
                val_size=val_size,
                test_size=test_size,
                verbose=verbose,
                id_col=id_col,
                time_col=time_col,
                target_col=target_col,
                h=h,
                **data_kwargs,
            )
        if df is None:
            raise ValueError("Must specify `df` with `refit!=False`.")
        validate_freq(df[time_col], self.freq)
        splits = ufp.backtest_splits(
            df,
            n_windows=n_windows,
            h=h,
            id_col=id_col,
            time_col=time_col,
            freq=self.freq,
            step_size=step_size,
            input_size=None,
        )
        results = []
        for i_window, (cutoffs, train, test) in enumerate(splits):
            should_fit = i_window == 0 or (refit > 0 and i_window % refit == 0)
            if should_fit:
                self.fit(
                    df=train,
                    static_df=static_df,
                    val_size=val_size,
                    use_init_models=False,
                    verbose=verbose,
                    id_col=id_col,
                    time_col=time_col,
                    target_col=target_col,
                    prediction_intervals=prediction_intervals,
                )
                predict_df: Optional[DataFrame] = None
            else:
                predict_df = train
            needed_futr_exog = self._get_needed_futr_exog()
            if needed_futr_exog:
                futr_df: Optional[DataFrame] = test
            else:
                futr_df = None
            preds = self.predict(
                df=predict_df,
                static_df=static_df,
                futr_df=futr_df,
                verbose=verbose,
                level=level,
                quantiles=quantiles,
                h=h,
                **data_kwargs,
            )
            preds = ufp.join(preds, cutoffs, on=id_col, how="left")
            fold_result = ufp.join(
                preds, test[[id_col, time_col, target_col]], on=[id_col, time_col]
            )
            results.append(fold_result)
        out = ufp.vertical_concat(results, match_categories=False)
        out = ufp.drop_index_if_pandas(out)
        # match order of cv with no refit
        first_out_cols = [id_col, time_col, "cutoff"]
        remaining_cols = [
            c for c in out.columns if c not in first_out_cols + [target_col]
        ]
        cols_order = first_out_cols + remaining_cols + [target_col]
        return ufp.sort(out[cols_order], by=[id_col, "cutoff", time_col])

    def predict_insample(
        self,
        step_size: int = 1,
        level: Optional[List[Union[int, float]]] = None,
        quantiles: Optional[List[float]] = None,
    ):
        """Predict insample with core.NeuralForecast.

        `core.NeuralForecast`'s `predict_insample` uses stored fitted `models`
        to predict historic values of a time series from the stored dataframe.

        Args:
            step_size (int): Step size between each window. Defaults to 1.
            level (list of ints or floats, optional): Confidence levels between 0 and 100. Defaults to None.
            quantiles (list of floats, optional): Alternative to level, target quantiles to predict. Defaults to None.

        Returns:
            fcsts_df (pandas.DataFrame): DataFrame with insample predictions for all fitted `models`.
        """
        if not self._fitted:
            raise Exception(
                "The models must be fitted first with `fit` or `cross_validation`."
            )
        test_size = self.models[0].get_test_size()

        quantiles_ = None
        level_ = None
        has_level = False
        if level is not None:
            has_level = True
            if quantiles is not None:
                raise ValueError("You can't set both level and quantiles.")
            level_ = sorted(list(set(level)))
            quantiles_ = level_to_quantiles(level_)
            if self._cs_df is not None:
                raise NotImplementedError(
                    "One or more models has been trained with conformal prediction intervals. They are not supported for insample predictions. Set level=None"
                )

        if quantiles is not None:
            if level is not None:
                raise ValueError("You can't set both level and quantiles.")
            quantiles_ = sorted(list(set(quantiles)))
            level_ = quantiles_to_level(quantiles_)
            if self._cs_df is not None:
                raise NotImplementedError(
                    "One or more models has been trained with conformal prediction intervals. They are not supported for insample predictions. Set quantiles=None"
                )

        for model in self.models:
            if model.MULTIVARIATE:
                raise NotImplementedError(
                    f"Model {model} is multivariate. Insample predictions are not supported for multivariate models."
                )

        # Process each series separately
        fcsts_dfs = []
        trimmed_datasets = []

        for i in range(self.dataset.n_groups):
            # Calculate series-specific length and offset
            series_length = self.dataset.indptr[i + 1] - self.dataset.indptr[i]
            _, forefront_offset = np.divmod(
                (series_length - test_size - self.h), step_size
            )

            if test_size > 0 or forefront_offset > 0:
                # Create single-series dataset
                series_dataset = TimeSeriesDataset(
                    temporal=self.dataset.temporal[
                        self.dataset.indptr[i] : self.dataset.indptr[i + 1]
                    ],
                    temporal_cols=self.dataset.temporal_cols,
                    static=self.dataset.static,
                    static_cols=self.dataset.static_cols,
                    indptr=np.array([0, series_length]),
                    y_idx=self.dataset.y_idx,
                )
                # Trim the series
                trimmed_series = TimeSeriesDataset.trim_dataset(
                    dataset=series_dataset,
                    right_trim=test_size,
                    left_trim=forefront_offset,
                )

                new_idxs = np.arange(
                    self.dataset.indptr[i] + forefront_offset,
                    self.dataset.indptr[i + 1] - test_size,
                )
                times = self.ds[new_idxs]
            else:
                trimmed_series = TimeSeriesDataset(
                    temporal=self.dataset.temporal[
                        self.dataset.indptr[i] : self.dataset.indptr[i + 1]
                    ],
                    temporal_cols=self.dataset.temporal_cols,
                    static=self.dataset.static,
                    static_cols=self.dataset.static_cols,
                    indptr=np.array([0, series_length]),
                    y_idx=self.dataset.y_idx,
                )
                times = self.ds[self.dataset.indptr[i] : self.dataset.indptr[i + 1]]

            series_fcsts_df = _insample_times(
                times=times,
                uids=self.uids[i : i + 1],
                indptr=trimmed_series.indptr,
                h=self.h,
                freq=self.freq,
                step_size=step_size,
                id_col=self.id_col,
                time_col=self.time_col,
            )

            fcsts_dfs.append(series_fcsts_df)
            trimmed_datasets.append(trimmed_series)

        # Combine all series forecasts DataFrames
        fcsts_df = ufp.vertical_concat(fcsts_dfs)

        h_backup = self.h
        fcst_list = []
        # Generate predictions for each dataset
        for i, trimmed_dataset in enumerate(trimmed_datasets):
            # Set test size to current series length
            self.h = trimmed_dataset.max_size
            fcsts, cols = self._generate_forecasts(
                dataset=trimmed_dataset,
                uids=self.uids[i : i + 1],
                quantiles_=quantiles_,
                level_=level_,
                has_level=has_level,
                step_size=step_size,
                h=None,
            )
            fcst_list.append(fcsts)

        fcsts = np.vstack(fcst_list)
        self.h = h_backup

        # Add original y values
        original_y = {
            self.id_col: ufp.repeat(self.uids, np.diff(self.dataset.indptr)),
            self.time_col: self.ds,
            self.target_col: self.dataset.temporal[:, 0].numpy(),
        }

        # Declare predictions pd.DataFrame
        if isinstance(fcsts_df, pl_DataFrame):
            fcsts = pl_DataFrame(dict(zip(cols, fcsts.T)))
            Y_df = pl_DataFrame(original_y)
        else:
            fcsts = pd.DataFrame(fcsts, columns=cols)
            Y_df = pd.DataFrame(original_y).reset_index(drop=True)

        fcsts_df = ufp.horizontal_concat([fcsts_df, fcsts])
        fcsts_df = ufp.join(fcsts_df, Y_df, how="left", on=[self.id_col, self.time_col])

        if self.scalers_:
            sizes = ufp.counts_by_id(fcsts_df, self.id_col)["counts"].to_numpy()
            indptr = np.append(0, sizes.cumsum())
            invert_cols = cols + [self.target_col]
            fcsts_df[invert_cols] = self._scalers_target_inverse_transform(
                fcsts_df[invert_cols].to_numpy(), indptr
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

        Args:
            path (str): Directory to save current status.
            model_index (list, optional): List to specify which models from list of self.models to save. Defaults to None.
            save_dataset (bool): Whether to save dataset or not. Defaults to True.
            overwrite (bool): Whether to overwrite files or not. Defaults to False.
        """
        # Standarize path without '/'
        if path[-1] == "/":
            path = path[:-1]

        # Model index list
        if model_index is None:
            model_index = list(range(len(self.models)))

        fs, _, _ = fsspec.get_fs_token_paths(path)
        if not fs.exists(path):
            fs.makedirs(path)
        else:
            # Check if directory is empty to protect overwriting files
            files = fs.ls(path)

            # Checking if the list is empty or not
            if files:
                if not overwrite:
                    raise Exception(
                        "Directory is not empty. Set `overwrite=True` to overwrite files."
                    )
                else:
                    fs.rm(path, recursive=True)
                    fs.mkdir(path)

        # Save models
        count_names = {"model": 0}
        alias_to_model = {}
        for i, model in enumerate(self.models):
            # Skip model if not in list
            if i not in model_index:
                continue

            model_name = repr(model)
            if model.__class__.__name__.lower() in MODEL_FILENAME_DICT:
                model_class_name = model.__class__.__name__.lower()
            elif model.__class__.__base__.__name__.lower() in MODEL_FILENAME_DICT:
                model_class_name = model.__class__.__base__.__name__.lower()
            else:
                raise ValueError(
                    f"Model {model.__class__.__name__} is not supported for saving."
                )
            alias_to_model[model_name] = model_class_name
            count_names[model_name] = count_names.get(model_name, -1) + 1
            model.save(f"{path}/{model_name}_{count_names[model_name]}.ckpt")
        with fsspec.open(f"{path}/alias_to_model.pkl", "wb") as f:
            pickle.dump(alias_to_model, f)

        # Save dataset
        if save_dataset and hasattr(self, "dataset"):
            if isinstance(self.dataset, _FilesDataset):
                raise ValueError(
                    "Cannot save distributed dataset.\n"
                    "You can set `save_dataset=False` and use the `df` argument in the predict method after loading "
                    "this model to use it for inference."
                )
            with fsspec.open(f"{path}/dataset.pkl", "wb") as f:
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
            "_fitted": self._fitted,
            "local_scaler_type": self.local_scaler_type,
            "scalers_": self.scalers_,
            "id_col": self.id_col,
            "time_col": self.time_col,
            "target_col": self.target_col,
        }
        for attr in ["prediction_intervals", "_cs_df"]:
            # conformal prediction related attributes was not available < 1.7.6
            config_dict[attr] = getattr(self, attr, None)

        if save_dataset:
            config_dict.update(
                {
                    "uids": self.uids,
                    "last_dates": self.last_dates,
                    "ds": self.ds,
                }
            )

        with fsspec.open(f"{path}/configuration.pkl", "wb") as f:
            pickle.dump(config_dict, f)

    @staticmethod
    def load(path, verbose=False, **kwargs):
        """Load NeuralForecast

        `core.NeuralForecast`'s method to load checkpoint from path.

        Args:
            path (str): Directory with stored artifacts.
            verbose (bool): Defaults to False.
            **kwargs: Additional keyword arguments to be passed to the function
                `load_from_checkpoint`.

        Returns:
            result (NeuralForecast): Instantiated `NeuralForecast` class.
        """
        # Standarize path without '/'
        if path[-1] == "/":
            path = path[:-1]

        fs, _, _ = fsspec.get_fs_token_paths(path)
        files = [f.split("/")[-1] for f in fs.ls(path) if fs.isfile(f)]

        # Load models
        models_ckpt = [f for f in files if f.endswith(".ckpt")]
        if len(models_ckpt) == 0:
            raise Exception("No model found in directory.")

        if verbose:
            print(10 * "-" + " Loading models " + 10 * "-")
        models = []
        try:
            with fsspec.open(f"{path}/alias_to_model.pkl", "rb") as f:
                alias_to_model = pickle.load(f)
        except FileNotFoundError:
            alias_to_model = {}

        for model in models_ckpt:
            model_name = "_".join(model.split("_")[:-1])
            model_class_name = alias_to_model.get(model_name, model_name)
            loaded_model = MODEL_FILENAME_DICT[model_class_name].load(
                f"{path}/{model}", **kwargs
            )
            loaded_model.alias = model_name
            models.append(loaded_model)
            if verbose:
                print(f"Model {model_name} loaded.")

        if verbose:
            print(10 * "-" + " Loading dataset " + 10 * "-")
        # Load dataset
        try:
            with fsspec.open(f"{path}/dataset.pkl", "rb") as f:
                dataset = pickle.load(f)
            if verbose:
                print("Dataset loaded.")
        except FileNotFoundError:
            dataset = None
            if verbose:
                print("No dataset found in directory.")

        if verbose:
            print(10 * "-" + " Loading configuration " + 10 * "-")
        # Load configuration
        try:
            with fsspec.open(f"{path}/configuration.pkl", "rb") as f:
                config_dict = pickle.load(f)
            if verbose:
                print("Configuration loaded.")
        except FileNotFoundError:
            raise Exception("No configuration found in directory.")

        # in 1.6.4, `local_scaler_type` / `scalers_` lived on the dataset.
        # in order to preserve backwards-compatibility, we check to see if these are found on the dataset
        # in case they cannot be found in `config_dict`
        default_scalar_type = getattr(dataset, "local_scaler_type", None)
        default_scalars_ = getattr(dataset, "scalers_", None)

        # Create NeuralForecast object
        neuralforecast = NeuralForecast(
            models=models,
            freq=config_dict["freq"],
            local_scaler_type=config_dict.get("local_scaler_type", default_scalar_type),
        )

        attr_to_default = {"id_col": "unique_id", "time_col": "ds", "target_col": "y"}
        for attr, default in attr_to_default.items():
            setattr(neuralforecast, attr, config_dict.get(attr, default))
        # only restore attribute if available
        for attr in ["prediction_intervals", "_cs_df"]:
            setattr(neuralforecast, attr, config_dict.get(attr, None))

        # Dataset
        if dataset is not None:
            neuralforecast.dataset = dataset
            restore_attrs = [
                "uids",
                "last_dates",
                "ds",
            ]
            for attr in restore_attrs:
                setattr(neuralforecast, attr, config_dict[attr])

        # Fitted flag
        neuralforecast._fitted = config_dict["_fitted"]

        neuralforecast.scalers_ = config_dict.get("scalers_", default_scalars_)

        return neuralforecast

    def _conformity_scores(
        self,
        df: DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        static_df: Optional[DataFrame],
    ) -> DataFrame:
        """Compute conformity scores.

        We need at least two cross validation errors to compute
        quantiles for prediction intervals (`n_windows=2`, specified by self.prediction_intervals).

        The exception is raised by the PredictionIntervals data class.

        Args:
            df (DataFrame): DataFrame with time series data.
            id_col (str): Column that identifies each serie.
            time_col (str): Column that identifies each timestep.
            target_col (str): Column that contains the target.
            static_df (Optional[DataFrame]): DataFrame with static exogenous variables.
        """
        if self.prediction_intervals is None:
            raise AttributeError(
                "Please rerun the `fit` method passing a valid prediction_interval setting to compute conformity scores"
            )

        min_size = ufp.counts_by_id(df, id_col)["counts"].min()
        min_samples = self.h * self.prediction_intervals.n_windows + 1
        if min_size < min_samples:
            raise ValueError(
                "Minimum required samples in each serie for the prediction intervals "
                f"settings are: {min_samples}, shortest serie has: {min_size}. "
                "Please reduce the number of windows, horizon or remove those series."
            )

        self._add_level = True
        cv_results = self.cross_validation(
            df=df,
            static_df=static_df,
            n_windows=self.prediction_intervals.n_windows,
            id_col=id_col,
            time_col=time_col,
            target_col=target_col,
        )
        self._add_level = False

        kept = [time_col, id_col, "cutoff"]
        # conformity score for each model
        for model in self._get_model_names(add_level=True):
            kept.append(model)

            # compute absolute error for each model
            abs_err = abs(cv_results[model] - cv_results[target_col])
            cv_results = ufp.assign_columns(cv_results, model, abs_err)
        dropped = list(set(cv_results.columns) - set(kept))
        return ufp.drop_columns(cv_results, dropped)

    def _generate_forecasts(
        self,
        dataset: TimeSeriesDataset,
        uids: Series,
        h: Union[int, None],
        quantiles_: Optional[List[float]] = None,
        level_: Optional[List[Union[int, float]]] = None,
        has_level: Optional[bool] = False,
        **data_kwargs,
    ) -> np.array:
        fcsts_list: List = []
        cols = []
        count_names = {"model": 0}
        for model in self.models:
            old_test_size = model.get_test_size()
            model.set_test_size(
                h if h is not None else self.h
            )  # To predict h steps ahead

            # Increment model name if the same model is used more than once
            model_name = repr(model)
            count_names[model_name] = count_names.get(model_name, -1) + 1
            if count_names[model_name] > 0:
                model_name += str(count_names[model_name])

            # Predict for every quantile or level if requested and the loss function supports it
            # case 1: DistributionLoss and MixtureLosses
            if (
                quantiles_ is not None
                and not isinstance(model.loss, (IQLoss, HuberIQLoss))
                and hasattr(model.loss, "update_quantile")
                and callable(model.loss.update_quantile)
            ):
                model_fcsts = model.predict(
                    dataset=dataset, quantiles=quantiles_, h=h, **data_kwargs
                )
                fcsts_list.append(model_fcsts)
                col_names = []
                for i, quantile in enumerate(quantiles_):
                    col_name = self._get_column_name(model_name, quantile, has_level)
                    if i == 0:
                        col_names.extend([f"{model_name}", col_name])
                    else:
                        col_names.extend([col_name])
                if hasattr(model.loss, "return_params") and model.loss.return_params:
                    cols.extend(
                        col_names
                        + [
                            model_name + param_name
                            for param_name in model.loss.param_names
                        ]
                    )
                else:
                    cols.extend(col_names)
            # case 2: IQLoss
            elif quantiles_ is not None and isinstance(
                model.loss, (IQLoss, HuberIQLoss)
            ):
                # IQLoss does not give monotonically increasing quantiles, so we apply a hack: compute all quantiles, and take the quantile over the quantiles
                quantiles_iqloss = [
                    0.01,
                    0.05,
                    0.1,
                    0.2,
                    0.3,
                    0.5,
                    0.7,
                    0.8,
                    0.9,
                    0.95,
                    0.99,
                ]
                fcsts_list_iqloss = []
                for i, quantile in enumerate(quantiles_iqloss):
                    model_fcsts = model.predict(
                        dataset=dataset, quantiles=[quantile], h=h, **data_kwargs
                    )
                    fcsts_list_iqloss.append(model_fcsts)
                fcsts_iqloss = np.concatenate(fcsts_list_iqloss, axis=-1)

                # Get the actual requested quantiles
                model_fcsts = np.quantile(fcsts_iqloss, quantiles_, axis=-1).T
                fcsts_list.append(model_fcsts)

                # Get the right column names
                col_names = []
                for i, quantile in enumerate(quantiles_):
                    col_name = self._get_column_name(model_name, quantile, has_level)
                    col_names.extend([col_name])
                cols.extend(col_names)
            # case 3: PointLoss via prediction intervals
            elif quantiles_ is not None and model.loss.outputsize_multiplier == 1:
                if self.prediction_intervals is None:
                    raise AttributeError(
                        f"You have trained {model_name} with loss={type(model.loss).__name__}(). \n"
                        " You then must set `prediction_intervals` during fit to use level or quantiles during predict."
                    )
                model_fcsts = model.predict(
                    dataset=dataset, quantiles=quantiles_, h=h, **data_kwargs
                )
                prediction_interval_method = get_prediction_interval_method(
                    self.prediction_intervals.method
                )
                fcsts_with_intervals, out_cols = prediction_interval_method(
                    model_fcsts,
                    self._cs_df,
                    model=model_name,
                    level=level_ if has_level else None,
                    cs_n_windows=self.prediction_intervals.n_windows,
                    n_series=len(uids),
                    horizon=self.h,
                    quantiles=quantiles_ if not has_level else None,
                )
                fcsts_list.append(fcsts_with_intervals)
                cols.extend([model_name] + out_cols)
            # base case: quantiles or levels are not supported or provided as arguments
            else:
                model_fcsts = model.predict(dataset=dataset, h=h, **data_kwargs)
                fcsts_list.append(model_fcsts)
                cols.extend(model_name + n for n in model.loss.output_names)
            model.set_test_size(old_test_size)  # Set back to original value
        fcsts = np.concatenate(fcsts_list, axis=-1)

        return fcsts, cols

    @staticmethod
    def _get_column_name(model_name, quantile, has_level) -> str:
        if not has_level:
            col_name = f"{model_name}_ql{quantile}"
        elif quantile < 0.5:
            level_lo = int(round(100 - 200 * quantile))
            col_name = f"{model_name}-lo-{level_lo}"
        elif quantile > 0.5:
            level_hi = int(round(100 - 200 * (1 - quantile)))
            col_name = f"{model_name}-hi-{level_hi}"
        else:
            col_name = f"{model_name}-median"

        return col_name
