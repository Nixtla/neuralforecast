__all__ = ['AirPassengers', 'AirPassengersDF', 'unique_id', 'ds', 'y', 'AirPassengersPanel', 'snaive', 'airline1_dummy',
           'airline2_dummy', 'AirPassengersStatic', 'generate_series', 'TimeFeature', 'SecondOfMinute', 'MinuteOfHour',
           'HourOfDay', 'DayOfWeek', 'DayOfMonth', 'DayOfYear', 'MonthOfYear', 'WeekOfYear',
           'time_features_from_frequency_str', 'augment_calendar_df', 'get_indexer_raise_missing',
           'PredictionIntervals', 'add_conformal_distribution_intervals', 'add_conformal_error_intervals',
           'get_prediction_interval_method', 'level_to_quantiles', 'quantiles_to_level', 'ShapModelWrapper',
           'create_input_tensor_for_series', 'create_multi_series_background_data', 'create_multi_series_feature_names']


import random
from itertools import chain
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from utilsforecast.compat import DFType


def generate_series(
    n_series: int,
    freq: str = "D",
    min_length: int = 50,
    max_length: int = 500,
    n_temporal_features: int = 0,
    n_static_features: int = 0,
    equal_ends: bool = False,
    seed: int = 0,
) -> pd.DataFrame:
    """Generate Synthetic Panel Series.

    Generates `n_series` of frequency `freq` of different lengths in the interval [`min_length`, `max_length`].
    If `n_temporal_features > 0`, then each serie gets temporal features with random values.
    If `n_static_features > 0`, then a static dataframe is returned along the temporal dataframe.
    If `equal_ends == True` then all series end at the same date.

    Args:
        n_series (int): Number of series for synthetic panel.
        freq (str, optional): Frequency of the data, panda's available frequencies. Defaults to "D".
        min_length (int, optional): Minimal length of synthetic panel's series. Defaults to 50.
        max_length (int, optional): Maximal length of synthetic panel's series. Defaults to 500.
        n_temporal_features (int, optional): Number of temporal exogenous variables for synthetic panel's series. Defaults to 0.
        n_static_features (int, optional): Number of static exogenous variables for synthetic panel's series. Defaults to 0.
        equal_ends (bool, optional): If True, series finish in the same date stamp `ds`. Defaults to False.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        pd.DataFrame: Synthetic panel with columns [`unique_id`, `ds`, `y`] and exogenous.
    """
    seasonalities = {"D": 7, "M": 12}
    season = seasonalities[freq]

    rng = np.random.RandomState(seed)
    series_lengths = rng.randint(min_length, max_length + 1, n_series)
    total_length = series_lengths.sum()

    dates = pd.date_range("2000-01-01", periods=max_length, freq=freq).values
    uids = [np.repeat(i, serie_length) for i, serie_length in enumerate(series_lengths)]
    if equal_ends:
        ds = [dates[-serie_length:] for serie_length in series_lengths]
    else:
        ds = [dates[:serie_length] for serie_length in series_lengths]

    y = np.arange(total_length) % season + rng.rand(total_length) * 0.5
    temporal_df = pd.DataFrame(
        dict(unique_id=chain.from_iterable(uids), ds=chain.from_iterable(ds), y=y)
    )

    random.seed(seed)
    for i in range(n_temporal_features):
        random.seed(seed)
        temporal_values = [
            [random.randint(0, 100)] * serie_length for serie_length in series_lengths
        ]
        temporal_df[f"temporal_{i}"] = np.hstack(temporal_values)
        temporal_df[f"temporal_{i}"] = temporal_df[f"temporal_{i}"].astype("category")
        if i == 0:
            temporal_df["y"] = temporal_df["y"] * (
                1 + temporal_df[f"temporal_{i}"].cat.codes
            )

    temporal_df["unique_id"] = temporal_df["unique_id"].astype("category")
    temporal_df["unique_id"] = temporal_df["unique_id"].cat.as_ordered()

    if n_static_features > 0:
        static_features = np.random.uniform(
            low=0.0, high=1.0, size=(n_series, n_static_features)
        )
        static_df = pd.DataFrame.from_records(
            static_features, columns=[f"static_{i}" for i in range(n_static_features)]
        )

        static_df["unique_id"] = np.arange(n_series)
        static_df["unique_id"] = static_df["unique_id"].astype("category")
        static_df["unique_id"] = static_df["unique_id"].cat.as_ordered()

        return temporal_df, static_df

    return temporal_df


AirPassengers = np.array(
    [
        112.0,
        118.0,
        132.0,
        129.0,
        121.0,
        135.0,
        148.0,
        148.0,
        136.0,
        119.0,
        104.0,
        118.0,
        115.0,
        126.0,
        141.0,
        135.0,
        125.0,
        149.0,
        170.0,
        170.0,
        158.0,
        133.0,
        114.0,
        140.0,
        145.0,
        150.0,
        178.0,
        163.0,
        172.0,
        178.0,
        199.0,
        199.0,
        184.0,
        162.0,
        146.0,
        166.0,
        171.0,
        180.0,
        193.0,
        181.0,
        183.0,
        218.0,
        230.0,
        242.0,
        209.0,
        191.0,
        172.0,
        194.0,
        196.0,
        196.0,
        236.0,
        235.0,
        229.0,
        243.0,
        264.0,
        272.0,
        237.0,
        211.0,
        180.0,
        201.0,
        204.0,
        188.0,
        235.0,
        227.0,
        234.0,
        264.0,
        302.0,
        293.0,
        259.0,
        229.0,
        203.0,
        229.0,
        242.0,
        233.0,
        267.0,
        269.0,
        270.0,
        315.0,
        364.0,
        347.0,
        312.0,
        274.0,
        237.0,
        278.0,
        284.0,
        277.0,
        317.0,
        313.0,
        318.0,
        374.0,
        413.0,
        405.0,
        355.0,
        306.0,
        271.0,
        306.0,
        315.0,
        301.0,
        356.0,
        348.0,
        355.0,
        422.0,
        465.0,
        467.0,
        404.0,
        347.0,
        305.0,
        336.0,
        340.0,
        318.0,
        362.0,
        348.0,
        363.0,
        435.0,
        491.0,
        505.0,
        404.0,
        359.0,
        310.0,
        337.0,
        360.0,
        342.0,
        406.0,
        396.0,
        420.0,
        472.0,
        548.0,
        559.0,
        463.0,
        407.0,
        362.0,
        405.0,
        417.0,
        391.0,
        419.0,
        461.0,
        472.0,
        535.0,
        622.0,
        606.0,
        508.0,
        461.0,
        390.0,
        432.0,
    ],
    dtype=np.float32,
)


AirPassengersDF = pd.DataFrame(
    {
        "unique_id": np.ones(len(AirPassengers)),
        "ds": pd.date_range(
            start="1949-01-01", periods=len(AirPassengers), freq=pd.offsets.MonthEnd()
        ),
        "y": AirPassengers,
    }
)


# Declare Panel Data
unique_id = np.concatenate(
    [["Airline1"] * len(AirPassengers), ["Airline2"] * len(AirPassengers)]
)
ds = np.tile(
    pd.date_range(
        start="1949-01-01", periods=len(AirPassengers), freq=pd.offsets.MonthEnd()
    ).to_numpy(),
    2,
)
y = np.concatenate([AirPassengers, AirPassengers + 300])

AirPassengersPanel = pd.DataFrame({"unique_id": unique_id, "ds": ds, "y": y})

# For future exogenous variables
# Declare SeasonalNaive12 and fill first 12 values with y
snaive = (
    AirPassengersPanel.groupby("unique_id")["y"]
    .shift(periods=12)
    .reset_index(drop=True)
)
AirPassengersPanel["trend"] = range(len(AirPassengersPanel))
AirPassengersPanel["y_[lag12]"] = snaive.fillna(AirPassengersPanel["y"])

# Declare Static Data
unique_id = np.array(["Airline1", "Airline2"])
airline1_dummy = [0, 1]
airline2_dummy = [1, 0]
AirPassengersStatic = pd.DataFrame(
    {"unique_id": unique_id, "airline1": airline1_dummy, "airline2": airline2_dummy}
)

AirPassengersPanel.groupby("unique_id").tail(4)


class TimeFeature:
    def __init__(self):
        pass

    def __call__(self, index: pd.DatetimeIndex):
        return print("Overwrite with corresponding feature")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    """Second of minute encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    """Minute of hour encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    """Hour of day encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    """Day of week encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    """Day of month encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    """Day of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    """Month of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    """Week of year encoded as value between [-0.5, 0.5]."""

    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.week - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq_str: str) -> List[TimeFeature]:
    """Returns a list of time features that will be appropriate for the given frequency string.

    Args:
        freq_str (str): Frequency string of the form [multiple][granularity] such as "12H", "5min", "1D" etc.

    Returns:
        List[TimeFeature]: List of time features appropriate for the frequency.
    """

    if freq_str not in ["Q", "M", "MS", "W", "D", "B", "H", "T", "S"]:
        raise Exception("Frequency not supported")

    if freq_str in ["Q", "M", "MS"]:
        return [cls() for cls in [MonthOfYear]]
    elif freq_str == "W":
        return [cls() for cls in [DayOfMonth, WeekOfYear]]
    elif freq_str in ["D", "B"]:
        return [cls() for cls in [DayOfWeek, DayOfMonth, DayOfYear]]
    elif freq_str == "H":
        return [cls() for cls in [HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]]
    elif freq_str == "T":
        return [
            cls() for cls in [MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear]
        ]
    else:
        return [
            cls()
            for cls in [
                SecondOfMinute,
                MinuteOfHour,
                HourOfDay,
                DayOfWeek,
                DayOfMonth,
                DayOfYear,
            ]
        ]


def augment_calendar_df(df, freq="H"):
    """Augment a dataframe with calendar features based on frequency.

    Frequency mappings:
    - Q - [month]
    - M - [month]
    - W - [Day of month, week of year]
    - D - [Day of week, day of month, day of year]
    - B - [Day of week, day of month, day of year]
    - H - [Hour of day, day of week, day of month, day of year]
    - T - [Minute of hour*, hour of day, day of week, day of month, day of year]
    - S - [Second of minute, minute of hour, hour of day, day of week, day of month, day of year]

    *minute returns a number from 0-3 corresponding to the 15 minute period it falls into.

    Args:
        df (pd.DataFrame): DataFrame to augment with calendar features.
        freq (str, optional): Frequency string for determining which features to add. Defaults to "H".

    Returns:
        Tuple[pd.DataFrame, List[str]]: Tuple of (augmented DataFrame, list of feature column names).
    """
    df = df.copy()

    freq_map = {
        "Q": ["month"],
        "M": ["month"],
        "MS": ["month"],
        "W": ["monthday", "yearweek"],
        "D": ["weekday", "monthday", "yearday"],
        "B": ["weekday", "monthday", "yearday"],
        "H": ["dayhour", "weekday", "monthday", "yearday"],
        "T": ["hourminute", "dayhour", "weekday", "monthday", "yearday"],
        "S": [
            "minutesecond",
            "hourminute",
            "dayhour",
            "weekday",
            "monthday",
            "yearday",
        ],
    }

    ds_col = pd.to_datetime(df.ds.values)
    ds_data = np.vstack(
        [feat(ds_col) for feat in time_features_from_frequency_str(freq)]
    ).transpose(1, 0)
    ds_data = pd.DataFrame(ds_data, columns=freq_map[freq])

    return pd.concat([df, ds_data], axis=1), freq_map[freq]


def get_indexer_raise_missing(idx: pd.Index, vals: List[str]) -> List[int]:
    """Get index positions for values, raising error if any are missing.

    Args:
        idx (pd.Index): Index to search in.
        vals (List[str]): Values to find indices for.

    Returns:
        List[int]: List of index positions.

    Raises:
        ValueError: If any values are missing from the index.
    """
    idxs = idx.get_indexer(vals)
    missing = [v for i, v in zip(idxs, vals) if i == -1]
    if missing:
        raise ValueError(f"The following values are missing from the index: {missing}")
    return idxs


class PredictionIntervals:
    """Class for storing prediction intervals metadata information."""

    def __init__(
        self,
        n_windows: int = 2,
        method: str = "conformal_distribution",
    ):
        """Initialize PredictionIntervals.

        Args:
            n_windows (int, optional): Number of windows to evaluate. Defaults to 2.
            method (str, optional): One of the supported methods for the computation of prediction intervals:
                conformal_error or conformal_distribution. Defaults to "conformal_distribution".
        """
        if n_windows < 2:
            raise ValueError(
                "You need at least two windows to compute conformal intervals"
            )
        allowed_methods = ["conformal_error", "conformal_distribution"]
        if method not in allowed_methods:
            raise ValueError(f"method must be one of {allowed_methods}")
        self.n_windows = n_windows
        self.method = method

    def __repr__(self):
        return (
            f"PredictionIntervals(n_windows={self.n_windows}, method='{self.method}')"
        )


def add_conformal_distribution_intervals(
    model_fcsts: np.array,
    cs_df: DFType,
    model: str,
    cs_n_windows: int,
    n_series: int,
    horizon: int,
    level: Optional[List[Union[int, float]]] = None,
    quantiles: Optional[List[float]] = None,
) -> Tuple[np.array, List[str]]:
    """Add conformal intervals based on conformal scores using distribution strategy.

    This strategy creates forecast paths based on errors and calculates quantiles using those paths.

    Args:
        model_fcsts (np.array): Model forecasts array.
        cs_df (DFType): DataFrame containing conformal scores.
        model (str): Model name.
        cs_n_windows (int): Number of conformal score windows.
        n_series (int): Number of series.
        horizon (int): Forecast horizon.
        level (Optional[List[Union[int, float]]], optional): Confidence levels for prediction intervals. Defaults to None.
        quantiles (Optional[List[float]], optional): Quantiles for prediction intervals. Defaults to None.

    Returns:
        Tuple[np.array, List[str]]: Tuple of (forecasts with intervals, column names).
    """
    assert (
        level is not None or quantiles is not None
    ), "Either level or quantiles must be provided"

    if quantiles is None and level is not None:
        alphas = [100 - lv for lv in level]
        cuts = [alpha / 200 for alpha in reversed(alphas)]
        cuts.extend(1 - alpha / 200 for alpha in alphas)
    elif quantiles is not None:
        cuts = quantiles

    scores = cs_df[model].to_numpy().reshape(n_series, cs_n_windows, horizon)
    scores = scores.transpose(1, 0, 2)
    # restrict scores to horizon
    scores = scores[:, :, :horizon]
    mean = model_fcsts.reshape(1, n_series, -1)
    scores = np.vstack([mean - scores, mean + scores])
    scores_quantiles = np.quantile(
        scores,
        cuts,
        axis=0,
    )
    scores_quantiles = scores_quantiles.reshape(len(cuts), -1).T
    if quantiles is None and level is not None:
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
    elif quantiles is not None:
        out_cols = [f"{model}-ql{q}" for q in quantiles]

    fcsts_with_intervals = np.hstack([model_fcsts, scores_quantiles])

    return fcsts_with_intervals, out_cols


def add_conformal_error_intervals(
    model_fcsts: np.array,
    cs_df: DFType,
    model: str,
    cs_n_windows: int,
    n_series: int,
    horizon: int,
    level: Optional[List[Union[int, float]]] = None,
    quantiles: Optional[List[float]] = None,
) -> Tuple[np.array, List[str]]:
    """Add conformal intervals based on conformal scores using error strategy.

    This strategy creates prediction intervals based on absolute errors.

    Args:
        model_fcsts (np.array): Model forecasts array.
        cs_df (DFType): DataFrame containing conformal scores.
        model (str): Model name.
        cs_n_windows (int): Number of conformal score windows.
        n_series (int): Number of series.
        horizon (int): Forecast horizon.
        level (Optional[List[Union[int, float]]], optional): Confidence levels for prediction intervals. Defaults to None.
        quantiles (Optional[List[float]], optional): Quantiles for prediction intervals. Defaults to None.

    Returns:
        Tuple[np.array, List[str]]: Tuple of (forecasts with intervals, column names).
    """
    assert (
        level is not None or quantiles is not None
    ), "Either level or quantiles must be provided"

    if quantiles is None and level is not None:
        alphas = [100 - lv for lv in level]
        cuts = [alpha / 200 for alpha in reversed(alphas)]
        cuts.extend(1 - alpha / 200 for alpha in alphas)
    elif quantiles is not None:
        cuts = quantiles

    mean = model_fcsts.ravel()
    scores = cs_df[model].to_numpy().reshape(n_series, cs_n_windows, horizon)
    scores = scores.transpose(1, 0, 2)
    # restrict scores to horizon
    scores = scores[:, :, :horizon]
    scores_quantiles = np.quantile(
        scores,
        cuts,
        axis=0,
    )
    scores_quantiles = scores_quantiles.reshape(len(cuts), -1)

    if quantiles is None and level is not None:
        lo_cols = [f"{model}-lo-{lv}" for lv in reversed(level)]
        hi_cols = [f"{model}-hi-{lv}" for lv in level]
        out_cols = lo_cols + hi_cols
    else:
        out_cols = [f"{model}-ql{q}" for q in cuts]

    scores_quantiles_ls = []
    for i, q in enumerate(cuts):
        if q < 0.5:
            scores_quantiles_ls.append(mean - scores_quantiles[::-1][i])
        elif q > 0.5:
            scores_quantiles_ls.append(mean + scores_quantiles[i])
        else:
            scores_quantiles_ls.append(mean)
    scores_quantiles = np.vstack(scores_quantiles_ls).T

    fcsts_with_intervals = np.hstack([model_fcsts, scores_quantiles])

    return fcsts_with_intervals, out_cols


def get_prediction_interval_method(method: str):
    """Get the prediction interval method function by name.

    Args:
        method (str): Name of the prediction interval method.

    Returns:
        Callable: The corresponding method function.

    Raises:
        ValueError: If the method is not supported.
    """
    available_methods = {
        "conformal_distribution": add_conformal_distribution_intervals,
        "conformal_error": add_conformal_error_intervals,
    }
    if method not in available_methods.keys():
        raise ValueError(
            f"prediction intervals method {method} not supported "
            f'please choose one of {", ".join(available_methods.keys())}'
        )
    return available_methods[method]


def level_to_quantiles(level: List[Union[int, float]]) -> List[float]:
    """Convert a list of confidence levels to quantiles.

    Args:
        level (List[Union[int, float]]): List of confidence levels (e.g., [80, 90]).

    Returns:
        List[float]: List of corresponding quantiles.
    """
    level_set = set(level)
    return sorted(
        list(
            set(sum([[(50 - l / 2) / 100, (50 + l / 2) / 100] for l in level_set], []))
        )
    )


def quantiles_to_level(quantiles: List[float]) -> List[Union[int, float]]:
    """Convert a list of quantiles to confidence levels.

    Args:
        quantiles (List[float]): List of quantiles (e.g., [0.1, 0.5, 0.9]).

    Returns:
        List[Union[int, float]]: List of corresponding confidence levels.
    """
    quantiles_set = set(quantiles)
    return sorted(
        set(
            [
                int(round(100 - 200 * (q * (q < 0.5) + (1 - q) * (q >= 0.5)), 2))
                for q in quantiles_set
            ]
        )
    )

# class ShapModelWrapper:
#     """
#     SHAP wrapper that uses nf.predict() to properly handle scaling
#     """

#     def __init__(self, nf_object, model, train_df, static_df, futr_df=None):
#         self.nf = nf_object  # Already fitted NeuralForecast object
#         self.model = model
#         self.train_df = train_df
#         self.static_df = static_df
#         self.futr_df = futr_df
#         self.freq = nf_object.freq

#         self.futr_exog_cols = model.futr_exog_list
#         self.hist_exog_cols = model.hist_exog_list
#         self.stat_exog_cols = model.stat_exog_list
#         self.input_size = model.input_size
#         self.h = model.h
#         self.model_alias = model.alias or type(model).__name__
#         self.model.trainer_kwargs["logger"] = False

#         # Calculate input dimensions
#         self.n_futr_features = (
#             len(self.futr_exog_cols) * (self.input_size + self.h)
#             if self.futr_exog_cols
#             else 0
#         )
#         self.n_hist_exog_features = (
#             len(self.hist_exog_cols) * self.input_size if self.hist_exog_cols else 0
#         )
#         self.n_hist_target_features = self.input_size
#         self.n_series_features = 1  # unique_id encoded as integer

#         # Create unique_id to integer mapping
#         available_unique_ids = sorted(self.train_df["unique_id"].unique())
#         self.unique_id_to_int = {uid: i for i, uid in enumerate(available_unique_ids)}
#         self.int_to_unique_id = {i: uid for uid, i in self.unique_id_to_int.items()}

#         # Store static features mapping if needed
#         self.static_mapping = {}
#         if self.stat_exog_cols and static_df is not None:
#             for unique_id in static_df["unique_id"].unique():
#                 static_values = static_df[static_df["unique_id"] == unique_id][
#                     self.stat_exog_cols
#                 ].values[0]
#                 self.static_mapping[unique_id] = static_values

#     def _predict_single_sample(self, x_flat_single, horizon_idx=None):
#         """Process a single sample - extracted from predict_batch for parallelization"""
#         # Parse the flattened input
#         idx = 0

#         # Series identifier
#         series_int = int(x_flat_single[idx])
#         unique_id = self.int_to_unique_id[series_int]
#         idx += self.n_series_features

#         # Future exogenous features (if present)
#         temp_futr_df = None
#         if self.futr_exog_cols and self.futr_df is not None:
#             futr_flat = x_flat_single[idx : idx + self.n_futr_features]
#             idx += self.n_futr_features

#             # Reshape to (input_size + h, n_futr_features)
#             futr_reshaped = futr_flat.reshape(
#                 self.input_size + self.h, len(self.futr_exog_cols)
#             )

#             # Get the future part only (last h timesteps)
#             futr_future = futr_reshaped[-self.h :, :]

#             # Create future DataFrame with dummy consecutive dates
#             series_train_data = self.train_df[self.train_df["unique_id"] == unique_id]
#             last_train_date = series_train_data["ds"].iloc[-1]

#             if pd.api.types.is_datetime64_any_dtype(series_train_data["ds"]):
#                 future_dates = pd.date_range(
#                     start=last_train_date, periods=self.h + 1, freq=self.freq
#                 )[1:]
#             else:
#                 future_dates = np.arange(
#                     last_train_date + 1, last_train_date + self.h + 1
#                 )

#             temp_futr_df = pd.DataFrame(futr_future, columns=self.futr_exog_cols)
#             temp_futr_df["ds"] = future_dates[: self.h]
#             temp_futr_df["unique_id"] = unique_id

#         # Create a copy of ALL training data
#         temp_train_df = self.train_df.copy()
#         series_mask = temp_train_df["unique_id"] == unique_id
#         series_indices = temp_train_df[series_mask].index

#         # Historical exogenous features (if present)
#         if self.hist_exog_cols:
#             hist_exog_flat = x_flat_single[idx : idx + self.n_hist_exog_features]
#             hist_exog_reshaped = hist_exog_flat.reshape(
#                 self.input_size, len(self.hist_exog_cols)
#             )
#             idx += self.n_hist_exog_features

#             last_indices = series_indices[-self.input_size :]
#             for i, col in enumerate(self.hist_exog_cols):
#                 temp_train_df.loc[last_indices, col] = hist_exog_reshaped[:, i]

#         # Historical target values
#         hist_target = x_flat_single[idx : idx + self.n_hist_target_features]

#         last_indices = series_indices[-self.input_size :]
#         temp_train_df.loc[last_indices, "y"] = hist_target

#         # Prepare futr_df for ALL series
#         if temp_futr_df is not None:
#             all_futr_df = []
#             for uid in temp_train_df["unique_id"].unique():
#                 if uid == unique_id:
#                     all_futr_df.append(temp_futr_df)
#                 else:
#                     if self.futr_df is not None:
#                         other_series_futr = self.futr_df[
#                             self.futr_df["unique_id"] == uid
#                         ].copy()
#                         if len(other_series_futr) > 0:
#                             other_series_futr = other_series_futr.iloc[: self.h]
#                             all_futr_df.append(other_series_futr)
#                         else:
#                             other_train_data = temp_train_df[
#                                 temp_train_df["unique_id"] == uid
#                             ]
#                             last_date = other_train_data["ds"].iloc[-1]

#                             if pd.api.types.is_datetime64_any_dtype(
#                                 other_train_data["ds"]
#                             ):
#                                 future_dates = pd.date_range(
#                                     start=last_date, periods=self.h + 1, freq=self.freq
#                                 )[1:]
#                             else:
#                                 future_dates = np.arange(
#                                     last_date + 1, last_date + self.h + 1
#                                 )

#                             dummy_futr = pd.DataFrame()
#                             for col in self.futr_exog_cols:
#                                 dummy_futr[col] = np.zeros(self.h)
#                             dummy_futr["ds"] = future_dates[: self.h]
#                             dummy_futr["unique_id"] = uid
#                             all_futr_df.append(dummy_futr)

#             combined_futr_df = (
#                 pd.concat(all_futr_df, ignore_index=True) if all_futr_df else None
#             )
#         else:
#             combined_futr_df = None

#         # Use the already fitted NeuralForecast object to predict
#         forecast = self.nf.predict(
#             df=temp_train_df, static_df=self.static_df, futr_df=combined_futr_df
#         )

#         # Extract predictions for this series
#         series_forecast = forecast[forecast["unique_id"] == unique_id].reset_index(
#             drop=True
#         )

#         if len(series_forecast) == 0:
#             raise ValueError(f"No predictions found for series {unique_id}")

#         if self.model_alias not in series_forecast.columns:
#             model_cols = [
#                 col for col in series_forecast.columns if col not in ["unique_id", "ds"]
#             ]
#             if model_cols:
#                 actual_model_col = model_cols[0]
#             else:
#                 raise ValueError(
#                     f"Model column {self.model_alias} not found in forecast columns: {series_forecast.columns.tolist()}"
#                 )
#         else:
#             actual_model_col = self.model_alias

#         if horizon_idx is not None:
#             if horizon_idx >= len(series_forecast):
#                 raise IndexError(
#                     f"horizon_idx {horizon_idx} is out of bounds. Only {len(series_forecast)} horizons available."
#                 )
#             pred = series_forecast[actual_model_col].iloc[horizon_idx]
#         else:
#             pred = series_forecast[actual_model_col].mean()

#         return pred

#     def predict_batch(self, X_flat, horizon_idx=None):
#         """
#         Convert flattened input to DataFrames and use nf.predict()
#         Now with parallel processing for better performance
#         """
#         batch_size = X_flat.shape[0]

#         # Use ThreadPoolExecutor for parallel processing
#         max_workers = min(4, batch_size)  # Don't create more threads than samples

#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Submit all tasks
#             futures = [
#                 executor.submit(self._predict_single_sample, X_flat[b], horizon_idx)
#                 for b in range(batch_size)
#             ]

#             # Collect results in order
#             predictions = [future.result() for future in futures]

#         return np.array(predictions)

class ShapModelWrapper:
    """
    SHAP wrapper that uses nf.predict() to properly handle scaling
    Optimized version using batch prediction with artificial unique_ids
    """

    def __init__(self, nf_object, model, train_df, static_df, futr_df=None):
        self.nf = nf_object  # Already fitted NeuralForecast object
        self.model = model
        self.train_df = train_df
        self.static_df = static_df
        self.futr_df = futr_df
        self.freq = nf_object.freq

        self.futr_exog_cols = model.futr_exog_list
        self.hist_exog_cols = model.hist_exog_list
        self.stat_exog_cols = model.stat_exog_list
        self.input_size = model.input_size
        self.h = model.h
        self.model_alias = model.alias or type(model).__name__
        self.model.trainer_kwargs["logger"] = False

        # Calculate input dimensions
        self.n_futr_features = (
            len(self.futr_exog_cols) * (self.input_size + self.h)
            if self.futr_exog_cols
            else 0
        )
        self.n_hist_exog_features = (
            len(self.hist_exog_cols) * self.input_size if self.hist_exog_cols else 0
        )
        self.n_hist_target_features = self.input_size
        self.n_series_features = 1  # unique_id encoded as integer

        # Create unique_id to integer mapping
        available_unique_ids = sorted(self.train_df["unique_id"].unique())
        self.unique_id_to_int = {uid: i for i, uid in enumerate(available_unique_ids)}
        self.int_to_unique_id = {i: uid for uid, i in self.unique_id_to_int.items()}

        # Store static features mapping if needed
        self.static_mapping = {}
        if self.stat_exog_cols and static_df is not None:
            for unique_id in static_df["unique_id"].unique():
                static_values = static_df[static_df["unique_id"] == unique_id][
                    self.stat_exog_cols
                ].values[0]
                self.static_mapping[unique_id] = static_values

    def _process_sample_to_dataframes(self, x_flat_single, artificial_id):
        """
        Process a single sample and create DataFrames for it
        Returns: (train_df_portion, futr_df_portion, original_series_id)
        """
        # Parse the flattened input
        idx = 0

        # Series identifier
        series_int = int(x_flat_single[idx])
        original_unique_id = self.int_to_unique_id[series_int]
        idx += self.n_series_features

        # Future exogenous features (if present)
        temp_futr_df = None
        if self.futr_exog_cols and self.futr_df is not None:
            futr_flat = x_flat_single[idx : idx + self.n_futr_features]
            idx += self.n_futr_features

            # Reshape to (input_size + h, n_futr_features)
            futr_reshaped = futr_flat.reshape(
                self.input_size + self.h, len(self.futr_exog_cols)
            )

            # Get the future part only (last h timesteps)
            futr_future = futr_reshaped[-self.h :, :]

            # Create future DataFrame with dummy consecutive dates
            series_train_data = self.train_df[self.train_df["unique_id"] == original_unique_id]
            last_train_date = series_train_data["ds"].iloc[-1]

            if pd.api.types.is_datetime64_any_dtype(series_train_data["ds"]):
                future_dates = pd.date_range(
                    start=last_train_date, periods=self.h + 1, freq=self.freq
                )[1:]
            else:
                future_dates = np.arange(
                    last_train_date + 1, last_train_date + self.h + 1
                )

            temp_futr_df = pd.DataFrame(futr_future, columns=self.futr_exog_cols)
            temp_futr_df["ds"] = future_dates[: self.h]
            temp_futr_df["unique_id"] = artificial_id

        # Create a copy of the specific series training data
        series_train_data = self.train_df[self.train_df["unique_id"] == original_unique_id].copy()
        
        # Change unique_id to artificial_id
        series_train_data["unique_id"] = artificial_id

        # Historical exogenous features (if present)
        if self.hist_exog_cols:
            hist_exog_flat = x_flat_single[idx : idx + self.n_hist_exog_features]
            hist_exog_reshaped = hist_exog_flat.reshape(
                self.input_size, len(self.hist_exog_cols)
            )
            idx += self.n_hist_exog_features

            # Update last input_size rows with new historical exogenous values
            for i, col in enumerate(self.hist_exog_cols):
                series_train_data.iloc[-self.input_size :, series_train_data.columns.get_loc(col)] = hist_exog_reshaped[:, i]

        # Historical target values
        hist_target = x_flat_single[idx : idx + self.n_hist_target_features]

        # Update last input_size rows with new target values
        series_train_data.iloc[-self.input_size :, series_train_data.columns.get_loc("y")] = hist_target

        return series_train_data, temp_futr_df, original_unique_id

    def _create_futr_df_for_all_series(self, combined_train_df, artificial_series_futr_data, artificial_ids):
        """
        Create complete futr_df that includes future data for ALL series in combined_train_df
        """
        if not self.futr_exog_cols:
            return None
            
        all_futr_dfs = []
        all_unique_ids_in_train = combined_train_df["unique_id"].unique()
        
        for uid in all_unique_ids_in_train:
            if uid in artificial_ids:
                # This is an artificial series - use the provided future data
                idx = artificial_ids.index(uid)
                temp_futr_df = artificial_series_futr_data[idx]
                if temp_futr_df is not None:
                    all_futr_dfs.append(temp_futr_df)
                else:
                    # Create dummy future data if none provided
                    series_train_data = combined_train_df[combined_train_df["unique_id"] == uid]
                    last_date = series_train_data["ds"].iloc[-1]
                    
                    if pd.api.types.is_datetime64_any_dtype(series_train_data["ds"]):
                        future_dates = pd.date_range(
                            start=last_date, periods=self.h + 1, freq=self.freq
                        )[1:]
                    else:
                        future_dates = np.arange(
                            last_date + 1, last_date + self.h + 1
                        )

                    dummy_futr = pd.DataFrame()
                    for col in self.futr_exog_cols:
                        dummy_futr[col] = np.zeros(self.h)
                    dummy_futr["ds"] = future_dates[:self.h]
                    dummy_futr["unique_id"] = uid
                    all_futr_dfs.append(dummy_futr)
            else:
                # This is an original series - use original futr_df data
                if self.futr_df is not None:
                    other_series_futr = self.futr_df[self.futr_df["unique_id"] == uid].copy()
                    if len(other_series_futr) > 0:
                        other_series_futr = other_series_futr.iloc[:self.h]
                        all_futr_dfs.append(other_series_futr)
                    else:
                        # Create dummy future data for original series without futr_df
                        series_train_data = combined_train_df[combined_train_df["unique_id"] == uid]
                        last_date = series_train_data["ds"].iloc[-1]

                        if pd.api.types.is_datetime64_any_dtype(series_train_data["ds"]):
                            future_dates = pd.date_range(
                                start=last_date, periods=self.h + 1, freq=self.freq
                            )[1:]
                        else:
                            future_dates = np.arange(
                                last_date + 1, last_date + self.h + 1
                            )

                        dummy_futr = pd.DataFrame()
                        for col in self.futr_exog_cols:
                            dummy_futr[col] = np.zeros(self.h)
                        dummy_futr["ds"] = future_dates[:self.h]
                        dummy_futr["unique_id"] = uid
                        all_futr_dfs.append(dummy_futr)
                else:
                    # No original futr_df provided - create dummy data
                    series_train_data = combined_train_df[combined_train_df["unique_id"] == uid]
                    last_date = series_train_data["ds"].iloc[-1]

                    if pd.api.types.is_datetime64_any_dtype(series_train_data["ds"]):
                        future_dates = pd.date_range(
                            start=last_date, periods=self.h + 1, freq=self.freq
                        )[1:]
                    else:
                        future_dates = np.arange(
                            last_date + 1, last_date + self.h + 1
                        )

                    dummy_futr = pd.DataFrame()
                    for col in self.futr_exog_cols:
                        dummy_futr[col] = np.zeros(self.h)
                    dummy_futr["ds"] = future_dates[:self.h]
                    dummy_futr["unique_id"] = uid
                    all_futr_dfs.append(dummy_futr)
        
        return pd.concat(all_futr_dfs, ignore_index=True) if all_futr_dfs else None

    def _create_combined_static_df(self, artificial_ids, original_series_ids):
        """Handle static features for artificial series"""
        if self.static_df is None:
            return None
            
        artificial_static_rows = []
        for artificial_id, original_id in zip(artificial_ids, original_series_ids):
            static_row = self.static_df[self.static_df["unique_id"] == original_id].copy()
            static_row["unique_id"] = artificial_id
            artificial_static_rows.append(static_row)
        
        return pd.concat([self.static_df] + artificial_static_rows, ignore_index=True)

    def _extract_prediction_from_forecast(self, forecast, artificial_id, horizon_idx):
        """Extract prediction for a specific artificial series"""
        series_forecast = forecast[forecast["unique_id"] == artificial_id].reset_index(drop=True)

        if len(series_forecast) == 0:
            raise ValueError(f"No predictions found for series {artificial_id}")

        if self.model_alias not in series_forecast.columns:
            model_cols = [
                col for col in series_forecast.columns if col not in ["unique_id", "ds"]
            ]
            if model_cols:
                actual_model_col = model_cols[0]
            else:
                raise ValueError(
                    f"Model column {self.model_alias} not found in forecast columns: {series_forecast.columns.tolist()}"
                )
        else:
            actual_model_col = self.model_alias

        if horizon_idx is not None:
            if horizon_idx >= len(series_forecast):
                raise IndexError(
                    f"horizon_idx {horizon_idx} is out of bounds. Only {len(series_forecast)} horizons available."
                )
            pred = series_forecast[actual_model_col].iloc[horizon_idx]
        else:
            pred = series_forecast[actual_model_col].mean()

        return pred

    def predict_batch(self, X_flat, horizon_idx=None):
        """
        Optimized batch prediction using artificial unique_ids
        """
        batch_size = X_flat.shape[0]
        
        # Create artificial unique_ids to avoid conflicts with existing ones
        artificial_ids = [f"shap_batch_{i}" for i in range(batch_size)]
        
        # Process all samples into DataFrames
        all_train_dfs = []
        all_futr_dfs = []
        original_series_ids = []
        
        for i, x_flat_single in enumerate(X_flat):
            artificial_id = artificial_ids[i]
            temp_train_df, temp_futr_df, original_id = self._process_sample_to_dataframes(
                x_flat_single, artificial_id
            )
            
            all_train_dfs.append(temp_train_df)
            all_futr_dfs.append(temp_futr_df)
            original_series_ids.append(original_id)
        
        # Combine training DataFrames (original + all artificial series)
        combined_train_df = pd.concat([self.train_df] + all_train_dfs, ignore_index=True)
        
        # Create combined future DataFrame
        combined_futr_df = None
        if self.futr_exog_cols:
            combined_futr_df = self._create_futr_df_for_all_series(
                combined_train_df, all_futr_dfs, artificial_ids
            )
        
        # Handle static features
        combined_static_df = self._create_combined_static_df(artificial_ids, original_series_ids)
        
        # Single batch prediction call!
        forecast = self.nf.predict(
            df=combined_train_df, 
            static_df=combined_static_df, 
            futr_df=combined_futr_df
        )
        
        # Extract predictions for each artificial series
        predictions = []
        for artificial_id in artificial_ids:
            pred = self._extract_prediction_from_forecast(forecast, artificial_id, horizon_idx)
            predictions.append(pred)
        
        return np.array(predictions)

def create_input_tensor_for_series(train_df, unique_id, wrapper_model, futr_df=None):
    """Create input tensor for a specific series"""
    train_series = train_df[train_df["unique_id"] == unique_id]

    input_components = []
    futr_exog_cols = wrapper_model.futr_exog_cols
    hist_exog_cols = wrapper_model.hist_exog_cols
    input_size = wrapper_model.input_size

    # Series identifier (encoded as integer)
    series_int = wrapper_model.unique_id_to_int[unique_id]
    input_components.append(np.array([series_int]))

    # Future exogenous features (historical + future) if they exist
    if futr_exog_cols:
        if futr_df is None:
            raise ValueError("You must pass a futr_df if futr_exog_list is specified.")
        futr_exog_series = futr_df[futr_df["unique_id"] == unique_id].reset_index(
            drop=True
        )
        # Get historical part
        futr_hist_data = train_series[futr_exog_cols].values[-input_size:]
        # Get future part
        futr_pred_data = futr_exog_series[futr_exog_cols].values
        # Combine and flatten
        full_futr_data = np.vstack([futr_hist_data, futr_pred_data]).flatten()
        input_components.append(full_futr_data)

    # Historical exogenous features (if they exist)
    if hist_exog_cols:
        hist_exog_data = train_series[hist_exog_cols].values[-input_size:].flatten()
        input_components.append(hist_exog_data)

    # Historical target values (always present)
    hist_target_data = train_series["y"].values[-input_size:]
    input_components.append(hist_target_data)

    # Combine all features
    complete_input = np.concatenate(input_components)
    return complete_input.reshape(1, -1)

def create_multi_series_background_data(
    train_df, wrapper_model, n_samples_per_series=10
):
    """Create background data including samples from all series"""
    background_samples = []

    futr_exog_cols = wrapper_model.futr_exog_cols
    hist_exog_cols = wrapper_model.hist_exog_cols
    input_size = wrapper_model.input_size
    horizon = wrapper_model.h

    for unique_id in train_df["unique_id"].unique():
        train_series = train_df[train_df["unique_id"] == unique_id]

        # Determine the range of valid indices
        start_idx = input_size
        if futr_exog_cols:
            end_idx = len(train_series) - horizon + 1
        else:
            end_idx = len(train_series)

        # Sample points from this series
        if end_idx > start_idx:
            sample_indices = np.linspace(
                start_idx,
                end_idx - 1,
                min(n_samples_per_series, end_idx - start_idx),
                dtype=int,
            )

            for i in sample_indices:
                sample_components = []

                # Series identifier
                series_int = wrapper_model.unique_id_to_int[unique_id]
                sample_components.append(np.array([series_int]))

                # Future exogenous features (if they exist)
                if futr_exog_cols:
                    futr_hist_data = (
                        train_series[futr_exog_cols]
                        .iloc[i - input_size : i]
                        .to_numpy()
                        .flatten()
                    )
                    futr_pred_data = (
                        train_series[futr_exog_cols]
                        .iloc[i : i + horizon]
                        .to_numpy()
                        .flatten()
                    )
                    full_futr_data = np.concatenate([futr_hist_data, futr_pred_data])
                    sample_components.append(full_futr_data)

                # Historical exogenous features (if they exist)
                if hist_exog_cols:
                    hist_exog_window = (
                        train_series[hist_exog_cols]
                        .iloc[i - input_size : i]
                        .to_numpy()
                        .flatten()
                    )
                    sample_components.append(hist_exog_window)

                # Historical target values
                hist_target_window = (
                    train_series["y"].iloc[i - input_size : i].to_numpy()
                )
                sample_components.append(hist_target_window)

                # Combine all features
                complete_sample = np.concatenate(sample_components)
                background_samples.append(complete_sample)

    return np.array(background_samples)

def create_multi_series_feature_names(wrapper_model):
    """Create feature names for multi-series input"""
    feature_names = ["series_id"]
    futr_exog_cols = wrapper_model.futr_exog_cols
    hist_exog_cols = wrapper_model.hist_exog_cols
    input_size = wrapper_model.input_size
    horizon = wrapper_model.h

    # Future exogenous features (if they exist) - FULL LENGTH
    if futr_exog_cols:
        for i in range(input_size):
            for col in futr_exog_cols:
                feature_names.append(f"{col}_hist_lag{i+1}")
        for i in range(horizon):
            for col in futr_exog_cols:
                feature_names.append(f"{col}_h{i+1}")

    # Historical exogenous features (if they exist)
    if hist_exog_cols:
        for i in range(input_size):
            for col in hist_exog_cols:
                feature_names.append(f"{col}_lag{i+1}")

    # Historical target values (always present)
    for i in range(input_size):
        feature_names.append(f"y_lag{i+1}")

    return feature_names
