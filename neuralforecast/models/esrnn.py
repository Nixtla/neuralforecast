__all__ = ["ESRNN"]

import json
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from torch import nn

from ..common._base_model import BaseModel
from ..common._esrnn import scan, native_lstm, use_native_lstm
from ..losses.pytorch import MAE, MSE, RMSE, HuberLoss, QuantileLoss
from ..tsdataset import TimeSeriesDataset
from .dilated_rnn import DRNN


class _ExponentialSmoothing(nn.Module):
    """Differentiable multiplicative ES with a separate parameter row per series.

    Seasonal trajectories store the initial m factors followed by updates for
    their next occurrence. Thus s[t] is the factor available *before* seeing
    y[t], and observing y[t] writes s[t+m]. Both double-seasonal updates use
    the other component's lagged factor, as in Smyl's multiplicative ES-RNN
    and https://github.com/kdgutier/esrnn_torch/blob/master/ESRNN/utils/ESRNN.py.
    Independent batched implementation with PyTorch and optional native kernels.
    """

    def __init__(self, n_series, season_lengths, backend="auto"):
        super().__init__()
        self.season_lengths = tuple(season_lengths)
        self.backend = backend
        self.smoothing = nn.Parameter(torch.zeros(n_series, 1 + len(season_lengths)))
        self.initial_seasons = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_series, m)) for m in season_lengths]
        )

    def forward_batch(self, y, series_indices):
        coefficients = self.smoothing[series_indices].sigmoid()
        dtype = torch.promote_types(y.dtype, coefficients.dtype)
        y = y.to(dtype)
        coefficients = coefficients.to(dtype)
        initial = [
            m * weights[series_indices].softmax(-1)
            for m, weights in zip(self.season_lengths, self.initial_seasons)
        ]
        initial = [v.to(dtype) for v in initial]
        while len(initial) < 2:
            initial.append(y.new_empty((len(y), 0)))
        levels, first, second = scan(y, coefficients, *initial, self.backend)
        return levels, [first, second][: len(self.season_lengths)]

    def forward(self, y, series_idx):
        rows = torch.as_tensor([series_idx], device=y.device)
        levels, seasons = self.forward_batch(y.unsqueeze(0), rows)
        return levels[0], [s[0] for s in seasons]

    def forecast_scale(self, levels, seasons, origin, h):
        # origin is a prefix length: y[:origin] is observed. Only the cycle
        # [origin, origin+m) has been computed by that prefix; repeat it.
        scale = levels[origin - 1].expand(h)
        for m, factors in zip(self.season_lengths, seasons):
            indices = origin + torch.arange(h, device=levels.device) % m
            scale = scale * factors[indices]
        return scale

    @staticmethod
    def level_penalty(levels):
        if len(levels) < 3:
            return levels.sum() * 0
        log_levels = levels.log()
        return (log_levels[2:] - 2 * log_levels[1:-1] + log_levels[:-2]).square().mean()


class _ESRNNDRNN(DRNN):
    """Preserve DRNN weights/semantics while fusing eligible CPU LSTM scans."""

    def __init__(self, *args, backend="auto", **kwargs):
        super().__init__(*args, **kwargs)
        self.backend = backend

    def _apply_cell(
        self, dilated_inputs, cell, batch_size, rate, hidden_size, hidden=None
    ):
        if (
            hidden is None
            and isinstance(cell, nn.LSTM)
            and use_native_lstm(dilated_inputs, self.backend)
        ):
            output = native_lstm(
                dilated_inputs,
                cell.weight_ih_l0,
                cell.weight_hh_l0,
                cell.bias_ih_l0,
                cell.bias_hh_l0,
            )
            # ESRNN consumes only chronological outputs, not exported states.
            return output, None
        return super()._apply_cell(
            dilated_inputs, cell, batch_size, rate, hidden_size, hidden
        )


class ESRNN(BaseModel):
    """Exponential smoothing with a global residual dilated LSTM.

    Local smoothing coefficients and initial seasonal cycles are learned for
    each fitted series, jointly with shared recurrent weights. Each recurrent
    step consumes a log-normalized lag window. Forecasts multiply positive
    neural corrections by the origin's ES level and extrapolated seasonality.

    Observations must be finite, strictly positive and contiguous. Prediction
    requires known series and history from their fitted starts. Exogenous
    features, sample weights and distributed training are unsupported.

    Args:
        h (int): Forecast horizon.
        input_size (int): Number of observations per recurrent input window.
            A negative value uses three times the horizon.
        season_length (int or sequence of int): Zero, one, or two seasonal
            periods greater than one. An empty sequence disables seasonality.
        n_series (int): Initial local parameter capacity. Fit infers this from
            the dataset; users ordinarily do not need to specify it.
        encoder_hidden_size (int): Hidden units in each dilated LSTM.
        dilations (sequence of sequences): Dilation rates in residual groups.
        level_variability_penalty (float): Weight of mean squared second
            differences of log level; defaults to 50.
        es_backend (str): Execution backend: 'auto' attempts cached CPU native
            compilation and falls back to PyTorch; 'native' requires compiled
            CPU float32/float64 kernels; 'torch' disables compilation. Native
            kernels support first-order gradients; use 'torch' for higher orders.
        loss (BasePointLoss): Training loss on positive normalized forecasts.
            Defaults to pinball loss at quantile 0.5.
        valid_loss (BasePointLoss): Validation loss on original-scale forecasts.
            Defaults to MAE. Supported losses: MAE, MSE, RMSE, HuberLoss,
            QuantileLoss.
        max_steps (int): Maximum optimizer steps.
        learning_rate (float): Optimizer learning rate.
        batch_size (int): Number of series per training batch.
        valid_batch_size (int, optional): Number of series per evaluation batch.
        windows_batch_size (int): Maximum origins used for the training loss
            per series. All preceding origins remain recurrent context.
            Use -1 for all origins.
        val_check_steps (int): Steps between validation passes.
        step_size (int): Stride between training origins.
        scaler_type (str): Must be 'identity'.
        **trainer_kwargs: BaseModel optimizer, scheduler, seed and Lightning
            Trainer options.

    References:
        Smyl (2020), A hybrid method of exponential smoothing and recurrent
        neural networks for time series forecasting,
        https://doi.org/10.1016/j.ijforecast.2019.03.017.
    """

    EXOGENOUS_FUTR = False
    EXOGENOUS_HIST = False
    EXOGENOUS_STAT = False
    MULTIVARIATE = False
    RECURRENT = False

    def __init__(
        self,
        h: int,
        input_size: int = -1,
        season_length: Union[int, Sequence[int]] = (),
        n_series: int = 1,
        encoder_hidden_size: int = 40,
        dilations: Optional[Sequence[Sequence[int]]] = None,
        level_variability_penalty: float = 50.0,
        loss=None,
        valid_loss=None,
        max_steps: int = 1000,
        learning_rate: float = 1e-3,
        batch_size: int = 16,
        valid_batch_size: Optional[int] = None,
        windows_batch_size: int = 128,
        val_check_steps: int = 100,
        step_size: int = 1,
        scaler_type: str = "identity",
        es_backend: str = "auto",
        **trainer_kwargs,
    ):
        if isinstance(season_length, int):
            season_length = () if season_length == 1 else (season_length,)
        season_length = tuple(season_length)
        if len(season_length) > 2 or any(
            not isinstance(m, int) or m < 2 for m in season_length
        ):
            raise ValueError(
                "season_length must contain zero, one, or two integer periods > 1."
            )
        if len(set(season_length)) != len(season_length):
            raise ValueError("season_length periods must be distinct.")
        if scaler_type != "identity":
            raise ValueError("ESRNN requires scaler_type='identity'.")
        if (
            windows_batch_size is not None
            and windows_batch_size != -1
            and windows_batch_size < 1
        ):
            raise ValueError("windows_batch_size must be positive, None, or -1.")
        if trainer_kwargs.get("start_padding_enabled", False) or trainer_kwargs.get(
            "exclude_insample_y", False
        ):
            raise ValueError(
                "ESRNN requires observed target history without artificial start padding."
            )
        if trainer_kwargs.get("inference_input_size", input_size) not in (
            None,
            input_size,
            3 * h if input_size < 1 else input_size,
        ):
            raise ValueError("ESRNN uses the fitted input_size for inference.")
        devices = trainer_kwargs.get("devices", 1)
        if devices not in (1, "auto", None) and not (
            isinstance(devices, (tuple, list)) and len(devices) == 1
        ):
            raise ValueError("ESRNN supports a single device only.")
        # BaseModel otherwise selects every CUDA device when devices is omitted.
        # Resolve automatic counts here so this model cannot silently enter DDP.
        trainer_kwargs["devices"] = 1 if devices in ("auto", None) else devices
        if trainer_kwargs.get("num_nodes", 1) != 1:
            raise ValueError("ESRNN supports a single node only.")
        if trainer_kwargs.get("strategy", "auto") not in ("auto", None):
            raise ValueError("ESRNN does not support distributed strategies.")
        loss = QuantileLoss(q=0.5) if loss is None else loss
        valid_loss = MAE() if valid_loss is None else valid_loss
        for objective in (loss, valid_loss):
            if type(objective) not in (MAE, MSE, RMSE, HuberLoss, QuantileLoss):
                raise ValueError(
                    "ESRNN supports single-output point losses: MAE, MSE, RMSE, HuberLoss, QuantileLoss."
                )
        for key in ("futr_exog_list", "hist_exog_list", "stat_exog_list"):
            if trainer_kwargs.get(key):
                raise ValueError("ESRNN does not support exogenous regressors.")
        if h < 1 or n_series < 1 or encoder_hidden_size < 1 or step_size < 1:
            raise ValueError(
                "h, n_series, encoder_hidden_size and step_size must be positive."
            )
        if level_variability_penalty < 0:
            raise ValueError("level_variability_penalty must be nonnegative.")
        dilations = (
            ((1, 2), (4, 8))
            if dilations is None
            else tuple(tuple(g) for g in dilations)
        )
        if not dilations or any(
            not group or any(not isinstance(d, int) or d < 1 for d in group)
            for group in dilations
        ):
            raise ValueError(
                "dilations must contain nonempty groups of positive integers."
            )
        base_options: dict[str, Any] = dict(
            inference_windows_batch_size=-1,
            start_padding_enabled=False,
        )
        base_options.update(trainer_kwargs)
        super().__init__(
            h=h,
            input_size=input_size,
            loss=loss,
            valid_loss=valid_loss,
            max_steps=max_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            windows_batch_size=windows_batch_size,
            val_check_steps=val_check_steps,
            step_size=step_size,
            scaler_type=scaler_type,
            **base_options,
        )
        if es_backend not in ("auto", "torch", "native"):
            raise ValueError("es_backend must be auto, torch, or native.")
        self.es_backend = es_backend
        self.season_lengths = season_length
        self.level_variability_penalty = level_variability_penalty
        self.es = _ExponentialSmoothing(n_series, season_length, backend=es_backend)
        self.encoder = nn.ModuleList(
            [
                _ESRNNDRNN(
                    self.input_size if i == 0 else encoder_hidden_size,
                    encoder_hidden_size,
                    len(group),
                    group,
                    cell_type="LSTM",
                    batch_first=True,
                    backend=es_backend,
                )
                for i, group in enumerate(dilations)
            ]
        )
        self.decoder = nn.Linear(encoder_hidden_size, h)
        self.series_ids_: list[Any] = []
        self.series_starts_: list[Any] = []
        self.series_frequencies_: list[Union[int, str]] = []
        self._batch_series_rows: list[int] = []

    @property
    def _first_origin(self):
        return self.input_size + max(self.season_lengths, default=0)

    def _forecast_batch(self, y, rows, origins, states=None):
        """One chronological RNN sequence per series, processed as a batch.

        Origins are prefix lengths. Padding only extends sequences to the
        right, so it cannot influence any selected, earlier recurrent output.
        """
        first = self._first_origin
        last = max(max(v) for v in origins)
        if min(min(v) for v in origins) < first:
            raise ValueError(
                "ESRNN origins need a complete input window and seasonal warmup."
            )
        levels, seasons = (
            self.es.forward_batch(y[:, :last], rows) if states is None else states
        )
        start = first - self.input_size
        stop = last - self.input_size + 1
        windows = y.unfold(1, self.input_size, 1)[:, start:stop]
        normalized = windows / levels[:, first - 1 : last].unsqueeze(-1)
        for factors in seasons:
            seasonal_windows = factors.unfold(1, self.input_size, 1)[:, start:stop]
            normalized = normalized / seasonal_windows
        hidden = normalized.log()
        for i, group in enumerate(self.encoder):
            encoded, _ = group(hidden)
            hidden = encoded if i == 0 else hidden + encoded
        width = max(len(v) for v in origins)
        positions = torch.tensor(
            [v + [v[0]] * (width - len(v)) for v in origins],
            device=y.device,
            dtype=torch.long,
        )
        selected = hidden.gather(
            1, (positions - first).unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )
        corrections = self.decoder(selected).clamp(-20, 20).exp()
        scale = levels.gather(1, positions - 1).unsqueeze(-1).expand(-1, -1, self.h)
        horizon = torch.arange(self.h, device=y.device)
        for period, factors in zip(self.season_lengths, seasons):
            phases = positions.unsqueeze(-1) + horizon.remainder(period)
            scale = scale * factors.gather(1, phases.flatten(1)).reshape_as(scale)
        return corrections, scale

    def _forecast_series(self, y, series_idx, origins):
        rows = torch.as_tensor([series_idx], device=y.device)
        normalized, scale = self._forecast_batch(y.unsqueeze(0), rows, [origins])
        return normalized[0], scale[0]

    def forward(self, windows_batch):
        """Forecast a chronological series with explicit local identity and origins."""
        normalized, scale = self._forecast_series(
            windows_batch["insample_y"],
            windows_batch["series_idx"],
            windows_batch["origins"],
        )
        return normalized * scale

    def get_extra_state(self):
        from .._serialization import encode_mapping

        # Safetensors requires tensor-only state. Keep local-series metadata as
        # tagged JSON bytes, using the same non-executable codec as hparams.
        metadata = dict(
            series_ids=self.series_ids_,
            series_starts=np.asarray(self.series_starts_),
            series_frequencies=self.series_frequencies_,
        )
        encoded, _ = encode_mapping(metadata, inline=True)
        return torch.tensor(
            list(json.dumps(encoded).encode("utf-8")), dtype=torch.uint8
        )

    def set_extra_state(self, state):
        from .._serialization import decode_mapping

        metadata = decode_mapping(
            json.loads(bytes(state.cpu().tolist()).decode("utf-8"))
        )
        self.series_ids_ = metadata["series_ids"]
        self.series_starts_ = list(metadata["series_starts"])
        self.series_frequencies_ = metadata["series_frequencies"]

    def _bind_dataset(self, dataset, fitting, tail):
        if (
            not isinstance(dataset, TimeSeriesDataset)
            or dataset.series_ids is None
            or dataset.series_times is None
        ):
            raise ValueError(
                "ESRNN requires an in-memory dataset created by TimeSeriesDataset.from_df with series identities and times."
            )
        if "sample_weight" in dataset.temporal_cols:
            raise ValueError("ESRNN does not support sample weights.")
        ids = dataset.series_ids
        if fitting:
            self.series_ids_ = list(ids)
            self.series_starts_ = []
            self.series_frequencies_ = []
        rows = {uid: i for i, uid in enumerate(self.series_ids_)}
        lengths = np.diff(dataset.indptr).tolist()
        for i, (uid, times, length) in enumerate(
            zip(ids, dataset.series_times, lengths)
        ):
            if uid not in rows:
                raise ValueError(
                    f"ESRNN cannot forecast unseen series {uid!r}; refit the model."
                )
            row = rows[uid]
            observed_length = length if fitting else length - tail
            y = dataset.temporal[
                dataset.indptr[i] : dataset.indptr[i] + observed_length, dataset.y_idx
            ]
            mask_idx = dataset.temporal_cols.get_loc("available_mask")
            mask = dataset.temporal[
                dataset.indptr[i] : dataset.indptr[i] + observed_length, mask_idx
            ]
            if not torch.isfinite(y).all() or (y <= 0).any():
                raise ValueError(
                    f"ESRNN requires finite, strictly positive observations for series {uid!r}."
                )
            if not (mask == 1).all():
                raise ValueError(
                    "ESRNN requires contiguous available observations; internal missing values are unsupported."
                )
            minimum = self._first_origin + (self.h + tail if fitting else 0)
            if length < minimum or observed_length < self._first_origin:
                raise ValueError(
                    f"ESRNN has insufficient history for {uid!r}: need at least {minimum} observations including held-out/future rows."
                )
            if not fitting and times[0] != self.series_starts_[row]:
                raise ValueError(
                    f"ESRNN requires history from the fitted start for {uid!r}; truncated history is unsupported."
                )
            integer_time = np.issubdtype(np.asarray(times).dtype, np.integer)
            if fitting:
                frequency = getattr(dataset, "frequency", None)
                if frequency is None:
                    if integer_time:
                        frequency = (times[1] - times[0]).item()
                    elif len(times) == 2:
                        delta = pd.Timedelta(times[1] - times[0])
                        if delta <= pd.Timedelta(0):
                            raise ValueError("ESRNN requires increasing timestamps.")
                        frequency = pd.tseries.frequencies.to_offset(delta).freqstr
                    else:
                        frequency = pd.infer_freq(times)
            else:
                frequency = self.series_frequencies_[row]
            if integer_time:
                valid = (
                    isinstance(frequency, (int, np.integer))
                    and frequency > 0
                    and np.all(np.diff(times) == frequency)
                )
            else:
                valid = frequency is not None and pd.DatetimeIndex(times).equals(
                    pd.date_range(times[0], periods=len(times), freq=frequency)
                )
            if not valid:
                raise ValueError(
                    f"ESRNN requires contiguous times without gaps or duplicates for series {uid!r}."
                )
            if fitting:
                self.series_starts_.append(times[0])
                self.series_frequencies_.append(frequency)
        self._batch_series_rows = [rows[uid] for uid in ids]
        self._batch_series_lengths = lengths

    def fit(
        self,
        dataset,
        val_size=0,
        test_size=0,
        random_seed=None,
        distributed_config=None,
    ):
        """Fit local ES and global weights using only the training partition."""
        if distributed_config is not None:
            raise ValueError("ESRNN does not support distributed training.")
        if val_size < 0 or test_size < 0:
            raise ValueError("val_size and test_size must be nonnegative.")
        if val_size and val_size < self.h:
            raise ValueError("ESRNN val_size must be zero or at least h.")
        self._bind_dataset(dataset, fitting=True, tail=val_size + test_size)
        self.es = _ExponentialSmoothing(
            len(self.series_ids_), self.season_lengths, backend=self.es_backend
        ).to(self.device)
        self.hparams["n_series"] = len(self.series_ids_)
        return super().fit(
            dataset,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            distributed_config=None,
        )

    def predict(
        self,
        dataset,
        test_size=None,
        step_size=1,
        random_seed=None,
        quantiles=None,
        h=None,
        explainer_config=None,
        **data_module_kwargs,
    ):
        """Forecast fitted IDs using full histories anchored at their fitted starts."""
        if not self.series_ids_:
            raise ValueError("ESRNN must be fitted before prediction.")
        if h is not None and h != self.h:
            raise ValueError("ESRNN prediction horizon must equal the fitted h.")
        if explainer_config is not None:
            raise ValueError("ESRNN does not support explanation hooks.")
        if step_size < 1:
            raise ValueError("step_size must be positive.")
        if test_size is not None:
            self.test_size = test_size
        if self.test_size < self.h:
            raise ValueError(
                "ESRNN prediction requires at least h held-out/future rows."
            )
        self._bind_dataset(dataset, fitting=False, tail=self.test_size)
        return super().predict(
            dataset,
            step_size=step_size,
            random_seed=random_seed,
            quantiles=quantiles,
            h=h,
            **data_module_kwargs,
        )

    def _batch_series(self, batch):
        for batch_row, dataset_row in enumerate(batch["series_idx"].tolist()):
            n = self._batch_series_lengths[dataset_row]
            yield (
                batch["temporal"][batch_row, batch["y_idx"], -n:],
                self._batch_series_rows[dataset_row],
            )

    def _pack_batch(self, batch, exclude=0):
        histories, rows = [], []
        for y, row in self._batch_series(batch):
            histories.append(y[: len(y) - exclude] if exclude else y)
            rows.append(row)
        lengths = [len(y) for y in histories]
        longest = max(lengths)
        # Positive right padding is never an observation or a loss target.
        y = torch.stack(
            [
                torch.nn.functional.pad(v, (0, longest - len(v)), value=1.0)
                for v in histories
            ]
        )
        return y, torch.tensor(rows, device=y.device), lengths

    def training_step(self, batch, batch_idx):
        y, rows, lengths = self._pack_batch(batch, self.val_size + self.test_size)
        origins = []
        for length in lengths:
            requested = list(
                range(self._first_origin, length - self.h + 1, self.step_size)
            )
            if (
                self.windows_batch_size is not None
                and self.windows_batch_size > 0
                and len(requested) > self.windows_batch_size
            ):
                selected = (
                    torch.randperm(len(requested))[: self.windows_batch_size]
                    .sort()
                    .values.tolist()
                )
                requested = [requested[i] for i in selected]
            origins.append(requested)
        # Reuse this single state scan for both forecasts and regularization.
        states = self.es.forward_batch(y, rows)
        normalized, scale = self._forecast_batch(y, rows, origins, states=states)
        objectives = []
        offsets = torch.arange(self.h, device=y.device)
        for i, requested in enumerate(origins):
            count = len(requested)
            positions = torch.tensor(requested, device=y.device).unsqueeze(-1) + offsets
            objective = self.loss(
                y=(y[i, positions] / scale[i, :count]).unsqueeze(-1),
                y_hat=normalized[i, :count].unsqueeze(-1),
            )
            if self.level_variability_penalty:
                objective = (
                    objective
                    + self.level_variability_penalty
                    * self.es.level_penalty(states[0][i, : lengths[i]])
                )
            objectives.append(objective)
        loss = torch.stack(objectives).mean()
        if not torch.isfinite(loss):
            raise ValueError("ESRNN training produced a nonfinite loss.")
        self.log(
            "train_loss",
            loss.detach(),
            batch_size=len(objectives),
            on_epoch=True,
            prog_bar=True,
        )
        self.train_trajectories.append((self.global_step, loss.detach().item()))
        return loss

    def validation_step(self, batch, batch_idx):
        if not self.val_size:
            return None
        y, rows, lengths = self._pack_batch(batch, self.test_size)
        origins = [
            list(range(n - self.val_size, n - self.h + 1, self.step_size))
            for n in lengths
        ]
        normalized, scale = self._forecast_batch(y, rows, origins)
        targets = [
            torch.stack([y[i, o : o + self.h] for o in requested])
            for i, requested in enumerate(origins)
        ]
        predictions = normalized * scale
        loss = self.valid_loss(
            y=torch.cat(targets).unsqueeze(-1),
            y_hat=predictions.flatten(0, 1).unsqueeze(-1),
        )
        self.log(
            "valid_loss",
            loss.detach(),
            batch_size=len(lengths),
            on_epoch=True,
            prog_bar=True,
        )
        window_count = sum(len(requested) for requested in origins)
        self.validation_step_outputs.append(
            torch.stack((loss.detach() * window_count, loss.new_tensor(window_count)))
        )
        return loss

    def predict_step(self, batch, batch_idx):
        # Exclude the final forecast horizon, whose targets can be NaN placeholders.
        y, rows, lengths = self._pack_batch(batch, self.h)
        origins = [
            list(range(n + self.h - self.test_size, n + 1, self.predict_step_size))
            for n in lengths
        ]
        normalized, scale = self._forecast_batch(y, rows, origins)
        return (normalized * scale).flatten(0, 1).unsqueeze(-1)
