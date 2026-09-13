"""SearchCast's official Ridge/scaling/augmentation primitives in NeuralForecast.

The adapter owns only window construction, time-ordered validation and NF state.
It uses one pooled series group and searches separate horizon groups. It never
runs the upstream dataset-specific CLI or its fixed 96/192/336/720 test protocol.
"""

import math
import sys

import numpy as np
import torch

from ..losses.pytorch import MSE
from ._exogenous import ExogenousModel
from ._forecast_source import forecast_source
from .research import _full_windows, _positive

__all__ = ["SearchCast"]


class SearchCast(ExogenousModel):
    """Closed-form Ridge plus Optuna search; fit is not neural-network training.

    Args:
        h: Any positive horizon, including 1..72.
        input_size: Maximum lookback searched; inference needs this much history.
        source_dir: Pinned SakanaAI/SearchCast checkout.
        n_trials: Optuna trials per horizon group.
        n_folds: Expanding chronological folds INSIDE the NF training partition.
        cv_val_size: Points per inner fold; defaults to max(h, shortest_train/5).
        horizon_group_size: Outputs sharing hyperparameters; final group may be shorter.
        ridge_alphas: Positive candidate Ridge penalties, tested by the official solver.
        **kwargs: NF point-forecast options. max_steps=0 and scaler_type='identity'.

    All unique_ids pool training windows (the upstream pool_series mode); they
    are not treated as jointly observed feature channels. Exogenous variables
    are rejected: the reviewed SearchCast method has no separate covariate API.
    NF validation/test rows never enter search or final refitting. After search,
    refit includes inner-validation rows, but excludes the outer NF holdouts.
    Complete finite histories, an in-memory dataset, and CPU Ridge are required.
    """

    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False

    def __init__(self, h, input_size, source_dir=None, n_trials=20, n_folds=1,
                 cv_val_size=None, horizon_group_size=24, ridge_alphas=None,
                 max_steps=0, loss=None, **kwargs):
        _positive(h=h, input_size=input_size, n_trials=n_trials, n_folds=n_folds,
                  horizon_group_size=horizon_group_size)
        if cv_val_size is not None:
            _positive(cv_val_size=cv_val_size)
            if cv_val_size < h:
                raise ValueError("cv_val_size must be at least h.")
        if max_steps != 0 or kwargs.get("early_stop_patience_steps", -1) > 0:
            raise ValueError("SearchCast uses closed-form fitting; max_steps=0, without early stopping.")
        if kwargs.get("scaler_type", "identity") != "identity":
            raise ValueError("SearchCast owns normalization; use scaler_type='identity'.")
        loss = MSE() if loss is None else loss
        if type(loss) is not MSE or getattr(loss, "horizon_weight", None) is not None:
            raise ValueError("SearchCast fits Ridge/MSE; use unweighted MSE() and valid_loss for evaluation.")
        alphas = np.logspace(-6, 4, 21) if ridge_alphas is None else np.asarray(ridge_alphas, dtype=float)
        if alphas.ndim != 1 or not len(alphas) or not np.isfinite(alphas).all() or (alphas <= 0).any():
            raise ValueError("ridge_alphas must be a nonempty sequence of positive finite values.")
        super().__init__(h=h, input_size=input_size, max_steps=0, loss=loss, **_full_windows(kwargs))
        self.source_dir = source_dir
        self.n_trials, self.n_folds = n_trials, n_folds
        self.cv_val_size, self.horizon_group_size = cv_val_size, horizon_group_size
        self.ridge_alphas = alphas.tolist()
        self._source_name = None
        groups = math.ceil(h / horizon_group_size)
        # Fixed-size buffers preserve learned Ridge state through NF.save/load.
        self.register_buffer("ridge_weights", torch.zeros(input_size + 1, h, dtype=torch.float64))
        self.register_buffer("ridge_config", torch.zeros(groups, 6, dtype=torch.long))
        self.register_buffer("ridge_stats", torch.zeros(groups, 4, dtype=torch.float64))
        self.register_buffer("ridge_fitted", torch.tensor(False))

    def _source_module(self):
        module = sys.modules.get(self._source_name)
        if module is None:
            module = forecast_source(self.source_dir, "SearchCast")
            self._source_name = module.__name__
        return module

    def _training_series(self, dataset, val_size, test_size):
        if not all(hasattr(dataset, key) for key in ("temporal", "temporal_cols", "indptr", "y_idx")):
            raise ValueError("SearchCast requires an in-memory TimeSeriesDataset.")
        if "sample_weight" in dataset.temporal_cols:
            raise ValueError("SearchCast does not support sample_weight.")
        mask_idx = dataset.temporal_cols.get_loc("available_mask")
        series = []
        for start, end in zip(dataset.indptr[:-1], dataset.indptr[1:]):
            stop = int(end) - val_size - test_size
            if stop <= start:
                raise ValueError("SearchCast has no training observations before the NF holdouts.")
            rows = dataset.temporal[int(start):stop]
            y = rows[:, dataset.y_idx].detach().cpu().double()
            if not torch.isfinite(y).all() or not (rows[:, mask_idx] == 1).all():
                raise ValueError("SearchCast requires complete finite training series.")
            series.append(y)
        if not series:
            raise ValueError("SearchCast requires at least one training series.")
        return series

    def _windows(self, series, lookback, cuts=None, block=None):
        xs, ys = [], []
        for i, values in enumerate(series):
            if cuts is None:
                region = values
            elif block is None:
                region = values[:cuts[i]]
            else:
                region = values[cuts[i] - lookback:cuts[i] + block]
            if len(region) < lookback + self.h:
                raise ValueError("Not enough history for the requested lookback, h and validation folds.")
            windows = region.unfold(0, lookback + self.h, self.step_size)
            xs.append(windows[:, :lookback])
            ys.append(windows[:, lookback:])
        return torch.cat(xs), torch.cat(ys)

    def _scaler(self, params, training):
        source = self._source_module()
        strategy = source.StandardStrategy() if params["scaler_method"] == "mean" else source.RobustStrategy()
        if params["scaler_scope"] == "local":
            return source.LocalNormScaler(strategy, params["lookback"], params["last_k"])
        scaler = source.GlobalScaler(strategy)
        scaler.fit(torch.cat(training).reshape(1, -1))
        return scaler

    def _fit(self, dataset, batch_size, valid_batch_size=1024, val_size=0,
             test_size=0, random_seed=None, shuffle_train=True, distributed_config=None):
        if distributed_config is not None:
            raise ValueError("Distributed SearchCast fitting is unsupported.")
        if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in (val_size, test_size)):
            raise ValueError("val_size and test_size must be nonnegative integers.")
        self._check_exog(dataset)
        self._restart_seed(random_seed)
        series = self._training_series(dataset, val_size, test_size)
        block = self.cv_val_size or max(self.h, min(map(len, series)) // 5)
        if min(map(len, series)) - self.n_folds * block < self.input_size + self.h:
            raise ValueError("Training partition too short for input_size+h and inner folds; reduce cv_val_size/n_folds.")
        source = self._source_module()
        try:
            import optuna
        except ImportError as exc:
            raise ImportError("SearchCast requires optuna.") from exc
        solver = source.RidgeSolver(torch.device("cpu"))
        alphas = torch.tensor(self.ridge_alphas, dtype=torch.float64)
        learned = []
        for begin in range(0, self.h, self.horizon_group_size):
            end = min(begin + self.horizon_group_size, self.h)

            def objective(trial):
                params = dict(
                    lookback=trial.suggest_int("lookback", min(32, self.input_size), self.input_size, log=True),
                    scaler_scope=trial.suggest_categorical("scaler_scope", ["local", "global"]),
                    scaler_method=trial.suggest_categorical("scaler_method", ["mean", "robust"]),
                    local_ratio=trial.suggest_float("local_ratio", 0.001, 1.0, log=True),
                    noise_type=trial.suggest_categorical("noise_type", ["none", "time", "freq"]),
                    aug_sigma=trial.suggest_float("aug_sigma", 0.001, 0.5, log=True),
                )
                params["last_k"] = max(1, int(params["lookback"] * params["local_ratio"]))
                scores = []
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(self.random_seed + begin + trial.number)
                    for fold in range(self.n_folds):
                        cuts = [len(s) - (self.n_folds - fold) * block for s in series]
                        x_train, y_train = self._windows(series, params["lookback"], cuts)
                        x_val, y_val = self._windows(series, params["lookback"], cuts, block)
                        scaler = self._scaler(params, [s[:cut] for s, cut in zip(series, cuts)])
                        weights = solver.solve(
                            x_train, y_train[:, begin:end], alphas, scaler=scaler,
                            aug_config={"noise_type": params["noise_type"], "sigma": params["aug_sigma"]},
                        )
                        preds = solver.predict(x_val, weights, scaler=scaler)
                        if params["scaler_scope"] == "global":
                            preds = scaler.inv_transform(preds)
                        scores.append((preds - y_val[None, :, begin:end]).square().mean((1, 2)))
                mse = torch.stack(scores).mean(0)
                if not torch.isfinite(mse).all():
                    raise ValueError("SearchCast returned non-finite validation scores.")
                best = int(mse.argmin())
                trial.set_user_attr("ridge_alpha", float(alphas[best]))
                return float(mse[best])

            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=self.random_seed + begin))
            study.optimize(objective, n_trials=self.n_trials, n_jobs=1, show_progress_bar=False)
            params = dict(study.best_params)
            params["last_k"] = max(1, int(params["lookback"] * params["local_ratio"]))
            alpha = study.best_trial.user_attrs["ridge_alpha"]
            x_train, y_train = self._windows(series, params["lookback"])
            scaler = self._scaler(params, series)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(self.random_seed + begin)
                weights = solver.solve(
                    x_train, y_train[:, begin:end], torch.tensor([alpha], dtype=torch.float64),
                    scaler=scaler,
                    aug_config={"noise_type": params["noise_type"], "sigma": params["aug_sigma"]},
                )[0]
            if not torch.isfinite(weights).all():
                raise ValueError("SearchCast returned non-finite fitted weights.")
            center = 0.0 if params["scaler_scope"] == "local" else float(scaler.center.item())
            scale = 1.0 if params["scaler_scope"] == "local" else float(scaler.scale.item())
            learned.append((begin, end, params, alpha, weights, center, scale))

        # Commit learned buffers only after every group succeeds.
        self.ridge_weights.zero_()
        for group, (begin, end, params, alpha, weights, center, scale) in enumerate(learned):
            self.ridge_weights[:weights.shape[0], begin:end].copy_(weights)
            self.ridge_config[group] = torch.tensor([
                params["lookback"], params["last_k"], int(params["scaler_scope"] == "global"),
                int(params["scaler_method"] == "robust"),
                ["none", "time", "freq"].index(params["noise_type"]), end - begin,
            ], device=self.ridge_config.device)
            self.ridge_stats[group] = torch.tensor(
                [center, scale, alpha, params["aug_sigma"]], dtype=torch.float64, device=self.ridge_stats.device,
            )
        self.ridge_fitted.fill_(True)
        self.val_size, self.test_size = val_size, test_size
        self.metrics = {}
        return self

    @torch.no_grad()
    def forward(self, windows_batch):
        if not bool(self.ridge_fitted):
            raise RuntimeError("SearchCast must be fitted before prediction.")
        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        source = self._source_module()
        solver = source.RidgeSolver(torch.device("cpu"))
        outputs = []
        begin = 0
        for group, config in enumerate(self.ridge_config.cpu().tolist()):
            lookback, last_k, global_scope, robust, _, width = config
            strategy = source.RobustStrategy() if robust else source.StandardStrategy()
            if global_scope:
                scaler = source.GlobalScaler(strategy)
                scaler.center = self.ridge_stats[group, 0].detach().cpu().reshape(1, 1)
                scaler.scale = self.ridge_stats[group, 1].detach().cpu().reshape(1, 1)
            else:
                scaler = source.LocalNormScaler(strategy, lookback, last_k)
            weights = self.ridge_weights[:lookback + 1, begin:begin + width].detach().cpu()[None]
            pred = solver.predict(y[:, -lookback:, 0].detach().cpu().double(), weights, scaler=scaler)[0]
            if global_scope:
                pred = scaler.inv_transform(pred)
            outputs.append(pred)
            begin += width
        return self._point_output(torch.cat(outputs, dim=-1), y)

    def predict(self, *args, **kwargs):
        if kwargs.get("explainer_config") is not None:
            raise ValueError("Gradient explanations are unavailable for closed-form SearchCast.")
        if kwargs.get("quantiles") is not None or "quantile" in kwargs:
            raise ValueError("SearchCast exposes point forecasts only.")
        return super().predict(*args, **kwargs)
