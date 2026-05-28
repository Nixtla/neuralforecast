__all__ = ['BaseAuto', 'RayOptions', 'OptunaOptions']


import warnings
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from os import cpu_count
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.basic_variant import BasicVariantGenerator


@dataclass
class RayOptions:
    """Container for Ray-only options forwarded to `tune.Tuner` / `tune.TuneConfig`.

    Attributes:
        run_config (ray.air.RunConfig, optional): Forwarded to `tune.Tuner`.
            When provided, it is used as-is, so `callbacks` and `verbose` must be
            set on it directly.
            See https://docs.ray.io/en/latest/tune/api/doc/ray.tune.RunConfig.html.
        scheduler (ray.tune.schedulers.TrialScheduler, optional): Trial scheduler
            forwarded to `tune.TuneConfig`. Use this to enable schedulers other
            than the default FIFO (e.g. ASHA, HyperBand, BOHB).
            See https://docs.ray.io/en/latest/tune/api/schedulers.html.
        cpus (int, optional): Number of cpus to use during optimization.
            Defaults to `os.cpu_count()` when unset.
        gpus (int, optional): Number of gpus to use during optimization.
            Defaults to `torch.cuda.device_count()` when unset.
    """

    run_config: Optional[Any] = None
    scheduler: Optional[Any] = None
    cpus: Optional[int] = None
    gpus: Optional[int] = None


@dataclass
class OptunaOptions:
    """Container for Optuna-only options forwarded to `optuna.create_study` / `study.optimize`.

    Attributes:
        study_kwargs (dict, optional): Additional keyword arguments forwarded to
            `optuna.Study.optimize`. Keys that overlap with arguments already
            passed by `BaseAuto` (`n_trials`, `show_progress_bar`, `callbacks`,
            `timeout`) take precedence over the defaults.
            See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize.
        create_study_kwargs (dict, optional): Additional keyword arguments
            forwarded to `optuna.create_study`. Keys that overlap with arguments
            already passed by `BaseAuto` (`sampler`, `direction`) take precedence
            over the defaults.
            See https://optuna.readthedocs.io/en/stable/reference/generated/optuna.create_study.html.
    """

    study_kwargs: Optional[dict] = None
    create_study_kwargs: Optional[dict] = None


def _warn_unused_options(backend, ray_options, optuna_options):
    """Warn when an options object is supplied for the wrong backend."""
    if backend == "ray":
        unused = optuna_options
        unused_name = "optuna_options"
    else:
        unused = ray_options
        unused_name = "ray_options"
    set_fields = [
        f.name for f in fields(unused) if getattr(unused, f.name) is not None
    ]
    if set_fields:
        warnings.warn(
            f"{set_fields} on `{unused_name}` are ignored when "
            f"`backend={backend!r}`.",
        )


class MockTrial:
    def suggest_int(*args, **kwargs):
        return "int"

    def suggest_categorical(self, name, choices):
        return choices

    def suggest_uniform(*args, **kwargs):
        return "uniform"

    def suggest_loguniform(*args, **kwargs):
        return "loguniform"

    def suggest_float(*args, **kwargs):
        if "log" in kwargs:
            return "quantized_log"
        elif "step" in kwargs:
            return "quantized_loguniform"
        return "float"


class BaseAuto(pl.LightningModule):
    """
    Class for Automatic Hyperparameter Optimization, it builds on top of `ray` to
    give access to a wide variety of hyperparameter optimization tools ranging
    from classic grid search, to Bayesian optimization and HyperBand algorithm.

    The validation loss to be optimized is defined by the `config['loss']` dictionary
    value, the config also contains the rest of the hyperparameter search space.

    It is important to note that the success of this hyperparameter optimization
    heavily relies on a strong correlation between the validation and test periods.

    Args:
        cls_model (PyTorch/PyTorchLightning model): See `neuralforecast.models` [collection here](./models.html).
        h (int): Forecast horizon
        loss (PyTorch module): Instantiated train loss class from [losses collection](./losses.pytorch.html).
        valid_loss (PyTorch module): Instantiated valid loss class from [losses collection](./losses.pytorch.html).
        config (dict or callable): Dictionary with ray.tune defined search space or function that takes an optuna trial and returns a configuration dict.
        search_alg (ray.tune.search variant or optuna.sampler): For ray see https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
            For optuna see https://optuna.readthedocs.io/en/stable/reference/samplers/index.html.
        num_samples (int): Number of hyperparameter optimization steps/samples.
        time_budget (int, optional): Time budget in seconds for the hyperparameter search.
        refit_with_val (bool): Refit of best model should preserve val_size.
        verbose (bool): Track progress.
        alias (str): Custom name of the model.
        backend (str): Backend to use for searching the hyperparameter space, can be either 'ray' or 'optuna'.
        callbacks (list of callable): List of functions to call during the optimization process.
            ray reference: https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html
            optuna reference: https://optuna.readthedocs.io/en/stable
        ray_options (RayOptions, optional): Container for Ray-only options. See
            `RayOptions` for the supported fields (`run_config`, `scheduler`,
            `cpus`, `gpus`). Only used with `backend='ray'`.
        optuna_options (OptunaOptions, optional): Container for Optuna-only options.
            See `OptunaOptions` for the supported fields (`study_kwargs`,
            `create_study_kwargs`). Only used with `backend='optuna'`.
        cpus: Deprecated, will be removed in v3.2.0. Pass
            `ray_options=RayOptions(cpus=...)` instead.
        gpus: Deprecated, will be removed in v3.2.0. Pass
            `ray_options=RayOptions(gpus=...)` instead.
    """

    def __init__(
        self,
        cls_model,
        h,
        loss,
        valid_loss,
        config,
        search_alg=BasicVariantGenerator(random_state=1),
        num_samples=10,
        time_budget=None,
        refit_with_val=False,
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
        ray_options=None,
        optuna_options=None,
        cpus=None,
        gpus=None,
    ):
        super(BaseAuto, self).__init__()
        with warnings.catch_warnings(record=False):
            warnings.filterwarnings("ignore")
            # the following line issues a warning about the loss attribute being saved
            # but we do want to save it
            # Ignore deprecated kwargs so they aren't persisted into checkpoints
            # and break loads after they're removed in v3.2.0.
            self.save_hyperparameters(ignore=["cpus", "gpus", "ray_options", "optuna_options"])

        if backend == "ray":
            if not isinstance(config, dict):
                raise ValueError(
                    "You have to provide a dict as `config` when using `backend='ray'`"
                )
            config_base = deepcopy(config)
        elif backend == "optuna":
            if not callable(config):
                raise ValueError(
                    "You have to provide a function that takes a trial and returns a dict as `config` when using `backend='optuna'`"
                )
            # extract constant values from the config fn for validations
            config_base = config(MockTrial())
        else:
            raise ValueError(
                f"Unknown backend {backend}. The supported backends are 'ray' and 'optuna'."
            )

        # Shallow-copy user-supplied options so subsequent mutations
        # (legacy coalescing, default resolution) don't leak back to the caller.
        ray_options = replace(ray_options) if ray_options is not None else RayOptions()
        optuna_options = (
            replace(optuna_options) if optuna_options is not None else OptunaOptions()
        )
        for _name, _val in (("cpus", cpus), ("gpus", gpus)):
            if _val is None:
                continue
            if backend != "ray":
                # On non-ray backends cpus/gpus aren't used, so skip the
                # deprecation (which would only point at RayOptions, also unused).
                warnings.warn(
                    f"`{_name}` is ignored when `backend={backend!r}`; "
                    f"it only applies to `backend='ray'`.",
                )
                continue
            if getattr(ray_options, _name) is not None:
                raise TypeError(
                    f"`{_name}` and `ray_options.{_name}` were both provided; "
                    f"pass only `ray_options=RayOptions({_name}=...)` — "
                    f"`{_name}` is deprecated."
                )
            warnings.warn(
                f"`{_name}` is deprecated and will be removed in v3.2.0; "
                f"pass `ray_options=RayOptions({_name}=...)` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            setattr(ray_options, _name, _val)
        _warn_unused_options(backend, ray_options, optuna_options)
        # Resolve None defaults for ray; optuna doesn't use cpus/gpus.
        if backend == "ray":
            if ray_options.cpus is None:
                ray_options.cpus = cpu_count()
            if ray_options.gpus is None:
                ray_options.gpus = torch.cuda.device_count()
        if config_base.get("h", None) is not None:
            raise Exception("Please use `h` init argument instead of `config['h']`.")
        if config_base.get("loss", None) is not None:
            raise Exception(
                "Please use `loss` init argument instead of `config['loss']`."
            )
        if config_base.get("valid_loss", None) is not None:
            raise Exception(
                "Please use `valid_loss` init argument instead of `config['valid_loss']`."
            )
        # This attribute helps to protect
        # model and datasets interactions protections
        if "early_stop_patience_steps" in config_base.keys():
            self.early_stop_patience_steps = 1
        else:
            self.early_stop_patience_steps = -1

        if callable(config):
            # reset config_base here to save params to override in the config fn
            config_base = {}

        # Add losses to config and protect valid_loss default
        config_base["h"] = h
        config_base["loss"] = loss
        if valid_loss is None:
            valid_loss = loss
        config_base["valid_loss"] = valid_loss

        if isinstance(config, dict):
            self.config = config_base
        else:

            def config_f(trial):
                return {**config(trial), **config_base}

            self.config = config_f

        self.h = h
        self.cls_model = cls_model
        self.loss = loss
        self.valid_loss = valid_loss

        self.num_samples = num_samples
        self.time_budget = time_budget
        self.search_alg = search_alg
        self.cpus = ray_options.cpus
        self.gpus = ray_options.gpus
        self.refit_with_val = refit_with_val or self.early_stop_patience_steps > 0
        self.verbose = verbose
        self.alias = alias
        self.backend = backend
        self.callbacks = callbacks
        self.ray_options = ray_options
        self.optuna_options = optuna_options

        # Base Class attributes
        self.EXOGENOUS_FUTR = cls_model.EXOGENOUS_FUTR
        self.EXOGENOUS_HIST = cls_model.EXOGENOUS_HIST
        self.EXOGENOUS_STAT = cls_model.EXOGENOUS_STAT
        self.MULTIVARIATE = cls_model.MULTIVARIATE
        self.RECURRENT = cls_model.RECURRENT

    def __repr__(self):
        return type(self).__name__ if self.alias is None else self.alias

    def _train_tune(self, config_step, cls_model, dataset, val_size, test_size):
        """BaseAuto._train_tune

        Internal function that instantiates a NF class model, then automatically
        explores the validation loss (ptl/val_loss) on which the hyperparameter
        exploration is based.

        Args:
            config_step (dict): Dict, initialization parameters of a NF model.
            cls_model (NeuralForecast model class): NeuralForecast model class, yet to be instantiated.
            dataset (NeuralForecast dataset): NeuralForecast dataset, to fit the model.
            val_size (int): Validation size for temporal cross-validation.
            test_size (int): Test size for temporal cross-validation.
        """
        metrics = {"loss": "ptl/val_loss", "train_loss": "train_loss"}
        callbacks = [TuneReportCallback(metrics, on="validation_end")]
        if "callbacks" in config_step.keys():
            callbacks.extend(config_step["callbacks"])
        config_step = {**config_step, **{"callbacks": callbacks}}

        # Protect dtypes from tune samplers
        if "batch_size" in config_step.keys():
            config_step["batch_size"] = int(config_step["batch_size"])
        if "windows_batch_size" in config_step.keys():
            config_step["windows_batch_size"] = int(config_step["windows_batch_size"])

        # Tune session receives validation signal
        # from the specialized PL TuneReportCallback
        _ = self._fit_model(
            cls_model=cls_model,
            config=config_step,
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
        )

    def _tune_model(
        self,
        cls_model,
        dataset,
        val_size,
        test_size,
        cpus,
        gpus,
        verbose,
        num_samples,
        search_alg,
        config,
        time_budget,
    ):
        train_fn_with_parameters = tune.with_parameters(
            self._train_tune,
            cls_model=cls_model,
            dataset=dataset,
            val_size=val_size,
            test_size=test_size,
        )

        # Device
        if gpus > 0:
            device_dict = {"gpu": gpus}
        else:
            device_dict = {"cpu": cpus}

        # on Windows, prevent long trial directory names
        import platform

        trial_dirname_creator = (
            (lambda trial: f"{trial.trainable_name}_{trial.trial_id}")
            if platform.system() == "Windows"
            else None
        )

        if self.ray_options.run_config is not None:
            if self.callbacks is not None:
                warnings.warn(
                    "`callbacks` is ignored when `ray_options.run_config` is provided; "
                    "set callbacks on the RunConfig instead.",
                )
            run_config = self.ray_options.run_config
        else:
            run_config = air.RunConfig(callbacks=self.callbacks, verbose=verbose)

        tuner = tune.Tuner(
            tune.with_resources(train_fn_with_parameters, device_dict),
            run_config=run_config,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                num_samples=num_samples,
                search_alg=search_alg,
                scheduler=self.ray_options.scheduler,
                trial_dirname_creator=trial_dirname_creator,
                time_budget_s=time_budget,
            ),
            param_space=config,
        )
        results = tuner.fit()
        return results

    @staticmethod
    def _ray_config_to_optuna(ray_config):
        def optuna_config(trial):
            out = {}
            for k, v in ray_config.items():
                if hasattr(v, "sampler"):
                    sampler = v.sampler
                    if isinstance(
                        sampler, tune.search.sample.Integer.default_sampler_cls
                    ):
                        v = trial.suggest_int(k, v.lower, v.upper)
                    elif isinstance(
                        sampler, tune.search.sample.Categorical.default_sampler_cls
                    ):
                        v = trial.suggest_categorical(k, v.categories)
                    elif isinstance(sampler, tune.search.sample.Uniform):
                        v = trial.suggest_uniform(k, v.lower, v.upper)
                    elif isinstance(sampler, tune.search.sample.LogUniform):
                        v = trial.suggest_loguniform(k, v.lower, v.upper)
                    elif isinstance(sampler, tune.search.sample.Quantized):
                        if isinstance(
                            sampler.get_sampler(), tune.search.sample.Float._LogUniform
                        ):
                            v = trial.suggest_float(k, v.lower, v.upper, log=True)
                        elif isinstance(
                            sampler.get_sampler(), tune.search.sample.Float._Uniform
                        ):
                            v = trial.suggest_float(k, v.lower, v.upper, step=sampler.q)
                    else:
                        raise ValueError(f"Couldn't translate {type(v)} to optuna.")
                out[k] = v
            return out

        return optuna_config

    def _optuna_tune_model(
        self,
        cls_model,
        dataset,
        val_size,
        test_size,
        verbose,
        num_samples,
        search_alg,
        config,
        distributed_config,
        time_budget,
    ):
        import optuna

        def objective(trial):
            user_cfg = config(trial)
            cfg = deepcopy(user_cfg)
            model = self._fit_model(
                cls_model=cls_model,
                config=cfg,
                dataset=dataset,
                val_size=val_size,
                test_size=test_size,
                distributed_config=distributed_config,
            )
            # `loss` and `valid_loss` are PyTorch modules and not JSON-serializable;
            # exclude them so the study can be persisted to backends like SQLite.
            # They are re-attached from `self.loss` / `self.valid_loss` in `fit`.
            persistable_cfg = {
                k: v for k, v in user_cfg.items() if k not in ("loss", "valid_loss")
            }
            trial.set_user_attr("ALL_PARAMS", persistable_cfg)
            metrics = model.metrics
            trial.set_user_attr(
                "METRICS",
                {
                    "loss": float(metrics["ptl/val_loss"]),
                    "train_loss": float(metrics["train_loss"]),
                },
            )
            return trial.user_attrs["METRICS"]["loss"]

        if isinstance(search_alg, optuna.samplers.BaseSampler):
            sampler = search_alg
        else:
            sampler = None

        create_kwargs = {"sampler": sampler, "direction": "minimize"}
        if self.optuna_options.create_study_kwargs is not None:
            overridden = sorted(
                set(self.optuna_options.create_study_kwargs) & set(create_kwargs)
            )
            if overridden:
                warnings.warn(
                    f"`optuna_options.create_study_kwargs` overrides default values for "
                    f"{overridden}; user-supplied values take precedence.",
                )
            create_kwargs.update(self.optuna_options.create_study_kwargs)
        study = optuna.create_study(**create_kwargs)
        optimize_kwargs = {
            "n_trials": num_samples,
            "show_progress_bar": verbose,
            "callbacks": self.callbacks,
            "timeout": time_budget,
        }
        if self.optuna_options.study_kwargs is not None:
            overridden = sorted(
                set(self.optuna_options.study_kwargs) & set(optimize_kwargs)
            )
            if overridden:
                warnings.warn(
                    f"`optuna_options.study_kwargs` overrides default values for "
                    f"{overridden}; user-supplied values take precedence.",
                )
            optimize_kwargs.update(self.optuna_options.study_kwargs)
        study.optimize(objective, **optimize_kwargs)
        return study

    def _fit_model(
        self, cls_model, config, dataset, val_size, test_size, distributed_config=None
    ):
        model = cls_model(**config)
        model = model.fit(
            dataset,
            val_size=val_size,
            test_size=test_size,
            distributed_config=distributed_config,
        )
        return model

    def fit(
        self,
        dataset,
        val_size=0,
        test_size=0,
        random_seed=None,
        distributed_config=None,
    ):
        """BaseAuto.fit

        Perform the hyperparameter optimization as specified by the BaseAuto configuration
        dictionary `config`.

        The optimization is performed on the `TimeSeriesDataset` using temporal cross validation with
        the validation set that sequentially precedes the test set.

        Args:
            dataset (NeuralForecast's `TimeSeriesDataset`): NeuralForecast's `TimeSeriesDataset` see details [here](./tsdataset.html)
            val_size (int): Size of temporal validation set (needs to be bigger than 0).
            test_size (int): Size of temporal test set (default 0).
            random_seed (int): Random seed for hyperparameter exploration algorithms, not yet implemented.

        Returns:
            self: Fitted instance of `BaseAuto` with best hyperparameters and results.
        """
        # we need val_size > 0 to perform
        # hyperparameter selection.
        search_alg = deepcopy(self.search_alg)
        val_size = val_size if val_size > 0 else self.h
        # When `_reuse_search` is set (by `NeuralForecast.fit` during conformal
        # interval calibration), refit with the previously found best config
        reuse_search = (
            getattr(self, "_reuse_search", False)
            and getattr(self, "results", None) is not None
        )
        if reuse_search:
            results = self.results
        elif self.backend == "ray":
            if distributed_config is not None:
                raise ValueError(
                    "distributed training is not supported for the ray backend."
                )
            results = self._tune_model(
                cls_model=self.cls_model,
                dataset=dataset,
                val_size=val_size,
                test_size=test_size,
                cpus=self.cpus,
                gpus=self.gpus,
                verbose=self.verbose,
                num_samples=self.num_samples,
                search_alg=search_alg,
                config=self.config,
                time_budget=self.time_budget,
            )
        else:
            results = self._optuna_tune_model(
                cls_model=self.cls_model,
                dataset=dataset,
                val_size=val_size,
                test_size=test_size,
                verbose=self.verbose,
                num_samples=self.num_samples,
                search_alg=search_alg,
                config=self.config,
                distributed_config=distributed_config,
                time_budget=self.time_budget,
            )

        if self.backend == "ray":
            best_config = results.get_best_result().config
        else:
            # Deepcopy so the final fit doesn't mutate the loss instances stored on
            # `self` (matches the historic behavior where optuna's `set_user_attr`
            # deepcopied the config dict).
            best_config = {
                **results.best_trial.user_attrs["ALL_PARAMS"],
                "loss": deepcopy(self.loss),
                "valid_loss": deepcopy(self.valid_loss),
            }
        self.model = self._fit_model(
            cls_model=self.cls_model,
            config=best_config,
            dataset=dataset,
            val_size=val_size * self.refit_with_val,
            test_size=test_size,
            distributed_config=distributed_config,
        )
        self.results = results

        # Added attributes for compatibility with NeuralForecast core
        self.futr_exog_list = self.model.futr_exog_list
        self.hist_exog_list = self.model.hist_exog_list
        self.stat_exog_list = self.model.stat_exog_list
        return self

    def predict(self, dataset, step_size=1, h=None, **data_kwargs):
        """BaseAuto.predict

        Predictions of the best performing model on validation.

        Args:
            dataset (NeuralForecast's `TimeSeriesDataset`): NeuralForecast's `TimeSeriesDataset` see details [here](./tsdataset.html)
            step_size (int): Steps between sequential predictions, (default 1).
            h (int): Prediction horizon, if None, uses the model's fitted horizon. Defaults to None.
            **data_kwarg: Additional parameters for the dataset module.

        Returns:
            y_hat: Numpy predictions of the `NeuralForecast` model.
        """
        return self.model.predict(dataset=dataset, step_size=step_size, h=h, **data_kwargs)

    def set_test_size(self, test_size):
        self.model.set_test_size(test_size)

    def get_test_size(self):
        return self.model.test_size

    def save(self, path):
        """BaseAuto.save

        Save the fitted model to disk.

        Args:
            path (str): Path to save the model.
        """
        self.model.save(path)
