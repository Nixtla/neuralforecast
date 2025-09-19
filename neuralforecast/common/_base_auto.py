__all__ = ['BaseAuto']


import warnings
from copy import deepcopy
from os import cpu_count

import pytorch_lightning as pl
import torch
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.basic_variant import BasicVariantGenerator


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
        cls_model (PyTorch/PyTorchLightning model): See `neuralforecast.models` [collection here](./models).
        h (int): Forecast horizon
        loss (PyTorch module): Instantiated train loss class from [losses collection](./losses.pytorch).
        valid_loss (PyTorch module): Instantiated valid loss class from [losses collection](./losses.pytorch).
        config (dict or callable): Dictionary with ray.tune defined search space or function that takes an optuna trial and returns a configuration dict.
        search_alg (ray.tune.search variant or optuna.sampler): For ray see https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
        For optuna see https://optuna.readthedocs.io/en/stable/reference/samplers/index.html.
        num_samples (int): Number of hyperparameter optimization steps/samples.
        Number of hyperparameter optimization steps/samples.
        cpus (int): Number of cpus to use during optimization. Only used with ray tune.
        gpus (int): Number of gpus to use during optimization, default all available. Only used with ray tune.
        refit_with_val (bool): Refit of best model should preserve val_size.
        verbose (bool): Track progress.
        alias (str): Custom name of the model.
        Custom name of the model.
        backend (str): Backend to use for searching the hyperparameter space, can be either 'ray' or 'optuna'.
        callbacks (list of callable): List of functions to call during the optimization process.
        List of functions to call during the optimization process.
        ray reference: https://docs.ray.io/en/latest/tune/tutorials/tune-metrics.html
        optuna reference: https://optuna.readthedocs.io/en/stable
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
        cpus=cpu_count(),
        gpus=torch.cuda.device_count(),
        refit_with_val=False,
        verbose=False,
        alias=None,
        backend="ray",
        callbacks=None,
    ):
        super(BaseAuto, self).__init__()
        with warnings.catch_warnings(record=False):
            warnings.filterwarnings("ignore")
            # the following line issues a warning about the loss attribute being saved
            # but we do want to save it
            self.save_hyperparameters()  # Allows instantiation from a checkpoint from class

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
        self.search_alg = search_alg
        self.cpus = cpus
        self.gpus = gpus
        self.refit_with_val = refit_with_val or self.early_stop_patience_steps > 0
        self.verbose = verbose
        self.alias = alias
        self.backend = backend
        self.callbacks = callbacks

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

        tuner = tune.Tuner(
            tune.with_resources(train_fn_with_parameters, device_dict),
            run_config=air.RunConfig(callbacks=self.callbacks, verbose=verbose),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                num_samples=num_samples,
                search_alg=search_alg,
                trial_dirname_creator=trial_dirname_creator,
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
            trial.set_user_attr("ALL_PARAMS", user_cfg)
            metrics = model.metrics
            trial.set_user_attr(
                "METRICS",
                {
                    "loss": metrics["ptl/val_loss"],
                    "train_loss": metrics["train_loss"],
                },
            )
            return trial.user_attrs["METRICS"]["loss"]

        if isinstance(search_alg, optuna.samplers.BaseSampler):
            sampler = search_alg
        else:
            sampler = None

        study = optuna.create_study(sampler=sampler, direction="minimize")
        study.optimize(
            objective,
            n_trials=num_samples,
            show_progress_bar=verbose,
            callbacks=self.callbacks,
        )
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
            dataset (NeuralForecast's `TimeSeriesDataset`): NeuralForecast's `TimeSeriesDataset` see details [here](./tsdataset)
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
        if self.backend == "ray":
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
            )
            best_config = results.get_best_result().config
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
            )
            best_config = results.best_trial.user_attrs["ALL_PARAMS"]
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
            dataset (NeuralForecast's `TimeSeriesDataset`): NeuralForecast's `TimeSeriesDataset` see details [here](./tsdataset)
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
