# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/common.base_auto.ipynb.

# %% auto 0
__all__ = ['BaseAuto']

# %% ../../nbs/common.base_auto.ipynb 5
from copy import deepcopy
from os import cpu_count

import torch
import pytorch_lightning as pl

from pytorch_lightning.callbacks import TQDMProgressBar
from ray import air, tune
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.search.basic_variant import BasicVariantGenerator

# %% ../../nbs/common.base_auto.ipynb 6
class BaseAuto(pl.LightningModule):
    """
    Class for Automatic Hyperparameter Optimization, it builds on top of `ray` to
    give access to a wide variety of hyperparameter optimization tools ranging
    from classic grid search, to Bayesian optimization and HyperBand algorithm.

    The validation loss to be optimized is defined by the `config['loss']` dictionary
    value, the config also contains the rest of the hyperparameter search space.

    It is important to note that the success of this hyperparameter optimization
    heavily relies on a strong correlation between the validation and test periods.

    **Parameters:**<br>
    `cls_model`: PyTorch/PyTorchLightning model, see `neuralforecast.models` [collection here](https://nixtla.github.io/neuralforecast/models.html).<br>
    `h`: int, forecast horizon.<br>
    `loss`: PyTorch module, instantiated train loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `valid_loss`: PyTorch module=`loss`, instantiated valid loss class from [losses collection](https://nixtla.github.io/neuralforecast/losses.pytorch.html).<br>
    `config`: dict, dictionary with ray.tune defined search space.<br>
    `search_alg`: ray.tune.search variant, BasicVariantGenerator, HyperOptSearch, DragonflySearch, TuneBOHB for details
        see [tune.search](https://docs.ray.io/en/latest/tune/api_docs/suggestion.html#).<br>
    `num_samples`: int, number of hyperparameter optimization steps/samples.<br>
    `cpus`: int, number of cpus to use during optimization, default all available.<br>
    `gpus`: int, number of gpus to use during optimization, default all available.<br>
    `refit_wo_val`: bool, number of gpus to use during optimization, default all available.<br>
    `verbose`: bool, wether print partial outputs.<br>
    `alias`: str, optional,  Custom name of the model.<br>
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
    ):
        super(BaseAuto, self).__init__()
        self.save_hyperparameters()  # Allows instantiation from a checkpoint from class

        if config.get("h", None) is not None:
            raise Exception("Please use `h` init argument instead of `config['h']`.")
        if config.get("loss", None) is not None:
            raise Exception(
                "Please use `loss` init argument instead of `config['loss']`."
            )
        if config.get("valid_loss", None) is not None:
            raise Exception(
                "Please use `valid_loss` init argument instead of `config['valid_loss']`."
            )

        # Deepcopy to avoid modifying the original config
        config_base = deepcopy(config)

        # Add losses to config and protect valid_loss default
        config_base["h"] = h
        config_base["loss"] = loss
        if valid_loss is None:
            valid_loss = loss
        config_base["valid_loss"] = valid_loss

        self.h = h
        self.cls_model = cls_model

        self.config = config_base
        self.loss = self.config["loss"]
        self.valid_loss = self.config["valid_loss"]

        self.num_samples = num_samples
        self.search_alg = search_alg
        self.cpus = cpus
        self.gpus = gpus
        self.refit_with_val = refit_with_val
        self.verbose = verbose
        self.alias = alias

        # Base Class attributes
        self.SAMPLING_TYPE = cls_model.SAMPLING_TYPE

    def __repr__(self):
        return type(self).__name__ if self.alias is None else self.alias

    def _train_tune(self, config_step, cls_model, dataset, val_size, test_size):
        """BaseAuto._train_tune

        Internal function that instantiates a NF class model, then automatically
        explores the validation loss (ptl/val_loss) on which the hyperparameter
        exploration is based.

        **Parameters:**<br>
        `config_step`: Dict, initialization parameters of a NF model.<br>
        `cls_model`: NeuralForecast model class, yet to be instantiated.<br>
        `dataset`: NeuralForecast dataset, to fit the model.<br>
        `val_size`: int, validation size for temporal cross-validation.<br>
        `test_size`: int, test size for temporal cross-validation.<br>
        """
        metrics = {"loss": "ptl/val_loss"}
        callbacks = [
            TQDMProgressBar(),
            TuneReportCallback(metrics, on="validation_end"),
        ]
        if "callbacks" in config_step.keys():
            callbacks += config_step["callbacks"]
        config_step = {**config_step, **{"callbacks": callbacks}}

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

        tuner = tune.Tuner(
            tune.with_resources(train_fn_with_parameters, device_dict),
            run_config=air.RunConfig(
                verbose=verbose,
                # checkpoint_config=air.CheckpointConfig(
                # num_to_keep=0,
                # keep_checkpoints_num=None
                # )
            ),
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                num_samples=num_samples,
                search_alg=search_alg,
            ),
            param_space=config,
        )
        results = tuner.fit()
        return results

    def _fit_model(self, cls_model, config, dataset, val_size, test_size):
        model = cls_model(**config)
        model.fit(dataset, val_size=val_size, test_size=test_size)
        return model

    def fit(self, dataset, val_size=0, test_size=0, random_seed=None):
        """BaseAuto.fit

        Perform the hyperparameter optimization as specified by the BaseAuto configuration
        dictionary `config`.

        The optimization is performed on the `TimeSeriesDataset` using temporal cross validation with
        the validation set that sequentially precedes the test set.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset` see details [here](https://nixtla.github.io/neuralforecast/tsdataset.html)<br>
        `val_size`: int, size of temporal validation set (needs to be bigger than 0).<br>
        `test_size`: int, size of temporal test set (default 0).<br>
        `random_seed`: int=None, random_seed for hyperparameter exploration algorithms, not yet implemented.<br>
        **Returns:**<br>
        `self`: fitted instance of `BaseAuto` with best hyperparameters and results<br>.
        """
        # we need val_size > 0 to perform
        # hyperparameter selection.
        search_alg = deepcopy(self.search_alg)
        val_size = val_size if val_size > 0 else self.h
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
        # self.model = self.cls_model(**best_config)
        # self.model.fit(
        #    dataset=dataset,
        #    val_size=val_size * (1 - self.refit_with_val),
        #    test_size=test_size,
        # )
        self.model = self._fit_model(
            cls_model=self.cls_model,
            config=best_config,
            dataset=dataset,
            val_size=val_size * (1 - self.refit_with_val),
            test_size=test_size,
        )
        self.results = results

    def predict(self, dataset, step_size=1, **data_kwargs):
        """BaseAuto.predict

        Predictions of the best performing model on validation.

        **Parameters:**<br>
        `dataset`: NeuralForecast's `TimeSeriesDataset` see details [here](https://nixtla.github.io/neuralforecast/tsdataset.html)<br>
        `step_size`: int, steps between sequential predictions, (default 1).<br>
        `**data_kwarg`: additional parameters for the dataset module.<br>
        `random_seed`: int=None, random_seed for hyperparameter exploration algorithms (not implemented).<br>
        **Returns:**<br>
        `y_hat`: numpy predictions of the `NeuralForecast` model.<br>
        """
        return self.model.predict(dataset=dataset, step_size=step_size, **data_kwargs)

    def set_test_size(self, test_size):
        self.model.set_test_size(test_size)

    def get_test_size(self):
        return self.model.test_size

    def save(self, path):
        """BaseAuto.save

        Save the fitted model to disk.

        **Parameters:**<br>
        `path`: str, path to save the model.<br>
        """
        self.model.trainer.save_checkpoint(path)
