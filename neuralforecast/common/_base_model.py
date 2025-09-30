__all__ = ["DistributedConfig", "BaseModel"]


import inspect
import math
import random
import warnings
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Union

import fsspec
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import neuralforecast.losses.pytorch as losses
from neuralforecast.common.enums import ExplainerEnum
from neuralforecast.tsdataset import (
    BaseTimeSeriesDataset,
    TimeSeriesDataModule,
    _DistributedTimeSeriesDataModule,
)

from ..losses.pytorch import BasePointLoss, DistributionLoss
from ..utils import get_indexer_raise_missing

DISTRIBUTION_LOSSES = (
    losses.DistributionLoss,
    losses.PMM,
    losses.GMM,
    losses.NBMM,
)
MULTIQUANTILE_LOSSES = (
    losses.MQLoss,
    losses.HuberMQLoss,
)
from ._scalers import TemporalNorm


@dataclass
class DistributedConfig:
    partitions_path: str
    num_nodes: int
    devices: int


@contextmanager
def _disable_torch_init():
    """Context manager used to disable pytorch's weight initialization.

    This is especially useful when loading saved models, since when initializing
    a model the weights are also initialized following some method
    (e.g. kaiming uniform), and that time is wasted since we'll override them with
    the saved weights."""

    def noop(*args, **kwargs):
        return

    kaiming_uniform = nn.init.kaiming_uniform_
    kaiming_normal = nn.init.kaiming_normal_
    xavier_uniform = nn.init.xavier_uniform_
    xavier_normal = nn.init.xavier_normal_

    nn.init.kaiming_uniform_ = noop
    nn.init.kaiming_normal_ = noop
    nn.init.xavier_uniform_ = noop
    nn.init.xavier_normal_ = noop
    try:
        yield
    finally:
        nn.init.kaiming_uniform_ = kaiming_uniform
        nn.init.kaiming_normal_ = kaiming_normal
        nn.init.xavier_uniform_ = xavier_uniform
        nn.init.xavier_normal_ = xavier_normal


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """Convert a tensor to numpy"""
    if tensor.dtype == torch.bfloat16:
        return tensor.float().numpy()

    return tensor.numpy()


class BaseModel(pl.LightningModule):
    EXOGENOUS_FUTR = True  # If the model can handle future exogenous variables
    EXOGENOUS_HIST = True  # If the model can handle historical exogenous variables
    EXOGENOUS_STAT = True  # If the model can handle static exogenous variables
    MULTIVARIATE = False  # If the model produces multivariate forecasts (True) or univariate (False)
    RECURRENT = (
        False  # If the model produces forecasts recursively (True) or direct (False)
    )

    def __init__(
        self,
        h: int,
        input_size: int,
        loss: Union[BasePointLoss, DistributionLoss, nn.Module],
        valid_loss: Union[BasePointLoss, DistributionLoss, nn.Module],
        learning_rate: float,
        max_steps: int,
        val_check_steps: int,
        batch_size: int,
        valid_batch_size: Union[int, None],
        windows_batch_size: int,
        inference_windows_batch_size: Union[int, None],
        start_padding_enabled: bool,
        training_data_availability_threshold: Union[float, List[float]] = 0.0,
        n_series: Union[int, None] = None,
        n_samples: Union[int, None] = 100,
        h_train: int = 1,
        inference_input_size: Union[int, None] = None,
        step_size: int = 1,
        num_lr_decays: int = 0,
        early_stop_patience_steps: int = -1,
        scaler_type: str = "identity",
        futr_exog_list: Union[List, None] = None,
        hist_exog_list: Union[List, None] = None,
        stat_exog_list: Union[List, None] = None,
        exclude_insample_y: Union[bool, None] = False,
        drop_last_loader: Union[bool, None] = False,
        random_seed: Union[int, None] = 1,
        alias: Union[str, None] = None,
        optimizer: Union[torch.optim.Optimizer, None] = None,
        optimizer_kwargs: Union[Dict, None] = None,
        lr_scheduler: Union[torch.optim.lr_scheduler.LRScheduler, None] = None,
        lr_scheduler_kwargs: Union[Dict, None] = None,
        dataloader_kwargs=None,
        **trainer_kwargs,
    ):
        super().__init__()

        # Multivarariate checks
        if self.MULTIVARIATE and n_series is None:
            raise Exception(
                f"{type(self).__name__} is a multivariate model. Please set n_series to the number of unique time series in your dataset."
            )
        if not self.MULTIVARIATE:
            n_series = 1
        self.n_series = n_series
        self.n_predicts = 1

        # Protections for previous recurrent models
        if input_size < 1:
            input_size = 3 * h
            warnings.warn(
                f"Input size too small. Automatically setting input size to 3 * horizon = {input_size}"
            )

        if inference_input_size is None:
            inference_input_size = input_size
        elif inference_input_size is not None and inference_input_size < 1:
            inference_input_size = input_size
            warnings.warn(
                f"Inference input size too small. Automatically setting inference input size to input_size = {input_size}"
            )

        # For recurrent models we need one additional input as we need to shift insample_y to use it as input
        if self.RECURRENT:
            input_size += 1
            inference_input_size += 1

        # Attributes needed for recurrent models
        self.horizon_backup = h
        self.input_size_backup = input_size
        self.n_samples = n_samples
        if self.RECURRENT:
            if (
                hasattr(loss, "horizon_weight")
                and loss.horizon_weight is not None
                and h_train != h
            ):
                warnings.warn(
                    f"Setting h_train={h} to match the horizon_weight length."
                )
                h_train = h
            self.h_train = h_train
            self.inference_input_size = inference_input_size
            self.rnn_state = None
            self.maintain_state = False

        with warnings.catch_warnings(record=False):
            warnings.filterwarnings("ignore")
            # the following line issues a warning about the loss attribute being saved
            # but we do want to save it
            self.save_hyperparameters()  # Allows instantiation from a checkpoint from class
        self.random_seed = random_seed
        pl.seed_everything(self.random_seed, workers=True)

        # Loss
        self.loss = loss
        if valid_loss is None:
            self.valid_loss = loss
        else:
            self.valid_loss = valid_loss
        self.train_trajectories: List = []
        self.valid_trajectories: List = []

        # Optimization
        if optimizer is not None and not issubclass(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer is not a valid subclass of torch.optim.Optimizer"
            )
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs if optimizer_kwargs is not None else {}

        # lr scheduler
        if lr_scheduler is not None and not issubclass(
            lr_scheduler, torch.optim.lr_scheduler.LRScheduler
        ):
            raise TypeError(
                "lr_scheduler is not a valid subclass of torch.optim.lr_scheduler.LRScheduler"
            )
        self.lr_scheduler = lr_scheduler
        self.lr_scheduler_kwargs = (
            lr_scheduler_kwargs if lr_scheduler_kwargs is not None else {}
        )

        # Variables
        self.futr_exog_list = list(futr_exog_list) if futr_exog_list is not None else []
        self.hist_exog_list = list(hist_exog_list) if hist_exog_list is not None else []
        self.stat_exog_list = list(stat_exog_list) if stat_exog_list is not None else []

        # Set data sizes
        self.futr_exog_size = len(self.futr_exog_list)
        self.hist_exog_size = len(self.hist_exog_list)
        self.stat_exog_size = len(self.stat_exog_list)

        # Check if model supports exogenous, otherwise raise Exception
        if not self.EXOGENOUS_FUTR and self.futr_exog_size > 0:
            raise Exception(
                f"{type(self).__name__} does not support future exogenous variables."
            )
        if not self.EXOGENOUS_HIST and self.hist_exog_size > 0:
            raise Exception(
                f"{type(self).__name__} does not support historical exogenous variables."
            )
        if not self.EXOGENOUS_STAT and self.stat_exog_size > 0:
            raise Exception(
                f"{type(self).__name__} does not support static exogenous variables."
            )

        # Protections for loss functions
        if isinstance(self.loss, (losses.IQLoss, losses.HuberIQLoss)):
            loss_type = type(self.loss)
            if not isinstance(self.valid_loss, loss_type):
                raise Exception(
                    f"Please set valid_loss={type(self.loss).__name__}() when training with {type(self.loss).__name__}"
                )
        if isinstance(self.loss, (losses.MQLoss, losses.HuberMQLoss)):
            if not isinstance(self.valid_loss, (losses.MQLoss, losses.HuberMQLoss)):
                raise Exception(
                    f"Please set valid_loss to MQLoss() or HuberMQLoss() when training with {type(self.loss).__name__}"
                )
        if isinstance(self.valid_loss, (losses.IQLoss, losses.HuberIQLoss)):
            valid_loss_type = type(self.valid_loss)
            if not isinstance(self.loss, valid_loss_type):
                raise Exception(
                    f"Please set loss={type(self.valid_loss).__name__}() when validating with {type(self.valid_loss).__name__}"
                )

        # Deny impossible loss / valid_loss combinations
        if (
            isinstance(self.loss, losses.BasePointLoss)
            and self.valid_loss.is_distribution_output
        ):
            raise Exception(
                f"Validation with distribution loss {type(self.valid_loss).__name__} is not possible when using loss={type(self.loss).__name__}. Please use a point valid_loss (MAE, MSE, ...)"
            )
        elif self.valid_loss.is_distribution_output and self.valid_loss is not loss:
            # Maybe we should raise a Warning or an Exception here, but meh for now.
            self.valid_loss = loss

        if isinstance(self.loss, (losses.relMSE, losses.Accuracy, losses.sCRPS)):
            raise Exception(
                f"{type(self.loss).__name__} cannot be used for training. Please use another loss function (MAE, MSE, ...)"
            )

        if isinstance(self.valid_loss, (losses.relMSE)):
            raise Exception(
                f"{type(self.valid_loss).__name__} cannot be used for validation. Please use another valid_loss (MAE, MSE, ...)"
            )

        ## Trainer arguments ##
        # Max steps, validation steps and check_val_every_n_epoch
        trainer_kwargs = {**trainer_kwargs, "max_steps": max_steps}

        if "max_epochs" in trainer_kwargs.keys():
            raise Exception("max_epochs is deprecated, use max_steps instead.")

        # Callbacks
        if early_stop_patience_steps > 0:
            if "callbacks" not in trainer_kwargs:
                trainer_kwargs["callbacks"] = []
            trainer_kwargs["callbacks"].append(
                EarlyStopping(
                    monitor="ptl/val_loss", patience=early_stop_patience_steps
                )
            )

        # Add GPU accelerator if available
        if trainer_kwargs.get("accelerator", None) is None:
            if torch.cuda.is_available():
                trainer_kwargs["accelerator"] = "gpu"
        if trainer_kwargs.get("devices", None) is None:
            if torch.cuda.is_available():
                trainer_kwargs["devices"] = -1

        # Avoid saturating local memory, disabled fit model checkpoints
        if trainer_kwargs.get("enable_checkpointing", None) is None:
            trainer_kwargs["enable_checkpointing"] = False

        # Set other attributes
        self.trainer_kwargs = trainer_kwargs
        self.h = h
        self.input_size = input_size
        self.windows_batch_size = windows_batch_size
        self.start_padding_enabled = start_padding_enabled
        self.predict_horizon = self.horizon_backup  # Used in recurrent prediction whereby predict h > h_train

        # Padder to complete train windows,
        # example y=[1,2,3,4,5] h=3 -> last y_output = [5,0,0]
        if start_padding_enabled:
            self.padder_train = nn.ConstantPad1d(
                padding=(self.input_size - 1, self.h), value=0.0
            )
        else:
            self.padder_train = nn.ConstantPad1d(padding=(0, self.h), value=0.0)

        # Batch sizes
        if self.MULTIVARIATE and n_series is not None:
            self.batch_size = max(batch_size, n_series)
            if valid_batch_size is not None:
                valid_batch_size = max(valid_batch_size, n_series)
        else:
            self.batch_size = batch_size

        if valid_batch_size is None:
            self.valid_batch_size = self.batch_size
        else:
            self.valid_batch_size = valid_batch_size

        if inference_windows_batch_size is None:
            self.inference_windows_batch_size = windows_batch_size
        else:
            self.inference_windows_batch_size = inference_windows_batch_size

        # Filtering training windows by available sample fractions
        if isinstance(training_data_availability_threshold, int):
            raise ValueError(
                "training_data_availability_threshold cannot be an integer - must be a float"
            )
        elif isinstance(training_data_availability_threshold, float):
            if (
                training_data_availability_threshold < 0.0
                or training_data_availability_threshold > 1.0
            ):
                raise ValueError(
                    f"training_data_availability_threshold must be between 0.0 and 1.0, got {training_data_availability_threshold}"
                )
            self.min_insample_fraction = training_data_availability_threshold
            self.min_outsample_fraction = training_data_availability_threshold
        elif (
            isinstance(training_data_availability_threshold, (list, tuple))
            and len(training_data_availability_threshold) == 2
        ):
            for i, value in enumerate(training_data_availability_threshold):
                if isinstance(value, int):
                    raise ValueError(
                        f"training_data_availability_threshold[{i}] cannot be an integer - must be a float"
                    )
                if not isinstance(value, float):
                    raise ValueError(
                        f"training_data_availability_threshold[{i}] must be a float"
                    )
                if value < 0.0 or value > 1.0:
                    raise ValueError(
                        f"training_data_availability_threshold[{i}] must be between 0.0 and 1.0, got {value}"
                    )

            self.min_insample_fraction = training_data_availability_threshold[0]
            self.min_outsample_fraction = training_data_availability_threshold[1]
        else:
            raise ValueError(
                "training_data_availability_threshold must be a float or a list/tuple of two floats"
            )

        # Optimization
        self.learning_rate = learning_rate
        self.max_steps = max_steps
        self.num_lr_decays = num_lr_decays
        self.lr_decay_steps = (
            max(max_steps // self.num_lr_decays, 1) if self.num_lr_decays > 0 else 10e7
        )
        self.early_stop_patience_steps = early_stop_patience_steps
        self.val_check_steps = val_check_steps
        self.windows_batch_size = windows_batch_size
        self.step_size = step_size

        # If the model does not support exogenous, it can't support exclude_insample_y
        if exclude_insample_y and not (
            self.EXOGENOUS_FUTR or self.EXOGENOUS_HIST or self.EXOGENOUS_STAT
        ):
            raise Exception(
                f"{type(self).__name__} does not support `exclude_insample_y=True`. Please set `exclude_insample_y=False`"
            )

        self.exclude_insample_y = exclude_insample_y

        # Scaler
        self.scaler = TemporalNorm(
            scaler_type=scaler_type,
            dim=1,  # Time dimension is 1.
            num_features=1 + len(self.hist_exog_list) + len(self.futr_exog_list),
        )

        # Fit arguments
        self.val_size = 0
        self.test_size = 0

        # Model state
        self.decompose_forecast = False

        # DataModule arguments
        self.dataloader_kwargs = dataloader_kwargs
        self.drop_last_loader = drop_last_loader
        # used by on_validation_epoch_end hook
        self.validation_step_outputs: List = []
        self.alias = alias

    def __repr__(self):
        return type(self).__name__ if self.alias is None else self.alias

    def _check_exog(self, dataset):
        temporal_cols = set(dataset.temporal_cols.tolist())
        static_cols = set(
            dataset.static_cols.tolist() if dataset.static_cols is not None else []
        )

        missing_hist = set(self.hist_exog_list) - temporal_cols
        missing_futr = set(self.futr_exog_list) - temporal_cols
        missing_stat = set(self.stat_exog_list) - static_cols
        if missing_hist:
            raise Exception(
                f"{missing_hist} historical exogenous variables not found in input dataset"
            )
        if missing_futr:
            raise Exception(
                f"{missing_futr} future exogenous variables not found in input dataset"
            )
        if missing_stat:
            raise Exception(
                f"{missing_stat} static exogenous variables not found in input dataset"
            )

    def _restart_seed(self, random_seed):
        if random_seed is None:
            random_seed = self.random_seed
        torch.manual_seed(random_seed)

    def _get_temporal_exogenous_cols(self, temporal_cols):
        return list(
            set(temporal_cols.tolist()) & set(self.hist_exog_list + self.futr_exog_list)
        )

    def _set_quantiles(self, quantiles=None):
        if quantiles is None and isinstance(
            self.loss, (losses.IQLoss, losses.HuberIQLoss)
        ):
            self.loss.update_quantile(q=[0.5])
        elif hasattr(self.loss, "update_quantile") and callable(
            self.loss.update_quantile
        ):
            self.loss.update_quantile(q=quantiles)

    def _fit_distributed(
        self,
        distributed_config,
        datamodule,
        val_size,
        test_size,
    ):
        assert distributed_config is not None
        from pyspark.ml.torch.distributor import TorchDistributor

        def train_fn(
            model_cls,
            model_params,
            datamodule,
            trainer_kwargs,
            num_tasks,
            num_proc_per_task,
            val_size,
            test_size,
        ):
            import pytorch_lightning as pl

            # we instantiate here to avoid pickling large tensors (weights)
            model = model_cls(**model_params)
            model.val_size = val_size
            model.test_size = test_size
            for arg in ("devices", "num_nodes"):
                trainer_kwargs.pop(arg, None)
            trainer = pl.Trainer(
                strategy="ddp",
                use_distributed_sampler=False,  # to ensure our dataloaders are used as-is
                num_nodes=num_tasks,
                devices=num_proc_per_task,
                **trainer_kwargs,
            )
            trainer.fit(model=model, datamodule=datamodule)
            model.metrics = trainer.callback_metrics
            model.__dict__.pop("_trainer", None)
            return model

        def is_gpu_accelerator(accelerator):
            from pytorch_lightning.accelerators.cuda import CUDAAccelerator

            return (
                accelerator == "gpu"
                or isinstance(accelerator, CUDAAccelerator)
                or (accelerator == "auto" and CUDAAccelerator.is_available())
            )

        local_mode = distributed_config.num_nodes == 1
        if local_mode:
            num_tasks = 1
            num_proc_per_task = distributed_config.devices
        else:
            num_tasks = distributed_config.num_nodes * distributed_config.devices
            num_proc_per_task = 1  # number of GPUs per task
        num_proc = num_tasks * num_proc_per_task
        use_gpu = is_gpu_accelerator(self.trainer_kwargs["accelerator"])
        model = TorchDistributor(
            num_processes=num_proc,
            local_mode=local_mode,
            use_gpu=use_gpu,
        ).run(
            train_fn,
            model_cls=type(self),
            model_params=self.hparams,
            datamodule=datamodule,
            trainer_kwargs=self.trainer_kwargs,
            num_tasks=num_tasks,
            num_proc_per_task=num_proc_per_task,
            val_size=val_size,
            test_size=test_size,
        )
        return model

    def _fit(
        self,
        dataset,
        batch_size,
        valid_batch_size=1024,
        val_size=0,
        test_size=0,
        random_seed=None,
        shuffle_train=True,
        distributed_config=None,
    ):
        self._check_exog(dataset)
        self._restart_seed(random_seed)

        self.val_size = val_size
        self.test_size = test_size
        is_local = isinstance(dataset, BaseTimeSeriesDataset)
        if is_local:
            datamodule_constructor = TimeSeriesDataModule
        else:
            datamodule_constructor = _DistributedTimeSeriesDataModule

        dataloader_kwargs = (
            self.dataloader_kwargs if self.dataloader_kwargs is not None else {}
        )
        datamodule = datamodule_constructor(
            dataset=dataset,
            batch_size=batch_size,
            valid_batch_size=valid_batch_size,
            drop_last=self.drop_last_loader,
            shuffle_train=shuffle_train,
            **dataloader_kwargs,
        )

        if self.val_check_steps > self.max_steps:
            warnings.warn(
                "val_check_steps is greater than max_steps, "
                "setting val_check_steps to max_steps."
            )
        val_check_interval = min(self.val_check_steps, self.max_steps)
        self.trainer_kwargs["val_check_interval"] = int(val_check_interval)
        self.trainer_kwargs["check_val_every_n_epoch"] = None

        if is_local:
            model = self
            trainer = pl.Trainer(**model.trainer_kwargs)
            trainer.fit(model, datamodule=datamodule)
            model.metrics = trainer.callback_metrics
            model.__dict__.pop("_trainer", None)
        else:
            model = self._fit_distributed(
                distributed_config,
                datamodule,
                val_size,
                test_size,
            )
        return model

    def on_fit_start(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)

    def configure_optimizers(self):
        if self.optimizer:
            optimizer_signature = inspect.signature(self.optimizer)
            optimizer_kwargs = deepcopy(self.optimizer_kwargs)
            if "lr" in optimizer_signature.parameters:
                if "lr" in optimizer_kwargs:
                    warnings.warn(
                        "ignoring learning rate passed in optimizer_kwargs, using the model's learning rate"
                    )
                optimizer_kwargs["lr"] = self.learning_rate
            optimizer = self.optimizer(params=self.parameters(), **optimizer_kwargs)
        else:
            if self.optimizer_kwargs:
                warnings.warn(
                    "ignoring optimizer_kwargs as the optimizer is not specified"
                )
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        lr_scheduler = {"frequency": 1, "interval": "step"}
        if self.lr_scheduler:
            lr_scheduler_signature = inspect.signature(self.lr_scheduler)
            lr_scheduler_kwargs = deepcopy(self.lr_scheduler_kwargs)
            if "optimizer" in lr_scheduler_signature.parameters:
                if "optimizer" in lr_scheduler_kwargs:
                    warnings.warn(
                        "ignoring optimizer passed in lr_scheduler_kwargs, using the model's optimizer"
                    )
                    del lr_scheduler_kwargs["optimizer"]
            lr_scheduler["scheduler"] = self.lr_scheduler(
                optimizer=optimizer, **lr_scheduler_kwargs
            )
        else:
            if self.lr_scheduler_kwargs:
                warnings.warn(
                    "ignoring lr_scheduler_kwargs as the lr_scheduler is not specified"
                )
            lr_scheduler["scheduler"] = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=self.lr_decay_steps, gamma=0.5
            )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def get_test_size(self):
        return self.test_size

    def set_test_size(self, test_size):
        self.test_size = test_size

    def on_validation_epoch_end(self):
        if self.val_size == 0:
            return
        losses = torch.stack(self.validation_step_outputs)
        avg_loss = losses.mean().detach().item()
        self.log(
            "ptl/val_loss",
            avg_loss,
            batch_size=losses.size(0),
            sync_dist=True,
        )
        self.valid_trajectories.append((self.global_step, avg_loss))
        self.validation_step_outputs.clear()  # free memory (compute `avg_loss` per epoch)

    def save(self, path):
        with fsspec.open(path, "wb") as f:
            torch.save(
                {"hyper_parameters": self.hparams, "state_dict": self.state_dict()},
                f,
            )

    @classmethod
    def load(cls, path, **kwargs):
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False
        with fsspec.open(path, "rb") as f, warnings.catch_warnings():
            # ignore possible warnings about weights_only=False
            warnings.filterwarnings("ignore", category=FutureWarning)
            content = torch.load(f, **kwargs)
        with _disable_torch_init():
            model = cls(**content["hyper_parameters"])
        if "assign" in inspect.signature(model.load_state_dict).parameters:
            model.load_state_dict(content["state_dict"], strict=True, assign=True)
        else:  # pytorch<2.1
            model.load_state_dict(content["state_dict"], strict=True)
        return model

    def _create_windows(self, batch, step):
        # Parse common data
        window_size = self.input_size + self.h
        temporal_cols = batch["temporal_cols"]
        temporal = batch["temporal"]

        if step == "train":
            if self.val_size + self.test_size > 0:
                cutoff = -self.val_size - self.test_size
                temporal = temporal[:, :, :cutoff]

            temporal = self.padder_train(temporal)

            if temporal.shape[-1] < window_size:
                raise Exception(
                    "Time series is too short for training, consider setting a smaller input size or set start_padding_enabled=True"
                )

            windows = temporal.unfold(
                dimension=-1, size=window_size, step=self.step_size
            )

            if self.MULTIVARIATE:
                # [n_series, C, Ws, L + h] -> [Ws, L + h, C, n_series]
                windows = windows.permute(2, 3, 1, 0)
            else:
                # [n_series, C, Ws, L + h] -> [Ws * n_series, L + h, C, 1]
                windows_per_serie = windows.shape[2]
                windows = windows.permute(0, 2, 3, 1)
                windows = windows.flatten(0, 1)
                windows = windows.unsqueeze(-1)

            # Calculate minimum required available points based on fractions
            min_insample_points = max(
                1, int(self.input_size * self.min_insample_fraction * self.n_series)
            )
            min_outsample_points = max(
                1, int(self.h * self.min_outsample_fraction * self.n_series)
            )

            # Sample based on available conditions
            available_idx = temporal_cols.get_loc("available_mask")
            insample_condition = windows[:, : self.input_size, available_idx]
            insample_condition = torch.sum(
                insample_condition, axis=(1, -1)
            )  # Sum over time & series dimension
            final_condition = insample_condition >= min_insample_points

            if self.h > 0:
                outsample_condition = windows[:, self.input_size :, available_idx]
                outsample_condition = torch.sum(
                    outsample_condition, axis=(1, -1)
                )  # Sum over time & series dimension
                final_condition = (outsample_condition >= min_outsample_points) & (
                    insample_condition >= min_insample_points
                )

            windows = windows[final_condition]

            # Parse Static data to match windows
            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)

            # Repeat static if univariate: [n_series, S] -> [Ws * n_series, S]
            if static is not None and not self.MULTIVARIATE:
                static = torch.repeat_interleave(
                    static, repeats=windows_per_serie, dim=0
                )
                static = static[final_condition]

            # Protection of empty windows
            if final_condition.sum() == 0:
                raise Exception("No windows available for training")

            return windows, static, static_cols

        elif step in ["predict", "val"]:

            if step == "predict":
                initial_input = temporal.shape[-1] - self.test_size
                if (
                    initial_input <= self.input_size
                ):  # There is not enough data to predict first timestamp
                    temporal = F.pad(
                        temporal,
                        pad=(self.input_size - initial_input, 0),
                        mode="constant",
                        value=0.0,
                    )
                predict_step_size = self.predict_step_size
                cutoff = -self.input_size - self.test_size
                temporal = temporal[:, :, cutoff:]

            elif step == "val":
                predict_step_size = self.step_size
                cutoff = -self.input_size - self.val_size - self.test_size
                if self.test_size > 0:
                    temporal = batch["temporal"][:, :, cutoff : -self.test_size]
                else:
                    temporal = batch["temporal"][:, :, cutoff:]
                if temporal.shape[-1] < window_size:
                    initial_input = temporal.shape[-1] - self.val_size
                    temporal = F.pad(
                        temporal,
                        pad=(self.input_size - initial_input, 0),
                        mode="constant",
                        value=0.0,
                    )

            if (
                (step == "predict")
                and (self.test_size == 0)
                and (len(self.futr_exog_list) == 0)
            ):
                temporal = F.pad(temporal, pad=(0, self.h), mode="constant", value=0.0)

            windows = temporal.unfold(
                dimension=-1, size=window_size, step=predict_step_size
            )

            static = batch.get("static", None)
            static_cols = batch.get("static_cols", None)

            if self.MULTIVARIATE:
                # [n_series, C, Ws, L + h] -> [Ws, L + h, C, n_series]
                windows = windows.permute(2, 3, 1, 0)
            else:
                # [n_series, C, Ws, L + h] -> [Ws * n_series, L + h, C, 1]
                windows_per_serie = windows.shape[2]
                windows = windows.permute(0, 2, 3, 1)
                windows = windows.flatten(0, 1)
                windows = windows.unsqueeze(-1)
                if static is not None:
                    static = torch.repeat_interleave(
                        static, repeats=windows_per_serie, dim=0
                    )

            return windows, static, static_cols
        else:
            raise ValueError(f"Unknown step {step}")

    def _normalization(self, windows, y_idx):
        # windows are already filtered by train/validation/test
        # from the `create_windows_method` nor leakage risk
        temporal = windows["temporal"]  # [Ws, L + h, C, n_series]
        temporal_cols = windows["temporal_cols"].copy()  # [Ws, L + h, C, n_series]

        # To avoid leakage uses only the lags
        temporal_data_cols = self._get_temporal_exogenous_cols(
            temporal_cols=temporal_cols
        )
        temporal_idxs = get_indexer_raise_missing(temporal_cols, temporal_data_cols)
        temporal_idxs = np.append(y_idx, temporal_idxs)
        temporal_data = temporal[:, :, temporal_idxs]
        temporal_mask = temporal[:, :, temporal_cols.get_loc("available_mask")].clone()
        if self.h > 0:
            temporal_mask[:, -self.h :] = 0.0

        # Normalize. self.scaler stores the shift and scale for inverse transform
        temporal_mask = temporal_mask.unsqueeze(
            2
        )  # Add channel dimension for scaler.transform.
        temporal_data = self.scaler.transform(x=temporal_data, mask=temporal_mask)

        # Replace values in windows dict
        temporal[:, :, temporal_idxs] = temporal_data
        windows["temporal"] = temporal

        return windows

    def _inv_normalization(self, y_hat, y_idx):
        # Receives window predictions [Ws, h, output, n_series]
        # Broadcasts scale if necessary and inverts normalization
        add_channel_dim = y_hat.ndim > 3
        y_loc, y_scale = self._get_loc_scale(y_idx, add_channel_dim=add_channel_dim)
        if hasattr(self, "explain") and self.explain and y_hat.shape[0] != y_loc.shape[0]:
            # n_repeats is always a multiple of the batch size
            n_repeats = y_hat.shape[0] // y_loc.shape[0]
            y_loc = y_loc.repeat(n_repeats, *([1] * (y_loc.ndim - 1)))
            y_scale = y_scale.repeat(n_repeats, *([1] * (y_scale.ndim - 1)))
        y_hat = self.scaler.inverse_transform(z=y_hat, x_scale=y_scale, x_shift=y_loc)

        return y_hat

    def _sample_windows(
        self, windows_temporal, static, static_cols, temporal_cols, step, w_idxs=None
    ):
        if step == "train" and self.windows_batch_size is not None:
            n_windows = windows_temporal.shape[0]
            w_idxs = np.random.choice(
                n_windows,
                size=self.windows_batch_size,
                replace=(n_windows < self.windows_batch_size),
            )
        windows_sample = windows_temporal
        if w_idxs is not None:
            windows_sample = windows_temporal[w_idxs]

            if static is not None and not self.MULTIVARIATE:
                static = static[w_idxs]

        windows_batch = dict(
            temporal=windows_sample,
            temporal_cols=temporal_cols,
            static=static,
            static_cols=static_cols,
        )
        return windows_batch

    def _parse_windows(self, batch, windows):
        # windows: [Ws, L + h, C, n_series]

        # Filter insample lags from outsample horizon
        y_idx = batch["y_idx"]
        mask_idx = batch["temporal_cols"].get_loc("available_mask")

        insample_y = windows["temporal"][:, : self.input_size, y_idx]
        insample_mask = windows["temporal"][:, : self.input_size, mask_idx]

        # Declare additional information
        outsample_y = None
        outsample_mask = None
        hist_exog = None
        futr_exog = None
        stat_exog = None

        if self.h > 0:
            outsample_y = windows["temporal"][:, self.input_size :, y_idx]
            outsample_mask = windows["temporal"][:, self.input_size :, mask_idx]

        # Recurrent models at t predict t+1, so we shift the input (insample_y) by one
        if self.RECURRENT:
            insample_y = torch.cat((insample_y, outsample_y[:, :-1]), dim=1)
            insample_mask = torch.cat((insample_mask, outsample_mask[:, :-1]), dim=1)
            self.maintain_state = False

        if len(self.hist_exog_list):
            hist_exog_idx = get_indexer_raise_missing(
                windows["temporal_cols"], self.hist_exog_list
            )
            if self.RECURRENT:
                hist_exog = windows["temporal"][:, :, hist_exog_idx]
                hist_exog[:, self.input_size :] = 0.0
                hist_exog = hist_exog[:, 1:]
            else:
                hist_exog = windows["temporal"][:, : self.input_size, hist_exog_idx]
            if not self.MULTIVARIATE:
                hist_exog = hist_exog.squeeze(-1)
            else:
                hist_exog = hist_exog.swapaxes(1, 2)

        if len(self.futr_exog_list):
            futr_exog_idx = get_indexer_raise_missing(
                windows["temporal_cols"], self.futr_exog_list
            )
            futr_exog = windows["temporal"][:, :, futr_exog_idx]
            if self.RECURRENT:
                futr_exog = futr_exog[:, 1:]
            if not self.MULTIVARIATE:
                futr_exog = futr_exog.squeeze(-1)
            else:
                futr_exog = futr_exog.swapaxes(1, 2)

        if len(self.stat_exog_list):
            static_idx = get_indexer_raise_missing(
                windows["static_cols"], self.stat_exog_list
            )
            stat_exog = windows["static"][:, static_idx]

        # TODO: think a better way of removing insample_y features
        if self.exclude_insample_y:
            insample_y = insample_y * 0

        return (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        )

    def _get_loc_scale(self, y_idx, add_channel_dim=False):
        # [B, L, C, n_series] -> [B, L, n_series]
        y_scale = self.scaler.x_scale[:, :, y_idx]
        y_loc = self.scaler.x_shift[:, :, y_idx]

        # [B, L, n_series] -> [B, L, n_series, 1]
        if add_channel_dim:
            y_scale = y_scale.unsqueeze(-1)
            y_loc = y_loc.unsqueeze(-1)

        return y_loc, y_scale

    def _compute_valid_loss(
        self, insample_y, outsample_y, output, outsample_mask, y_idx
    ):
        if self.loss.is_distribution_output:
            y_loc, y_scale = self._get_loc_scale(y_idx)
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            if isinstance(
                self.valid_loss, (losses.sCRPS, losses.MQLoss, losses.HuberMQLoss)
            ):
                _, _, quants = self.loss.sample(distr_args=distr_args)
                output = quants
            elif isinstance(self.valid_loss, losses.BasePointLoss):
                distr = self.loss.get_distribution(distr_args=distr_args)
                output = distr.mean

        # Validation Loss evaluation
        if self.valid_loss.is_distribution_output:
            valid_loss = self.valid_loss(
                y=outsample_y, distr_args=distr_args, mask=outsample_mask
            )
        else:
            output = self._inv_normalization(y_hat=output, y_idx=y_idx)
            valid_loss = self.valid_loss(
                y=outsample_y, y_hat=output, y_insample=insample_y, mask=outsample_mask
            )
        return valid_loss

    def _validate_step_recurrent_batch(
        self, insample_y, insample_mask, futr_exog, hist_exog, stat_exog, y_idx
    ):
        # Remember state in network and set horizon to 1
        self.rnn_state = None
        self.maintain_state = True
        self.h = 1

        # Initialize results array
        n_outputs = self.loss.outputsize_multiplier
        y_hat = torch.zeros(
            (insample_y.shape[0], self.horizon_backup, self.n_series * n_outputs),
            device=insample_y.device,
            dtype=insample_y.dtype,
        )

        # First step prediction
        tau = 0

        # Set exogenous
        hist_exog_current = None
        if self.hist_exog_size > 0:
            hist_exog_current = hist_exog[:, : self.input_size + tau]

        futr_exog_current = None
        if self.futr_exog_size > 0:
            futr_exog_current = futr_exog[:, : self.input_size + tau]

        # First forecast step
        y_hat[:, tau], insample_y = self._validate_step_recurrent_single(
            insample_y=insample_y[:, : self.input_size + tau],
            insample_mask=insample_mask[:, : self.input_size + tau],
            hist_exog=hist_exog_current,
            futr_exog=futr_exog_current,
            stat_exog=stat_exog,
            y_idx=y_idx,
        )

        # Horizon prediction recursively
        for tau in range(1, self.horizon_backup):
            # Set exogenous
            if self.hist_exog_size > 0:
                hist_exog_current = hist_exog[:, self.input_size + tau - 1].unsqueeze(1)

            if self.futr_exog_size > 0:
                futr_exog_current = futr_exog[:, self.input_size + tau - 1].unsqueeze(1)

            y_hat[:, tau], insample_y = self._validate_step_recurrent_single(
                insample_y=insample_y,
                insample_mask=None,
                hist_exog=hist_exog_current,
                futr_exog=futr_exog_current,
                stat_exog=stat_exog,
                y_idx=y_idx,
            )

        # Reset state and horizon
        self.maintain_state = False
        self.rnn_state = None
        self.h = self.horizon_backup

        return y_hat

    def _validate_step_recurrent_single(
        self, insample_y, insample_mask, hist_exog, futr_exog, stat_exog, y_idx
    ):
        # Input sequence
        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L, n_series]
            insample_mask=insample_mask,  # [Ws, L, n_series]
            futr_exog=futr_exog,  # univariate: [Ws, L, F]; multivariate: [Ws, F, L, n_series]
            hist_exog=hist_exog,  # univariate: [Ws, L, X]; multivariate: [Ws, X, L, n_series]
            stat_exog=stat_exog,
        )  # univariate: [Ws, S]; multivariate: [n_series, S]

        # Model Predictions
        output_batch_unmapped = self(windows_batch)
        output_batch = self.loss.domain_map(output_batch_unmapped)

        # Inverse normalization and sampling
        if self.loss.is_distribution_output:
            # Sample distribution
            y_loc, y_scale = self._get_loc_scale(y_idx)
            distr_args = self.loss.scale_decouple(
                output=output_batch, loc=y_loc, scale=y_scale
            )
            # When validating, the output is the mean of the distribution which is an attribute
            distr = self.loss.get_distribution(distr_args=distr_args)

            # Scale back to feed back as input
            insample_y = self.scaler.scaler(distr.mean, y_loc, y_scale)
        else:
            # Todo: for now, we assume that in case of a BasePointLoss with ndim==4, the last dimension
            # contains a set of predictions for the target (e.g. MQLoss multiple quantiles), for which we use the
            # mean as feedback signal for the recurrent predictions. A more precise way is to increase the
            # insample input size of the recurrent network by the number of outputs so that each output
            # can be fed back to a specific input channel.
            if output_batch.ndim == 4:
                output_batch = output_batch.mean(dim=-1)

            insample_y = output_batch

        # Remove horizon dim: [B, 1, N * n_outputs] -> [B, N * n_outputs]
        y_hat = output_batch_unmapped.squeeze(1)
        return y_hat, insample_y

    def _predict_step_recurrent_batch(
        self, insample_y, insample_mask, futr_exog, hist_exog, stat_exog, y_idx
    ):
        # Remember state in network and set horizon to 1
        self.rnn_state = None
        self.maintain_state = True
        self.h = 1

        # Initialize results array
        n_outputs = len(self.loss.output_names)
        y_hat = torch.zeros(
            (insample_y.shape[0], self.predict_horizon, self.n_series, n_outputs),
            device=insample_y.device,
            dtype=insample_y.dtype,
        )

        # First step prediction
        tau = 0

        # Set exogenous
        hist_exog_current = None
        if self.hist_exog_size > 0:
            hist_exog_current = hist_exog[:, : self.input_size + tau]

        futr_exog_current = None
        if self.futr_exog_size > 0:
            futr_exog_current = futr_exog[:, : self.input_size + tau]

        # First forecast step
        y_hat[:, tau], insample_y = self._predict_step_recurrent_single(
            insample_y=insample_y[:, : self.input_size + tau],
            insample_mask=insample_mask[:, : self.input_size + tau],
            hist_exog=hist_exog_current,
            futr_exog=futr_exog_current,
            stat_exog=stat_exog,
            y_idx=y_idx,
        )

        # Horizon prediction recursively
        for tau in range(1, self.predict_horizon):
            # Set exogenous
            if self.hist_exog_size > 0:
                hist_exog_current = hist_exog[:, self.input_size + tau - 1].unsqueeze(1)

            if self.futr_exog_size > 0:
                futr_exog_current = futr_exog[:, self.input_size + tau - 1].unsqueeze(1)

            y_hat[:, tau], insample_y = self._predict_step_recurrent_single(
                insample_y=insample_y,
                insample_mask=None,
                hist_exog=hist_exog_current,
                futr_exog=futr_exog_current,
                stat_exog=stat_exog,
                y_idx=y_idx,
            )

        # Reset state and horizon
        self.maintain_state = False
        self.rnn_state = None
        self.h = self.horizon_backup

        # Squeeze for univariate case
        if not self.MULTIVARIATE:
            y_hat = y_hat.squeeze(2)

        return y_hat

    def _predict_step_recurrent_single(
        self, insample_y, insample_mask, hist_exog, futr_exog, stat_exog, y_idx
    ):
        # Input sequence
        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L, n_series]
            insample_mask=insample_mask,  # [Ws, L, n_series]
            futr_exog=futr_exog,  # univariate: [Ws, L, F]; multivariate: [Ws, F, L, n_series]
            hist_exog=hist_exog,  # univariate: [Ws, L, X]; multivariate: [Ws, X, L, n_series]
            stat_exog=stat_exog,
        )  # univariate: [Ws, S]; multivariate: [n_series, S]

        # Model Predictions
        output_batch_unmapped = self(windows_batch)
        output_batch = self.loss.domain_map(output_batch_unmapped)

        # Inverse normalization and sampling
        if self.loss.is_distribution_output:
            # Sample distribution
            y_loc, y_scale = self._get_loc_scale(y_idx)
            distr_args = self.loss.scale_decouple(
                output=output_batch, loc=y_loc, scale=y_scale
            )
            # When predicting, we need to sample to get the quantiles. The mean is an attribute.
            _, _, quants = self.loss.sample(
                distr_args=distr_args, num_samples=self.n_samples
            )
            mean = self.loss.distr_mean

            # Scale back to feed back as input
            insample_y = self.scaler.scaler(mean, y_loc, y_scale)

            # Save predictions
            y_hat = torch.concat((mean.unsqueeze(-1), quants), axis=-1)

            if self.loss.return_params:
                distr_args = torch.stack(distr_args, dim=-1)
                if distr_args.ndim > 4:
                    distr_args = distr_args.flatten(-2, -1)
                y_hat = torch.concat((y_hat, distr_args), axis=-1)
        else:
            # Todo: for now, we assume that in case of a BasePointLoss with ndim==4, the last dimension
            # contains a set of predictions for the target (e.g. MQLoss multiple quantiles), for which we use the
            # mean as feedback signal for the recurrent predictions. A more precise way is to increase the
            # insample input size of the recurrent network by the number of outputs so that each output
            # can be fed back to a specific input channel.
            if output_batch.ndim == 4:
                output_batch = output_batch.mean(dim=-1)

            insample_y = output_batch
            y_hat = self._inv_normalization(y_hat=output_batch, y_idx=y_idx)
            y_hat = y_hat.unsqueeze(-1)

        # Remove horizon dim: [B, 1, N, n_outputs] -> [B, N, n_outputs]
        y_hat = y_hat.squeeze(1)
        return y_hat, insample_y

    def _predict_step_direct_batch(
        self, insample_y, insample_mask, hist_exog, futr_exog, stat_exog, y_idx
    ):
        windows_batch = dict(
            insample_y=insample_y,
            insample_mask=insample_mask,
            futr_exog=futr_exog,
            hist_exog=hist_exog,
            stat_exog=stat_exog,
        )

        # Model Predictions
        output_batch = self(windows_batch)
        output_batch = self.loss.domain_map(output_batch)

        # Inverse normalization and sampling
        if self.loss.is_distribution_output:
            y_loc, y_scale = self._get_loc_scale(y_idx)
            # Always compute distribution args (needed for both explain and normal mode)
            distr_args = self.loss.scale_decouple(
                output=output_batch, loc=y_loc, scale=y_scale
            )
            # Normal mode: full distribution processing with sampling
            _, sample_mean, quants = self.loss.sample(distr_args=distr_args)
            y_hat = torch.concat((sample_mean, quants), axis=-1)
            
            if self.loss.return_params:
                distr_args = torch.stack(distr_args, dim=-1)
                if distr_args.ndim > 4:
                    distr_args = distr_args.flatten(-2, -1)
                y_hat = torch.concat((y_hat, distr_args), axis=-1)
        else:
            y_hat = self._inv_normalization(y_hat=output_batch, y_idx=y_idx)

        return y_hat

    def _predict_step_recurrent(self, batch, batch_idx):
        self.input_size = self.inference_input_size
        temporal_cols = batch["temporal_cols"]
        windows_temporal, static, static_cols = self._create_windows(
            batch, step="predict"
        )
        n_windows = len(windows_temporal)
        y_idx = batch["y_idx"]

        # Number of windows in batch
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))
        y_hats = []

        explain_state = hasattr(self, "explain") and self.explain
        if explain_state:
            insample_explanations = []
            futr_exog_explanations = []
            hist_exog_explanations = []
            stat_exog_explanations = []
            baseline_predictions = []

        for i in range(n_batches):
            # Create and normalize windows [Ws, L+H, C]
            w_idxs = np.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
            )
            windows = self._sample_windows(
                windows_temporal,
                static,
                static_cols,
                temporal_cols,
                step="predict",
                w_idxs=w_idxs,
            )
            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
            insample_y, insample_mask, _, _, hist_exog, futr_exog, stat_exog = (
                self._parse_windows(batch, windows)
            )

            y_hat = self._predict_step_recurrent_batch(
                insample_y=insample_y,
                insample_mask=insample_mask,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
                y_idx=y_idx,
            )

            if explain_state:
                (
                    insample_explanation,
                    futr_exog_explanation,
                    hist_exog_explanation,
                    stat_exog_explanation,
                    baseline_prediction,
                ) = self._explain_batch(
                    insample_y=insample_y,
                    insample_mask=insample_mask,
                    futr_exog=futr_exog,
                    hist_exog=hist_exog,
                    stat_exog=stat_exog,
                    y_idx=y_idx,
                    y_hat_shape=y_hat.shape,
                )
                insample_explanations.append(insample_explanation)
                if futr_exog_explanation is not None:
                    futr_exog_explanations.append(futr_exog_explanation)
                if hist_exog_explanation is not None:
                    hist_exog_explanations.append(hist_exog_explanation)
                if stat_exog_explanation is not None:
                    stat_exog_explanations.append(stat_exog_explanation)
                baseline_predictions.append(baseline_prediction)

            y_hats.append(y_hat)
        y_hat = torch.cat(y_hats, dim=0)
        self.input_size = self.input_size_backup

        if explain_state:
            insample_explanations = torch.cat(insample_explanations, dim=0)
            if futr_exog_explanations:
                futr_exog_explanations = torch.cat(futr_exog_explanations, dim=0)
            if hist_exog_explanations:
                hist_exog_explanations = torch.cat(hist_exog_explanations, dim=0)
            if stat_exog_explanations:
                stat_exog_explanations = torch.cat(stat_exog_explanations, dim=0)
            if baseline_predictions and baseline_predictions[0] is not None:
                baseline_predictions = torch.cat(baseline_predictions, dim=0)
            else:
                baseline_predictions = None
            return (
                y_hat,
                insample_explanations,
                futr_exog_explanations,
                hist_exog_explanations,
                stat_exog_explanations,
                baseline_predictions,
            )        
        else:
            return y_hat
        
    def _compute_explanations_for_step(
        self,
        batch,
        temporal_cols,
        y_idx,
        recursive_step=0,
        y_hat_shape=None,
    ):
        """Compute explanations for a single prediction step."""
        # Create windows and normalize for explanations
        windows_temporal, static, static_cols = self._create_windows(
            batch, step="predict"
        )
        n_windows = len(windows_temporal)

        # Process windows in batches
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        step_insample_explanations = []
        step_futr_exog_explanations = []
        step_hist_exog_explanations = []
        step_stat_exog_explanations = []
        step_baseline_predictions = []

        for j in range(n_batches):
            w_idxs = np.arange(
                j * windows_batch_size, min((j + 1) * windows_batch_size, n_windows)
            )
            windows = self._sample_windows(
                windows_temporal,
                static,
                static_cols,
                temporal_cols,
                step="predict",
                w_idxs=w_idxs,
            )
            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
            insample_y, insample_mask, _, _, hist_exog, futr_exog, stat_exog = (
                self._parse_windows(batch, windows)
            )

            # Compute explanations
            (
                insample_explanation,
                futr_exog_explanation,
                hist_exog_explanation,
                stat_exog_explanation,
                baseline_prediction,
            ) = self._explain_batch(
                insample_y=insample_y,
                insample_mask=insample_mask,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
                y_idx=y_idx,
                y_hat_shape=y_hat_shape,
                recursive_step=recursive_step,
            )

            if insample_explanation is not None:
                step_insample_explanations.append(insample_explanation)
                if futr_exog_explanation is not None:
                    step_futr_exog_explanations.append(futr_exog_explanation)
                if hist_exog_explanation is not None:
                    step_hist_exog_explanations.append(hist_exog_explanation)
                if stat_exog_explanation is not None:
                    step_stat_exog_explanations.append(stat_exog_explanation)
                if baseline_prediction is not None:
                    step_baseline_predictions.append(baseline_prediction)

        # Concatenate window batches
        insample = torch.cat(step_insample_explanations, dim=0) if step_insample_explanations else None
        futr_exog = torch.cat(step_futr_exog_explanations, dim=0) if step_futr_exog_explanations else None
        hist_exog = torch.cat(step_hist_exog_explanations, dim=0) if step_hist_exog_explanations else None
        stat_exog = torch.cat(step_stat_exog_explanations, dim=0) if step_stat_exog_explanations else None
        baseline = torch.cat(step_baseline_predictions, dim=0) if step_baseline_predictions else None

        return insample, futr_exog, hist_exog, stat_exog, baseline, y_hat_shape

    def _predict_step_direct(self, batch, batch_idx, recursive=False):
        temporal_cols = batch["temporal_cols"]
        explain_state = hasattr(self, "explain") and self.explain
        if recursive:
            # We need to predict recursively, so we use the median quantile if it exists to feed back as insample_y
            median_idx = self._maybe_get_quantile_idx(quantile=0.5)
            y_idx = batch["y_idx"]
            y_hats = []
            total_test_size = self.test_size
            futr_temporal = batch["temporal"][:, :, -total_test_size + self.h :]
            batch["temporal"] = batch["temporal"][:, :, : -total_test_size + self.h]
            self.test_size = self.h
            
            # Initialize explanation storage if explaining
            if explain_state:
                all_insample_explanations = []
                all_futr_exog_explanations = []
                all_hist_exog_explanations = []
                all_stat_exog_explanations = []
                all_baseline_predictions = []
            
            for i in range(self.n_predicts):
                # Temporarily disable explanations for the recursive call
                if explain_state:
                    self.explain = False
                
                # Make predictions for this step
                y_hat = self._predict_step_direct(batch, batch_idx, recursive=False)
                
                # Restore explanation state
                if explain_state:
                    self.explain = True
                
                y_hats.append(y_hat)
                
                # Generate explanations
                if explain_state:
                    (insample, futr_exog, hist_exog, stat_exog, baseline, _) = \
                        self._compute_explanations_for_step(
                            batch=batch,
                            temporal_cols=temporal_cols,
                            y_idx=y_idx,
                            recursive_step=i,
                            y_hat_shape=y_hat.shape,
                        )
                    
                    if insample is not None:
                        all_insample_explanations.append(insample)
                        if futr_exog is not None:
                            all_futr_exog_explanations.append(futr_exog)
                        if hist_exog is not None:
                            all_hist_exog_explanations.append(hist_exog)
                        if stat_exog is not None:
                            all_stat_exog_explanations.append(stat_exog)
                        if baseline is not None:
                            all_baseline_predictions.append(baseline)
                
                # Update temporal with predictions for next iteration
                if i < self.n_predicts - 1:
                    y_hat_median = y_hat
                    if median_idx is not None:
                        y_hat_median = y_hat[..., median_idx]
                    
                    # Update temporal of the batch with predictions
                    temporal = batch["temporal"]
                    if self.MULTIVARIATE:
                        y_hat_median = y_hat_median.swapaxes(0, 2)
                        y_hat_median = y_hat_median.swapaxes(1, 2)
                        y_hat_median = y_hat_median.squeeze(1)
                    else:
                        y_hat_median = y_hat_median.squeeze(-1)

                    temporal[:, y_idx, -self.h :] = y_hat_median

                    # Concatenate next futr_temporal
                    idx = i * self.h
                    next_futr_temporal = futr_temporal[:, :, idx : idx + self.h]
                    temporal = torch.cat((temporal, next_futr_temporal), dim=-1)

                    # Update batch
                    batch["temporal"] = temporal

            y_hat = torch.cat(y_hats, dim=1)
            self.test_size = total_test_size
            
            # Return with concatenated explanations if explaining
            if explain_state:
                # Concatenate explanations across all recursive steps
                insample_explanations = torch.cat(all_insample_explanations, dim=1) if all_insample_explanations else None
                futr_exog_explanations = torch.cat(all_futr_exog_explanations, dim=1) if all_futr_exog_explanations else None
                hist_exog_explanations = torch.cat(all_hist_exog_explanations, dim=1) if all_hist_exog_explanations else None
                stat_exog_explanations = torch.cat(all_stat_exog_explanations, dim=1) if all_stat_exog_explanations else None
                baseline_predictions = torch.cat(all_baseline_predictions, dim=1) if all_baseline_predictions else None
                
                return (
                    y_hat,
                    insample_explanations,
                    futr_exog_explanations,
                    hist_exog_explanations,
                    stat_exog_explanations,
                    baseline_predictions,
                )
            else:
                return y_hat
        
        else:
            # Non-recursive case remains unchanged
            windows_temporal, static, static_cols = self._create_windows(
                batch,
                step="predict",
            )
            n_windows = len(windows_temporal)
            y_idx = batch["y_idx"]

            # Number of windows in batch
            windows_batch_size = self.inference_windows_batch_size
            if windows_batch_size < 0:
                windows_batch_size = n_windows
            n_batches = int(np.ceil(n_windows / windows_batch_size))
            y_hats = []

            if explain_state:
                insample_explanations = []
                futr_exog_explanations = []
                hist_exog_explanations = []
                stat_exog_explanations = []
                baseline_predictions = []

            for i in range(n_batches):
                # Create and normalize windows [Ws, L+H, C]
                w_idxs = np.arange(
                    i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
                )
                windows = self._sample_windows(
                    windows_temporal,
                    static,
                    static_cols,
                    temporal_cols,
                    step="predict",
                    w_idxs=w_idxs,
                )
                windows = self._normalization(windows=windows, y_idx=y_idx)

                # Parse windows
                insample_y, insample_mask, _, _, hist_exog, futr_exog, stat_exog = (
                    self._parse_windows(batch, windows)
                )

                y_hat = self._predict_step_direct_batch(
                    insample_y=insample_y,
                    insample_mask=insample_mask,
                    futr_exog=futr_exog,
                    hist_exog=hist_exog,
                    stat_exog=stat_exog,
                    y_idx=y_idx,
                )

                if explain_state:
                    (
                        insample_explanation,
                        futr_exog_explanation,
                        hist_exog_explanation,
                        stat_exog_explanation,
                        baseline_prediction,
                    ) = self._explain_batch(
                        insample_y=insample_y,
                        insample_mask=insample_mask,
                        futr_exog=futr_exog,
                        hist_exog=hist_exog,
                        stat_exog=stat_exog,
                        y_idx=y_idx,
                        y_hat_shape=y_hat.shape,
                        recursive_step=0,  # Non-recursive is always step 0
                    )
                    insample_explanations.append(insample_explanation)
                    if futr_exog_explanation is not None:
                        futr_exog_explanations.append(futr_exog_explanation)
                    if hist_exog_explanation is not None:
                        hist_exog_explanations.append(hist_exog_explanation)
                    if stat_exog_explanation is not None:
                        stat_exog_explanations.append(stat_exog_explanation)
                    baseline_predictions.append(baseline_prediction)

                y_hats.append(y_hat)

            y_hat = torch.cat(y_hats, dim=0)
            
            if explain_state:
                insample_explanations = torch.cat(insample_explanations, dim=0)
                if futr_exog_explanations:
                    futr_exog_explanations = torch.cat(futr_exog_explanations, dim=0)
                if hist_exog_explanations:
                    hist_exog_explanations = torch.cat(hist_exog_explanations, dim=0)
                if stat_exog_explanations:
                    stat_exog_explanations = torch.cat(stat_exog_explanations, dim=0)
                if baseline_predictions and baseline_predictions[0] is not None:
                    baseline_predictions = torch.cat(baseline_predictions, dim=0)
                else:
                    baseline_predictions = None
                return (
                    y_hat,
                    insample_explanations,
                    futr_exog_explanations,
                    hist_exog_explanations,
                    stat_exog_explanations,
                    baseline_predictions,
                )

        return y_hat

    def _maybe_get_quantile_idx(self, quantile: float) -> Union[int, None]:
        if isinstance(self.loss, DISTRIBUTION_LOSSES + MULTIQUANTILE_LOSSES):
            try:
                idx_quantile = (self.loss.quantiles == quantile).nonzero().item()
                offset = 1 if isinstance(self.loss, DISTRIBUTION_LOSSES) else 0
                return idx_quantile + offset
            except:
                raise ValueError("Model was not trained with a median quantile.")
        return None

    def training_step(self, batch, batch_idx):
        # Set horizon to h_train in case of recurrent model to speed up training
        if self.RECURRENT:
            self.h = self.h_train

        # windows: [Ws, L + h, C, n_series] or [Ws, L + h, C]
        y_idx = batch["y_idx"]

        temporal_cols = batch["temporal_cols"]
        windows_temporal, static, static_cols = self._create_windows(
            batch, step="train"
        )
        windows = self._sample_windows(
            windows_temporal, static, static_cols, temporal_cols, step="train"
        )
        original_outsample_y = torch.clone(
            windows["temporal"][:, self.input_size :, y_idx]
        )
        windows = self._normalization(windows=windows, y_idx=y_idx)

        # Parse windows
        (
            insample_y,
            insample_mask,
            outsample_y,
            outsample_mask,
            hist_exog,
            futr_exog,
            stat_exog,
        ) = self._parse_windows(batch, windows)

        windows_batch = dict(
            insample_y=insample_y,  # [Ws, L, n_series]
            insample_mask=insample_mask,  # [Ws, L, n_series]
            futr_exog=futr_exog,  # univariate: [Ws, L, F]; multivariate: [Ws, F, L, n_series]
            hist_exog=hist_exog,  # univariate: [Ws, L, X]; multivariate: [Ws, X, L, n_series]
            stat_exog=stat_exog,
        )  # univariate: [Ws, S]; multivariate: [n_series, S]

        # Model Predictions
        output = self(windows_batch)
        output = self.loss.domain_map(output)

        if self.loss.is_distribution_output:
            y_loc, y_scale = self._get_loc_scale(y_idx)
            outsample_y = original_outsample_y
            distr_args = self.loss.scale_decouple(
                output=output, loc=y_loc, scale=y_scale
            )
            loss = self.loss(y=outsample_y, distr_args=distr_args, mask=outsample_mask)
        else:
            loss = self.loss(
                y=outsample_y, y_hat=output, y_insample=insample_y, mask=outsample_mask
            )

        if torch.isnan(loss):
            print("Model Parameters", self.hparams)
            print("insample_y", torch.isnan(insample_y).sum())
            print("outsample_y", torch.isnan(outsample_y).sum())
            raise Exception("Loss is NaN, training stopped.")

        train_loss_log = loss.detach().item()
        self.log(
            "train_loss",
            train_loss_log,
            batch_size=outsample_y.size(0),
            prog_bar=True,
            on_epoch=True,
        )
        self.train_trajectories.append((self.global_step, train_loss_log))

        self.h = self.horizon_backup

        return loss

    def validation_step(self, batch, batch_idx):
        if self.val_size == 0:
            return np.nan

        temporal_cols = batch["temporal_cols"]
        windows_temporal, static, static_cols = self._create_windows(batch, step="val")
        n_windows = len(windows_temporal)
        y_idx = batch["y_idx"]

        # Number of windows in batch
        windows_batch_size = self.inference_windows_batch_size
        if windows_batch_size < 0:
            windows_batch_size = n_windows
        n_batches = int(np.ceil(n_windows / windows_batch_size))

        valid_losses = []
        batch_sizes = []
        for i in range(n_batches):
            # Create and normalize windows [Ws, L + h, C, n_series]
            w_idxs = np.arange(
                i * windows_batch_size, min((i + 1) * windows_batch_size, n_windows)
            )
            windows = self._sample_windows(
                windows_temporal,
                static,
                static_cols,
                temporal_cols,
                step="val",
                w_idxs=w_idxs,
            )
            original_outsample_y = torch.clone(
                windows["temporal"][:, self.input_size :, y_idx]
            )

            windows = self._normalization(windows=windows, y_idx=y_idx)

            # Parse windows
            (
                insample_y,
                insample_mask,
                _,
                outsample_mask,
                hist_exog,
                futr_exog,
                stat_exog,
            ) = self._parse_windows(batch, windows)

            if self.RECURRENT:
                output_batch = self._validate_step_recurrent_batch(
                    insample_y=insample_y,
                    insample_mask=insample_mask,
                    futr_exog=futr_exog,
                    hist_exog=hist_exog,
                    stat_exog=stat_exog,
                    y_idx=y_idx,
                )
            else:
                windows_batch = dict(
                    insample_y=insample_y,  # [Ws, L, n_series]
                    insample_mask=insample_mask,  # [Ws, L, n_series]
                    futr_exog=futr_exog,  # univariate: [Ws, L, F]; multivariate: [Ws, F, L, n_series]
                    hist_exog=hist_exog,  # univariate: [Ws, L, X]; multivariate: [Ws, X, L, n_series]
                    stat_exog=stat_exog,
                )  # univariate: [Ws, S]; multivariate: [n_series, S]

                # Model Predictions
                output_batch = self(windows_batch)

            output_batch = self.loss.domain_map(output_batch)
            valid_loss_batch = self._compute_valid_loss(
                insample_y=insample_y,
                outsample_y=original_outsample_y,
                output=output_batch,
                outsample_mask=outsample_mask,
                y_idx=batch["y_idx"],
            )
            valid_losses.append(valid_loss_batch)
            batch_sizes.append(len(output_batch))

        valid_loss = torch.stack(valid_losses)
        batch_sizes = torch.tensor(batch_sizes, device=valid_loss.device)
        batch_size = torch.sum(batch_sizes)
        valid_loss = torch.sum(valid_loss * batch_sizes) / batch_size

        if torch.isnan(valid_loss):
            raise Exception("Loss is NaN, training stopped.")

        valid_loss_log = valid_loss.detach()
        self.log(
            "valid_loss",
            valid_loss_log.item(),
            batch_size=batch_size,
            prog_bar=True,
            on_epoch=True,
        )
        self.validation_step_outputs.append(valid_loss_log)
        return valid_loss

    def predict_step(self, batch, batch_idx):
        if self.RECURRENT:
            return self._predict_step_recurrent(batch, batch_idx)
        else:
            return self._predict_step_direct(
                batch, batch_idx, recursive=self.n_predicts > 1
            )

    def fit(
        self,
        dataset,
        val_size=0,
        test_size=0,
        random_seed=None,
        distributed_config=None,
    ):
        """Fit.

        The `fit` method, optimizes the neural network's weights using the
        initialization parameters (`learning_rate`, `windows_batch_size`, ...)
        and the `loss` function as defined during the initialization.
        Within `fit` we use a PyTorch Lightning `Trainer` that
        inherits the initialization's `self.trainer_kwargs`, to customize
        its inputs, see [PL's trainer arguments](https://pytorch-lightning.readthedocs.io/en/stable/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer).

        The method is designed to be compatible with SKLearn-like classes
        and in particular to be compatible with the StatsForecast library.

        By default the `model` is not saving training checkpoints to protect
        disk memory, to get them change `enable_checkpointing=True` in `__init__`.

        Args:
            dataset (TimeSeriesDataset): NeuralForecast's `TimeSeriesDataset`, see [documentation](./tsdataset).
            val_size (int): Validation size for temporal cross-validation.
            random_seed (int): Random seed for pytorch initializer and numpy generators, overwrites model.__init__'s.
            test_size (int): Test size for temporal cross-validation.

        Returns:
            None
        """
        return self._fit(
            dataset=dataset,
            batch_size=self.batch_size,
            valid_batch_size=self.valid_batch_size,
            val_size=val_size,
            test_size=test_size,
            random_seed=random_seed,
            distributed_config=distributed_config,
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
        """Predict.

        Neural network prediction with PL's `Trainer` execution of `predict_step`.

        Args:
            dataset (TimeSeriesDataset): NeuralForecast's `TimeSeriesDataset`, see [documentation](./tsdataset).
            test_size (int): Test size for temporal cross-validation.
            step_size (int): Step size between each window.
            random_seed (int): Random seed for pytorch initializer and numpy generators, overwrites model.__init__'s.
            quantiles (list): Target quantiles to predict.
            h (int): Prediction horizon, if None, uses the model's fitted horizon. Defaults to None.
            explainer_config (dict): configuration for explanations.
            **data_module_kwargs (dict): PL's TimeSeriesDataModule args, see [documentation](https://pytorch-lightning.readthedocs.io/en/1.6.1/extensions/datamodules.html#using-a-datamodule).

        Returns:
            None
        """

        self._check_exog(dataset)
        self._restart_seed(random_seed)
        if "quantile" in data_module_kwargs:
            warnings.warn(
                "The 'quantile' argument will be deprecated, use 'quantiles' instead."
            )
            if quantiles is not None:
                raise ValueError("You can't specify quantile and quantiles.")
            quantiles = [data_module_kwargs.pop("quantile")]
        self._set_quantiles(quantiles)

        self.predict_step_size = step_size
        self.decompose_forecast = False

        # Protect when case of multiple gpu. PL does not support return preds with multiple gpu.
        pred_trainer_kwargs = self.trainer_kwargs.copy()
        if (pred_trainer_kwargs.get("accelerator", None) == "gpu") and (
            torch.cuda.device_count() > 1
        ):
            pred_trainer_kwargs["devices"] = [0]
            pred_trainer_kwargs["strategy"] = "auto"

        # Determine the number of predictions to make in case h > self.h
        if h is None:
            self.predict_horizon = self.horizon_backup
        else:
            self.predict_horizon = h

        self.n_predicts = 1
        if h is not None and h > self.h:
            if not self.RECURRENT:
                self.n_predicts = math.ceil(h / self.h)
                assert (
                    self.test_size > self.h
                ), f"Test size should be larger than horizon h={self.h} for direct recursive prediction."
            else:
                self.h = h

        datamodule = TimeSeriesDataModule(
            dataset=dataset,
            valid_batch_size=self.valid_batch_size,
            **data_module_kwargs,
        )

        # We need to re-enable grad for explanations
        self.explain = explainer_config is not None
        if self.explain:
            pred_trainer_kwargs["inference_mode"] = False
            self.explainer_config = explainer_config
            trainer = pl.Trainer(**pred_trainer_kwargs)
            out = trainer.predict(self, datamodule=datamodule)
            fcsts = []
            insample_explanations = []
            futr_exog_explanations = []
            hist_exog_explanations = []
            stat_exog_explanations = []
            baseline_predictions = []
            for tensors in out:
                (
                    fcst,
                    insample_explanation,
                    futr_exog_explanation,
                    hist_exog_explanation,
                    stat_exog_explanation,
                    baseline_prediction
                ) = tensors
                fcsts.append(fcst)
                insample_explanations.append(insample_explanation)
                if self.futr_exog_list:
                    futr_exog_explanations.append(futr_exog_explanation)
                if self.hist_exog_list:
                    hist_exog_explanations.append(hist_exog_explanation)
                if self.stat_exog_list:
                    stat_exog_explanations.append(stat_exog_explanation)
                baseline_predictions.append(baseline_prediction)

            fcsts = torch.vstack(fcsts)
            insample_explanations = torch.vstack(insample_explanations)
            if futr_exog_explanations:
                futr_exog_explanations = torch.vstack(futr_exog_explanations)
            if hist_exog_explanations:
                hist_exog_explanations = torch.vstack(hist_exog_explanations)
            if stat_exog_explanations:
                stat_exog_explanations = torch.vstack(stat_exog_explanations)
            if baseline_predictions and baseline_predictions[0] is not None:
                baseline_predictions = torch.vstack(baseline_predictions)
            else:
                baseline_predictions = None
            self.explanations = {
                'insample_explanations': insample_explanations,
                'futr_exog_explanations': futr_exog_explanations if futr_exog_explanations is not None else None,
                'hist_exog_explanations': hist_exog_explanations if hist_exog_explanations is not None else None,
                'stat_exog_explanations': stat_exog_explanations if stat_exog_explanations is not None else None,
                'baseline_predictions': baseline_predictions
            }
        else:
            trainer = pl.Trainer(**pred_trainer_kwargs)
            fcsts = trainer.predict(self, datamodule=datamodule)
            fcsts = torch.vstack(fcsts)
            self.explanations = None
            if h is not None:
                fcsts = fcsts[:, :h]

        if self.MULTIVARIATE:
            # [B, h, n_series (, Q)] -> [n_series, B, h (, Q)]
            fcsts = fcsts.swapaxes(0, 2)
            fcsts = fcsts.swapaxes(1, 2)

        fcsts = tensor_to_numpy(fcsts).flatten()
        fcsts = fcsts.reshape(-1, len(self.loss.output_names))

        # Reset n_predicts
        self.n_predicts = 1
        self.h = self.horizon_backup
        self.predict_horizon = self.horizon_backup

        return fcsts

    def decompose(
        self,
        dataset,
        step_size=1,
        random_seed=None,
        quantiles=None,
        **data_module_kwargs,
    ):
        """Decompose Predictions.

        Decompose the predictions through the network's layers.
        Available methods are `ESRNN`, `NHITS`, `NBEATS`, and `NBEATSx`.

        Args:
            dataset (TimeSeriesDataset): NeuralForecast's `TimeSeriesDataset`, see [documentation here](./tsdataset).
            step_size (int): Step size between each window of temporal data.
            random_seed (int): Random seed for pytorch initializer and numpy generators, overwrites model.__init__'s.
            quantiles (list): Target quantiles to predict.
            **data_module_kwargs (dict): PL's TimeSeriesDataModule args, see [documentation](https://pytorch-lightning.readthedocs.io/en/1.6.1/extensions/datamodules.html#using-a-datamodule).

        Returns:
            None
        """
        # Restart random seed
        if random_seed is None:
            random_seed = self.random_seed
        torch.manual_seed(random_seed)
        self._set_quantiles(quantiles)

        self.predict_step_size = step_size
        self.decompose_forecast = True
        datamodule = TimeSeriesDataModule(
            dataset=dataset,
            valid_batch_size=self.valid_batch_size,
            **data_module_kwargs,
        )
        trainer = pl.Trainer(**self.trainer_kwargs)
        fcsts = trainer.predict(self, datamodule=datamodule)
        self.decompose_forecast = False  # Default decomposition back to false
        fcsts = torch.vstack(fcsts)
        return tensor_to_numpy(fcsts)
    
    def _predict_step_wrapper(
        self,
        insample_y,
        insample_mask,
        futr_exog,
        hist_exog,
        stat_exog,
        y_idx,
        output_horizon,
        output_series,
        output_index,
    ):
        """Forward pass for tensorized inputs."""

        # Dumb trick to ensure that insample_mask is used to produce the output for calculating explanations.
        insample_y = insample_y - insample_mask
        insample_y = insample_y + insample_mask
        if self.RECURRENT:
            y_hat = self._predict_step_recurrent_batch(
                insample_y=insample_y,
                insample_mask=insample_mask,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
                y_idx=y_idx,
            )
        else:
            y_hat = self._predict_step_direct_batch(
                insample_y=insample_y,
                insample_mask=insample_mask,
                futr_exog=futr_exog,
                hist_exog=hist_exog,
                stat_exog=stat_exog,
                y_idx=y_idx,
            )
        if y_hat.ndim == 3:
            y_hat = y_hat.unsqueeze(-1)  # Add output dimension if needed

        return y_hat[:, output_horizon, output_series, output_index]

    def _explain_batch(
        self,
        insample_y,
        insample_mask,
        futr_exog,
        hist_exog,
        stat_exog,
        y_idx,
        y_hat_shape,
        recursive_step=0,
    ):
        add_dim = False
        if len(y_hat_shape) == 3:
            y_hat_shape = y_hat_shape + (1,)
            add_dim = True
        
        # Determine which horizons to explain in this recursive step
        step_start = recursive_step * self.h
        step_end = min((recursive_step + 1) * self.h, self.predict_horizon)
        
        # Get the horizons to explain
        all_horizons = self.explainer_config.get("horizons", list(range(self.predict_horizon)))
        
        # Filter to only horizons in this recursive step
        horizons_to_explain = [h for h in all_horizons if step_start <= h < step_end]
        
        # Convert to local horizon indices
        local_horizons = [h - step_start for h in horizons_to_explain]
        
        if not local_horizons:
            empty_shape = list(y_hat_shape)
            empty_shape[1] = 0  # No horizons
            empty_shape[3] = len(self.explainer_config.get("output_index", list(range(y_hat_shape[-1]))))
            
            # Insample explanations
            insample_explanations = torch.empty(
                size=(*empty_shape, insample_y.shape[1], 2),
                device=insample_y.device,
                dtype=insample_y.dtype,
            )
            
            # Future exogenous explanations
            futr_exog_explanations = None
            if futr_exog is not None:
                if futr_exog.ndim == 3:
                    futr_exog_explanations = torch.empty(
                        size=(*empty_shape, futr_exog.shape[1], futr_exog.shape[2]),
                        device=futr_exog.device,
                        dtype=futr_exog.dtype,
                    )
                else:
                    futr_exog_explanations = torch.empty(
                        size=(*empty_shape, futr_exog.shape[2], futr_exog.shape[1]),
                        device=futr_exog.device,
                        dtype=futr_exog.dtype,
                    )
            
            # Historical exogenous explanations
            hist_exog_explanations = None
            if hist_exog is not None:
                if hist_exog.ndim == 3:
                    hist_exog_explanations = torch.empty(
                        size=(*empty_shape, hist_exog.shape[1], hist_exog.shape[2]),
                        device=hist_exog.device,
                        dtype=hist_exog.dtype,
                    )
                else:
                    hist_exog_explanations = torch.empty(
                        size=(*empty_shape, hist_exog.shape[2], hist_exog.shape[1]),
                        device=hist_exog.device,
                        dtype=hist_exog.dtype,
                    )
            
            # Static exogenous explanations
            stat_exog_explanations = None
            if stat_exog is not None:
                stat_exog_explanations = torch.empty(
                    size=(*empty_shape, stat_exog.shape[1]),
                    device=stat_exog.device,
                    dtype=stat_exog.dtype,
                )
            
            # Baseline predictions
            baseline_predictions = None
            
            return (
                insample_explanations,
                futr_exog_explanations,
                hist_exog_explanations,
                stat_exog_explanations,
                baseline_predictions
            )
        
        # Attribute the input
        series = list(range(self.n_series))
        output_index = self.explainer_config.get(
            "output_index", list(range(y_hat_shape[-1]))
        )

        # Start with required inputs
        insample_y.requires_grad_()
        insample_mask.requires_grad_()
        input_batch = (insample_y, insample_mask)
        param_positions = {"insample_y": 0, "insample_mask": 1}
        
        shape = list(y_hat_shape)
        shape[1] = len(local_horizons)
        shape[3] = len(output_index)            
        insample_explanations = torch.empty(
            size=(*shape, insample_y.shape[1], 2),
            device=insample_y.device,
            dtype=insample_y.dtype,
        )

        # Keep track of which parameter is at which position in input_batch
        pos = 2  # Starting position after insample_y and insample_mask

        # Add optional parameters and track their positions
        futr_exog_explanations = None
        if futr_exog is not None:
            futr_exog.requires_grad_()
            input_batch = input_batch + (futr_exog,)
            param_positions["futr_exog"] = pos
            pos += 1
            if futr_exog.ndim == 3:
                futr_exog_explanations = torch.empty(
                    size=(*shape, futr_exog.shape[1], futr_exog.shape[2]),
                    device=futr_exog.device,
                    dtype=futr_exog.dtype,
                )
            else:
                futr_exog_explanations = torch.empty(
                    size=(*shape, futr_exog.shape[2], futr_exog.shape[1]),
                    device=futr_exog.device,
                    dtype=futr_exog.dtype,
                )

        hist_exog_explanations = None
        if hist_exog is not None:
            hist_exog.requires_grad_()
            input_batch = input_batch + (hist_exog,)
            param_positions["hist_exog"] = pos
            pos += 1
            if hist_exog.ndim == 3:
                hist_exog_explanations = torch.empty(
                    size=(*shape, hist_exog.shape[1], hist_exog.shape[2]),
                    device=hist_exog.device,
                    dtype=hist_exog.dtype,
                )
            else:
                hist_exog_explanations = torch.empty(
                    size=(*shape, hist_exog.shape[2], hist_exog.shape[1]),
                    device=hist_exog.device,
                    dtype=hist_exog.dtype,
                )

        stat_exog_explanations = None
        if stat_exog is not None:
            stat_exog.requires_grad_()
            input_batch = input_batch + (stat_exog,)
            param_positions["stat_exog"] = pos
            pos += 1
            stat_exog_explanations = torch.empty(
                size=(*shape, stat_exog.shape[1]),
                device=stat_exog.device,
                dtype=stat_exog.dtype,
            )

        # Loop over horizons, series and output_indices
        for i, local_horizon in enumerate(local_horizons):
            for j, series_idx in enumerate(series):
                for k, output_idx in enumerate(output_index):
                    forward_fn = lambda *args: self._predict_step_wrapper(
                        insample_y=args[param_positions["insample_y"]],
                        insample_mask=args[param_positions["insample_mask"]],
                        futr_exog=(
                            args[param_positions["futr_exog"]]
                            if "futr_exog" in param_positions
                            else None
                        ),
                        hist_exog=(
                            args[param_positions["hist_exog"]]
                            if "hist_exog" in param_positions
                            else None
                        ),
                        stat_exog=(
                            args[param_positions["stat_exog"]]
                            if "stat_exog" in param_positions
                            else None
                        ),
                        y_idx=y_idx,
                        output_horizon=local_horizon,
                        output_series=series_idx,
                        output_index=output_idx,
                    )
                    attributor = self.explainer_config["explainer"](forward_fn)
                    attributions = attributor.attribute(input_batch)

                    insample_attr = attributions[0].squeeze(-1)
                    insample_explanations[:, i, j, k, :, 0] = insample_attr

                    insample_mask_attr = attributions[1].squeeze(-1)
                    insample_explanations[:, i, j, k, :, 1] = insample_mask_attr

                    if "futr_exog" in param_positions:
                        futr_exog_attr = attributions[param_positions["futr_exog"]]
                        futr_exog_explanations[:, i, j, k] = futr_exog_attr

                    if "hist_exog" in param_positions:
                        hist_exog_attr = attributions[param_positions["hist_exog"]]
                        hist_exog_explanations[:, i, j, k] = hist_exog_attr

                    if "stat_exog" in param_positions:
                        stat_exog_attr = attributions[param_positions["stat_exog"]]
                        stat_exog_explanations[:, i, j, k] = stat_exog_attr

        explainer_class = self.explainer_config["explainer"]
        explainer_name = explainer_class.__name__ if hasattr(explainer_class, '__name__') else str(explainer_class)
        additive_explainers = ExplainerEnum.AdditiveExplainers

        if explainer_name in additive_explainers:
            if self.RECURRENT:
                baseline_predictions = self._predict_step_recurrent_batch(
                    insample_y=insample_y * 0,
                    insample_mask=insample_mask * 0,
                    futr_exog=futr_exog * 0 if futr_exog is not None else None,
                    hist_exog=hist_exog * 0 if hist_exog is not None else None,
                    stat_exog=stat_exog * 0 if stat_exog is not None else None,
                    y_idx=y_idx,
                )                
            else:
                baseline_predictions = self._predict_step_direct_batch(
                    insample_y=insample_y * 0,
                    insample_mask=insample_mask * 0,
                    futr_exog=futr_exog * 0 if futr_exog is not None else None,
                    hist_exog=hist_exog * 0 if hist_exog is not None else None,
                    stat_exog=stat_exog * 0 if stat_exog is not None else None,
                    y_idx=y_idx,
                )
            if add_dim:
                baseline_predictions = baseline_predictions.unsqueeze(-1)
            
            baseline_predictions = baseline_predictions.index_select(1, torch.tensor(local_horizons, device=baseline_predictions.device))
            baseline_predictions = baseline_predictions.index_select(2, torch.tensor(series, device=baseline_predictions.device))
            baseline_predictions = baseline_predictions.index_select(3, torch.tensor(output_index, device=baseline_predictions.device))
        else:
            baseline_predictions = None

        return (
            insample_explanations,
            futr_exog_explanations,
            hist_exog_explanations,
            stat_exog_explanations,
            baseline_predictions
        )
