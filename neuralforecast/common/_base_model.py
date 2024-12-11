# AUTOGENERATED! DO NOT EDIT! File to edit: ../../nbs/common.base_model.ipynb.

# %% auto 0
__all__ = ['DistributedConfig', 'BaseModel']

# %% ../../nbs/common.base_model.ipynb 2
import inspect
import random
import warnings
from contextlib import contextmanager
from dataclasses import dataclass

import fsspec
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from neuralforecast.tsdataset import (
    TimeSeriesDataModule,
    BaseTimeSeriesDataset,
    _DistributedTimeSeriesDataModule,
)
from ..losses.pytorch import IQLoss

# %% ../../nbs/common.base_model.ipynb 3
@dataclass
class DistributedConfig:
    partitions_path: str
    num_nodes: int
    devices: int

# %% ../../nbs/common.base_model.ipynb 4
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

# %% ../../nbs/common.base_model.ipynb 5
class BaseModel(pl.LightningModule):
    EXOGENOUS_FUTR = True
    EXOGENOUS_HIST = True
    EXOGENOUS_STAT = True

    def __init__(
        self,
        random_seed,
        loss,
        valid_loss,
        futr_exog_list,
        hist_exog_list,
        stat_exog_list,
        max_steps,
        early_stop_patience_steps,
        **trainer_kwargs,
    ):
        super().__init__()
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
        self.train_trajectories = []
        self.valid_trajectories = []

        # customized by set_configure_optimizers()
        self.config_optimizers = None

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

        # Implicit Quantile Loss
        if isinstance(self.loss, IQLoss):
            if not isinstance(self.valid_loss, IQLoss):
                raise Exception(
                    "Please set valid_loss to IQLoss() when training with IQLoss"
                )
        if isinstance(self.valid_loss, IQLoss) and not isinstance(self.loss, IQLoss):
            raise Exception("Please set loss to IQLoss() when validating with IQLoss")

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

        self.trainer_kwargs = trainer_kwargs

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

    def _set_quantile_for_iqloss(self, **data_module_kwargs):
        if "quantile" in data_module_kwargs:
            if not isinstance(self.loss, IQLoss):
                raise Exception(
                    "Please train with loss=IQLoss() to make use of the quantile argument."
                )
            else:
                self.quantile = data_module_kwargs["quantile"]
                data_module_kwargs.pop("quantile")
                self.loss.update_quantile(q=self.quantile)
        elif isinstance(self.loss, IQLoss):
            self.quantile = 0.5
            self.loss.update_quantile(q=self.quantile)

        return data_module_kwargs

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

        if self.num_workers_loader != 0:  # value is not at its default
            warnings.warn(
                "The `num_workers_loader` argument is deprecated and will be removed in a future version. "
                "Please provide num_workers through `dataloader_kwargs`, e.g. "
                f"`dataloader_kwargs={{'num_workers': {self.num_workers_loader}}}`",
                category=FutureWarning,
            )
            dataloader_kwargs["num_workers"] = self.num_workers_loader

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
        if self.config_optimizers is not None:
            # return the customized optimizer settings if specified
            return self.config_optimizers

        # default choice
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, step_size=self.lr_decay_steps, gamma=0.5
            ),
            "frequency": 1,
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def set_configure_optimizers(
        self,
        optimizer=None,
        scheduler=None,
        interval="step",
        frequency=1,
        monitor="val_loss",
        strict=True,
        name=None,
    ):
        """Helper function to customize the lr_scheduler_config as detailed in
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers

        Calling set_configure_optimizers() with valid `optimizer`, `scheduler` shall modify the returned
        dictionary of key='optimizer', key='lr_scheduler' in configure_optimizers().
        Note that the default choice of `interval` in set_configure_optiizers() is 'step',
        which differs from the choice of 'epoch' used in lightning_module.
        """
        lr_scheduler_config = {
            "interval": interval,
            "frequency": frequency,
            "monitor": monitor,
            "strict": strict,
            "name": name,
        }

        if scheduler is not None and optimizer is not None:
            if not isinstance(scheduler, torch.optim.lr_scheduler.LRScheduler):
                raise TypeError(
                    "scheduler is not a valid instance of torch.optim.lr_scheduler.LRScheduler"
                )
            if not isinstance(optimizer, torch.optim.Optimizer):
                raise TypeError(
                    "optimizer is not a valid instance of torch.optim.Optimizer"
                )

            lr_scheduler_config["scheduler"] = scheduler
            self.config_optimizers = {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }
        else:
            # falls back to default option as specified in configure_optimizers()
            self.config_optimizers = None

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
        with fsspec.open(path, "rb") as f:
            content = torch.load(f, **kwargs)
        with _disable_torch_init():
            model = cls(**content["hyper_parameters"])
        if "assign" in inspect.signature(model.load_state_dict).parameters:
            model.load_state_dict(content["state_dict"], strict=True, assign=True)
        else:  # pytorch<2.1
            model.load_state_dict(content["state_dict"], strict=True)
        return model
