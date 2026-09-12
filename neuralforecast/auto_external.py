"""Automatic HPO wrappers for trainable external forecasting adapters."""

from math import ceil

from ray import tune
from ray.tune.search.basic_variant import BasicVariantGenerator

from .common._base_auto import BaseAuto
from .losses.pytorch import MAE
from .models.apt import APT
from .models.crosslinear import CrossLinear
from .models.dag import DAG
from .models.dualformer import Dualformer
from .models.glaff import GLAFF
from .models.gpt4mts import GPT4MTS
from .models.kite import KITE
from .models.langtime import LangTime
from .models.seesawnet import SeesawNet
from .models.spectf import SpecTF
from .models.tgforecaster import TGForecaster
from .models.tinytimemixer import TinyTimeMixer
from .models.timerxl import TimerXL
from .models.unitime import UniTime
from .models.vot import VoT

__all__ = [
    "AutoCrossLinear",
    "AutoTimerXL",
    "AutoTinyTimeMixer",
    "AutoDAG",
    "AutoKITE",
    "AutoGLAFF",
    "AutoAPT",
    "AutoVoT",
    "AutoGPT4MTS",
    "AutoUniTime",
    "AutoLangTime",
    "AutoSpecTF",
    "AutoTGForecaster",
    "AutoSeesawNet",
    "AutoDualformer",
]


def _round_up(value, multiple):
    return multiple * ceil(value / multiple)


def _input_sizes(h, multiple=8, minimum=32):
    if not isinstance(h, int) or isinstance(h, bool) or h < 1:
        raise ValueError("h must be a positive integer.")
    raw = (max(minimum, 2 * h), max(2 * minimum, 4 * h), max(4 * minimum, 8 * h))
    return sorted({_round_up(value, multiple) for value in raw})


def _divisors(value, candidates=(1, 2, 4, 8, 16)):
    return [
        candidate
        for candidate in candidates
        if candidate <= value and value % candidate == 0
    ]


def _heads(width, candidates=(1, 2, 4, 8, 12, 16)):
    choices = _divisors(width, candidates)
    if not choices:
        raise ValueError(f"No supported attention head count divides width={width}.")
    return choices


def _training_space(low=1e-5, high=3e-3, scaler="identity"):
    scaler_space = scaler if isinstance(scaler, str) else tune.choice(list(scaler))
    return {
        "learning_rate": tune.loguniform(low, high),
        "max_steps": tune.choice([500, 1000, 2000]),
        "num_lr_decays": tune.choice([0, 1, 3]),
        "batch_size": tune.choice([16, 32, 64]),
        "windows_batch_size": tune.choice([16, 32, 64]),
        "early_stop_patience_steps": 5,
        "val_check_steps": 100,
        "random_seed": 1,
        "scaler_type": scaler_space,
    }


def _merge_fixed(config, fixed):
    if callable(config):
        return lambda trial: {**config(trial), **fixed}
    return {**config, **fixed}


class _ExternalAuto(BaseAuto):
    def __init__(
        self,
        model,
        h,
        default_config,
        fixed=None,
        loss=MAE(),
        valid_loss=None,
        config=None,
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
    ):
        search = default_config if config is None else config
        if config is None and backend == "optuna":
            search = self._ray_config_to_optuna(search)
        search = _merge_fixed(search, fixed or {})
        super().__init__(
            cls_model=model,
            h=h,
            loss=loss,
            valid_loss=valid_loss,
            config=search,
            search_alg=search_alg,
            num_samples=num_samples,
            time_budget=time_budget,
            refit_with_val=refit_with_val,
            verbose=verbose,
            alias=alias,
            backend=backend,
            callbacks=callbacks,
            ray_options=ray_options,
            optuna_options=optuna_options,
        )


def _crosslinear_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=16)),
        "patch_len": tune.choice([4, 8, 16]),
        "hidden_size": tune.choice([32, 64, 128]),
        "d_ff": tune.choice([64, 128, 256]),
        "alpha": tune.uniform(0.2, 0.8),
        "beta": tune.uniform(0.2, 0.8),
        **_training_space(low=1e-4, high=1e-2, scaler=("identity", "robust")),
    }


class AutoCrossLinear(_ExternalAuto):
    def __init__(self, h, hist_exog_list=None, config=None, **kwargs):
        super().__init__(
            CrossLinear,
            h,
            _crosslinear_space(h),
            {"hist_exog_list": hist_exog_list},
            config=config,
            **kwargs,
        )


def _timerxl_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=16)),
        "patch_len": tune.choice([4, 8, 16]),
        "hidden_size": tune.choice([64, 128, 256]),
        "n_heads": tune.choice([2, 4, 8]),
        "n_layers": tune.choice([1, 2, 3]),
        "d_ff": tune.choice([128, 256, 512]),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        "use_norm": tune.choice([True, False]),
        **_training_space(low=1e-5, high=3e-3, scaler="identity"),
    }


class AutoTimerXL(_ExternalAuto):
    def __init__(self, h, hist_exog_list=None, config=None, **kwargs):
        super().__init__(
            TimerXL,
            h,
            _timerxl_space(h),
            {"hist_exog_list": hist_exog_list},
            config=config,
            **kwargs,
        )


def _tinytimemixer_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=16)),
        "patch_len": tune.choice([4, 8, 16]),
        "hidden_size": tune.choice([32, 64, 128]),
        "n_layers": tune.choice([2, 3, 4]),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        **_training_space(low=1e-5, high=3e-3, scaler="identity"),
    }


class AutoTinyTimeMixer(_ExternalAuto):
    def __init__(
        self,
        h,
        hist_exog_list=None,
        futr_exog_list=None,
        config=None,
        **kwargs,
    ):
        super().__init__(
            TinyTimeMixer,
            h,
            _tinytimemixer_space(h),
            {
                "hist_exog_list": hist_exog_list,
                "futr_exog_list": futr_exog_list,
            },
            config=config,
            **kwargs,
        )


def _dag_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=16)),
        "hidden_size": tune.choice([64, 128, 256]),
        "n_heads": tune.choice([2, 4, 8]),
        "encoder_layers": tune.choice([1, 2, 3]),
        "patch_len": tune.choice([4, 8, 16]),
        "stride": tune.choice([2, 4, 8]),
        "d_ff": tune.choice([128, 256, 512]),
        "dropout": tune.choice([0.05, 0.1, 0.2]),
        "alpha": tune.choice([0.1, 0.2, 0.5]),
        "beta": tune.choice([0.05, 0.1, 0.2]),
        **_training_space(low=1e-5, high=3e-3, scaler=("identity", "robust")),
    }


class AutoDAG(_ExternalAuto):
    def __init__(self, h, source_dir, futr_exog_list, config=None, **kwargs):
        if not futr_exog_list:
            raise ValueError("AutoDAG requires nonempty futr_exog_list.")
        super().__init__(
            DAG,
            h,
            _dag_space(h),
            {"source_dir": source_dir, "futr_exog_list": futr_exog_list},
            config=config,
            **kwargs,
        )


def _kite_space(h):
    if h < 2:
        raise ValueError("AutoKITE requires h >= 2.")
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=8)),
        "hidden_size": tune.choice([32, 64, 128]),
        "n_heads": tune.choice([2, 4, 8]),
        "depth": tune.choice([1, 2, 3]),
        "num_sampling_steps": tune.choice([10, 20, 40]),
        "num_samples": tune.choice([10, 20, 50]),
        "omega": tune.choice([0.5, 1.0, 2.0]),
        "p_uncond": tune.choice([0.0, 0.1, 0.2]),
        **_training_space(low=1e-5, high=1e-3, scaler=("identity", "robust")),
    }


class AutoKITE(_ExternalAuto):
    def __init__(
        self,
        h,
        source_dir,
        hist_exog_list=None,
        futr_exog_list=None,
        config=None,
        **kwargs,
    ):
        if bool(hist_exog_list) == bool(futr_exog_list):
            raise ValueError(
                "AutoKITE requires exactly one nonempty hist_exog_list "
                "or futr_exog_list."
            )
        super().__init__(
            KITE,
            h,
            _kite_space(h),
            {
                "source_dir": source_dir,
                "hist_exog_list": hist_exog_list,
                "futr_exog_list": futr_exog_list,
            },
            config=config,
            **kwargs,
        )


def _glaff_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=8)),
        "hidden_size": tune.choice([16, 32, 64]),
        "n_heads": tune.choice([2, 4, 8]),
        "encoder_layers": tune.choice([1, 2, 3]),
        "d_ff": tune.choice([32, 64, 128]),
        "dropout": tune.choice([0.05, 0.1, 0.2]),
        "q": tune.choice([0.6, 0.75, 0.9]),
        "moving_avg_window": tune.choice([7, 13, 25]),
        **_training_space(low=1e-5, high=3e-3, scaler="identity"),
    }


class AutoGLAFF(_ExternalAuto):
    def __init__(self, h, source_dir, futr_exog_list, config=None, **kwargs):
        if len(futr_exog_list or []) != 6:
            raise ValueError("AutoGLAFF requires exactly six futr_exog_list columns.")
        super().__init__(
            GLAFF,
            h,
            _glaff_space(h),
            {"source_dir": source_dir, "futr_exog_list": futr_exog_list},
            config=config,
            **kwargs,
        )


def _apt_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=8)),
        "timestamp_dim": tune.choice([8, 16, 32]),
        "timestamp_hidden": tune.choice([16, 32, 64]),
        "num_prototypes": tune.choice([4, 8, 16]),
        "top_k": tune.choice([1, 2, 3, 4]),
        "warmup_steps": tune.choice([0, 5, 10, 20]),
        "station_lambda": tune.loguniform(1e-4, 1e-1),
        "moving_avg_window": tune.choice([7, 13, 25]),
        **_training_space(low=1e-5, high=3e-3, scaler="identity"),
    }


class AutoAPT(_ExternalAuto):
    def __init__(
        self,
        h,
        source_dir,
        futr_exog_list,
        time_of_day_size=24,
        config=None,
        **kwargs,
    ):
        if len(futr_exog_list or []) != 2:
            raise ValueError("AutoAPT requires exactly two futr_exog_list columns.")
        super().__init__(
            APT,
            h,
            _apt_space(h),
            {
                "source_dir": source_dir,
                "futr_exog_list": futr_exog_list,
                "time_of_day_size": time_of_day_size,
            },
            config=config,
            **kwargs,
        )


def _vot_space(h):
    patches = _divisors(h)
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=max(patches))),
        "d_model": tune.choice([64, 128, 256]),
        "n_heads": tune.choice([2, 4, 8]),
        "d_ff": tune.choice([128, 256, 512]),
        "e_layers": tune.choice([1, 2, 3]),
        "patch_len": tune.choice(patches),
        "stride": tune.choice(patches),
        "dropout": tune.choice([0.05, 0.1, 0.2]),
        **_training_space(low=1e-5, high=1e-3, scaler="identity"),
    }


class AutoVoT(_ExternalAuto):
    def __init__(self, h, source_dir, hist_exog_list, config=None, **kwargs):
        width = len(hist_exog_list or [])
        if width < 8 or width % 8:
            raise ValueError(
                "AutoVoT hist_exog_list width must be a positive multiple of eight."
            )
        super().__init__(
            VoT,
            h,
            _vot_space(h),
            {"source_dir": source_dir, "hist_exog_list": hist_exog_list},
            config=config,
            **kwargs,
        )


def _gpt4mts_space(h, width, backbone_path):
    if width < 2:
        raise ValueError("AutoGPT4MTS requires at least two text embedding columns.")
    space = {
        "input_size": tune.choice(_input_sizes(h, multiple=8)),
        "n_heads": tune.choice(_heads(width)),
        "patch_len": tune.choice([4, 8]),
        "stride": tune.choice([2, 4, 8]),
        **_training_space(low=1e-5, high=1e-3, scaler="identity"),
    }
    if backbone_path is None:
        space["gpt_layers"] = tune.choice([1, 2, 4])
        space["freeze_backbone"] = False
    else:
        space["n_heads"] = None  # The checkpoint defines its attention heads.
        space["gpt_layers"] = 2
        space["freeze_backbone"] = tune.choice([False, True])
    return space


class AutoGPT4MTS(_ExternalAuto):
    def __init__(
        self,
        h,
        source_dir,
        hist_exog_list,
        backbone_path=None,
        config=None,
        **kwargs,
    ):
        width = len(hist_exog_list or [])
        super().__init__(
            GPT4MTS,
            h,
            _gpt4mts_space(h, width, backbone_path),
            {
                "source_dir": source_dir,
                "hist_exog_list": hist_exog_list,
                "backbone_path": backbone_path,
            },
            config=config,
            **kwargs,
        )


def _unitime_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=16)),
        "patch_len": tune.choice([8, 16]),
        "gpt_layers": 2,
        "decoder_layers": tune.choice([1, 2]),
        "max_tokens": 128,
        "dropout": tune.choice([0.05, 0.1, 0.2]),
        **_training_space(low=1e-5, high=1e-3, scaler="identity"),
    }


class AutoUniTime(_ExternalAuto):
    def __init__(
        self,
        h,
        source_dir,
        backbone_path,
        contexts,
        stat_exog_list,
        config=None,
        **kwargs,
    ):
        if len(stat_exog_list or []) != 1:
            raise ValueError("AutoUniTime requires one context_id stat_exog column.")
        super().__init__(
            UniTime,
            h,
            _unitime_space(h),
            {
                "source_dir": source_dir,
                "backbone_path": backbone_path,
                "contexts": contexts,
                "stat_exog_list": stat_exog_list,
            },
            config=config,
            **kwargs,
        )


def _langtime_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=16)),
        "patch_len": tune.choice([4, 8, 16]),
        "d_model": tune.choice([32, 64, 128]),
        "n_heads": tune.choice([2, 4, 8]),
        "d_ff": tune.choice([64, 128, 256]),
        "e_layers": tune.choice([1, 2, 3]),
        "dropout": tune.choice([0.05, 0.1, 0.2]),
        **_training_space(low=1e-5, high=1e-3, scaler="identity"),
    }


class AutoLangTime(_ExternalAuto):
    def __init__(
        self,
        h,
        source_dir,
        backbone_path,
        contexts,
        stat_exog_list,
        config=None,
        **kwargs,
    ):
        if len(stat_exog_list or []) != 1:
            raise ValueError("AutoLangTime requires one context_id stat_exog column.")
        super().__init__(
            LangTime,
            h,
            _langtime_space(h),
            {
                "source_dir": source_dir,
                "backbone_path": backbone_path,
                "contexts": contexts,
                "stat_exog_list": stat_exog_list,
            },
            config=config,
            **kwargs,
        )


def _spectf_space(h):
    sizes = [size for size in _input_sizes(h, multiple=8) if size <= 9998]
    if not sizes:
        raise ValueError("AutoSpecTF has no valid input_size <= 9998 for this h.")
    return {
        "input_size": tune.choice(sizes),
        "mm_emb_size": tune.choice([16, 32, 64]),
        "mm_hidden_size": tune.choice([32, 64, 128]),
        "text_emb": tune.choice([4, 6, 8, 12]),
        "dropout": tune.choice([0.0, 0.1, 0.2]),
        "text_dropout": tune.choice([0.0, 0.1, 0.2]),
        **_training_space(low=1e-5, high=1e-3, scaler="identity"),
    }


class AutoSpecTF(_ExternalAuto):
    def __init__(self, h, source_dir, hist_exog_list, config=None, **kwargs):
        if not hist_exog_list:
            raise ValueError("AutoSpecTF requires nonempty hist_exog_list.")
        super().__init__(
            SpecTF,
            h,
            _spectf_space(h),
            {"source_dir": source_dir, "hist_exog_list": hist_exog_list},
            config=config,
            **kwargs,
        )


def _tgforecaster_space(h, text_dim):
    patches = _divisors(h)
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=max(patches))),
        "text_dim": text_dim,
        "n_heads": tune.choice(_heads(text_dim)),
        "encoder_layers": tune.choice([1, 2, 3]),
        "cross_layers": tune.choice([1, 2]),
        "mixer_self_layers": tune.choice([1, 2]),
        "patch_len": tune.choice(patches),
        "dropout": tune.choice([0.05, 0.1, 0.2]),
        **_training_space(low=1e-5, high=1e-3, scaler="identity"),
    }


class AutoTGForecaster(_ExternalAuto):
    def __init__(self, h, source_dir, futr_exog_list, config=None, **kwargs):
        width = len(futr_exog_list or [])
        if width < 4 or width % 2:
            raise ValueError(
                "AutoTGForecaster requires paired news/description embedding columns."
            )
        text_dim = width // 2
        super().__init__(
            TGForecaster,
            h,
            _tgforecaster_space(h, text_dim),
            {
                "source_dir": source_dir,
                "futr_exog_list": futr_exog_list,
                "text_dim": text_dim,
            },
            config=config,
            **kwargs,
        )


def _seesawnet_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=16)),
        "hidden_size": tune.choice([64, 128, 256]),
        "d_ff": tune.choice([128, 256, 512]),
        "n_heads": tune.choice([2, 4, 8]),
        "patch_len": tune.choice([4, 8, 16]),
        "stride": tune.choice([2, 4, 8]),
        "pd_layers": tune.choice([1, 2]),
        "cr_layers": tune.choice([1, 2, 3]),
        "down_sample_rate": tune.choice([0.5, 0.75, 1.0]),
        "dropout": tune.choice([0.05, 0.1, 0.2]),
        **_training_space(low=1e-5, high=3e-3, scaler=("identity", "robust")),
    }


class AutoSeesawNet(_ExternalAuto):
    def __init__(
        self,
        h,
        source_dir,
        hist_exog_list=None,
        config=None,
        **kwargs,
    ):
        super().__init__(
            SeesawNet,
            h,
            _seesawnet_space(h),
            {"source_dir": source_dir, "hist_exog_list": hist_exog_list},
            config=config,
            **kwargs,
        )


def _dualformer_space(h):
    return {
        "input_size": tune.choice(_input_sizes(h, multiple=8, minimum=8)),
        "hidden_size": tune.choice([64, 128, 256]),
        "d_ff": tune.choice([128, 256, 512]),
        "n_heads": tune.choice([2, 4, 8]),
        "e_layers": tune.choice([2, 3, 4]),
        "dropout": tune.choice([0.05, 0.1, 0.2]),
        **_training_space(low=1e-5, high=3e-3, scaler=("identity", "robust")),
    }


class AutoDualformer(_ExternalAuto):
    def __init__(
        self,
        h,
        source_dir,
        hist_exog_list=None,
        config=None,
        **kwargs,
    ):
        super().__init__(
            Dualformer,
            h,
            _dualformer_space(h),
            {"source_dir": source_dir, "hist_exog_list": hist_exog_list},
            config=config,
            **kwargs,
        )
