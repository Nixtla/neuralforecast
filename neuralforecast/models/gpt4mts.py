"""GPT4MTS context-conditioned forecasting adapter."""

from pathlib import Path
from types import SimpleNamespace

import torch

from ._context_source import official_module
from ._context_utils import _local_path, _options, _positive
from ._exogenous import ExogenousModel

__all__ = ["GPT4MTS"]


class GPT4MTS(ExogenousModel):
    """Official GPT4MTS historical text-embedding fusion with a GPT-2 backbone.

    An explicit local GPT-2 checkpoint is optional. With backbone_path=None the
    original GPT-2 architecture is initialized randomly and must be trained.
    Historical text coordinates must match the selected language encoder width.
    With a checkpoint, n_heads=None uses its configured attention head count.
    """

    EXOGENOUS_FUTR = False

    def __init__(
        self,
        h,
        input_size,
        source_dir,
        backbone_path=None,
        gpt_layers=2,
        n_heads=12,
        patch_len=8,
        stride=4,
        freeze_backbone=False,
        **kwargs,
    ):
        _positive(
            gpt_layers=gpt_layers,
            patch_len=patch_len,
            stride=stride,
        )
        if n_heads is not None:
            _positive(n_heads=n_heads)
        if input_size < patch_len:
            raise ValueError("input_size must be at least patch_len.")
        super().__init__(h=h, input_size=input_size, **_options(kwargs))
        width = self.hist_exog_size
        if width < 2 or (n_heads is not None and width % n_heads):
            raise ValueError(
                "Text embedding width must be >=2 and divisible by n_heads."
            )
        if backbone_path is not None:
            from transformers import GPT2Config

            backbone_path = _local_path(backbone_path)
            cfg = GPT2Config.from_pretrained(
                backbone_path,
                local_files_only=True,
            )
            if cfg.n_embd != width or gpt_layers > cfg.n_layer:
                raise ValueError(
                    "GPT-2 width/layer count does not match "
                    "the requested text configuration."
                )
            config = cfg.to_dict()
            n_heads = cfg.n_head
        else:
            if freeze_backbone:
                raise ValueError(
                    "Cannot freeze a randomly initialized language backbone."
                )
            config = dict(n_embd=width, n_layer=gpt_layers, n_head=n_heads)
        _positive(n_heads=n_heads)
        if width < 2 or width % n_heads:
            raise ValueError(
                "Text embedding width must be >=2 and divisible by n_heads."
            )
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.backbone_path = backbone_path
        args = SimpleNamespace(
            is_gpt=True,
            revin=False,
            patch_size=patch_len,
            pretrain=backbone_path is not None,
            stride=stride,
            seq_len=input_size,
            pred_len=h,
            d_model=width,
            gpt_layers=gpt_layers,
            freeze=freeze_backbone,
            backbone_path=backbone_path,
            backbone_config=config,
        )
        self.model = official_module(
            self.source_dir,
            "GPT4MTS",
        ).GPT4MTS(args, torch.device("cpu"))
        token_count = 2 * ((input_size - patch_len) // stride + 2)
        if token_count > self.model.gpt2.config.n_positions:
            raise ValueError(
                "Combined text/target patches exceed GPT-2 position capacity."
            )

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        return self._point_output(self.model(y, 0, hist), y)
