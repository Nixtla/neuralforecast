"""UniTime context-conditioned forecasting adapter."""

from pathlib import Path
from types import SimpleNamespace

import torch

from ._context_source import official_module
from ._context_utils import (
    _context_indices,
    _context_schema,
    _contexts,
    _local_path,
    _options,
    _positive,
)
from ._exogenous import ExogenousModel

__all__ = ["UniTime"]


class UniTime(ExogenousModel):
    """Official UniTime with per-series external domain descriptions.

    One static integer context_id selects a string in contexts. A local GPT-2
    tokenizer/checkpoint is required. NF trains the forecast horizon, not the
    original multi-domain masked reconstruction curriculum.
    """

    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False
    EXOGENOUS_STAT = True

    def __init__(
        self,
        h,
        input_size,
        source_dir,
        backbone_path,
        contexts,
        patch_len=16,
        gpt_layers=2,
        decoder_layers=1,
        max_tokens=128,
        dropout=0.1,
        stat_exog_list=None,
        **kwargs,
    ):
        _positive(
            patch_len=patch_len,
            gpt_layers=gpt_layers,
            decoder_layers=decoder_layers,
            max_tokens=max_tokens,
        )
        if input_size % patch_len or not 0 <= dropout < 1:
            raise ValueError(
                "input_size must divide into patches; "
                "dropout must be in [0,1)."
            )
        self.contexts = _contexts(contexts)
        super().__init__(
            h=h,
            input_size=input_size,
            stat_exog_list=stat_exog_list,
            **_options(kwargs),
        )
        _context_schema(self)
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.backbone_path = _local_path(backbone_path)
        self.patch_len = patch_len
        args = SimpleNamespace(
            mask_rate=0.0,
            patch_len=patch_len,
            max_token_num=max_tokens,
            max_backcast_len=input_size,
            max_forecast_len=h,
            logger=None,
            model_path=self.backbone_path,
            lm_layer_num=gpt_layers,
            lm_ft_type="full",
            ts_embed_dropout=dropout,
            dec_trans_layer_num=decoder_layers,
            dec_head_dropout=dropout,
        )
        self.model = official_module(
            self.source_dir,
            "UniTime",
        ).UniTime(args)
        if gpt_layers > self.model.backbone.config.n_layer:
            raise ValueError("gpt_layers exceeds the checkpoint depth.")
        limit = min(max_tokens, self.model.backbone.config.n_positions)
        if max_tokens > self.model.backbone.config.n_positions:
            raise ValueError(
                "max_tokens exceeds GPT-2 position capacity."
            )
        for context in self.contexts:
            if (
                len(self.model.tokenizer.encode(context))
                + input_size // patch_len
                > limit
            ):
                raise ValueError(
                    "Description plus target patches exceeds token capacity; "
                    "shorten it explicitly."
                )

    def forward(self, windows_batch):
        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        ids = _context_indices(self, windows_batch, y)
        outputs = []
        for i, idx in enumerate(ids.tolist()):
            info = (
                idx,
                self.input_size,
                self.patch_len,
                self.contexts[idx],
            )
            output = self.model(
                info,
                y[i : i + 1].clone(),
                mask[i : i + 1].to(y.dtype),
            )
            outputs.append(
                output[
                    :,
                    self.input_size : self.input_size + self.h,
                ]
            )
        return self._point_output(torch.cat(outputs), y)
