"""LangTime context-conditioned forecasting adapter."""

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

__all__ = ["LangTime"]


class LangTime(ExogenousModel):
    """Official LangTime GPT-2 supervised forecaster with external instructions.

    One static context_id selects a description. The original patch encoder,
    linear adapter, special-token routing and forecast head execute unchanged;
    PPO/reasoner training and backcast loss are outside this NF integration.
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
        patch_len=8,
        d_model=64,
        n_heads=4,
        d_ff=128,
        e_layers=2,
        dropout=0.1,
        stat_exog_list=None,
        **kwargs,
    ):
        from transformers import GPT2Config, GPT2Tokenizer

        _positive(
            patch_len=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
        )
        if (
            input_size % patch_len
            or d_model % n_heads
            or not 0 <= dropout < 1
        ):
            raise ValueError(
                "Invalid patch/attention/dropout configuration."
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
        cfg = GPT2Config.from_pretrained(
            self.backbone_path,
            local_files_only=True,
        )
        self.tokenizer = GPT2Tokenizer.from_pretrained(
            self.backbone_path,
            local_files_only=True,
        )
        tokens = [
            "<|TS_ENC|>",
            "<|ts_emb|>",
            "<|ts_mask|>",
            "<|ts_out|>",
        ]
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": tokens}
        )
        special = [
            self.tokenizer.convert_tokens_to_ids(x) for x in tokens
        ]
        if max(special) >= cfg.vocab_size + 16:
            raise ValueError(
                "Tokenizer vocabulary does not match the GPT-2 checkpoint."
            )
        args = SimpleNamespace(
            backbone="gpt2",
            backbone_path=self.backbone_path,
            backbone_config=SimpleNamespace(
                hidden_size=cfg.n_embd,
                intermediate_size=4 * cfg.n_embd,
            ),
            use_flash_attn=False,
            ts_enc="patch",
            adapter_type="linear",
            q_num=input_size // patch_len,
            training_mode="full",
            seq_len=input_size,
            pred_len=h,
            single_pred_len=h,
            pretrain_seq_lens=[input_size],
            patch_size=patch_len,
            d_model=d_model,
            n_heads=n_heads,
            num_kv_heads=n_heads,
            d_ff=d_ff,
            e_layers=e_layers,
            factor=1,
            dropout=dropout,
            output_attention=False,
            activation="gelu",
        )
        self.model = official_module(
            self.source_dir,
            "LangTime",
        ).LTPratrainedModel(args, *special)
        # Use the original control-token ordering; descriptive text varies by series.
        self._prompts = []
        for context in self.contexts:
            ids = (
                self.tokenizer.encode(context)
                + [special[0]] * (input_size // patch_len)
                + [special[1]]
                + self.tokenizer.encode(" Predict next values: ")
                + [special[3]]
            )
            if (
                len(ids) > cfg.n_positions
                or any(
                    s in self.tokenizer.encode(context)
                    for s in special
                )
            ):
                raise ValueError(
                    "Context is too long or contains reserved "
                    "LangTime control tokens."
                )
            self._prompts.append(ids)

    def forward(self, windows_batch):
        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        ids = _context_indices(self, windows_batch, y)
        outputs = []
        for i, idx in enumerate(ids.tolist()):
            prompt = torch.tensor(
                self._prompts[idx],
                dtype=torch.long,
                device=y.device,
            )[None, None]
            output = self.model(
                y[i : i + 1],
                None,
                prompt,
                torch.ones_like(prompt),
                mask_rate=0,
            )
            outputs.append(output[:, -self.h :])
        return self._point_output(torch.cat(outputs), y)
