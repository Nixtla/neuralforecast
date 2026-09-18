"""Aurora context-conditioned pretrained forecasting adapter."""

from pathlib import Path

import torch

from ..losses.pytorch import MQLoss
from ._context_source import official_module
from ._context_utils import (
    _context_indices,
    _context_schema,
    _contexts,
    _local_path,
    _options,
    _positive,
)
from ._exogenous import PretrainedExogenousModel

__all__ = ["Aurora"]


class Aurora(PretrainedExogenousModel):
    """Official Aurora local-checkpoint inference with BERT text conditions.

    stat_exog_list contains one context_id selecting contexts. No future text,
    model weights or tokenizer are downloaded automatically. MAE/MSE returns the
    sample mean; loss=MQLoss(...) returns empirical marginal quantiles instead.
    """

    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False
    EXOGENOUS_STAT = True

    NATIVE_QUANTILES = True

    def __init__(
        self,
        h,
        input_size,
        source_dir,
        model_id,
        tokenizer_path,
        contexts,
        inference_token_len=48,
        stat_exog_list=None,
        **kwargs,
    ):
        _positive(inference_token_len=inference_token_len)
        self.contexts = _contexts(contexts)
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.tokenizer_path = str(
            Path(tokenizer_path).expanduser().resolve()
        )
        if not Path(self.tokenizer_path).is_dir():
            raise FileNotFoundError(
                "tokenizer_path must be a local BERT tokenizer directory."
            )
        super().__init__(
            h=h,
            input_size=input_size,
            model_id=_local_path(model_id),
            stat_exog_list=stat_exog_list,
            **_options(kwargs),
        )
        _context_schema(self)
        self.inference_token_len = inference_token_len
        if isinstance(self.loss, MQLoss) and self.num_samples < 2:
            raise ValueError(
                "Aurora quantiles require num_samples >= 2."
            )

    def _load_backend(self):
        from transformers import BertTokenizerFast

        module = official_module(self.source_dir, "Aurora")
        model, info = module.AuroraForPrediction.from_pretrained(
            self.model_id,
            local_files_only=True,
            use_safetensors=True,
            output_loading_info=True,
        )
        if (
            info["missing_keys"]
            or info["mismatched_keys"]
            or info["unexpected_keys"]
        ):
            raise ValueError(
                "The Aurora checkpoint does not match "
                "the pinned official architecture."
            )
        tokenizer = BertTokenizerFast.from_pretrained(
            self.tokenizer_path,
            local_files_only=True,
        )
        for context in self.contexts:
            if len(tokenizer.encode(context)) > 125:
                raise ValueError(
                    "Aurora context exceeds 125 BERT tokens; "
                    "shorten it explicitly."
                )
        return model.to(self.backend_device).eval(), tokenizer

    def forward(self, windows_batch):
        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        ids = _context_indices(self, windows_batch, y)
        model, tokenizer = self._get_backend()
        outputs = []
        # The upstream pseudo-image period selector pools a batch. Isolate windows.
        with torch.inference_mode():
            for i, idx in enumerate(ids.tolist()):
                text = tokenizer(
                    self.contexts[idx],
                    return_tensors="pt",
                ).to(self.backend_device)
                sample = model.generate(
                    inputs=y[i : i + 1, :, 0].to(self.backend_device),
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get(
                        "token_type_ids",
                        torch.zeros_like(text["input_ids"]),
                    ),
                    max_output_length=self.h,
                    num_samples=self.num_samples,
                    inference_token_len=self.inference_token_len,
                )
                if sample.shape != (1, self.num_samples, self.h):
                    raise ValueError(
                        f"Unexpected Aurora sample shape: {tuple(sample.shape)}"
                    )
                if not torch.isfinite(sample).all():
                    raise ValueError(
                        "Aurora returned non-finite forecast samples."
                    )
                if isinstance(self.loss, MQLoss):
                    # Marginal empirical quantiles, not residual/conformal intervals.
                    qs = self.loss.quantiles.to(
                        device=sample.device,
                        dtype=torch.float32,
                    )
                    output = torch.quantile(
                        sample.float(),
                        qs,
                        dim=1,
                    ).permute(1, 2, 0)
                else:
                    output = sample.mean(1)
                outputs.append(output.to(y.device))
        if isinstance(self.loss, MQLoss):
            return self._quantile_output(torch.cat(outputs), y)
        return self._point_output(torch.cat(outputs), y)
