"""Remaining official text-conditioned and numerical-covariate integrations.

Text models require real external text/embeddings, not arbitrary market columns.
See docs/context_models.md for pinned source setup and each integration boundary.
"""

from pathlib import Path
from types import SimpleNamespace

import torch

from ._context_source import official_module
from ._exogenous import ExogenousModel, PretrainedExogenousModel

__all__ = ["VoT", "GPT4MTS", "UniTime", "LangTime", "Aurora", "ChatTime", "TabPFNTS"]


def _positive(**values):
    for name, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")


def _options(kwargs):
    if kwargs.get("scaler_type", "identity") != "identity":
        raise ValueError("Use identity scaling; embeddings/context IDs must not be rescaled.")
    if kwargs.get("start_padding_enabled", False):
        raise ValueError("These integrations require complete history.")
    kwargs.setdefault("training_data_availability_threshold", [1.0, 1.0])
    return kwargs


def _local_path(path, directory=True):
    if not isinstance(path, str) or not path:
        raise ValueError("An explicit local checkpoint path is required; no implicit downloads.")
    resolved = Path(path).expanduser().resolve()
    if not (resolved.is_dir() if directory else resolved.is_file()):
        raise FileNotFoundError(f"Missing local checkpoint: {resolved}")
    if directory and not any(resolved.glob("*.safetensors")):
        raise ValueError("The checkpoint directory must contain safetensors weights.")
    return str(resolved)


def _contexts(values):
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("contexts must be a nonempty list of external descriptions.")
    if any(not isinstance(x, str) or not x.strip() for x in values):
        raise ValueError("Every context must be a nonempty string.")
    return list(values)


def _context_indices(model, batch, y):
    value = batch.get("stat_exog")
    if value is None or value.shape != (len(y), 1):
        raise ValueError("stat_exog must contain one context_id per series/window.")
    ids = value[:, 0]
    if not torch.isfinite(ids).all() or (ids != ids.round()).any():
        raise ValueError("context_id must be a finite integer.")
    if (ids < 0).any() or (ids >= len(model.contexts)).any():
        raise ValueError("context_id must index the contexts list.")
    return ids.long()


def _context_schema(model):
    if model.stat_exog_size != 1:
        raise ValueError("Provide exactly one stat_exog_list column containing context_id.")


def _native_quantile_output(output, y, h, quantiles):
    output = torch.as_tensor(output, device=y.device, dtype=y.dtype)
    expected = (y.shape[0], h, 1 + len(quantiles))
    if output.shape != expected:
        raise ValueError(f"Backend returned unexpected forecast shape {tuple(output.shape)}.")
    if not torch.isfinite(output).all():
        raise ValueError("Backend returned non-finite forecasts.")
    return output


class VoT(ExogenousModel):
    """Official VoT PatchTST text-fusion forecasting stage, trained with point loss.

    hist_exog_list contains precomputed event/text embedding coordinates.
    Reasoning/text extraction and the original contrastive pretraining stages
    remain external; this runs the official forecast=2 path with both branches.
    """

    EXOGENOUS_FUTR = False

    def __init__(self, h, input_size, source_dir, d_model=128, n_heads=4,
                 d_ff=256, e_layers=2, patch_len=4, stride=4, dropout=0.1, **kwargs):
        _positive(d_model=d_model, n_heads=n_heads, d_ff=d_ff, e_layers=e_layers,
                  patch_len=patch_len, stride=stride)
        if d_model % n_heads or not 0 <= dropout < 1 or patch_len > min(input_size, h):
            raise ValueError("Invalid attention/dropout/patch configuration.")
        super().__init__(h=h, input_size=input_size, **_options(kwargs))
        if self.hist_exog_size < 8 or self.hist_exog_size % 8:
            raise ValueError("VoT text embedding width must be a positive multiple of eight.")
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        config = SimpleNamespace(task_name="long_term_forecast", seq_len=input_size,
            pred_len=h, enc_in=1, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
            e_layers=e_layers, dropout=dropout, factor=1, output_attention=False,
            activation="gelu", multimodal=True, tr_sea=True, clip_t=0.07,
            llm_dim=self.hist_exog_size)
        self.model = official_module(self.source_dir, "VoT").Model(config, patch_len, stride)

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        # ponytail: upstream text rescaling averages over the batch; independent
        # window calls prevent later windows affecting earlier forecasts.
        output = torch.cat([self.model(y[i:i+1], None, None, None, hist[i:i+1], forecast=2)
                            for i in range(len(y))])
        return self._point_output(output, y)


class GPT4MTS(ExogenousModel):
    """Official GPT4MTS historical text-embedding fusion with a GPT-2 backbone.

    An explicit local GPT-2 checkpoint is optional. With backbone_path=None the
    original GPT-2 architecture is initialized randomly and must be trained.
    Historical text coordinates must match the selected language encoder width.
    """

    EXOGENOUS_FUTR = False

    def __init__(self, h, input_size, source_dir, backbone_path=None,
                 gpt_layers=2, n_heads=12, patch_len=8, stride=4,
                 freeze_backbone=False, **kwargs):
        _positive(gpt_layers=gpt_layers, n_heads=n_heads, patch_len=patch_len, stride=stride)
        if input_size < patch_len:
            raise ValueError("input_size must be at least patch_len.")
        super().__init__(h=h, input_size=input_size, **_options(kwargs))
        width = self.hist_exog_size
        if width < 2 or width % n_heads:
            raise ValueError("Text embedding width must be >=2 and divisible by n_heads.")
        config = dict(n_embd=width, n_layer=gpt_layers, n_head=n_heads)
        if backbone_path is not None:
            from transformers import GPT2Config
            backbone_path = _local_path(backbone_path)
            cfg = GPT2Config.from_pretrained(backbone_path, local_files_only=True)
            if cfg.n_embd != width or gpt_layers > cfg.n_layer:
                raise ValueError("GPT-2 width/layer count does not match the requested text configuration.")
            config = cfg.to_dict()
        elif freeze_backbone:
            raise ValueError("Cannot freeze a randomly initialized language backbone.")
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.backbone_path = backbone_path
        args = SimpleNamespace(is_gpt=True, revin=False, patch_size=patch_len,
            pretrain=backbone_path is not None, stride=stride, seq_len=input_size,
            pred_len=h, d_model=width, gpt_layers=gpt_layers, freeze=freeze_backbone,
            backbone_path=backbone_path, backbone_config=config)
        self.model = official_module(self.source_dir, "GPT4MTS").GPT4MTS(args, torch.device("cpu"))
        token_count = 2 * ((input_size - patch_len) // stride + 2)
        if token_count > self.model.gpt2.config.n_positions:
            raise ValueError("Combined text/target patches exceed GPT-2 position capacity.")

    def forward(self, windows_batch):
        y, mask, hist, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        return self._point_output(self.model(y, 0, hist), y)


class UniTime(ExogenousModel):
    """Official UniTime with per-series external domain descriptions.

    One static integer context_id selects a string in contexts. A local GPT-2
    tokenizer/checkpoint is required. NF trains the forecast horizon, not the
    original multi-domain masked reconstruction curriculum.
    """

    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False
    EXOGENOUS_STAT = True

    def __init__(self, h, input_size, source_dir, backbone_path, contexts,
                 patch_len=16, gpt_layers=2, decoder_layers=1,
                 max_tokens=128, dropout=0.1, stat_exog_list=None, **kwargs):
        _positive(patch_len=patch_len, gpt_layers=gpt_layers,
                  decoder_layers=decoder_layers, max_tokens=max_tokens)
        if input_size % patch_len or not 0 <= dropout < 1:
            raise ValueError("input_size must divide into patches; dropout must be in [0,1).")
        self.contexts = _contexts(contexts)
        super().__init__(h=h, input_size=input_size, stat_exog_list=stat_exog_list,
                         **_options(kwargs))
        _context_schema(self)
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.backbone_path = _local_path(backbone_path)
        self.patch_len = patch_len
        args = SimpleNamespace(mask_rate=0.0, patch_len=patch_len,
            max_token_num=max_tokens, max_backcast_len=input_size, max_forecast_len=h,
            logger=None, model_path=self.backbone_path, lm_layer_num=gpt_layers,
            lm_ft_type="full", ts_embed_dropout=dropout, dec_trans_layer_num=decoder_layers,
            dec_head_dropout=dropout)
        self.model = official_module(self.source_dir, "UniTime").UniTime(args)
        if gpt_layers > self.model.backbone.config.n_layer:
            raise ValueError("gpt_layers exceeds the checkpoint depth.")
        limit = min(max_tokens, self.model.backbone.config.n_positions)
        if max_tokens > self.model.backbone.config.n_positions:
            raise ValueError("max_tokens exceeds GPT-2 position capacity.")
        for context in self.contexts:
            if len(self.model.tokenizer.encode(context)) + input_size // patch_len > limit:
                raise ValueError("Description plus target patches exceeds token capacity; shorten it explicitly.")

    def forward(self, windows_batch):
        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        ids = _context_indices(self, windows_batch, y)
        outputs = []
        for i, idx in enumerate(ids.tolist()):
            info = (idx, self.input_size, self.patch_len, self.contexts[idx])
            output = self.model(info, y[i:i+1].clone(), mask[i:i+1].to(y.dtype))
            outputs.append(output[:, self.input_size:self.input_size + self.h])
        return self._point_output(torch.cat(outputs), y)


class LangTime(ExogenousModel):
    """Official LangTime GPT-2 supervised forecaster with external instructions.

    One static context_id selects a description. The original patch encoder,
    linear adapter, special-token routing and forecast head execute unchanged;
    PPO/reasoner training and backcast loss are outside this NF integration.
    """

    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False
    EXOGENOUS_STAT = True

    def __init__(self, h, input_size, source_dir, backbone_path, contexts,
                 patch_len=8, d_model=64, n_heads=4, d_ff=128,
                 e_layers=2, dropout=0.1, stat_exog_list=None, **kwargs):
        from transformers import GPT2Config, GPT2Tokenizer
        _positive(patch_len=patch_len, d_model=d_model, n_heads=n_heads,
                  d_ff=d_ff, e_layers=e_layers)
        if input_size % patch_len or d_model % n_heads or not 0 <= dropout < 1:
            raise ValueError("Invalid patch/attention/dropout configuration.")
        self.contexts = _contexts(contexts)
        super().__init__(h=h, input_size=input_size, stat_exog_list=stat_exog_list,
                         **_options(kwargs))
        _context_schema(self)
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.backbone_path = _local_path(backbone_path)
        cfg = GPT2Config.from_pretrained(self.backbone_path, local_files_only=True)
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.backbone_path, local_files_only=True)
        tokens = ["<|TS_ENC|>", "<|ts_emb|>", "<|ts_mask|>", "<|ts_out|>"]
        self.tokenizer.add_special_tokens({"additional_special_tokens": tokens})
        special = [self.tokenizer.convert_tokens_to_ids(x) for x in tokens]
        if max(special) >= cfg.vocab_size + 16:
            raise ValueError("Tokenizer vocabulary does not match the GPT-2 checkpoint.")
        args = SimpleNamespace(backbone="gpt2", backbone_path=self.backbone_path,
            backbone_config=SimpleNamespace(hidden_size=cfg.n_embd, intermediate_size=4*cfg.n_embd),
            use_flash_attn=False, ts_enc="patch", adapter_type="linear", q_num=input_size//patch_len,
            training_mode="full", seq_len=input_size, pred_len=h, single_pred_len=h,
            pretrain_seq_lens=[input_size], patch_size=patch_len, d_model=d_model,
            n_heads=n_heads, num_kv_heads=n_heads, d_ff=d_ff, e_layers=e_layers,
            factor=1, dropout=dropout, output_attention=False, activation="gelu")
        self.model = official_module(self.source_dir, "LangTime").LTPratrainedModel(args, *special)
        # Use the original control-token ordering; descriptive text varies by series.
        self._prompts = []
        for context in self.contexts:
            ids = (self.tokenizer.encode(context) + [special[0]] * (input_size//patch_len)
                   + [special[1]] + self.tokenizer.encode(" Predict next values: ") + [special[3]])
            if len(ids) > cfg.n_positions or any(s in self.tokenizer.encode(context) for s in special):
                raise ValueError("Context is too long or contains reserved LangTime control tokens.")
            self._prompts.append(ids)

    def forward(self, windows_batch):
        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        ids = _context_indices(self, windows_batch, y)
        outputs = []
        for i, idx in enumerate(ids.tolist()):
            prompt = torch.tensor(self._prompts[idx], dtype=torch.long, device=y.device)[None, None]
            output = self.model(y[i:i+1], None, prompt, torch.ones_like(prompt), mask_rate=0)
            outputs.append(output[:, -self.h:])
        return self._point_output(torch.cat(outputs), y)


class Aurora(PretrainedExogenousModel):
    """Official Aurora local-checkpoint inference with BERT text conditions.

    stat_exog_list contains one context_id selecting contexts. No future text,
    model weights or tokenizer are downloaded automatically. Point forecasts use
    the sample mean; requested quantiles use the same generated sample paths.
    """

    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False
    EXOGENOUS_STAT = True
    NATIVE_QUANTILES = True

    def __init__(self, h, input_size, source_dir, model_id, tokenizer_path, contexts,
                 inference_token_len=48, stat_exog_list=None, **kwargs):
        _positive(inference_token_len=inference_token_len)
        self.contexts = _contexts(contexts)
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        self.tokenizer_path = str(Path(tokenizer_path).expanduser().resolve())
        if not Path(self.tokenizer_path).is_dir():
            raise FileNotFoundError("tokenizer_path must be a local BERT tokenizer directory.")
        super().__init__(h=h, input_size=input_size, model_id=_local_path(model_id),
                         stat_exog_list=stat_exog_list, **_options(kwargs))
        _context_schema(self)
        self.inference_token_len = inference_token_len

    def _load_backend(self):
        from transformers import BertTokenizerFast
        module = official_module(self.source_dir, "Aurora")
        model, info = module.AuroraForPrediction.from_pretrained(
            self.model_id, local_files_only=True, use_safetensors=True, output_loading_info=True)
        if info["missing_keys"] or info["mismatched_keys"] or info["unexpected_keys"]:
            raise ValueError("The Aurora checkpoint does not match the pinned official architecture.")
        tokenizer = BertTokenizerFast.from_pretrained(self.tokenizer_path, local_files_only=True)
        for context in self.contexts:
            if len(tokenizer.encode(context)) > 125:
                raise ValueError("Aurora context exceeds 125 BERT tokens; shorten it explicitly.")
        return model.to(self.backend_device).eval(), tokenizer

    def forward(self, windows_batch):
        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        ids = _context_indices(self, windows_batch, y)
        model, tokenizer = self._get_backend()
        quantiles = self.loss.quantiles
        outputs = []
        # The upstream pseudo-image period selector pools a batch. Isolate windows.
        with torch.inference_mode():
            for i, idx in enumerate(ids.tolist()):
                text = tokenizer(self.contexts[idx], return_tensors="pt").to(self.backend_device)
                sample = model.generate(inputs=y[i:i+1, :, 0].to(self.backend_device),
                    text_input_ids=text["input_ids"], text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", torch.zeros_like(text["input_ids"])),
                    max_output_length=self.h, num_samples=self.num_samples,
                    inference_token_len=self.inference_token_len)
                if sample.shape != (1, self.num_samples, self.h):
                    raise ValueError(f"Unexpected Aurora sample shape: {tuple(sample.shape)}")
                point = sample.mean(1)
                if quantiles is None:
                    outputs.append(point.to(y.device))
                    continue
                qs = torch.tensor(quantiles, device=sample.device, dtype=sample.dtype)
                quantile_values = torch.quantile(sample, qs, dim=1).permute(1, 2, 0)
                outputs.append(torch.cat((point.unsqueeze(-1), quantile_values), dim=-1).to(y.device))
        output = torch.cat(outputs)
        if quantiles is None:
            return self._point_output(output, y)
        return _native_quantile_output(output, y, self.h, quantiles)


class ChatTime(PretrainedExogenousModel):
    """Official ChatTime local Llama forecasting API, conditioned on descriptions.

    stat_exog_list contains one context_id. Only a compatible trained ChatTime
    checkpoint is meaningful; a generic Llama checkpoint is not a forecaster.
    Invalid/unparseable numerical generations raise instead of filling forecasts.
    """

    EXOGENOUS_HIST = False
    EXOGENOUS_FUTR = False
    EXOGENOUS_STAT = True

    def __init__(self, h, input_size, source_dir, model_id, contexts,
                 stat_exog_list=None, **kwargs):
        self.contexts = _contexts(contexts)
        self.source_dir = str(Path(source_dir).expanduser().resolve())
        super().__init__(h=h, input_size=input_size, model_id=_local_path(model_id),
                         stat_exog_list=stat_exog_list, **_options(kwargs))
        _context_schema(self)

    def _load_backend(self):
        backend = official_module(self.source_dir, "ChatTime").ChatTime(
            self.model_id, hist_len=self.input_size, pred_len=self.h, num_samples=self.num_samples)
        backend.model.to(self.backend_device).eval()
        return backend

    def forward(self, windows_batch):
        y, mask, _, _ = self._inputs(windows_batch)
        self._complete_history(mask)
        ids = _context_indices(self, windows_batch, y)
        backend = self._get_backend()
        with torch.inference_mode():
            output = [backend.predict(y[i, :, 0].detach().cpu().numpy().copy(), context=self.contexts[idx])
                      for i, idx in enumerate(ids.tolist())]
        import numpy as np
        return self._point_output(np.stack(output), y)


class TabPFNTS(PretrainedExogenousModel):
    """Official TabPFN-TS pipeline in LOCAL mode with known-future covariates.

    model_id is an explicit local compatible TabPFN regressor checkpoint file.
    Only the running index is engineered; provide true calendar values through
    futr_exog_list because NF windows do not carry the original datetime index.
    Native pipeline quantiles are returned when requested.
    """

    EXOGENOUS_HIST = False
    NATIVE_QUANTILES = True

    def __init__(self, h, input_size, model_id, **kwargs):
        super().__init__(h=h, input_size=input_size, model_id=_local_path(model_id, directory=False),
                         **_options(kwargs))
        if not self.futr_exog_size:
            raise ValueError("TabPFNTS requires known-future numerical covariates.")

    def _load_backend(self):
        import os
        os.environ["TABPFN_DISABLE_TELEMETRY"] = "1"
        from tabpfn_time_series import TabPFNTSPipeline, TabPFNMode
        from tabpfn_time_series.features import RunningIndexFeature
        return TabPFNTSPipeline(max_context_length=self.input_size,
            temporal_features=[RunningIndexFeature()], tabpfn_mode=TabPFNMode.LOCAL,
            tabpfn_output_selection="median",
            tabpfn_model_config={"model_path": self.model_id, "device": self.backend_device})

    def forward(self, windows_batch):
        import numpy as np
        import pandas as pd
        y, mask, _, futr = self._inputs(windows_batch)
        self._complete_history(mask)
        pipeline = self._get_backend()
        quantiles = self.loss.quantiles
        requested_quantiles = [0.5] if quantiles is None else quantiles
        dates = pd.date_range("2000-01-01", periods=self.input_size + self.h, freq="s")
        output = []
        for i in range(len(y)):
            frame = pd.DataFrame(futr[i].detach().cpu().numpy(),
                                 columns=[f"covariate_{j}" for j in range(self.futr_exog_size)])
            frame["timestamp"] = dates  # Relative row positions, not calendar features.
            history = frame.iloc[:self.input_size].copy()
            history["target"] = y[i, :, 0].detach().cpu().numpy()
            future = frame.iloc[self.input_size:].copy()  # Never includes target labels.
            result = pipeline.predict_df(context_df=history, future_df=future,
                                         quantiles=requested_quantiles)
            if quantiles is None:
                output.append(result["target"].to_numpy())
            else:
                output.append(result[["target", *quantiles]].to_numpy())
        output = np.stack(output)
        if quantiles is None:
            return self._point_output(output, y)
        return _native_quantile_output(output, y, self.h, quantiles)
