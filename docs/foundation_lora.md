# Foundation LoRA

`neuralforecast.foundation_lora` exposes LoRA only where a reviewed official implementation provides a real parameter-efficient fine-tuning path. It does not add a forecast correction layer and call that LoRA.

## Supported models

| Model | Verified LoRA path | Status |
| --- | --- | --- |
| Chronos2 | official `Chronos2Pipeline.fit(..., finetune_mode="lora")` | supported |
| TimesFM 2.5 | official HuggingFace `TimesFm2_5ModelForPrediction` + PEFT example | supported |
| Moirai / MoiraiMoE / Moirai2 | `uni2ts` has training infrastructure, but no reviewed official LoRA path | unsupported |
| TimesFM3 | current official 3.0 inference API; no reviewed official LoRA path | unsupported |
| Toto | inference adapter path; no reviewed official LoRA path | unsupported |
| ChronosX / BaguanTS / RAG4CTS | fixed external checkpoint/source contracts; no reviewed official LoRA path | unsupported |
| Aurora / ChatTime | LLM/context protocol excluded from the commodity benchmark | excluded |
| TabPFNTS | local in-context regression adapter | unsupported |

Chronos2 support follows Amazon's official Chronos implementation. TimesFM 2.5 support follows Google's official `timesfm-forecasting/examples/finetuning/` LoRA workflow.

## Optional dependencies

LoRA requires PEFT. TimesFM 2.5 also requires a Transformers version exposing `TimesFm2_5ModelForPrediction`.

```bash
pip install peft transformers
```

These packages stay optional and are not added to NeuralForecast's mandatory dependencies. The wrapper raises `ImportError` when a required optional package is missing. Chronos2 therefore cannot silently fall back from requested LoRA to full fine-tuning.

## Search space

`get_foundation_lora_config(model, h=16, fixed=...)` starts from the model's existing inference context search and adds:

- learning rate: log-uniform `1e-5` to `3e-4`;
- LoRA rank: `4, 8, 16`;
- LoRA alpha: `8, 16, 32`;
- LoRA dropout: `0.0, 0.05, 0.1`;
- fine-tuning batch size: `8, 16, 32`.

### Chronos2 target modules

The target modules match the official Chronos2 default LoRA configuration:

- `self_attention.q`;
- `self_attention.k`;
- `self_attention.v`;
- `self_attention.o`;
- `output_patch_embedding.output_layer`.

### TimesFM 2.5 target modules

The official TimesFM fine-tuning example uses:

```text
target_modules = all-linear
```

The LoRA protocol therefore applies PEFT to every linear layer. The default LoRA checkpoint is `google/timesfm-2.5-200m-transformers`, which is the Transformers representation used by the official fine-tuning example. This is separate from the `google/timesfm-2.5-200m-pytorch` inference adapter default.

`xreg_ridge` is removed from the TimesFM LoRA search because the current no-exogenous commodity experiment does not use the XReg inference path.

## Training contract

Both protocols receive training rows only. The benchmark's following 16-week validation block is evaluation-only.

Chronos2 calls the official pipeline with:

```text
finetune_mode = lora
validation_inputs = None
```

TimesFM 2.5 follows the official random-window training approach. Every optimizer step samples `(input_size, horizon)` windows from the available training series, computes the native Transformers forecasting loss, clips gradients, updates the PEFT parameters and advances a cosine scheduler. Seed 42 controls window sampling. Validation targets never enter this loop.

Example:

```python
from neuralforecast.foundation_lora import (
    fit_foundation_lora,
    get_foundation_lora_config,
)
from neuralforecast.models import Chronos2

space = get_foundation_lora_config(
    "Chronos2",
    h=16,
    fixed={
        "model_id": "amazon/chronos-2",
        "backend_device": "cuda",
    },
)

# Resolve one Ray configuration before calling the protocol.
model = fit_foundation_lora(
    Chronos2,
    config,
    train_df,
    h=16,
    steps=1000,
    output_dir="results/chronos2-lora",
)
```

## Successive-halving boundary

The benchmark only claims checkpoint continuation when optimizer and scheduler state can be resumed correctly.

LoRA candidates use cumulative 100/250/500-step Phase 1 rungs with 10/5/1 configurations. The Chronos2 integration temporarily supplies a trainer subclass around the official fit call to resume native HuggingFace training checkpoints. TimesFM stores adapter weights, optimizer, scheduler, sampling RNG, and global RNG state. Both preserve validation patience and the best adapter weights separately from the last training state.

The optional `checkpoint` and `schedule_steps` arguments to `fit_foundation_lora` select continuation state and the fixed learning-rate schedule horizon. Fitted adapters expose `resume_checkpoint`. An early-stopped fold is reused by the benchmark without additional training.

The selected configuration is retrained from the base model on every Phase 2 fold, up to 500 steps. Both phases use train plus a single validation block, validate every 10 steps, stop after five non-improving checks, and restore best validation weights. Phase 2 admission requires strictly beating last-observation naive on Phase 1 pooled validation RMSE.
