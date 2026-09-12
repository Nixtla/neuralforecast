# Foundation LoRA

`neuralforecast.foundation_lora` exposes LoRA only where this fork can call a verified native fine-tuning API. It does not add a generic correction layer around pretrained forecasts.

## Supported model

| Model | LoRA path | Status |
| --- | --- | --- |
| Chronos2 | official `Chronos2Pipeline.fit(..., finetune_mode="lora")` | supported |
| Moirai / MoiraiMoE / Moirai2 | separate `uni2ts` inference worker in this fork | unsupported |
| TimesFM / TimesFM3 | inference adapter path | unsupported |
| Toto | inference adapter path | unsupported |
| ChronosX / BaguanTS / RAG4CTS | fixed external checkpoint/source contracts | unsupported |
| Aurora / ChatTime | LLM/context protocol excluded from the commodity benchmark | excluded |
| TabPFNTS | local in-context regression adapter | unsupported |

Chronos2 support follows the official Amazon Chronos implementation, which exposes native full and LoRA fine-tuning in `Chronos2Pipeline.fit`.

## Optional dependency

LoRA requires `peft` in the benchmark environment:

```bash
pip install peft
```

`peft` stays optional and is not added to NeuralForecast's mandatory dependencies. The wrapper raises `ImportError` when it is missing. This prevents the upstream Chronos fallback from silently changing a requested LoRA run into full fine-tuning.

## Search space

`get_foundation_lora_config("Chronos2", h=16, fixed=...)` starts from the existing Chronos2 inference context-length space and adds:

- learning rate: log-uniform `1e-5` to `3e-4`;
- LoRA rank: `4, 8, 16`;
- LoRA alpha: `8, 16, 32`;
- LoRA dropout: `0.0, 0.05, 0.1`;
- fine-tuning batch size: `8, 16, 32`.

The target modules match the official Chronos2 default LoRA targets:

- `self_attention.q`;
- `self_attention.k`;
- `self_attention.v`;
- `self_attention.o`;
- `output_patch_embedding.output_layer`.

Model ID, revision and device remain fixed inputs rather than HPO dimensions.

## Training contract

`fit_foundation_lora` receives training rows only. It calls the official pipeline with:

```text
finetune_mode = lora
validation_inputs = None
```

The benchmark's 16-week validation block therefore remains evaluation-only and cannot control early stopping or checkpoint selection.

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

The current official `Chronos2Pipeline.fit` method creates its own Hugging Face trainer and calls `trainer.train()` without exposing `resume_from_checkpoint`. This module does not emulate resumable SH by restarting optimizer state and calling it a checkpoint continuation.

For the commodity benchmark, Chronos2-LoRA should therefore be evaluated as a one-rung, 1000-step Phase 1 protocol until the upstream fine-tuning API exposes a verified resumable path or this fork adds a narrow native trainer hook. The selected LoRA configuration is then retrained from the base checkpoint on each Phase 2 fold.

This limitation is protocol metadata and should stay visible in the integrated leaderboard (`training_protocol=LoRA`).
