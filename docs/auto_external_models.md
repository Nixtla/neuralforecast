# Auto HPO for external forecasting adapters

The fork's trainable external-source adapters have model-aware `Auto*` wrappers in
`neuralforecast.auto_external`. The wrappers reuse NeuralForecast `BaseAuto`, keep
source/checkpoint/schema inputs fixed, and search only parameters that can be
changed without altering the external-data contract.

## Trainable models

| Model | Auto wrapper | Main search dimensions |
|---|---|---|
| CrossLinear | `AutoCrossLinear` | context, patch, width, MLP, alpha/beta, optimizer |
| TimerXL | `AutoTimerXL` | context, patch, width, heads/layers, FFN, norm, optimizer |
| TinyTimeMixer | `AutoTinyTimeMixer` | context, patch, width/layers, dropout, optimizer |
| DAG | `AutoDAG` | context, attention, patch/stride, alpha/beta, optimizer |
| KITE | `AutoKITE` | context, flow width/depth, sampling, guidance, optimizer |
| GLAFF | `AutoGLAFF` | context, encoder width/depth, robust q, moving average |
| APT | `AutoAPT` | context, timestamp/prototype sizes, warmup, regularization |
| VoT | `AutoVoT` | context, attention width/depth, patch/stride, optimizer |
| GPT4MTS | `AutoGPT4MTS` | context, valid head counts, patch/stride, GPT depth when trainable |
| UniTime | `AutoUniTime` | context, patch, decoder depth, dropout, optimizer |
| LangTime | `AutoLangTime` | context, patch, TS encoder width/depth, dropout, optimizer |
| SpecTF | `AutoSpecTF` | context, spectral/text projection widths, dropout, optimizer |
| TGForecaster | `AutoTGForecaster` | context, valid heads, encoder/fusion depth, patch, optimizer |
| SeesawNet | `AutoSeesawNet` | context, attention/FFN, patch/stride, block depth, sampling rate |
| Dualformer | `AutoDualformer` | context, attention/FFN width, encoder depth, dropout, optimizer |

The built-in spaces use early stopping and a fixed random seed. Random seed is not
an HPO variable, so model selection does not reward a lucky initialization. For a
final benchmark, repeat the selected configuration across several fixed seeds.

Example:

```python
from neuralforecast import NeuralForecast
from neuralforecast.auto_external import AutoDualformer

model = AutoDualformer(
    h=12,
    source_dir="/opt/sources/Dualformer",
    hist_exog_list=["inventory", "fx"],
    backend="optuna",
    num_samples=30,
    time_budget=3600,
)

nf = NeuralForecast(models=[model], freq="W")
nf.fit(df=train_df, val_size=52)
```

Required source paths, checkpoint paths, context strings and covariate column lists
are pinned by each Auto wrapper. They are not sampled by HPO. User-supplied
`config` can replace the built-in search space while those fixed values remain
pinned.

## Inference/foundation models

Inference-only adapters intentionally do not use `BaseAuto`. Their `fit()` validates
inputs and stores holdout sizes without training weights or producing Lightning
validation metrics. Passing them through `BaseAuto` would therefore create a false
training-HPO contract.

Use `get_inference_tuning_config` to obtain a separate Ray or Optuna search space:

```python
from neuralforecast.inference_tuning import get_inference_tuning_config

space = get_inference_tuning_config(
    "RAG4CTS",
    h=12,
    backend="optuna",
    fixed={
        "input_size": 128,
        "source_dir": "/opt/sources/RAG4CTS",
        "futr_exog_list": ["inventory", "fx"],
    },
)
```

Evaluate each sampled configuration in an external chronological validation loop.
The inference spaces cover these adapters:

- `Chronos2`: context length.
- `Moirai`, `MoiraiMoE`: context, explicit patch size, sample count.
- `TimesFM`: context length.
- `Toto`: context and sample count.
- `Moirai2`: context only; official patch/sample settings remain fixed.
- `ChronosX`: sample count only; checkpoint `hidden_dim`, `num_layers` and context
  are fixed because they must match the fine-tuned checkpoint.
- `BaguanTS`: retrieval context, neighbors and repeat count with fixed input size.
- `RAG4CTS`: query size, neighbors and retrieval stride with fixed input size.
- `TimesFM3`: context and symmetric averaging.
- `Aurora`: context, inference token length and sample count.
- `ChatTime`: context and sample count.
- `TabPFNTS`: context length.

`model_id`, local source paths, backend interpreters, tokenizer paths, external
contexts and covariate schemas are never searched by default. `max_steps` is pinned
to zero for every inference space.

## Benchmark protocol

For architecture comparisons, use the same temporal folds and metric for every
model. HPO belongs inside the training partition; the outer validation/test window
must remain unseen by HPO. A low score from a non-converged trainable trial should
be treated as a failed configuration, not evidence that the architecture itself is
weak.

These spaces are practical defaults for this fork. They do not claim to reproduce
each paper's full benchmark grid, and they do not guarantee convergence on every
dataset.
