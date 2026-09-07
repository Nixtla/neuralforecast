# Eight exogenous forecasting additions

These additions are **not interchangeable implementations** of one model. Two
are native architecture ports, one uses IBM's trainable official implementation,
and five connect to official pretrained inference backends. None is a placeholder
or a renamed baseline. Checkpoint quality and paper benchmark reproduction are
separate from interface correctness.

## Implementations and supported inputs

| Catalog name | Python class | Integration | Historical covariates | Known-future covariates |
|---|---|---|---|---|
| CrossLinear | `CrossLinear` | Native official MS-path port | Yes | No |
| Timer-XL | `TimerXL` | Native covariate-path port, final-horizon training | Yes | No |
| Tiny Time Mixers (TTM) | `TinyTimeMixer` | Official IBM architecture, trained from scratch | Yes | Yes |
| Chronos-2 | `Chronos2` | Official Chronos-2 pretrained inference | Yes | Yes |
| Moirai | `Moirai` | Official Moirai-1.1, isolated uni2ts interpreter | Yes | Yes |
| Moirai-MoE | `MoiraiMoE` | Official Moirai-MoE-1.0, isolated uni2ts interpreter | Yes | Yes |
| TimesFM | `TimesFM` | Official **2.5** checkpoint plus its XReg implementation | No | Yes |
| Toto | `Toto` | Official **1.0** pretrained forecaster | Joint auxiliary channels | Yes |

All classes are exported from `neuralforecast.models`, and their class names are
registered for `NeuralForecast.save` / `NeuralForecast.load`. They currently expose
**point forecasts with MAE/MSE only**. Static/categorical inputs, distribution
losses and unsupported covariate types fail explicitly. No Auto tuning wrappers
are added in this change.

For `futr_exog_list`, provide historical values in the training frame **and**
known future values in `futr_df`. Never put an unavailable future price, target,
or exogenous observation into that frame. CrossLinear and TimerXL deliberately
reject future lists because their ports implement historical-covariate paths.

## Native training example

Install this branch into a standard NeuralForecast environment. CrossLinear and
TimerXL add no dependencies beyond the repository's existing runtime stack.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import CrossLinear, TimerXL

models = [
    CrossLinear(h=4, input_size=32, patch_len=8,
                hist_exog_list=["inventory", "exchange_rate"], max_steps=300),
    TimerXL(h=4, input_size=32, patch_len=8,
            hist_exog_list=["inventory", "exchange_rate"], max_steps=300),
]
nf = NeuralForecast(models=models, freq="W-FRI")
# train_df columns: unique_id, ds, y, inventory, exchange_rate
nf.fit(df=train_df)
forecast_df = nf.predict()
```

CrossLinear matches the source's target-last MS pathway, convolutional correlation
embedding, patch MLP, learned value/position mixing and target de-normalization.
Complete contexts are required; missing/padded target histories are rejected.

TimerXL retains channel-major tokens, half-head rotary projection, learned binary
variate bias and the official covariate attention mask. Its output token has width
`h`, and NF trains the **last target token's horizon**, rather than the upstream
all-token pretraining objective. It does **not** load Timer-XL pretrained weights.
`input_size` must be divisible by `patch_len`; `hidden_size / n_heads` by four.
Complete contexts are required.

## Official optional backends

Install only the backend being used. Do not force all upstream packages into one
environment, or weaken the repository's dependency constraints to satisfy them.
Backend package versions must supply the reviewed APIs listed below.

```bash
# Chronos-2:
python -m pip install chronos-forecasting
# Trainable IBM TTM:
python -m pip install granite-tsfm
# TimesFM-2.5 + its XReg dependencies:
python -m pip install 'timesfm[torch]' jax scikit-learn
# Toto-1.0 API (not an unrelated package named toto):
python -m pip install toto-ts
```

TTM uses `TinyTimeMixerForPrediction(TinyTimeMixerConfig(...))`, without a weight
download. Channel mixing and forecast-channel mixing are explicitly enabled.
The future target/unknown channels are zeros, while only declared exogenous
channels receive their known future values. NF supplies the training loss, not
these dummy future target values. Train the model before interpreting forecasts;
this is not zero-shot pretrained TTM inference.

```python
from neuralforecast.models import TinyTimeMixer, Chronos2

trainable_ttm = TinyTimeMixer(
    h=4, input_size=32, patch_len=8,
    hist_exog_list=["inventory"], futr_exog_list=["holiday"], max_steps=300,
)
pretrained_chronos = Chronos2(
    h=4, input_size=32,
    hist_exog_list=["inventory"], futr_exog_list=["holiday"],
    backend_device="cpu",  # e.g. "cuda:0" when that device is available
)
# Both work inside NeuralForecast(models=[...], freq=...).
# nf.predict(futr_df=...) must receive holiday for every requested horizon row.
```

`Chronos2`, `Moirai`, `MoiraiMoE`, `TimesFM`, and `Toto` are **inference-only**.
Their `fit` prepares NF bookkeeping and validates declared features; it does not
fine-tune the backend. `max_steps>0` and gradient explanations are rejected. They
load optional libraries/weights on the first forward call. No training targets
are sent to a pretrained backend for adaptation.

Set `model_id` to a compatible local checkpoint directory or a Hugging Face model
ID. Set `revision` to an immutable Hugging Face revision for repeatability. NF
checkpoints store these **references**, not the external model weights. Offline
reload therefore requires a populated cache/local checkpoint and the relevant
backend package (and interpreter path for uni2ts). Large weights are never added
to this Git repository. A first-run download needs network access and disk space.

Chronos-2 explicitly disables cross-learning across NF windows. TimesFM XReg is
fitted **one window at a time**, avoiding regression pooling across later and
earlier rolling windows. TimesFM uses the 2.5 API with `return_backcast=True` and
requires complete historical inputs. Historical-only features are not fabricated
into known-future values. Toto uses the legacy 1.0 API because the reviewed 2.0
release does not yet support exogenous variables; known-future channels are
placed last as required by its decoder. Toto history-only covariates are jointly
predicted auxiliary channels, not a separate past-only feature encoder.

## Moirai dependency isolation

The reviewed `uni2ts/pyproject.toml` requires `torch>=2.1,<2.5`; this repository
requires `torch>=2.9.1`. Installing both normally in one environment is impossible.
The adapter therefore invokes an explicitly configured separate interpreter.
It exchanges arrays through a private temporary directory using NPZ **without
pickle**, uses no shell, enforces a timeout, and cleans up after success/failure.

```bash
# Run outside your active NF environment; use Python 3.11 for uni2ts' NumPy/SciPy.
python3.11 -m venv .venv-uni2ts
.venv-uni2ts/bin/python -m pip install --upgrade pip
.venv-uni2ts/bin/python -m pip install \
  'git+https://github.com/SalesforceAIResearch/uni2ts.git'
```

```python
from pathlib import Path
from neuralforecast.models import Moirai, MoiraiMoE

# absolute() intentionally preserves the venv Python symlink, unlike resolve().
interpreter = str(Path(".venv-uni2ts/bin/python").absolute())
models = [
    Moirai(h=4, input_size=32, backend_python=interpreter,
           hist_exog_list=["inventory"], futr_exog_list=["holiday"]),
    MoiraiMoE(h=4, input_size=32, backend_python=interpreter,
              hist_exog_list=["inventory"], futr_exog_list=["holiday"]),
]
```

The simple isolation implementation loads weights once **per batch subprocess**.
That is slower than a persistent server for large rolling backtests. Adjust
`inference_windows_batch_size`, `num_samples`, `backend_device` and
`backend_timeout` to control memory/runtime. Persistent workers are not included.

## Checks and verification scope

```bash
# Native kernels, official-API contracts, and full NF fit/predict/save/load:
python -m pytest -q -o addopts='' tests/test_exogenous_models.py -m 'not optional'
# Additionally execute the actual installed IBM TTM architecture (no checkpoint):
python -m pytest -q -o addopts='' tests/test_exogenous_models.py -m optional
```

Contract tests replace heavyweight pretrained packages with test doubles. They
check exact input layouts, covariate routing, masks, no future-target leakage,
window independence, distinct Moirai/MoE dispatch and real subprocess exchange.
They do **not** certify official checkpoint numerics, pretrained accuracy,
GPU/mixed-precision behavior, or reproduce published benchmark results. Native
architecture tests use real PyTorch forward/backward passes. Integration tests
exercise the actual installed NF framework, including save/load registration.

The CPU GitHub Actions workflow runs core integration tests and an independent
optional TTM job. Check its recorded results; the presence of a workflow file is
not evidence that CI has run or passed. No paper-level performance claim is made.

## Reviewed upstream API sources

- CrossLinear: https://github.com/mumiao2000/CrossLinear/blob/main/models/CrossLinear.py
  (`594ad517e8e3f188d77ce09d995fc6f3062827aa`).
- Timer-XL: https://github.com/thuml/OpenLTM/blob/main/models/timer_xl.py
  (`a554cf3b64ed1513953430e4ad8495b5e24d1b5a`); see license document for helper blobs.
- Chronos-2: https://github.com/amazon-science/chronos-forecasting/blob/main/src/chronos/chronos2/pipeline.py
  (`d2b3f02e62fe663c0888414bad017f41d8ce57d3`).
- Moirai: https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/model/moirai/forecast.py
  (`d8a938697ac6285df3d97dcdec4075b263756be7`).
- Moirai-MoE: https://github.com/SalesforceAIResearch/uni2ts/blob/main/src/uni2ts/model/moirai_moe/forecast.py
  (`43d4fd95f82fb96022b4cb5df5d79f609a91e906`).
- TimesFM-2.5: https://github.com/google-research/timesfm/blob/master/src/timesfm/timesfm_2p5/timesfm_2p5_base.py
  (`151abcb886cd6c49516a3ae99b310194307d9fb0`).
- IBM TTM: https://github.com/ibm-granite/granite-tsfm/blob/main/tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py
  (`86290a69faf498b324f90d91296006fc244e4849`).
- Toto-1.0: https://github.com/DataDog/toto/blob/main/toto/inference/forecaster.py
  (`a933a0f5122199120defc1fe181efefa6d4b5fee`).

IDs above are **file/blob hashes**, not installable repository commit revisions.
The links can move; retain these hashes when auditing upstream API compatibility.
