# Official model adapters for horizons 1–72

```python
from neuralforecast.models import TimesFM3, Chronos2, SeesawNet, Dualformer, SearchCast
```

`Chronos2` already exists in `foundation.py`; this addition reuses that class.
`TimesFM` continues to mean TimesFM **2.5**. Use the distinct `TimesFM3` class for
Google's 3.0 weights and native covariate API. All four new names are registered
with `NeuralForecast.save` / `NeuralForecast.load`.

## Implemented scope

| NF class | Execution | Historical covariates | Known-future covariates | Outputs |
|---|---|---|---|---|
| `TimesFM3` | Official pretrained inference; `fit` does not fine-tune | Native past-only arrays | Native past-and-future arrays | Native median, or a subset of deciles with `MQLoss` |
| `Chronos2` (existing) | Official pretrained inference; `fit` does not fine-tune | Official `past_covariates` | Official `future_covariates` | Native median |
| `SeesawNet` | Train the official network with NF's MAE/MSE | Additional encoder channels | Rejected | One target per NF series |
| `Dualformer` | Train the official network with NF's MAE/MSE | Additional encoder channels | Rejected | One target per NF series |
| `SearchCast` | Official Ridge/scaler/augmentation primitives plus Optuna; no gradient training | Rejected | Rejected | Direct forecasts, with separate hyperparameters per horizon group |

These adapters use **one target per NF `unique_id`**, not NF's joint multi-target
`MULTIVARIATE=True` layout. SeesawNet and Dualformer retain their multichannel
encoders when `hist_exog_list` is provided, then supervise and return the target
channel only. This is an explicit adaptation, not a reproduction of the papers'
all-channel losses or benchmark scores. In particular, SeesawNet uses the
requested NF point loss, not its benchmark script's TFMAE objective.

Static/categorical covariates are unsupported. Undeclared data are not used;
unsupported declared inputs fail instead of being silently consumed or replaced
with invented future values. The new adapters require complete finite history
(no start padding). Chronos2 retains its existing mask handling.

## Install the official dependencies

Start from this NeuralForecast checkout:

```bash
python -m pip install -e .
python -m pip install chronos-forecasting einops matplotlib scikit-learn reformer-pytorch

# TimesFM 3: use the reviewed source rather than assuming a PyPI release has v3.
python -m pip install 'timesfm[torch] @ git+https://github.com/google-research/timesfm.git@8cb0628371af142e16b8c232cc9fbf667ffb12f9'

# Research models: retain each as a clean, pinned git checkout.
git clone https://github.com/dreamone-Lee/SeesawNet.git external/SeesawNet
git -C external/SeesawNet checkout f0b49c8de2dceb866ffc8eab40ded169e4b2dd5d

git clone https://github.com/Akira-221/Dualformer.git external/Dualformer
git -C external/Dualformer checkout ebd4ccf8bc5634f0c965d0b8d5797d1b926daa19

git clone https://github.com/SakanaAI/SearchCast.git external/SearchCast
git -C external/SearchCast checkout 9a12b22525d787c0e0f919b2bd5b26fec5d64d03
```

Optuna is already a NeuralForecast dependency. The optional research dependencies
are not added to the core package requirements. `reformer-pytorch` is necessary
because both official attention modules import it even when their selected model
does not use LSH attention.

The compatibility loader verifies the pinned commit and rejects modified tracked
files. Supply only trusted local source directories. It is not a Python sandbox.
It scopes the upstream `layers`/`utils` imports so both architectures coexist
without permanently modifying `sys.path` or overwriting unrelated modules.
Use a single-process trainer (`devices=1`); spawned distributed training and
concurrent imports of unrelated packages into those absolute namespaces have
not been validated.

### Weight and source licenses

TimesFM 3.0's default weights (`google/timesfm-3.0-pytorch`) are governed by
**`timesfm-non-commercial-license-v1.0`: non-commercial, non-production use only**,
according to the [official README](https://github.com/google-research/timesfm/tree/8cb0628371af142e16b8c232cc9fbf667ffb12f9).
The Apache-2.0 source license does not remove the separate weight restriction.
The adapter warns when loading this default checkpoint. No pretrained weights
or upstream research source files are bundled in this repository.

Review each external checkout/checkpoint's own license before use or redistribution.
This PR does not assign NeuralForecast's Apache license to SeesawNet or Dualformer
source for which a grant has not been established here.

## Run all five at `h=72`

`df` must contain `unique_id`, `ds`, and `y`, with at least 700 complete hourly
observations per series for the configuration below. Provide your own data;
this example does not download benchmark datasets or checkpoints until a
foundation-model prediction actually runs.

```python
from neuralforecast import NeuralForecast
from neuralforecast.models import TimesFM3, Chronos2, SeesawNet, Dualformer, SearchCast

h = 72
runtime = dict(accelerator="cpu", devices=1, logger=False)
models = [
    TimesFM3(h=h, input_size=96, backend_device="cpu", **runtime),
    Chronos2(h=h, input_size=96, backend_device="cpu", **runtime),
    SeesawNet(h=h, input_size=96, source_dir="external/SeesawNet",
              max_steps=1000, **runtime),
    Dualformer(h=h, input_size=96, source_dir="external/Dualformer",
               max_steps=1000, **runtime),
    SearchCast(h=h, input_size=96, source_dir="external/SearchCast",
               n_trials=20, n_folds=2, cv_val_size=72,
               horizon_group_size=24, **runtime),
]
nf = NeuralForecast(models=models, freq="h")
nf.fit(df=df, val_size=72)
forecasts = nf.predict()  # t+1, ..., t+72 for each model and series
nf.save(path="checkpoints/short_horizon", overwrite=False, save_dataset=True)
```

`h` is measured in **observations**, not automatically in hours. ETTh1 hourly
`h=72` is 72 hours; standard Weather at ten-minute frequency `h=72` is 12 hours.
This integration does not establish SOTA on either dataset: train/validation/test
splits, scaling, context budgets, and per-lead metrics still need a controlled
benchmark. Foundation-model pretraining and training-from-scratch are different
experimental conditions.

### Covariate usage

```python
# Past-only sensor observations; don't pretend their future values are known.
a = TimesFM3(h=72, input_size=96, hist_exog_list=["density", "temperature"])
b = Chronos2(h=72, input_size=96, hist_exog_list=["density", "temperature"])
c = SeesawNet(h=72, input_size=96, source_dir="external/SeesawNet",
              hist_exog_list=["density", "temperature"])
d = Dualformer(h=72, input_size=96, source_dir="external/Dualformer",
               hist_exog_list=["density", "temperature"])
```

For TimesFM3/Chronos2, `futr_exog_list` must have historical AND genuinely known
future values through NF's `futr_df`. Do not mix the same name into historical
and future lists. SearchCast's reviewed method models series through shared or
separate Ridge fits, not through a separate numerical-exogenous-input API; that
unsupported pathway is rejected.

### Dualformer device compatibility

The reviewed upstream AutoCorrelation creates delay indices with unconditional
`.cuda()` in two methods. The loader changes only those allocations to
`.to(values.device)` inside its privately loaded module. This enables CPU and
non-default GPU placement without changing the attention computation. The
source checkout stays unmodified; a source mismatch raises an error. Tests
compare both patched aggregation paths against independent roll/gather formulas.

### TimesFM3 details

The adapter calls `timesfm3.TimesFM3Forecaster(ModelConfig(...)).predict_batch`,
not TimesFM2.5's `forecast_with_covariates`/XReg. `input_size <= 15360` and at most
32 total target/covariate channels are enforced, avoiding implicit truncation or
covariate subsampling. `make_positive=False` avoids silently clamping forecasts
for standardized or signed targets. Symmetric averaging is optional and defaults
to false. Native median is used for point forecasts (even with `loss=MSE()`);
it is not relabelled as a conditional mean.

```python
from neuralforecast.losses.pytorch import MQLoss
model = TimesFM3(h=72, input_size=96, loss=MQLoss(quantiles=[0.1, 0.5, 0.9]))
```

Only native deciles 0.1–0.9 are supported. Unsupported 0.05/0.95 quantiles are
rejected, rather than fabricated by interpolation. For offline reload, cache the
external checkpoint or use `model_id` as a local path. NF checkpoints store the
model reference/revision, not the external pretrained weights.

### SearchCast details

The implementation delegates scaling, augmentation and closed-form Ridge to
[the pinned official source](https://github.com/SakanaAI/SearchCast/blob/9a12b22525d787c0e0f919b2bd5b26fec5d64d03/optuna_ridge.py).
It supplies NF windows and an Optuna loop instead of running the upstream
fixed-dataset CLI. This is not a renamed `nn.Linear` or a fallback forecast.

- All NF series form one pooled group. Each output group searches lookback,
  local/global scaling, mean/robust centering, local trailing ratio, time/frequency
  noise, noise amplitude, and Ridge regularization. The last output group may be
  shorter, so `h=1`, `h=25`, and `h=72` work without divisibility restrictions.
- `input_size` bounds searched lookback; it is not forced on every horizon group.
  The default Ridge penalty grid is `logspace(-6, 4, 21)` as upstream. This
  adapter searches integer lookbacks in `[min(32,input_size), input_size]`, not
  the paper's fixed 32–2048 grid.
- NF's `val_size + test_size` is removed **before** any fitting/search. Inner
  expanding folds are drawn only from that remaining training prefix. Final
  refitting includes inner validation rows, but still excludes the outer NF
  validation/test rows. An inner fold has at least `h` points.
- The upstream local scaler **centers history and appends standard deviation**;
  it does not divide history by that scale. The official implementation is used
  unchanged. Global scalers are fit only on the fold's training data.
- Learned Ridge weights and selected settings are registered buffers, so NF
  save/load retains the fit. The pinned source directory and optional packages
  must still exist on reload. `max_steps=0`, unweighted `MSE()` and
  `scaler_type="identity"` are required. Missing values, sample weights,
  gradient explanations and distributed fitting are unsupported.
- CPU solves are intentional: the upstream batched-series solver calls CUDA
  memory APIs unconditionally. This uses the CPU-compatible official `solve`
  method, not a silent substitute. Large lookbacks/trial counts remain costly.

## Tests and what they establish

```bash
export SEESAWNET_SOURCE="$PWD/external/SeesawNet"
export DUALFORMER_SOURCE="$PWD/external/Dualformer"
export SEARCHCAST_SOURCE="$PWD/external/SearchCast"
python -m pytest -q -o addopts='' tests/test_short_horizon_models.py
```

Without these environment variables, source-backed tests are skipped. The
included CI workflow sets all three and checks out the exact commits, then runs
actual architecture forward/backward tests, NF fit/predict/save/load, SearchCast
chronological validation and holdout-isolation checks. Pretrained API contract
tests cover every `h=1..72`, covariate routing and quantile validation without
weight downloads. **Those contract tests do not test real pretrained checkpoint
execution, forecast quality, GPU throughput, or paper benchmark reproduction.**
