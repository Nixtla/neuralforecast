# SpecTF and TGForecaster: explicit text-covariate integrations

These two models were selected from `Forecasting Papers / 예측모델`: external
variable support `O`, official code present, and `my_forecast` status `X` on
2026-09-08. Neither is in the baseline `main` commit
`e5e78363f477d56ed7c86636430b23bcfdedbfcf`.

They execute the actual official architectures, not renamed generic baselines.
The integrations use the existing NeuralForecast point-loss trainer, window
validation, and checkpoint registry. Text encoding is an external preparation
step; no language model, embedding service, dataset, or pretrained checkpoint is
downloaded when importing or running these models.

| Model | Integration | External input supported here |
|---|---|---|
| `SpecTF` | Official `TextEncoder` + `FreqModelHistPred`, history-fusion configuration | A dense historical text embedding per time step, via `hist_exog_list` |
| `TGForecaster` | Official `models/TGTSF_torch.py` | News and description embeddings already known at the forecast origin, via `futr_exog_list` |

**Text embeddings are not arbitrary price, exchange-rate, or inventory
regressors.** The mathematical input is numerical, but its coordinates must come
from an appropriate fixed text encoder. These adapters do not claim that ordinary
market columns reproduce the papers' text-conditioned models.

## Setup

In an editable checkout of this fork/branch:

```sh
python -m pip install -e . 'reformer-pytorch==1.4.4' 'einops==0.8.1'
python scripts/fetch_research_sources.py "$HOME/nf-text-sources" SpecTF TGForecaster
```

The existing source-fetch script downloads reviewed source revisions explicitly,
refuses to overwrite existing directories, and excludes model weights and dataset
directories through sparse checkout. Optional dependencies do not enter this
fork's mandatory dependency list. `reformer-pytorch` is imported by SpecTF's
official attention file even though this configuration uses full-frequency
attention, not its LSH attention implementation.

| Source | Revision | Entrypoint |
|---|---|---|
| https://github.com/hiepnh137/SpecTF | `85185c7b883fed7de40098d76ce6782ba1eba016` | `models/SpecTF.py` |
| https://github.com/VEWOXIC/TGTSF | `fdf10ceea422c0bf13b0013c9d6ca48179bf9de7` | `models/TGTSF_torch.py` |

The loader validates the Git blob hash of each of the four source files required
by each model, including their local dependencies. It changes only local import
paths in memory, using private module names without modifying `sys.path`,
`sys.meta_path`, or global `layers`/`utils` packages. Package-wide benchmark
initializers are not executed. Modified files are rejected even after a cached
load. This is compatibility checking and namespace isolation, not a security
sandbox. Use trusted checkouts and dependencies. Upstream code and weights are
not redistributed or relicensed by this change; their own terms still apply.

## Historical text: SpecTF

```python
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import SpecTF

sources = Path.home() / "nf-text-sources"
# Example schema: a fixed encoder produces 384 coordinates per historical step.
text_columns = [f"text_{i}" for i in range(384)]
model = SpecTF(
    h=8, input_size=32, source_dir=str(sources / "SpecTF"),
    hist_exog_list=text_columns, max_steps=200, scaler_type="identity",
)
nf = NeuralForecast(models=[model], freq="D")
# df: unique_id, ds, y, text_0 ... text_383; one row per series/time step.
nf.fit(df=df, val_size=8)
prediction = nf.predict()  # Future text is not required or consumed.
```

Only the official text/history-fusion configuration is exposed. Its optional
calendar-feature path, sum/product/only-text ablations, and future text are not
exposed here. The `freq="h"` used internally configures an unused time-feature
projection; no hourly calendar data are invented or consumed. Use the actual
series frequency in `NeuralForecast(freq=...)`. The model's spectral width must
be even, and `input_size <= 9998` respects the official positional buffer.

## Origin-known text: TGForecaster

```python
from neuralforecast.models import TGForecaster

width = 384
columns = [f"news_{i}" for i in range(width)] + [f"description_{i}" for i in range(width)]
model = TGForecaster(
    h=8, input_size=32, source_dir=str(sources / "TGForecaster"),
    text_dim=width, futr_exog_list=columns, patch_len=4, max_steps=200,
)
nf = NeuralForecast(models=[model], freq="D")
nf.fit(df=df, val_size=8)
# futr_df: unique_id, ds and ALL news/description columns for each horizon row.
prediction = nf.predict(futr_df=futr_df)
```

Column order is news embedding followed by description embedding, both with
`text_dim` coordinates. `df` must contain their historical values for NF's future
covariate schema; the official TGTSF model consumes only the forecast segment of
these columns. The adapter mean-pools all per-step vectors within each
non-overlapping forecast patch, rather than silently discarding intermediate
rows. News and description use the same embedding width/space. `h` and
`input_size` must be divisible by `patch_len`; `text_dim` must divide by
`n_heads`. The temporal and text widths are equal, as required by this upstream
fusion implementation.

One pooled news vector per patch is supported. Variable-length article sets,
article-level missing masks, raw text, and automatic encoding are not exposed.
Prepare a finite embedding of empty text when a step has no article. If a patch
has one known description, repeating that vector across its rows preserves it.

**Do not place future news that was published after the forecast origin into
`futr_df`.** This path is for preannounced schedules, metadata, or explicit
origin-known scenarios. Historical backtests must recreate the information
available at each forecast origin, not use hindsight article embeddings.

## Boundaries and persistence

Both models are trainable from scratch through `NeuralForecast.fit`. Both expose
one target per series and MAE/MSE point losses. Fully observed windows are
required; start padding, separate static/categorical inputs, unsupported feature
lists, non-finite inputs, and non-identity NF scaling are rejected. No Auto tuning
wrapper, probabilistic head, distributed support, or GPU/MPS validation is claimed.

`NeuralForecast.save` stores the trained weights and source path;
`NeuralForecast.load` reconstructs the same official architecture and loads those
weights. Keep the checked source directory and optional packages available on
reload. Source directories and environments are not bundled into checkpoints.
The tests include a reload in a fresh Python interpreter, not just a warm import.

## Validation

```sh
NF_TEXT_SOURCE_ROOT="$HOME/nf-text-sources" python -m pytest -q -o addopts='' \
  tests/test_text_models.py tests/test_exogenous_models.py -m 'not optional'
```

The new tests execute the actual official architectures without backend doubles:
finite output and gradients, sensitivity to external embeddings, no future target
input, all-row patch pooling, invalid inputs, namespace isolation, modified-source
rejection, and NF fit/predict/save/cold-load. Previous-batch tests also run; some
of those pre-existing tests intentionally use backend contract doubles.

Local validation initially passed 55 tests (23 new + 32 prior), with one optional
TTM test deselected, on Python 3.13.5 / Torch 2.10.0+cpu / Lightning 2.6.5.
That local Lightning version is outside the package's declared range, so it is
not a supported-environment claim. The dedicated `Text exogenous models` CI uses
Python 3.11, Torch 2.9.1+cpu, and Lightning 2.5.6, and uploads JUnit results.
Consult the CI result for the exact commit. Passing these functional tests does
not establish forecasting accuracy, real-data performance, or paper replication.
