# Eight additional exogenous forecasting integrations

This batch adds eight models not present in the main branch after PR #1. Selection used the live `Forecasting Papers` / `예측모델` table: external-variable support `O`, an official code link, and `my_forecast` status `X`. Models from PR #1 were excluded even where the sheet's status was stale.

These integrations reuse official implementations rather than substituting generic networks under paper names. Six require explicit local source checkouts. The other two use their official installable packages. Importing NeuralForecast neither downloads source code nor installs optional packages.

## Supported configurations

| Sheet name / class | NF training | Inputs supported here | Implementation boundary |
|---|---|---|---|
| DAG / `DAG` | Yes | Nonempty `futr_exog_list`, with past and future numerical values | Official dual-correlation model and auxiliary reconstruction losses; not a causal-discovery estimator |
| KITE / `KITE` | Yes | Exactly one of `hist_exog_list` or `futr_exog_list` | Original conditional flow-matching objective; point forecast averages sampled trajectories |
| GLAFF / `GLAFF` | Yes | Six known-future numerical timestamp features | Official GLAFF plugin composed with the existing NF DLinear backbone |
| APT / `APT` | Yes | Two known-future calendar columns, time of day then day of week | Official shared-prototype affine module and regularizers, composed with NF DLinear |
| Moirai 2.0 / `Moirai2` | Inference only | Past-only and known-future numerical covariates | Original Moirai 2.0; returns the median quantile, not the mean across quantiles |
| ChronosX / `ChronosX` | Inference only | Past-only and known-future numerical covariates | Original IIB+OIB model; requires a compatible locally fine-tuned checkpoint |
| Baguan-TS / `BaguanTS` | Inference only | Known-future numerical covariates | Original TS-tabular forecasting and retrieval; requires a local checkpoint and matching model YAML |
| RAG4CTS / `RAG4CTS` | Inference only | Known-future numerical covariates | Original fixed-k retrieval/input-construction pipeline with Chronos-2; no right-context target hints |

`futr_exog_list` always requires both historical values in `df` and known future values in `futr_df`. Do not put observations unavailable at the prediction cutoff into `futr_df`. An external market indicator with unknown future values belongs in `hist_exog_list`, which not all integrations support. GLAFF/APT's calendar support must not be interpreted as unrestricted support for economic variables or text.

These are univariate-target-per-series integrations, not NF `MULTIVARIATE` models. Static features and categorical embedding APIs are unsupported. Outputs are point forecasts; MAE/MSE are supported for scoring. No Auto tuning wrappers, probabilistic NF heads or distributed/GPU performance guarantees are added. KITE uses its original flow objective regardless of the default MAE metadata; select `valid_loss` to change forecast scoring.

## Install the main environment and source checkouts

Use a checkout of this fork/branch and a Python 3.11 environment. The examples below use POSIX paths.

```sh
python -m pip install -e .
python -m pip install einops reformer-pytorch matplotlib
python scripts/fetch_research_sources.py "$HOME/nf-research-sources"
# Only for RAG4CTS's backend:
python -m pip install chronos-forecasting
```

The source-fetch helper requires Git, uses the revisions below, sparse-checks out the relevant code and refuses to overwrite existing paths. It does not fetch pretrained checkpoints. Source entrypoint hashes are checked at loading. This is a compatibility check, not a sandbox or verification of every transitive source file. Use trusted checkouts. Replacing code at a reviewed path requires a new review/hash update.

| Source checkout | Pinned revision | Main entrypoint |
|---|---|---|
| [decisionintelligence/DAG](https://github.com/decisionintelligence/DAG) | `0758990e2c73bb54138ea3e7b11a35cbc5476bcc` | `ts_benchmark/baselines/dag/models/dag_model.py` |
| [decisionintelligence/KITE](https://github.com/decisionintelligence/KITE) | `3140ee824cbd80c5ec7fdf2b54666210519d0b39` | `ts_benchmark/baselines/kite/models/KITEModel.py` |
| [ForestsKing/GLAFF](https://github.com/ForestsKing/GLAFF) | `4dedf10e0028b519780645ef5824810f4b1bdb55` | `plugin/Plugin/model.py` |
| [blisky-li/APT](https://github.com/blisky-li/APT) | `98a4c9c017666207b02029f842eff92818f0eab8` | `baselines/Normalization/normalization/APT.py` and original loss functions |
| [RAG4CTS-Project/RAG4CTS](https://github.com/RAG4CTS-Project/RAG4CTS) | `28143b3c3b13ef6be5eba3d28e051fc7bd662cf3` | `RAG4CTS/rag_pipeline.py` and `retriever.py` |
| [jxgogo/Baguan-TS](https://github.com/jxgogo/Baguan-TS) | `8b55d93eb52c2f2a9d393dfebdd33fc5a7466b8e` | `BaguanTS.py`, `src/pipeline/factory.py` |
| [SalesforceAIResearch/uni2ts](https://github.com/SalesforceAIResearch/uni2ts) | `cfd46d4510ed8896f263116f32928eede05b0a75` | `src/uni2ts/model/moirai2/forecast.py` |
| [amazon-science/chronos-forecasting, chronosx branch](https://github.com/amazon-science/chronos-forecasting/tree/chronosx) | `2b52bfc500e3ebab1ce846f0a2a60ee5ef2a14a7` | `src/chronosx/chronosx.py` |

Upstream code and weights are not redistributed in this change. Those projects retain their own code/model licenses and notices; this fork's Apache license does not relicense externally loaded sources. Inspect each upstream's applicable terms before redistribution or deployment. An accessible repository alone does not establish those permissions.

## Trainable models

```python
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import DAG, KITE, GLAFF, APT

sources = Path.home() / "nf-research-sources"
model = KITE(
    h=12, input_size=48, source_dir=str(sources / "KITE"),
    hist_exog_list=["market_indicator"], max_steps=200,
    windows_batch_size=32, num_sampling_steps=20, num_samples=20,
)
# df contains unique_id, ds, y and market_indicator. Its future values are not required.
nf = NeuralForecast(models=[model], freq="D")
nf.fit(df=df, val_size=12)
predictions = nf.predict()
```

Other model construction examples, for data with their respective schemas:

```python
dag = DAG(h=12, input_size=48, source_dir=str(sources / "DAG"),
          futr_exog_list=["known_schedule"], max_steps=200)
glaff = GLAFF(h=12, input_size=48, source_dir=str(sources / "GLAFF"),
              futr_exog_list=["time_1", "time_2", "time_3", "time_4", "time_5", "time_6"])
apt = APT(h=12, input_size=48, source_dir=str(sources / "APT"),
          futr_exog_list=["tod", "dow"], time_of_day_size=24, warmup_steps=10)
```

GLAFF expects the original six timestamp features, with the same representation for past and future data; it uses `scaler_type="identity"`. APT requires the exact order `[time_of_day, day_of_week]`, encoded as `index / cardinality - 0.5`. For hourly data use `hour / 24 - 0.5` and `dayofweek / 7 - 0.5`. Invalid calendar bins are rejected, not silently clamped.

DAG trains with the source's forecast loss plus its auxiliary loss. KITE's training step invokes the original flow loss with supervised future targets; its inference call never receives those targets. KITE requires horizon >= 2 and does not support a mixture of past-only and known-future feature lists in one instance. Its historical-only path disables future conditioning, with a shape-only zero tensor avoiding an upstream `zeros_like(None)` error.

APT's original epoch-based stages are translated to NF steps: `warmup_steps` regularizer-only steps with DLinear frozen, one forecast-loss step with DLinear still frozen, then joint forecast-loss training. This preserves the staged objectives but is not a claim to reproduce the original epoch schedule. Warmup needs at least two sampled windows. Near-zero affine scales raise an error instead of changing the mathematical inverse. This integration selects dependent/shared prototypes and no additional RevIN module.

All four trainable models require complete history and forecast training windows (`training_data_availability_threshold=1.0`, no start padding). DAG/KITE/APT reject sample weights because their specialized objectives do not implement per-example weighting. GLAFF uses the usual NF point-loss path.

## Isolated inference environments

Moirai2 requires the same separation as the first batch's Moirai adapters because uni2ts's Torch constraint conflicts with this fork. ChronosX pins an older, separate Chronos/Transformers stack. Do not install those into the main environment by ignoring dependency constraints.

```sh
python3.11 -m venv "$HOME/.venvs/nf-uni2ts"
"$HOME/.venvs/nf-uni2ts/bin/python" -m pip install \
  'uni2ts @ git+https://github.com/SalesforceAIResearch/uni2ts@cfd46d4510ed8896f263116f32928eede05b0a75'

python3.11 -m venv "$HOME/.venvs/nf-chronosx"
"$HOME/.venvs/nf-chronosx/bin/python" -m pip install \
  'chronosx @ git+https://github.com/amazon-science/chronos-forecasting@2b52bfc500e3ebab1ce846f0a2a60ee5ef2a14a7'

python3.11 -m venv "$HOME/.venvs/nf-baguan"
"$HOME/.venvs/nf-baguan/bin/python" -m pip install \
  'torch>=2.5' 'lightning==2.5.1' 'einops==0.8.1' numpy scipy pyyaml scikit-learn
```

For CPU-only installations select the CPU Torch index first, using versions compatible with the relevant backend. The dedicated workflow records the concrete CPU test environments. GPU acceleration and training dependencies of the original research projects may require additional setup.

```python
from neuralforecast.models import Moirai2, ChronosX, BaguanTS, RAG4CTS

moirai2 = Moirai2(
    h=12, input_size=128,
    hist_exog_list=["market_indicator"], futr_exog_list=["known_schedule"],
    backend_python=str(Path.home() / ".venvs/nf-uni2ts/bin/python"),
)
chronosx = ChronosX(
    h=12, input_size=128, futr_exog_list=["known_schedule"],
    model_id="/absolute/path/to/finetuned-chronosx-safetensors-directory",
    backend_python=str(Path.home() / ".venvs/nf-chronosx/bin/python"),
    hidden_dim=256, num_layers=1,
)
baguan = BaguanTS(
    h=12, input_size=128, futr_exog_list=["known_schedule"],
    source_dir=str(sources / "BaguanTS"),
    config_path="/absolute/path/to/matching-model-config.yml",
    model_id="/absolute/path/to/trained-baguan.ckpt",
    backend_python=str(Path.home() / ".venvs/nf-baguan/bin/python"),
    context_size=64, neighbors=5,
)
rag = RAG4CTS(h=12, input_size=128, source_dir=str(sources / "RAG4CTS"),
              futr_exog_list=["known_schedule"], query_size=32, neighbors=3)
```

Use these with `NeuralForecast.fit(df=...)` to register history, then `predict(futr_df=...)`. Their NF fit is bookkeeping only: it does not train/fine-tune weights. Positive `max_steps` and gradient explanations are rejected. Backend device defaults to CPU. Use a complete path to the separate environment's Python, not just `python` from PATH.

Moirai2 defaults to `Salesforce/moirai-2.0-R-small`; a local official checkpoint can replace the ID. Its patch/quantile architecture is fixed by the original module. Inherited non-default patch/sample options are rejected, and output is the actual median quantile.

ChronosX requires a checkpoint containing both input/output injection-block parameters, `config.json` and safetensors. The presence check does not establish its training quality. Base Chronos checkpoints are not substituted. Covariate ordering must match training: past-only features, then known-future features; values are followed by missing-indicator channels, following the official loader. Unavailable future past-only channels use value -1 with missing indicator 1. Injection width/depth must match the supplied checkpoint.

BaguanTS requires a tensor-only checkpoint and a trusted matching YAML. The worker reconstructs the original initializer's used attributes because the upstream constructor references an undefined, unused `StandardScaler`. It replaces unsafe checkpoint loading with `torch.load(weights_only=True)` and retains the original `predict` method. It does not manufacture pretrained weights. `h < context_size <= input_size` is enforced. Its retrieval examples end within the supplied history.

RAG4CTS uses the original `coarse_wcos_fine_weuc` retriever and `predict_with_fixed_k`. Reference windows end before the query's historical segment. Each NF window has its own bank, with no cross-window fitting or pooling. Future query target placeholders are zero-weighted for retrieval and are never sent as target hints to Chronos-2. Right padding, alignment and adaptive k-search are disabled. `input_size >= 2*query_size+h` is required. Columns `y`, `id` and `time` are reserved. This is a fixed-k forecasting integration, not a reproduction of right-context reconstruction benchmarks.

## Persistence and performance

All eight classes are exported from `neuralforecast.models` and registered for NF save/load. Trainable models save their model parameters; loading still requires the recorded source checkout and optional packages. Inference checkpoints save configuration/model references, not the external model weights, source checkouts or backend virtual environments. Keep those paths/checkpoints available, or adjust configuration when moving environments.

Isolated adapters start a worker and load the model per inference batch. This favors dependency isolation over throughput. Small inference batches reduce memory but increase startup overhead. A persistent worker would be a separate optimization. Do not use distributed execution or automatic benchmarking loops without testing resource use first.

## Validation scope

`tests/test_research_models.py` covers real official-source/NF fit-predict-save-load for four trainable models; official auxiliary/flow/staged-objective checks; actual RAG retrieval with a controlled prediction backend; registration, schema checks and future-label exclusion. `tests/test_research_backend_smoke.py` separates backend-transport doubles from actual official architectures initialized at small size and saved as random checkpoints. These smoke checkpoints test executable API/serialization compatibility, not forecast accuracy or pretrained model quality.

The dedicated `Research exogenous models batch 2` workflow also runs first-batch regression tests. Inspect its results for the exact commit being deployed. A passing targeted workflow is not a claim that all repository/platform tests pass. No real pretrained checkpoint evaluation, full-size GPU benchmark or published accuracy reproduction is claimed by this change.
