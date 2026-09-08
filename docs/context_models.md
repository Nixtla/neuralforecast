# Remaining external-context forecasting integrations

This change addresses all nine previously unstarted candidates in the live
`Forecasting Papers / 예측모델` catalog. Seven integrations are implemented below;
SCENARIODIFF and KairosAgent are investigated but blocked, as recorded at the end.
SpecTF/TGForecaster belong to PR #5 and are deliberately not duplicated.

Baseline main: `e5e78363f477d56ed7c86636430b23bcfdedbfcf`. Registration in this
branch is not main deployment, nor evidence of a paper's published accuracy.

## What is implemented

| Class | Actual official implementation | External inputs here | NF fitting |
|---|---|---|---|
| `VoT` | PatchTST_clip's forecast=2 path, trend and seasonal text fusion | Historical event/text embeddings | Supervised point-loss training |
| `GPT4MTS` | Official GPT4MTS patch/text fusion with GPT-2 | Historical text embeddings matching the backbone width | Supervised point-loss training |
| `UniTime` | Official UniTime and UniTimeGPT2 | Per-series domain descriptions selected by a static context ID | Forecast-horizon training |
| `LangTime` | Official LTPratrainedModel, patch encoder, linear adapter and GPT-2 | Per-series external instructions selected by a static context ID | Forecast-horizon training |
| `Aurora` | Official AuroraForPrediction.generate, including its BERT/ViT/flow architecture | Per-series external text selected by a static context ID | Inference only; sample mean point forecast and sample quantiles |
| `ChatTime` | Official discretizer, serializer, prompt and Llama forecasting API | Per-series external text selected by a static context ID | Inference only, original sample median |
| `TabPFNTS` | Official TabPFN-TS preprocessing, feature/predictor pipeline in LOCAL mode | Numerical known-future columns, including their historical values | Local inference/in-context regression; native pipeline quantiles |

These are different architectures, not renamed generic regressors. Text embedding
coordinates are numerical tensors but must come from an appropriate fixed text
encoder; they are **not** a claim of arbitrary exchange-rate/inventory-regressor
support. Aurora's image is generated from the target, not an independent external
image input. Text encoding/reasoning happens outside these NF adapters.

The trainable adapters and ChatTime remain point-output integrations. Aurora and
TabPFNTS additionally expose their backend-native predictive uncertainty through
`NeuralForecast.predict(quantiles=[...])`: Aurora computes requested quantiles from
the same generated sample trajectories used for its point forecast, while TabPFNTS
forwards the requested quantiles to the official pipeline. This does not introduce
trainable probabilistic NF loss heads, Auto tuning wrappers, categorical embedding
APIs or untested distributed/gradient-explanation paths. Complete history and
identity NF scaling are required. Preprocess embeddings consistently across train
and test. Only the four description-based classes use `stat_exog_list` as shown
above; these IDs select external text, not arbitrary numeric static features.
Invalid IDs and non-finite forecasts fail explicitly.

## Explicit local setup

Use an editable checkout of this fork's integration branch. Keep optional language
model dependencies separate from projects that need incompatible Transformers
versions (in particular older ChronosX environments).

```sh
python -m pip install -e .
python -m pip install 'transformers==4.48.3' 'huggingface-hub==0.36.0' \
  'reformer-pytorch==1.4.4' 'einops==0.8.1' 'accelerate>=1,<2' sentencepiece scikit-learn matplotlib
python scripts/fetch_context_sources.py "$HOME/nf-context-sources"
```

Aurora also requires torchvision compatible with the installed Torch. The CPU CI
pairs Torch 2.9.1 with torchvision 0.24.1. TabPFNTS is optional and uses the reviewed
official package revision:

```sh
python -m pip install \
  'tabpfn-time-series @ git+https://github.com/PriorLabs/tabpfn-time-series@e1c9d6afdf949e41a175009241300665eec07573'
```

The source script requires Git and network access, pins revisions, refuses existing
output model directories, and sparse-checks out only needed code/configuration and
notices. It downloads neither training data nor checkpoints. Model constructors
never install dependencies, clone code or fetch weights. Supply local compatible
checkpoints/tokenizers explicitly. The non-pretrained GPT4MTS option needs no
checkpoint, but its randomly initialized GPT-2 must be trained.

| Source | Revision |
|---|---|
| https://github.com/decisionintelligence/VoT | `f769621da9efde3e82a1975f478efdec83dcbd10` |
| https://github.com/Flora-jia-jfr/GPT4MTS-Prompt-based-Large-Language-Model-for-Multimodal-Time-series-Forecasting | `c85e74b2fd7048f5ef601453e5bcb9f6073cd3b4` |
| https://github.com/liuxu77/UniTime | `09acfbe9c63fc67db7539590094ac5f328230654` |
| https://github.com/niuwz/LangTime | `1feac37753b221a99c293eb0fa7d223a841283b3` |
| https://github.com/decisionintelligence/Aurora | `a247760abbc9d17a861bc365c032368d317815f2` |
| https://github.com/ForestsKing/ChatTime | `8c2d4c209302d2b2bd6cc3c154586842341e3247` |
| https://github.com/PriorLabs/tabpfn-time-series | `e1c9d6afdf949e41a175009241300665eec07573` |

The source loader checks every imported local Python file and Aurora's constructor
JSON/vocabulary resources against recorded Git blob hashes, including transitive local imports.
Only reviewed local imports and the small compatibility edits below are changed
in memory. No global `layers`/`utils` import replacement is used. This is import
isolation and compatibility checking, **not a security sandbox**. Use trusted
sources/dependencies. External source and weights are not redistributed or
relicensed; their own applicable licenses and model-access terms remain in force.
In particular, supplying an accessible checkpoint does not grant commercial rights.

## Input examples

```python
from pathlib import Path
from neuralforecast import NeuralForecast
from neuralforecast.models import VoT, GPT4MTS, UniTime, LangTime

sources = Path.home() / "nf-context-sources"
# embedding_0 ... embedding_767 are a fixed encoder's historical text vectors.
embedding_columns = [f"embedding_{i}" for i in range(768)]
vot = VoT(h=12, input_size=48, source_dir=str(sources / "VoT"),
          hist_exog_list=embedding_columns, patch_len=4, stride=4, max_steps=200)
gpt = GPT4MTS(h=12, input_size=48, source_dir=str(sources / "GPT4MTS"),
              hist_exog_list=embedding_columns, backbone_path="/local/gpt2",
              gpt_layers=2, max_steps=200)
contexts = ["Daily electricity demand at a manufacturing site.",
            "Daily demand at a retail location with weekend closures."]
unitime = UniTime(h=12, input_size=48, source_dir=str(sources / "UniTime"),
                  backbone_path="/local/gpt2", contexts=contexts,
                  stat_exog_list=["context_id"], patch_len=8, max_tokens=128)
langtime = LangTime(h=12, input_size=48, source_dir=str(sources / "LangTime"),
                    backbone_path="/local/gpt2", contexts=contexts,
                    stat_exog_list=["context_id"], patch_len=8)
# For description models, static_df includes unique_id and an integer context_id
# in [0, len(contexts)). Supply df's unique_id, ds and y as usual.
nf = NeuralForecast(models=[unitime], freq="D")
nf.fit(df=df, static_df=static_df)
predictions = nf.predict()
```

Choose each model's matching input schema rather than passing incompatible lists
to all models. VoT and GPT4MTS require historical embeddings for every time step;
future text lists and static IDs are rejected for those classes. VoT requires an
embedding width divisible by eight. GPT4MTS's text width must equal GPT-2's hidden
width. UniTime/LangTime require `input_size` divisible by `patch_len`. Descriptions
must fit language-model position capacity; they are not silently truncated.

```python
from neuralforecast.models import Aurora, ChatTime, TabPFNTS

aurora = Aurora(h=12, input_size=96, source_dir=str(sources / "Aurora"),
                model_id="/local/official-aurora", tokenizer_path="/local/bert-tokenizer",
                contexts=contexts, stat_exog_list=["context_id"], num_samples=100)
chattime = ChatTime(h=12, input_size=96, source_dir=str(sources / "ChatTime"),
                    model_id="/local/official-chattime", contexts=contexts,
                    stat_exog_list=["context_id"], num_samples=8)
tabpfn = TabPFNTS(h=12, input_size=96,
                  model_id="/local/tabpfn-v3-regressor-v3_default.ckpt",
                  futr_exog_list=["holiday", "planned_production"])

# Both Aurora and TabPFNTS support backend-native quantile output.
quantile_forecast = NeuralForecast(models=[aurora], freq="D")
quantile_forecast.fit(df=df, static_df=static_df)
quantiles = quantile_forecast.predict(quantiles=[0.1, 0.5, 0.9])
```

For TabPFNTS, provide each feature's historical values in `df` and origin-known
future values in `futr_df`. Unknown future market observations must not be used.
The adapter forces LOCAL mode and disables telemetry; no cloud-client prediction
is selected. It constructs a relative time index solely for ordering and uses only
the official running-index generator. **Actual calendar dates are not fabricated**:
provide real calendar/seasonal columns yourself. Each NF window is fit separately,
with no cross-window reference or target pooling. Past-only and static features
are rejected. The example checkpoint is a path schema, not a bundled file.

Aurora/ChatTime require local safetensors directories with architecture-compatible,
trained weights and matching tokenizers. Generic language-model weights are not
ChatTime forecast weights. Their `fit` registers history; positive `max_steps`
does not fine-tune these backends and is rejected. Aurora's default point forecast
is the sample mean and requested quantiles are calculated from its generated sample
paths; ChatTime uses its original median across parsed numerical samples.
Unparseable or non-finite outputs raise instead of being replaced by baseline forecasts.

## Deliberate integration boundaries and compatibility edits

- VoT uses its actual supervised forecasting path, not event extraction, a reasoner
  service or the original contrastive pretraining schedule. Upstream text
  rescaling averages the batch; per-window calls avoid cross-window leakage.
- GPT4MTS reuses its original fusion and GPT-2 architecture. The hardcoded download
  path becomes an explicit local path, and random GPT-2 configuration is explicit.
  A singleton-dimension `squeeze()` is removed from pooled text patches, preserving
  the batch and patch axes for batch=1/horizon=1. A regression test covers this.
- UniTime uses original domain-description tokenization and its decoder, but NF
  optimizes only the requested future horizon, not its masked reconstruction and
  multi-domain curriculum. Descriptions are selected per series/window.
- LangTime uses the original GPT-2/patch/linear-adapter route and control tokens.
  Its hardcoded GPT-2 width is read from the local checkpoint (still 768 for base
  GPT-2). PPO, instruction search and reconstruction-loss training stay external.
- Aurora uses the original generated-image/text encoding, prototype retrieval and
  flow sampler. Its period selector aggregates batches, so windows run separately.
  The wrapper exposes static descriptions, not timestamped news streams or an
  arbitrary dynamic numeric input. BERT context is limited to 125 tokens. Native
  quantiles are empirical quantiles over the generated sample trajectories.
- ChatTime retains original serialization and autoregressive chunking. NumPy 2's
  removed `np.NaN` spelling becomes `np.nan`; loading is local safetensors/float32
  before explicit device placement instead of automatic half-precision dispatch.
  The checkpoint/tokenizer are trusted local inputs; arbitrary remote model code
  is not enabled by the adapter.
- TabPFNTS passes requested quantile levels to the official LOCAL pipeline and
  returns its `target` point forecast plus the corresponding quantile columns.

## Persistence and validation

All seven classes are exported and registered for NF checkpoint lookup. Trainable
models save their learned parameters; source checkouts and backbone/tokenizer
paths must still exist for reconstruction. Inference-only NF checkpoints store
configuration and paths, **not** external model weights. Static descriptions are
stored in model configuration, so do not publish checkpoints containing private
text. Source execution and model loading happen locally, but downstream code must
still respect each dataset/model license and privacy policy.

`tests/test_context_models.py` separates real-model and controlled-component tests:
actual VoT/GPT4MTS/UniTime/LangTime forward/backward, condition sensitivity,
future-label exclusion, window isolation, NF fitting, persistence and reload in a
fresh interpreter; actual small BERT/ViT/Aurora trained briefly with its own loss
and saved/reloaded as safetensors; actual ChatTime API and tiny Llama/tokenizer
loading with **controlled generated responses**; official TabPFN-TS preprocessing,
workers and NF persistence with **controlled gated-regressor methods**. The latter
two are routing/compatibility tests, not pretrained numerical forecast execution.
`tests/test_context_probabilistic.py` additionally verifies the NF point-plus-quantile
column contract for Aurora samples and TabPFN-TS pipeline quantiles, including reset
to point-only prediction. Tiny models/checkpoints use synthetic inputs and are not
published model weights.

The dedicated workflow uses Python 3.11, Torch 2.9.1 CPU and Lightning 2.5.6, checks
package consistency, and uploads JUnit results. Check the workflow for the exact
commit before deployment. Full repository/platform CI, GPU/MPS, realistic latency,
pretrained checkpoint accuracy and paper-score reproduction are not asserted.

## Investigated but not implemented

| Candidate | Evidence and blocker, checked 2026-09-08 | Unblocking requirement |
|---|---|---|
| KairosAgent | https://foundation-model-research.github.io/KairosAgent/ lists the paper but no verified forecasting implementation was found. The similarly named Kairos TSFM is a different project. | Author-confirmed executable repository and its actual forecast/input interface |
| SCENARIODIFF | The catalog's official link is https://anonymous.4open.science/r/ScenarioDiff_ICDM-2C4C . The code endpoint could not be retrieved in this environment, so implementation, dependencies and checkpoint interface were not audited. This is an access blocker, not a claim that code does not exist. | Accessible official source snapshot and compatible training/checkpoint instructions |

No placeholder classes for these two are registered. The catalog's main-based
`my_forecast` flags are not changed by an unmerged feature branch.
