# Auto search-space audit

Baseline: `eb48022afd9500787866731ec1fd8846dff1b912` (`main`, 2026-09-12).

## Scope and decision rule

All **51 exported Auto wrappers** were reviewed: 35 native default spaces, 15 external trainable wrappers, and the user-configured AutoHINT wrapper. The 13 inference-only adapters in `inference_tuning.py` expose a separate API and are outside this Auto-class change.

The review follows each default space through its builder, Ray/Optuna translation, the model constructor and the consumed forward/training path. Bounded additions cover implemented capacity, regularization, normalization, window sampling and learning-rate decay. These are starter ranges; this audit does not establish forecast accuracy or convergence on a user's dataset. The expanded space also needs an appropriate trial budget.

A disabled search uses `None` when that value is supported. Required values are resolved explicitly: multivariate series batches use `n_series`; StemGNN requires the literal `n_stacks=2`. Conditional parameters can remain useful in other configurations (for example, recurrent dropout at depth > 1, or an MLP's hidden width with hidden layers). They are distinguished from parameters that are unused throughout the default path.

## Shared changes

- Add `num_lr_decays` choices `[0, 1, 3]` to all 50 automatically built trainable spaces. Custom `config` arguments and AutoHINT's user-supplied space remain unchanged.
- Add **188 native model/parameter search entries**, including the 35 native learning-rate decay entries; the external common builder adds 15 more model/parameter entries.
- Remove the five ineffective `context_size` domains, replacing them with `None`. Replace their single-choice inference-length domains with `inference_input_size=None` (follow the training context).
- Disable series-batch search for the 11 joint/multivariate models. Their builders use the required series count; BaseModel and the Ray dtype bridge also preserve explicitly supplied `None` for the default-config-copy path.
- Keep window stride, optimizer updates and batch cardinality separate. `max_steps` retains its existing budgets; no SH/ASHA scheduler is installed by this change.
- Make six formerly float-valued `max_steps` grids explicit integer choices from 500 through 1500 at intervals of 100.
- Convert Ray's exclusive integer upper bound to Optuna's inclusive upper bound using `upper - 1`. Use `suggest_float` for uniform/log-uniform domains.
- Remove duplicate `None`/`identity` scaler alternatives where both were present.
- Enforce structural context minima for short horizons, patch widths, FFT top-k and downsampling. Training-data length is still a dataset-specific constraint; callers should restrict `input_size` to their first training fold's available history.
- Fix FEDformer's decoder Fourier blocks to use its actual `label_len + h`; the previous hard-coded half-context failed for other decoder ratios.

The inherited optimizer class, loss/validation metric, validation cadence, early-stopping protocol, recurrent/direct strategy, accelerator, data availability policy, feature lists and checkpoint/source paths remain caller-controlled. In particular, adding `optimizer_kwargs` alone would be ineffective with the default optimizer=None implementation. No architecture search was invented for linear models that expose no additional architecture controls.

## Native review matrix

Every row also retains its pre-existing effective search entries. "Added" lists keys newly present as domains in the class dictionary.

| Auto wrapper | Added search entries | Disabled / fixed domains | Consumption review and retained constraints |
|---|---|---|---|
| AutoRNN | `encoder_bias`, `encoder_dropout`, `decoder_layers`, `windows_batch_size`, `scaler_type`, `num_lr_decays` | context_size → None; singleton inference length → None | encoder_activation is stored but never passed to nn.RNN; keep it out of HPO. Recurrent/direct mode and h_train stay fixed. |
| AutoLSTM | `encoder_bias`, `encoder_dropout`, `decoder_layers`, `windows_batch_size`, `scaler_type`, `num_lr_decays` | context_size → None; singleton inference length → None | Dropout is active between stacked recurrent layers; a one-layer encoder has no recurrent dropout. Recurrent/direct mode stays fixed. |
| AutoGRU | `encoder_bias`, `encoder_dropout`, `decoder_layers`, `windows_batch_size`, `scaler_type`, `num_lr_decays` | context_size → None; singleton inference length → None | encoder_activation is deprecated and ignored. Dropout is conditional on multiple encoder layers. |
| AutoTCN | `windows_batch_size`, `scaler_type`, `kernel_size`, `dilations`, `encoder_activation`, `decoder_layers`, `num_lr_decays` | context_size → None; singleton inference length → None | context_size is only assigned to an unused attribute; kernel size, dilations, activation and decoder depth are consumed. |
| AutoDeepAR | `decoder_hidden_layers`, `decoder_hidden_size`, `num_lr_decays` | — | Decoder width is conditional on decoder_hidden_layers > 0. trajectory_samples is only stored; do not add it. h_train is a training protocol choice. |
| AutoDilatedRNN | `windows_batch_size`, `scaler_type`, `decoder_layers`, `num_lr_decays` | context_size → None; singleton inference length → None | context_size is only assigned; cell_type and dilations were already searched. |
| AutoBiTCN | `num_lr_decays` | — | Existing hidden_size/dropout controls cover this constructor. input_size >= 2 prevents a zero-layer temporal stack. |
| AutoxLSTM | `encoder_bias`, `decoder_layers`, `decoder_dropout`, `decoder_activation`, `num_lr_decays` | — | Keep the mLSTM backend choice fixed; sLSTM/mLSTM need different optional kernels. Runtime checks require xlstm/mlstm_kernels. |
| AutoMLP | `num_lr_decays` | — | Existing hidden_size/num_layers cover its architecture; no extra architecture knob was invented. |
| AutoNBEATS | `n_blocks`, `mlp_units`, `activation`, `shared_weights`, `n_harmonics`, `n_basis`, `basis`, `num_lr_decays` | — | n_polynomials is ignored and set to None. Positive dropout_prob_theta raises NotImplementedError; keep zero. h=1 uses identity stacks and None for inactive basis controls. |
| AutoNBEATSx | `n_blocks`, `mlp_units`, `activation`, `shared_weights`, `n_harmonics`, `n_polynomials`, `dropout_prob_theta`, `num_lr_decays` | — | Dropout is implemented here. h=1 uses identity stacks and None for inactive harmonic/polynomial controls. Covariate lists remain data-schema inputs. |
| AutoNHITS | `n_blocks`, `mlp_units`, `activation`, `dropout_prob_theta`, `pooling_mode`, `interpolation_mode`, `num_lr_decays` | — | Keep three aligned stack lists. Use linear/nearest interpolation, avoiding cubic multi-output incompatibilities. |
| AutoDLinear | `num_lr_decays` | — | moving_avg_window was already searched; other constructor options are shared training/data controls. |
| AutoNLinear | `num_lr_decays` | — | No additional architecture dimensions in the implemented linear head. |
| AutoTiDE | `num_lr_decays` | — | Encoder/decoder depths, widths, dropout and layernorm were already searched. Temporal covariate width is conditional on supplied covariates. |
| AutoDeepNPTS | `batch_norm`, `num_lr_decays` | — | input_size >= 2 prevents singleton softmax weights from making the network output independent of its learned weights. |
| AutoKAN | `n_hidden_layers`, `scale_noise`, `scale_base`, `scale_spline`, `grid_range`, `num_lr_decays` | — | enable_standalone_scale_spline is not forwarded to KANLinear; grid_eps only affects update_grid, which the normal forward path does not call. Keep both out of HPO. |
| AutoTFT | `dropout`, `attn_dropout`, `n_rnn_layers`, `rnn_type`, `grn_activation`, `num_lr_decays` | — | Attention head divisibility is preserved by the width/head candidates. Static-state and categorical-embedding schemas stay fixed. |
| AutoVanillaTransformer | `dropout`, `encoder_layers`, `decoder_layers`, `conv_hidden_size`, `activation`, `decoder_input_size_multiplier`, `num_lr_decays` | — | The minimum input length supports both decoder context ratios. Head counts divide all sampled hidden widths. |
| AutoInformer | `dropout`, `encoder_layers`, `decoder_layers`, `conv_hidden_size`, `activation`, `decoder_input_size_multiplier`, `factor`, `distil`, `num_lr_decays` | — | The minimum context supports distillation and both decoder ratios. factor is consumed by ProbAttention. |
| AutoAutoformer | `dropout`, `encoder_layers`, `decoder_layers`, `conv_hidden_size`, `activation`, `decoder_input_size_multiplier`, `factor`, `MovingAvg_window`, `num_lr_decays` | — | Minimum input length 8 supports the largest searched auto-correlation factor in the short decoder sequence. |
| AutoFEDformer | `dropout`, `encoder_layers`, `decoder_layers`, `conv_hidden_size`, `activation`, `decoder_input_size_multiplier`, `MovingAvg_window`, `modes`, `mode_select`, `num_lr_decays` | — | Only Fourier and eight heads are implemented; keep these fixed. Decoder Fourier dimensions now use label_len + h for all ratios, including odd input lengths. |
| AutoPatchTST | `encoder_layers`, `linear_hidden_size`, `dropout`, `head_dropout`, `attn_dropout`, `stride`, `activation`, `res_attention`, `batch_normalization`, `learn_pos_embed`, `num_lr_decays` | — | Minimum input length 24 covers the longest patch. fc_dropout is inactive in the default non-pretraining head; do not add it. RevIN affine/subtract-last remain conditional fixed controls. |
| AutoiTransformer | `e_layers`, `d_ff`, `dropout`, `use_norm`, `num_lr_decays`, `windows_batch_size` | batch_size → None | d_layers is only assigned and factor is ignored by FullAttention; keep them out of HPO. |
| AutoTimeXer | `e_layers`, `d_ff`, `dropout`, `use_norm`, `patch_len`, `num_lr_decays`, `windows_batch_size` | batch_size → None | factor is ignored by FullAttention. Input candidates are multiples of 16 and support every searched patch length. |
| AutoTimesNet | `encoder_layers`, `dropout`, `top_k`, `num_kernels`, `num_lr_decays` | — | Minimum input length 16 supplies enough FFT bins for top_k up to 5. |
| AutoStemGNN | `dropout_rate`, `leaky_rate`, `num_lr_decays`, `windows_batch_size` | batch_size → None; singleton n_stacks → 2 | n_stacks must be exactly 2 and is now the literal 2. None would violate the model contract. |
| AutoTSMixer | `revin`, `num_lr_decays`, `windows_batch_size` | batch_size → None | All series are required together; batch_size=None disables the redundant search and resolves to n_series. Search windows_batch_size instead. |
| AutoTSMixerx | `revin`, `num_lr_decays`, `windows_batch_size` | batch_size → None | Same all-series batching rule; exogenous schemas and categorical cardinalities remain fixed. |
| AutoMLPMultivariate | `num_lr_decays`, `windows_batch_size` | batch_size → None | Existing hidden_size/num_layers cover its architecture. Add time-window batching, keeping all series together. |
| AutoSOFTS | `e_layers`, `d_ff`, `dropout`, `use_norm`, `num_lr_decays`, `windows_batch_size` | batch_size → None | Remove duplicate None/identity scaler alternatives; both select identity scaling. |
| AutoSOFTSSharp | `e_layers`, `d_ff`, `dropout`, `use_norm`, `num_lr_decays`, `windows_batch_size` | batch_size → None | pe_keep_prob was already searched. Remove duplicate identity-scaler aliases. |
| AutoTimeMixer | `dropout`, `e_layers`, `moving_avg`, `channel_independence`, `down_sampling_window`, `down_sampling_method`, `use_norm`, `num_lr_decays`, `windows_batch_size` | batch_size → None | Keep moving_avg decomposition. top_k only affects dft_decomp; decoder_input_size_multiplier is validated but unused by forward. Do not search those controls. Input lengths are multiples of 16 for the largest two-level downsampling. |
| AutoRMoK | `dropout`, `revin_affine`, `num_lr_decays`, `windows_batch_size` | batch_size → None | Expert polynomial/wavelet choices were already searched; add implemented dropout and RevIN affine controls. |
| AutoXLinear | `temporal_ff`, `channel_ff`, `temporal_dropout`, `channel_dropout`, `embed_dropout`, `head_dropout`, `num_lr_decays`, `windows_batch_size` | batch_size → None | All four dropout modules and both feed-forward widths are used by the implementation. |

## External trainable review matrix

Every row adds the common `num_lr_decays` search. The existing model-specific schemas and pinned-source/checkpoint requirements are preserved. Configuration and conversion checks cover all 15 wrappers; optional backends are not replaced by generic forecasting models.

| Auto wrapper | Review |
|---|---|
| AutoCrossLinear | All exposed patch/width/mixing-initialization parameters are already searched; native port forward/backward tested. |
| AutoTimerXL | Patch, hidden width, heads, layers, feed-forward width, dropout and normalization are already searched; native port forward/backward tested. |
| AutoTinyTimeMixer | All exposed architecture parameters are searched; complete patch divisibility is retained. Actual TTM execution needs granite-tsfm. |
| AutoDAG | All exposed architecture and fusion parameters are searched. source_dir and future-covariate columns are fixed schema/source requirements. |
| AutoKITE | Architecture and frequency/fusion choices are searched; the required single covariate schema stays fixed. |
| AutoGLAFF | Architecture/fusion parameters are searched; its six calendar features and their order stay fixed. |
| AutoAPT | Architecture dimensions are searched; time_of_day_size is an input encoding/cardinality contract. |
| AutoVoT | Attention heads are divisors of the supplied embedding width; hidden/text schema dimensions stay fixed. |
| AutoGPT4MTS | Pretrained n_heads is ignored by the loaded GPT2Config. Set it to None and resolve the actual checkpoint head count. Keep from-scratch head/depth search. Checkpoint depth stays fixed pending explicit checkpoint metadata. |
| AutoUniTime | Patch, decoder depth and dropout are searched. Checkpoint truncation depth and max_tokens remain bounded deployment/context choices requiring checkpoint/tokenizer metadata. |
| AutoLangTime | Trainable fusion dimensions are searched. The language checkpoint and context-index mapping remain fixed. |
| AutoSpecTF | Trainable architecture/text fusion dimensions are searched; historical text coordinate schema stays fixed. |
| AutoTGForecaster | Fusion/head dimensions are searched. Paired news/description embedding widths are derived from the covariate schema. |
| AutoSeesawNet | All exposed architecture/window parameters are already searched; reviewed source path stays fixed. |
| AutoDualformer | All exposed architecture/patch/attention parameters are already searched; reviewed source path stays fixed. |

## AutoHINT

AutoHINT requires `cls_model`, `S` and a caller-supplied `config` including `reconciliation`. Its underlying model and hierarchy determine meaningful search dimensions. The existing user space is preserved and a regression test checks reconciliation choices and the absence of automatic additions.

## Validation and limitations

Local CPU execution used Python 3.13.5, PyTorch 2.10.0+cpu, Ray 2.58.0 and Optuna 4.8.0.

- Targeted Auto suite: **348 passed, 4 skipped**. The skipped tests require xLSTM/mLSTM kernels (three cases) and Transformers for the checkpoint-config boundary test (one case).
- Existing BaseAuto Optuna/configuration regressions: **11 passed, 8 deselected**. Live Ray-cluster tests were excluded from this completed run; one earlier broader selection entered Ray and timed out in the container.
- Additional native sweep: **1,088 passed, 0 failed** = 34 executable native models × horizons `[1, 2, 7, 12]` × eight seeds. Each case instantiated sampled architecture choices, computed training loss, backpropagated finite gradients and evaluated validation loss. `max_steps=2` and `windows_batch_size=2` bound smoke-test cost; this is not full HPO or a convergence benchmark.
- A targeted before/after check against the unchanged baseline reproduced **13 regression failures** (context no-ops, integer bounds, Fourier decoder lengths, and None handling). Those tests pass with the fixes.
- Native CrossLinear and TimerXL ports also execute forward/backward checks. The remaining external-source/checkpoint backends and xLSTM are reviewed at source/configuration level here. Full external training, GPU runs, real pretrained-weight inference and 40-fold TSCV have not been executed.

Reproduce the primary checks from a configured development environment:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -o addopts='' -q \
  tests/test_auto_search_spaces.py tests/test_auto_external.py \
  tests/test_auto_external_registry.py

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -o addopts='' -q \
  tests/test_common/test_base_auto.py \
  -k 'optuna or validation_default or instantiation or missing_required'
```

The PR workflow runs these CPU checks. It installs Transformers so that the local checkpoint-config test can run with a real GPT2Config and a mocked external model constructor. That boundary test does not load pretrained model weights. The entire repository test suite and its global coverage threshold were not run; `-o addopts=''` deliberately scopes the checks above.
