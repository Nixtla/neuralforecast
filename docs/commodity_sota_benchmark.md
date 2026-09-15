# Commodity SOTA benchmark

`experiments/commodity_sota/run.py` runs a two-phase weekly benchmark over the model adapters available in this fork. The benchmark keeps model-specific search spaces from `neuralforecast.auto` and `neuralforecast.inference_tuning`, then applies one chronological evaluation contract.

## Evaluation contract

- Weekly frequency: Sunday-ending mean (`W-SUN`).
- Forecast horizon: 16 weeks by default.
- Cross-validation: expanding window, one-week step.
- Validation: the next 16 weeks after each training cutoff.
- Validation rows are excluded from gradient updates and used for early stopping,
  best-weight selection, and scoring in both phases. There is no separate Test block.
- Random seed: 42.
- Metric: pooled RMSE over every forecast point.
- No separate untouched holdout is created. Phase 2 RMSE is a model-selection CV score.

The first CV cutoff is the earliest point where every discoverable candidate can retain at least one feasible input size. Phase 1 uses the first, middle and last folds from that full fold set.

## Phase 1

New benchmark configurations apply numerical policy version 1. Sampled learning
rates for NBEATS, NBEATSx, Autoformer and xLSTM are limited to at most 0.001;
explicit fixed learning rates remain user overrides. These four models and
FEDformer use gradient-norm clipping at 1.0 and reject non-finite gradients
before optimizer updates. FEDformer additionally identifies the first module
returning a non-finite output. This is diagnostic protection, not a claim that
its intermittent NaN failures have been resolved.

New xLSTM configurations use a bounded-exponential parallel mLSTM backend.
Rescaling numerator, denominator and epsilon together preserves the reference
formula while avoiding overflow in `exp(-max_log_D)`. The external `xlstm`
package is not patched globally. Existing queued benchmark configurations keep
the legacy backend and do not acquire new clipping or diagnostics mid-experiment.
The numerical-policy version is included in the preparation fingerprint, so a
new experiment requires a fresh smoke report.

### Trainable Auto models

The runner instantiates the existing `Auto*` wrapper, reuses its search space, removes `max_steps` from HPO, fixes `random_seed=42`, and filters input-size choices against the first representative fold. A shared validation stopper controls training in both phases.

Ten sampled configurations enter synchronous successive halving:

| Rung | Cumulative optimizer steps | Survivors |
| ---: | ---: | ---: |
| 1 | 100 | 10 |
| 2 | 250 | 5 |
| 3 | 500 | 1 |

A configuration is one compound trial across the three representative folds. Its score is pooled RMSE over `3 × 16 = 48` validation points. Surviving fold jobs resume full training state, including optimizer, scheduler, RNGs, and validation patience. The learning-rate schedule uses a fixed 500-step horizon. Best validation weights are stored separately from the last training state. An early-stopped fold reuses its best prediction without further training; it is not automatically disqualified. LoRA uses the same three rungs.

### Inference/foundation models

Inference-only adapters use `get_inference_tuning_config`. Ten inference configurations are evaluated on the same three representative folds and the best pooled-RMSE configuration is retained. These jobs use `max_steps=0`.

Models requiring unavailable source paths, checkpoints, schemas or optional dependencies are recorded in `eligibility.csv` or `failures.csv`. LLM-oriented adapters are excluded by the experiment runner. Joint/multivariate models are excluded when the experiment has no exogenous inputs.

`SearchCast` is excluded from this runner because it owns an internal optimization loop and currently has no fixed-configuration Phase 2 path.

## Global GPU queue

The default `--scheduler dynamic` uses independent subprocesses with a selected
`CUDA_VISIBLE_DEVICES`. Ray remains available for search-space sampling and the
optional `--scheduler ray` legacy executor. No pre-measured jobs-per-GPU profile
is required. Multiple models may share a GPU when reservations fit.

The queue checks GPU memory and host RAM every second, reserves two CPU threads
per job, and leaves 15% headroom. Unknown configurations are executed as real
experiment work in isolation before their resource estimates are reused. Larger
history and step budgets must have been observed before reusing an estimate.
Observed peaks receive a 25% margin; unavailable process-level GPU accounting
uses conservative whole-device observations. CPU and memory reservations are
admission accounting, not OS-enforced resource partitions.

Old waiting jobs drain resources to avoid starvation. CUDA OOM jobs retry at most
twice in isolation; other failures are recorded without resource retries. Every
attempt has separate files and a stable logical W&B run ID. Only final attempts
enter evaluation tables. `scheduler/state.json` records queued/running jobs,
reservations, observed peaks, and retries. Shutdown terminates process groups,
including external backend children. Completed result files survive shutdown;
automatic recovery of an entire interrupted experiment is not implemented.

## Phase 2

Only candidates whose selected configuration has a strictly lower pooled validation RMSE than naive across the three representative folds advance, up to ten in score order. Naive repeats each fold's last training observation for all 16 validation weeks. Ties fail the gate. If none qualify, Phase 2 is skipped with status `no_models_above_naive`.

Trainable candidates reuse the selected hyperparameters and train a fresh model from seed 42 on every fold. Both phases validate every 10 steps and stop after five consecutive checks without a strictly lower validation MSE. Phase 2 uses up to 500 optimizer steps. Best weights are restored before validation prediction. Inference candidates reuse their selected inference configuration. HPO is not repeated, and train loss has no additional reduction threshold.

The final integrated leaderboard is ordered by:

1. pooled RMSE across all fold/horizon predictions;
2. standard deviation of fold RMSE;
3. worst fold RMSE.

Both phases split each fold into Train and the following 16 Validation weeks.
Existing evaluation dates and conservative input-size bounds are retained: the
683-row gasoline snapshot has 572 folds, with 96 training weeks in the first fold.
There is no additional block taken out of Train. Validation targets influence
checkpoint selection but not gradient updates or the forecast context. Zero-shot
candidates predict the same validation windows.

Policy options are `--phase2-max-steps 500`, `--phase2-val-check-steps 10`,
`--phase2-patience 5`, and `--phase2-val-size 16`. Validation size must equal the
forecast horizon. The interval and patience flags apply to both phases; the maximum
steps flag controls Phase 2, while Phase 1 always uses 100/250/500. The protocol
version and budgets are included in the smoke-report fingerprint. A new experiment
requires a fresh output directory and smoke report. W&B records both loss curves,
actual steps, best step, stopping reason, and naive selection results. Scores are
validation/model-selection scores, not independent test scores.

## Data

The runner accepts a CSV containing a date column and a target column. By default it
selects the target only and resamples it to weekly Sunday-ending means. Missing target
values are forward-filled within their own Train or Validation slice, so replacement
never uses a later observation. A slice that begins with a missing target is rejected.
Validation targets are filled separately and never enter gradient updates.

`--auto-hist-exog` enables automatic historical-exogenous selection from every other
CSV column. Missingness is measured over the complete bounded experiment period;
columns at or above `--exog-max-missing-ratio` (default `0.05`) are removed. Pearson
correlation is then measured only over the first fold's training rows, and columns
whose absolute correlation is below `--exog-max-abs-corr` (default `0.8`) are used as
one fixed `hist_exog_list` for every fold. The upper bound is exclusive: columns at
exactly `0.8` are removed. Remaining gaps are forward-filled from past observations.
A column missing at the period start is removed. Models without
historical-exogenous support are listed as skipped rather than mixed into the same
leaderboard. `exogenous_selection.csv` records every selection decision.

Use `--end-date` together with `--start-date` to reproduce an earlier experiment's
exact date range. The range, thresholds and selected feature names are part of the
experiment fingerprint.

Example:

```bash
python experiments/commodity_sota/run.py \
  --data data/gasoline.csv \
  --date-col ds \
  --target Oil_EIA_NY_Harbor_Conventional_Gasoline_Spot_Price_Daily_USD_Per_Gallon \
  --start-date 2013-08-11 \
  --model-config model_config.json \
  --output results/gasoline
```

`model_config.json` supplies fixed arguments that are outside HPO, such as trusted source directories, checkpoint IDs or backend interpreters:

```json
{
  "Chronos2": {
    "model_id": "amazon/chronos-2",
    "backend_device": "cuda"
  },
  "Dualformer": {
    "source_dir": "/opt/sources/Dualformer"
  }
}
```

Candidates whose required fixed inputs are absent are skipped with the reason recorded.

## Outputs

The output directory contains:

- `eligibility.csv`: candidate discovery and skip reasons;
- `phase1_trials.csv`: rung-level compound-trial scores;
- `phase1_ranking.csv`: one selected configuration per candidate;
- `phase1_leaderboard.csv`: selected configurations plus Naive and five metrics;
- `phase2_predictions.csv`: fold/horizon predictions for the top ten;
- `leaderboard.csv`: final integrated ranking including Naive, MAE, MAPE (%), MSE,
  RMSE, DA (%), and RMSE relative to Naive;
- `metric_definitions.json`: metric conventions and direction reference baselines;
- `failures.csv`: OOM and runtime failures;
- `run_config.json`: fold and SH protocol metadata;
- checkpoint directories used by Phase 1 and Phase 2.

Foundation LoRA support is documented separately in `docs/foundation_lora.md` when that optional protocol is installed.

All metrics pool individual fold/horizon points. DA compares rise/fall/flat signs
relative to the last training observation, held fixed across the forecast horizon.
The definitions include always-up, always-down and always-flat DA for comparison.
MAPE omits zero actuals and reports its denominator in `mape_n`; it is missing if
all actuals are zero. A final model row requires every expected fold/horizon point;
partial models are excluded and duplicate points or inconsistent actuals fail.
Naive does not count as a completed model or consume a Phase 2 place. If Phase 2 is
skipped, its leaderboard still records the inexpensive Naive reference.

With `--wandb`, both phases publish `phase1/leaderboard_with_naive` and
`phase2/leaderboard_with_naive`, metric definitions, and per-model summary metrics.
The CSVs and definitions are included in the result artifact automatically.

### Offline min_delta review

`experiments/commodity_sota/analyze_min_delta.py` reads full W&B validation histories
for a time-stratified sample and replays relative improvement thresholds. It does
not modify the training policy, train models, or write to W&B. It retains the true
best loss while resetting patience only when cumulative progress exceeds the
threshold relative to the last significant improvement. It verifies that replay
with zero threshold reproduces the recorded stopping step and best loss.

```bash
.venv/bin/python experiments/commodity_sota/analyze_min_delta.py \
  --experiment results/gasoline --output results/gasoline-min-delta-review
```

The output contains source curves, per-run comparisons, summaries, and methodology.
Step savings are not wall-time savings. These are comparisons of recorded validation
curves, not accuracy measurements of forecasts from alternate checkpoints.

## PostgreSQL gasoline snapshot and tracked runs

Export the fixed target (starting 2013-08-11) without changing the source database:

```bash
.venv/bin/python experiments/commodity_sota/export_postgres.py
```

Create immutable wide snapshots for the historical-exogenous follow-up experiments:

```bash
.venv/bin/python experiments/commodity_sota/export_postgres.py \
  --dataset gasoline --include-exogenous \
  --start-date 2013-08-11 --end-date 2026-09-06 \
  --output data/gasoline-exog.csv
.venv/bin/python experiments/commodity_sota/export_postgres.py \
  --dataset wti --include-exogenous \
  --start-date 2015-03-15 --end-date 2026-09-06 \
  --output data/wti-exog.csv
```

The manifest records source metadata, exact bounds, headers, per-column missing
ratios and the snapshot SHA-256. Exogenous nulls are permitted because the runner
applies the documented selection and causal filling policy; the target must remain
complete and finite.

The exporter resolves the SQL column through `collector.columns` and restores its
full header in CSV. It refuses to overwrite an existing snapshot. The adjacent
`gasoline.manifest.json` records the source, extraction time, dates, row count and
SHA-256. Training uses this snapshot while the hourly collector continues updating
PostgreSQL independently.

Use the same data/target/model-config arguments for `--preflight` and
`--smoke-test`. Preflight lists the full candidate/fold inventory. Smoke runs one
configuration on the last fold, using two optimizer steps for scratch/LoRA models.
These modes never initialize W&B and reject `--wandb`. A `smoke_results.json`
report can be supplied to the main run via `--validated-candidates`; its data,
start-date, horizon and model-config fingerprint must match. Failed candidates
remain visible in eligibility and preparation reports.

Enable tracking only on this experiment runner:

```bash
# WANDB_API_KEY is provided to this process by the launcher, never a CLI argument.
.venv/bin/python experiments/commodity_sota/run.py \
  --data data/gasoline.csv --date-col ds \
  --target Oil_EIA_NY_Harbor_Conventional_Gasoline_Spot_Price_Daily_USD_Per_Gallon \
  --start-date 2013-08-11 --model-config model_config.json \
  --output results/gasoline \
  --validated-candidates results/gasoline-dynamic-smoke/smoke_results.json \
  --wandb --wandb-entity Beat-Sun --wandb-project uni-gasoline
```

Each experiment has one W&B group, one coordinator run and one run per
phase/model/config/fold. SH rungs reuse the same ID; Phase 2 has separate IDs.
Tracking records configurations, available learning curves, RMSE, durations and
failures. Full result CSVs are uploaded as a result artifact; source CSV and model
weights are not uploaded. Backend auto-reporting is disabled so LoRA does not
create unrelated HuggingFace runs. Importing the tracking module has no side effects.
Local per-job result JSON and phase CSVs are saved during execution. Run URLs are
recorded in `wandb_run.json`. Data paths and checkpoint paths are absolute for Ray.

The common training cutoff reserves both input context and the forecast horizon
for scratch training. This prevents admitting a first fold with no trainable window.
`TimesFM-LoRA` in model_config may override the shared `TimesFM` fixed values when
an explicit LoRA checkpoint differs from the inference checkpoint.

### First-difference experiments

A project name ending in `-diff` enables first differences after weekly aggregation.
`uni-gasoline-diff` uses the existing target-only CSV and unchanged fold dates.
Each fold is filled using the existing slice-local policy, then training loses its
first row and validation is differenced from the last training observation.
The common differencing helper also transforms explicitly selected time-varying
exogenous columns; dates and identifiers are never differenced. This does not add
an exogenous input pipeline to the current univariate runner.

Training and early stopping use differences. Forecasts are restored using the last
training price plus cumulative predicted differences, without future actuals.
HPO ranking, Naive admission, and all five leaderboard metrics use restored prices.
Naive remains a constant last-training-price forecast (zero predicted differences).
Transform metadata is included in fingerprints and W&B; level experiment smoke
reports and checkpoints cannot be reused. `weekly.pkl` retains original levels,
with transform metadata in its DataFrame attributes; workers transform per fold.

Use `--wandb-project uni-gasoline-diff` for both preflight and smoke invocations,
with fresh `results/gasoline-diff-preflight` and
`results/gasoline-diff-dynamic-smoke` outputs. After smoke validation, launch with:

```bash
.venv/bin/python experiments/commodity_sota/launch_tracked.py --project uni-gasoline-diff
```

The launcher reuses process-environment or local netrc W&B credentials when
available, otherwise prompts privately. Results and the systemd service are
separate from the original experiment.
