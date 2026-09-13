# Commodity SOTA benchmark

`experiments/commodity_sota/run.py` runs a two-phase weekly benchmark over the model adapters available in this fork. The benchmark keeps model-specific search spaces from `neuralforecast.auto` and `neuralforecast.inference_tuning`, then applies one chronological evaluation contract.

## Evaluation contract

- Weekly frequency: Sunday-ending mean (`W-SUN`).
- Forecast horizon: 16 weeks by default.
- Cross-validation: expanding window, one-week step.
- Validation: the next 16 weeks after each training cutoff.
- Validation rows are evaluation-only. Early stopping is disabled for trainable candidates.
- Random seed: 42.
- Metric: pooled RMSE over every forecast point.
- No separate untouched holdout is created. Phase 2 RMSE is a model-selection CV score.

The first CV cutoff is the earliest point where every discoverable candidate can retain at least one feasible input size. Phase 1 uses the first, middle and last folds from that full fold set.

## Phase 1

### Trainable Auto models

The runner instantiates the existing `Auto*` wrapper, reuses its search space, removes `max_steps` from HPO, fixes `random_seed=42`, disables early stopping and filters input-size choices against the first representative fold.

Ten sampled configurations enter synchronous successive halving:

| Rung | Cumulative optimizer steps | Survivors |
| ---: | ---: | ---: |
| 1 | 125 | 10 |
| 2 | 250 | 5 |
| 3 | 500 | 2 |
| 4 | 1000 | 1 |

A configuration is one compound trial across the three representative folds. Its score is pooled RMSE over `3 × 16 = 48` forecast points. Surviving fold jobs resume from Lightning checkpoints, preserving model, optimizer, scheduler and global-step state.

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

The ten best Phase 1 candidates advance. Trainable candidates reuse the selected hyperparameters and train a fresh model from seed 42 on every fold. Phase 2 uses up to 500 optimizer steps, validation every 10 steps, and patience of five consecutive checks without a strictly lower validation MSE. The best weights are restored before Test prediction. Inference candidates reuse their selected inference configuration. HPO is not repeated.

The final integrated leaderboard is ordered by:

1. pooled RMSE across all fold/horizon predictions;
2. standard deviation of fold RMSE;
3. worst fold RMSE.

Phase 2 splits each fold chronologically into Train, 16 Validation weeks, and
16 Test weeks. The earliest Train contains 80 weeks for the gasoline configuration;
Test starts after 96 weeks of history. This produces 572 common Test folds for the
683-row snapshot. Phase 1 uses the same Test cutoffs with its original successive
halving budgets and input-size search bounds. Validation is excluded from gradient
updates and Test is excluded from fitting and checkpoint selection. Known
Validation observations may be used as context for the subsequent Test forecast;
there is no refit on Train+Validation. Both LoRA backends follow the same stopping
policy; zero-shot candidates only predict the same Test windows.

Policy options are `--phase2-max-steps 500`, `--phase2-val-check-steps 10`,
`--phase2-patience 5`, and `--phase2-val-size 16`. Validation size must equal the
forecast horizon. The policy is included in the smoke-report fingerprint. Changing
it requires renewed validation. W&B records validation loss, actual steps, best
step, stopping reason, and Test RMSE separately. Test RMSE remains a CV
model-selection score, not an untouched final holdout.

## Data

The runner accepts a CSV containing a date column and a target column. It selects the target only and resamples it to weekly Sunday-ending means. Missing values are forward-filled only within their own Train, Validation, or Test slice, so replacement never uses a later observation. A slice that begins with a missing target is rejected because no past observation is available to fill it. Validation targets are filled separately for scoring and never enter training.

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
- `phase2_predictions.csv`: fold/horizon predictions for the top ten;
- `leaderboard.csv`: final integrated ranking;
- `failures.csv`: OOM and runtime failures;
- `run_config.json`: fold and SH protocol metadata;
- checkpoint directories used by Phase 1 and Phase 2.

Foundation LoRA support is documented separately in `docs/foundation_lora.md` when that optional protocol is installed.

## PostgreSQL gasoline snapshot and tracked runs

Export the fixed target (starting 2013-08-11) without changing the source database:

```bash
.venv/bin/python experiments/commodity_sota/export_postgres.py
```

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
