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

Every executable unit is submitted to Ray as a one-GPU job. Phase 1 starts the current rung for every model at once. A model advances as soon as all three fold jobs for its current rung finish, while Ray continues running ready work from other models. This keeps two-GPU hosts busy without coupling one model's rung barrier to every other model.

## Phase 2

The ten best Phase 1 candidates advance. Trainable candidates reuse the selected hyperparameters and train a fresh model from seed 42 for 1000 optimizer steps on every fold. Inference candidates reuse their selected inference configuration. HPO is not repeated.

The final integrated leaderboard is ordered by:

1. pooled RMSE across all fold/horizon predictions;
2. standard deviation of fold RMSE;
3. worst fold RMSE.

## Data

The runner accepts a CSV containing a date column and a target column. It selects the target only and resamples it to weekly Sunday-ending means. Missing values inside a training fold are interpolated using that fold's training rows. Validation targets are filled separately for scoring and never enter training.

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
