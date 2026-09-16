"""Create a shared W&B saved workspace for completed Phase 2 experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from backfill_phase2_leaderboards import collect_snapshot
from neuralforecast.benchmark_tracking import run_id


def collect_completed(results):
    """Combine validated final results, preserving each experiment's own ranks."""
    frames, sources, definitions = [], [], {}
    for path in sorted(Path(results).glob("*/run_config.json")):
        config = json.loads(path.read_text())
        if config.get("status") != "completed" or not config.get("wandb"):
            continue
        config, snapshot = collect_snapshot(path.parent)
        if snapshot is None or snapshot[3] != "completed":
            continue
        board, metrics, total, _ = snapshot
        metadata = json.loads((path.parent / "wandb_run.json").read_text())
        for key in ("entity", "project", "group"):
            if metadata[key] != config["wandb"][key]:
                raise ValueError(f"{path.parent.name}: inconsistent W&B {key}")
        board = board.copy()
        board.insert(0, "experiment", path.parent.name)
        board["evaluation_folds"] = config["phase2_folds"]
        board["source_run"] = metadata["url"]
        frames.append(board)
        definitions[path.parent.name] = metrics
        sources.append(
            {
                "experiment": path.parent.name,
                "project": metadata["project"],
                "group": metadata["group"],
                "selected_models": total,
                "fingerprint": config["fingerprint"],
                "source_run": metadata["url"],
            }
        )
    if not frames:
        raise ValueError("No completed tracked experiments found")
    combined = pd.concat(frames, ignore_index=True).sort_values(["experiment", "rank"])
    leading = [
        "experiment",
        "rank",
        "candidate",
        "rmse",
        "mae",
        "mape_pct",
        "mse",
        "da_pct",
        "rmse_vs_naive",
        "beats_naive",
        "evaluation_folds",
    ]
    combined = combined[leading + [col for col in combined if col not in leading]]
    winners = (
        combined.loc[combined.protocol.ne("naive")]
        .groupby("experiment", sort=False)
        .head(1)
        .sort_values("rmse_vs_naive")
        .reset_index(drop=True)
    )
    return combined.reset_index(drop=True), winners, sources, definitions


def publish_dashboard(results, entity, project):
    """Publish one snapshot run and a saved view without editing other views."""
    import wandb
    import wandb_workspaces.reports.v2 as wr
    import wandb_workspaces.workspaces as ws

    combined, winners, sources, definitions = collect_completed(results)
    output = Path(results) / f"{project}-dashboard"
    output.mkdir(exist_ok=True)
    state_path = output / "dashboard.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    identity = run_id("completed-phase2-dashboard", "summary", "benchmark", 0, -1)
    title = "Completed Phase 2 Dashboard"
    records = json.loads(combined.to_json(orient="records", double_precision=15))
    digest = hashlib.sha256(
        json.dumps([records, sources], sort_keys=True).encode()
    ).hexdigest()
    with wandb.init(
        entity=entity,
        project=project,
        id=identity,
        name=title,
        job_type="completed-phase2-dashboard",
        group="completed-phase2",
        resume="allow",
        dir=str(output),
        save_code=False,
        config={"sources": sources, "evaluation_split": "validation"},
        allow_val_change=True,
    ) as run:
        if run.summary.get("dashboard/digest") != digest:

            def table(frame):
                rows = json.loads(frame.to_json(orient="records", double_precision=15))
                return wandb.Table(
                    columns=list(frame.columns),
                    data=[[row[col] for col in frame] for row in rows],
                    log_mode="MUTABLE",
                )

            tables = {
                "completed/leaderboard": table(combined),
                "completed/best_models": table(winners),
            }
            for experiment, frame in combined.groupby("experiment", sort=True):
                tables[f"experiments/{experiment}"] = table(frame)
            run.log(tables)
            run.summary.update(
                {
                    "dashboard/digest": digest,
                    "dashboard/experiment_count": len(sources),
                    "dashboard/model_count": int(combined.protocol.ne("naive").sum()),
                    "dashboard/baseline_count": int(
                        combined.protocol.eq("naive").sum()
                    ),
                    "dashboard/metric_definitions": definitions,
                    "status": "completed",
                }
            )
        run_url = run.url

    def bar(table_name, label, value, chart_title, layout):
        panel = wr.CustomChart.from_table(
            table_name,
            chart_fields={"label": label, "value": value},
            chart_strings={"title": chart_title},
        )
        panel.chart_name = "wandb/bar/v0"
        panel.layout = layout
        return panel

    sections = [
        ws.Section(
            name="Overview",
            is_open=True,
            panels=[
                wr.MarkdownPanel(
                    markdown=(
                        f"## Completed Phase 2 experiments\n"
                        f"**{len(sources)} experiments · "
                        f"{int(combined.protocol.ne('naive').sum())} model results · "
                        f"{len(sources)} Naive baselines**\n\n"
                        "Ranks and raw errors are compared within each experiment. "
                        "RMSE / Naive below 1 means the model beats its own baseline. "
                        "Scores use original-price validation windows; MAPE and "
                        "directional accuracy are percentages."
                    ),
                    layout=wr.Layout(x=0, y=0, w=24, h=4),
                ),
                wr.WeavePanelSummaryTable(
                    table_name="completed/best_models",
                    layout=wr.Layout(x=0, y=4, w=16, h=10),
                ),
                bar(
                    "completed/best_models",
                    "experiment",
                    "rmse_vs_naive",
                    "Best model RMSE / Naive (lower is better)",
                    wr.Layout(x=16, y=4, w=8, h=10),
                ),
            ],
        ),
        ws.Section(
            name="All completed results",
            is_open=True,
            panels=[
                wr.WeavePanelSummaryTable(
                    table_name="completed/leaderboard",
                    layout=wr.Layout(w=24, h=18),
                ),
            ],
        ),
        ws.Section(
            name="RMSE by experiment",
            is_open=True,
            panels=[
                bar(
                    f"experiments/{source['experiment']}",
                    "candidate",
                    "rmse",
                    f"{source['experiment']} — Phase 2 RMSE",
                    wr.Layout(x=(i % 2) * 12, y=(i // 2) * 9, w=12, h=9),
                )
                for i, source in enumerate(sources)
            ],
        ),
    ]
    if state.get("workspace_url") and state.get("entity") == entity:
        workspace = ws.Workspace.from_url(state["workspace_url"])
        if workspace.project != project:
            raise ValueError("Saved workspace belongs to a different project")
        workspace.sections = sections
    else:
        workspace = ws.Workspace(
            entity=entity,
            project=project,
            name=title,
            sections=sections,
            runset_settings=ws.RunsetSettings(query=title),
        )
    workspace.save()
    state = dict(
        entity=entity,
        project=project,
        run_url=run_url,
        workspace_url=workspace.url,
        digest=digest,
        sources=sources,
    )
    state_path.write_text(json.dumps(state, indent=2))
    combined.to_csv(output / "leaderboard.csv", index=False)
    winners.to_csv(output / "best_models.csv", index=False)
    print(
        json.dumps(
            {
                "workspace_url": workspace.url,
                "run_url": run_url,
                "experiments": len(sources),
                "rows": len(combined),
            },
            indent=2,
        )
    )
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--entity", default="Beat-Sun")
    parser.add_argument("--project", default="Riotinto")
    args = parser.parse_args()
    publish_dashboard(args.results, args.entity, args.project)


if __name__ == "__main__":
    main()
