"""Export immutable target-only or wide commodity experiment snapshots."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql

DATASETS = {
    "gasoline": {
        "table": "gasoline_ml",
        "sheet_id": 484490350,
        "target": (
            "Oil_EIA_NY_Harbor_Conventional_Gasoline_"
            "Spot_Price_Daily_USD_Per_Gallon_lag_0"
        ),
        "start": "2013-08-11",
        "label": "Gasoline",
    },
    "wti": {
        "table": "wti_crude_oil_ml",
        "sheet_id": 2026091101,
        "target": "Oil_EIA_Cushing_WTI_Spot_Price_Daily_USD_Per_Barrel_lag_0",
        "start": "2015-03-15",
        "label": "WTI Crude Oil",
    },
}


def _columns(conn, dataset, include_exogenous):
    metadata = conn.execute(
        "SELECT header, column_name FROM collector.columns "
        "WHERE sheet_id=%s AND active ORDER BY ordinal",
        (dataset["sheet_id"],),
    ).fetchall()
    metadata = [(header, column) for header, column in metadata if column != "week_end"]
    target = [item for item in metadata if item[0] == dataset["target"]]
    if len(target) != 1:
        raise ValueError("Dataset metadata must resolve the target exactly once")
    if not include_exogenous:
        return target
    table_columns = {
        row[0]
        for row in conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name=%s",
            (dataset["table"],),
        )
    }
    columns = [item for item in metadata if item[1] in table_columns]
    if len(columns) < 2:
        raise ValueError("Wide export requires at least one exogenous column")
    return columns


def _snapshot(database, dataset, start_date, end_date, include_exogenous):
    with psycopg.connect(
        database, options="-c default_transaction_read_only=on"
    ) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        columns = _columns(conn, dataset, include_exogenous)
        predicates = [sql.SQL("week_end >= %s")]
        parameters = [start_date]
        if end_date:
            predicates.append(sql.SQL("week_end <= %s"))
            parameters.append(end_date)
        if not include_exogenous:
            predicates.append(sql.SQL("header = %s"))
            parameters.append(dataset["target"])
        query = sql.SQL(
            "SELECT week_end, header, value FROM public.{} WHERE {} ORDER BY week_end, header"
        ).format(
            sql.Identifier(dataset["table"]),
            sql.SQL(" AND ").join(predicates),
        )
        triples = conn.execute(query, parameters).fetchall()
    by_date = {}
    for week_end, header, value in triples:
        by_date.setdefault(week_end, {})[header] = value
    rows = [
        (week_end, *(by_date[week_end].get(header) for header, _ in columns))
        for week_end in sorted(by_date)
    ]
    return columns, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=sorted(DATASETS), default="gasoline")
    parser.add_argument("--output")
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--include-exogenous", action="store_true")
    parser.add_argument(
        "--database", default="dbname=commodity host=/var/run/postgresql"
    )
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    start_date = args.start_date or dataset["start"]
    path = Path(args.output or f"data/{args.dataset}.csv")
    if path.exists():
        raise FileExistsError(f"Refusing to replace snapshot: {path}")
    columns, rows = _snapshot(
        args.database,
        dataset,
        start_date,
        args.end_date,
        args.include_exogenous,
    )
    if not rows:
        raise ValueError("Snapshot is empty")
    target_index = 1 + next(
        index for index, (header, _) in enumerate(columns) if header == dataset["target"]
    )
    if any(row[target_index] is None or not row[target_index].is_finite() for row in rows):
        raise ValueError("Target contains missing or non-finite values")
    if str(rows[0][0]) != start_date:
        raise ValueError("Snapshot does not begin on the requested start date")
    if args.end_date and str(rows[-1][0]) != args.end_date:
        raise ValueError("Snapshot does not end on the requested end date")
    if any((right[0] - left[0]).days != 7 for left, right in zip(rows, rows[1:])):
        raise ValueError("Dates must be a consecutive weekly series")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ds", *(header for header, _ in columns)])
        writer.writerows(rows)
    missing = {
        header: sum(row[index] is None for row in rows) / len(rows)
        for index, (header, _) in enumerate(columns, start=1)
    }
    manifest = {
        "source": f"commodity.public.{dataset['table']}",
        "dataset": args.dataset,
        "label": dataset["label"],
        "target": dataset["target"],
        "include_exogenous": args.include_exogenous,
        "columns": len(columns),
        "rows": len(rows),
        "start": str(rows[0][0]),
        "end": str(rows[-1][0]),
        "missing_ratio": missing,
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
