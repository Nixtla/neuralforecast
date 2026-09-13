"""Export the gasoline target as an immutable experiment CSV snapshot."""

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import psycopg
from psycopg import sql

TARGET = "Oil_EIA_NY_Harbor_Conventional_Gasoline_Spot_Price_Daily_USD_Per_Gallon"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/gasoline.csv")
    parser.add_argument(
        "--database", default="dbname=commodity host=/var/run/postgresql"
    )
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(f"Refusing to replace snapshot: {path}")
    with psycopg.connect(
        args.database, options="-c default_transaction_read_only=on"
    ) as conn:
        conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ")
        names = conn.execute(
            "SELECT column_name FROM collector.columns WHERE sheet_id=484490350 AND header=%s AND active",
            (TARGET,),
        ).fetchall()
        if len(names) != 1:
            raise ValueError("Target metadata must resolve to exactly one column")
        rows = conn.execute(
            sql.SQL(
                "SELECT week_end,{} FROM public.gasoline_ml WHERE week_end >= %s ORDER BY week_end"
            ).format(sql.Identifier(names[0][0])),
            ("2013-08-11",),
        ).fetchall()
    if not rows or any(v is None for _, v in rows):
        raise ValueError("Snapshot is empty or contains missing target values")
    if any((b[0] - a[0]).days != 7 for a, b in zip(rows, rows[1:])):
        raise ValueError("Dates must be a consecutive weekly series")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ds", TARGET])
        writer.writerows(rows)
    manifest = {
        "source": "commodity.public.gasoline_ml",
        "target": TARGET,
        "rows": len(rows),
        "start": str(rows[0][0]),
        "end": str(rows[-1][0]),
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
