from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List

import pandas as pd


@dataclass
class FactorRow:
    date: datetime
    mkt_rf: float
    smb: float
    hml: float
    rmw: float
    cma: float
    mom: float
    rf: float
    source: str


class FactorsLoader:
    def __init__(self, warehouse_root: str) -> None:
        self.warehouse_root = Path(warehouse_root)

    def load_factors(
        self, start: datetime, end: datetime, family: str = "fama_french_5", source: str = "ff_library"
    ) -> Path:
        factors_dir = self.warehouse_root / "factors" / family
        factors_dir.mkdir(parents=True, exist_ok=True)

        rows: List[FactorRow] = []
        for offset in range((end.date() - start.date()).days + 1):
            date = start + timedelta(days=offset)
            rows.append(
                FactorRow(
                    date=date,
                    mkt_rf=0.01,
                    smb=0.002,
                    hml=-0.001,
                    rmw=0.0005,
                    cma=-0.0002,
                    mom=0.003,
                    rf=0.0001,
                    source=source,
                )
            )

        df = pd.DataFrame([r.__dict__ for r in rows])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")

        output_paths = []
        for date, group in df.groupby(df["date"].dt.date):
            date_dir = factors_dir / f"dt={date:%Y-%m-%d}"
            date_dir.mkdir(parents=True, exist_ok=True)
            file_path = date_dir / f"part-{uuid.uuid4().hex}.parquet"
            group.to_parquet(file_path, index=False)
            output_paths.append(file_path)

        manifest = factors_dir / "manifest.json"
        manifest_record = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "family": family,
            "source": source,
            "files": [str(p) for p in output_paths],
            "written_at": datetime.utcnow().isoformat(),
        }
        manifest.write_text(json.dumps(manifest_record, indent=2))
        return factors_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load factor datasets (e.g., Fama-French)")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    parser.add_argument("--family", default="fama_french_5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start) if args.start else datetime.utcnow() - timedelta(days=30)
    end = datetime.fromisoformat(args.end) if args.end else datetime.utcnow()

    loader = FactorsLoader(args.warehouse)
    loader.load_factors(start=start, end=end, family=args.family)


if __name__ == "__main__":
    main()
