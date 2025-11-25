from __future__ import annotations

import argparse
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class PriceBar:
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    adj_close: float
    volume: int
    currency: str
    corporate_action_flag: Optional[str]
    source: str


class MarketLoader:
    def __init__(self, warehouse_root: str) -> None:
        self.warehouse_root = Path(warehouse_root)

    def load_prices(
        self, symbols: Iterable[str], start: datetime, end: datetime, source: str = "vendor"
    ) -> Path:
        prices_dir = self.warehouse_root / "market" / "prices"
        prices_dir.mkdir(parents=True, exist_ok=True)

        records: List[PriceBar] = []
        for symbol in symbols:
            for offset in range((end.date() - start.date()).days + 1):
                date = start + timedelta(days=offset)
                bar = PriceBar(
                    symbol=symbol.upper(),
                    date=date,
                    open=100 + offset,
                    high=101 + offset,
                    low=99 + offset,
                    close=100.5 + offset,
                    adj_close=100.4 + offset,
                    volume=1_000_000 + offset,
                    currency="USD",
                    corporate_action_flag=None,
                    source=source,
                )
                records.append(bar)

        df = pd.DataFrame([r.__dict__ for r in records])
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")

        output_paths = []
        for date, group in df.groupby(df["date"].dt.date):
            for symbol, sym_group in group.groupby("symbol"):
                date_dir = prices_dir / f"dt={date:%Y-%m-%d}" / f"symbol={symbol}"
                date_dir.mkdir(parents=True, exist_ok=True)
                file_path = date_dir / f"part-{uuid.uuid4().hex}.parquet"
                sym_group.to_parquet(file_path, index=False)
                output_paths.append(file_path)

        manifest = prices_dir / "manifest.json"
        manifest_record = {
            "symbols": list(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": source,
            "files": [str(p) for p in output_paths],
            "written_at": datetime.utcnow().isoformat(),
        }
        manifest.write_text(json.dumps(manifest_record, indent=2))
        return prices_dir

    def load_corporate_actions(
        self, symbols: Iterable[str], start: datetime, end: datetime, source: str = "vendor"
    ) -> Path:
        actions_dir = self.warehouse_root / "market" / "corp_actions"
        actions_dir.mkdir(parents=True, exist_ok=True)

        rows: List[Dict[str, Optional[str]]] = []
        for symbol in symbols:
            rows.append(
                {
                    "symbol": symbol.upper(),
                    "ex_date": start.date().isoformat(),
                    "action_type": "split",
                    "ratio": "2:1",
                    "cash_amount": None,
                    "notes": "Simulated split",
                    "source": source,
                }
            )

        df = pd.DataFrame(rows)
        df["ex_date"] = pd.to_datetime(df["ex_date"]).dt.tz_localize("UTC")

        output_paths = []
        for date, group in df.groupby(df["ex_date"].dt.date):
            for symbol, sym_group in group.groupby("symbol"):
                date_dir = actions_dir / f"dt={date:%Y-%m-%d}" / f"symbol={symbol}"
                date_dir.mkdir(parents=True, exist_ok=True)
                file_path = date_dir / f"part-{uuid.uuid4().hex}.parquet"
                sym_group.to_parquet(file_path, index=False)
                output_paths.append(file_path)

        manifest = actions_dir / "manifest.json"
        manifest_record = {
            "symbols": list(symbols),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "source": source,
            "files": [str(p) for p in output_paths],
            "written_at": datetime.utcnow().isoformat(),
        }
        manifest.write_text(json.dumps(manifest_record, indent=2))
        return actions_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load market prices and corporate actions")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "TSLA"])
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start) if args.start else datetime.utcnow() - timedelta(days=5)
    end = datetime.fromisoformat(args.end) if args.end else datetime.utcnow()

    loader = MarketLoader(args.warehouse)
    loader.load_prices(symbols=args.symbols, start=start, end=end)
    loader.load_corporate_actions(symbols=args.symbols, start=start, end=end)


if __name__ == "__main__":
    main()
