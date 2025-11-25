from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


@dataclass
class AlphaConfig:
    aggregation: str = "weighted"
    decay_half_life_hours: Optional[float] = 6.0
    outlier_clip: float = 3.0
    lag_minutes: int = 30


class AlphaConstructor:
    def __init__(self, warehouse_root: str, config: AlphaConfig) -> None:
        self.warehouse_root = Path(warehouse_root)
        self.config = config

    def _load_features(self, date: datetime, symbols: Iterable[str], model_version: str) -> pd.DataFrame:
        features_dir = self.warehouse_root / "twitter" / "features" / model_version / f"dt={date:%Y-%m-%d}"
        parts = list(features_dir.glob("symbol=*/part-*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No features found at {features_dir}")
        frames = [pd.read_parquet(p) for p in parts]
        df = pd.concat(frames, ignore_index=True)
        df = df[df["symbol"].isin([s.upper() for s in symbols])]
        return df

    def _decay_weights(self, timestamps: pd.Series, ref_time: datetime) -> pd.Series:
        if self.config.decay_half_life_hours is None:
            return pd.Series(1.0, index=timestamps.index)
        half_life = timedelta(hours=self.config.decay_half_life_hours)
        deltas = ref_time - pd.to_datetime(timestamps)
        return 0.5 ** (deltas / half_life)

    def construct(
        self,
        date: datetime,
        symbols: Iterable[str],
        model_version: str,
        agg_time: Optional[datetime] = None,
    ) -> pd.DataFrame:
        df = self._load_features(date, symbols, model_version)
        df["created_at_utc"] = pd.to_datetime(df["created_at_utc"])
        agg_time = agg_time or datetime.combine(date.date(), datetime.min.time()).replace(tzinfo=timezone.utc)

        df = df[df["created_at_utc"] <= agg_time - timedelta(minutes=self.config.lag_minutes)]
        df["weight"] = self._decay_weights(df["created_at_utc"], ref_time=agg_time)

        if self.config.aggregation == "mean":
            grouped = df.groupby(["symbol"])
            agg_df = grouped.apply(
                lambda g: pd.Series(
                    {
                        "alpha": (g["sentiment"] * g["weight"]).mean(),
                        "n_tweets": len(g),
                    }
                )
            ).reset_index()
        else:  # weighted by engagement defaults
            df["engagement"] = df[["like_count", "retweet_count", "reply_count", "quote_count"]].sum(axis=1)
            df["engagement"] = df["engagement"].replace(0, 1)
            df["combined_weight"] = df["weight"] * df["engagement"]
            grouped = df.groupby(["symbol"])
            agg_df = grouped.apply(
                lambda g: pd.Series(
                    {
                        "alpha": (g["sentiment"] * g["combined_weight"]).sum()
                        / g["combined_weight"].sum(),
                        "n_tweets": len(g),
                    }
                )
            ).reset_index()

        if self.config.outlier_clip:
            agg_df["alpha"] = agg_df["alpha"].clip(-self.config.outlier_clip, self.config.outlier_clip)

        agg_df["as_of"] = agg_time
        signals_dir = self.warehouse_root / "alpha" / f"dt={agg_time:%Y-%m-%d}"
        signals_dir.mkdir(parents=True, exist_ok=True)
        output_path = signals_dir / "alpha.parquet"
        agg_df.to_parquet(output_path, index=False)
        return agg_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Construct alpha signals from LLM features")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--date", required=False)
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "TSLA"])
    parser.add_argument("--aggregation", default="weighted")
    parser.add_argument("--lag-minutes", type=int, default=30)
    parser.add_argument("--decay-half-life-hours", type=float, default=6)
    parser.add_argument("--outlier-clip", type=float, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = datetime.fromisoformat(args.date) if args.date else datetime.now(timezone.utc)
    config = AlphaConfig(
        aggregation=args.aggregation,
        decay_half_life_hours=args.decay_half_life_hours,
        outlier_clip=args.outlier_clip,
        lag_minutes=args.lag_minutes,
    )
    constructor = AlphaConstructor(warehouse_root=args.warehouse, config=config)
    constructor.construct(date=date, symbols=args.symbols, model_version=args.model_version)


if __name__ == "__main__":
    main()
