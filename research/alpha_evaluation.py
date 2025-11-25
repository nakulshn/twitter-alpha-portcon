from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Tuple

import numpy as np
import pandas as pd


@dataclass
class EvaluationConfig:
    horizon_days: int = 1
    factor_family: str = "fama_french_5"


class AlphaEvaluator:
    def __init__(self, warehouse_root: str, config: EvaluationConfig) -> None:
        self.warehouse_root = Path(warehouse_root)
        self.config = config

    def _load_alpha(self, start: datetime, end: datetime) -> pd.DataFrame:
        frames = []
        for offset in range((end.date() - start.date()).days + 1):
            date = start + timedelta(days=offset)
            path = self.warehouse_root / "alpha" / f"dt={date:%Y-%m-%d}" / "alpha.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _load_prices(self, start: datetime, end: datetime) -> pd.DataFrame:
        prices_dir = self.warehouse_root / "market" / "prices"
        parts = list(prices_dir.glob("dt=*/symbol=*/part-*.parquet"))
        frames = [pd.read_parquet(p) for p in parts]
        df = pd.concat(frames, ignore_index=True)
        mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        return df.loc[mask]

    def _load_factors(self, start: datetime, end: datetime) -> pd.DataFrame:
        factors_dir = self.warehouse_root / "factors" / self.config.factor_family
        parts = list(factors_dir.glob("dt=*/part-*.parquet"))
        frames = [pd.read_parquet(p) for p in parts]
        df = pd.concat(frames, ignore_index=True)
        mask = (df["date"] >= pd.to_datetime(start)) & (df["date"] <= pd.to_datetime(end))
        return df.loc[mask]

    def _forward_returns(self, prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.sort_values(["symbol", "date"])
        prices["next_close"] = prices.groupby("symbol")["close"].shift(-self.config.horizon_days)
        prices["fwd_return"] = prices["next_close"] / prices["close"] - 1
        return prices.dropna(subset=["fwd_return"])

    def _merge_alpha_returns(self, alpha: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
        returns["date_only"] = pd.to_datetime(returns["date"]).dt.date
        alpha["as_of_date"] = pd.to_datetime(alpha["as_of"]).dt.date
        merged = alpha.merge(
            returns,
            left_on=["symbol", "as_of_date"],
            right_on=["symbol", "date_only"],
            how="inner",
        )
        return merged

    def _ols(self, y: pd.Series, X: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        X = pd.DataFrame({"intercept": 1.0, **{col: X[col] for col in X.columns}})
        beta = pd.Series(np.linalg.pinv(X.values) @ y.values, index=X.columns)
        residuals = y - X @ beta
        return beta, residuals

    def evaluate(self, start: datetime, end: datetime) -> pd.DataFrame:
        alpha = self._load_alpha(start, end)
        prices = self._load_prices(start, end + timedelta(days=self.config.horizon_days))
        factors = self._load_factors(start, end)

        returns = self._forward_returns(prices)
        merged = self._merge_alpha_returns(alpha, returns)

        factor_cols = [c for c in ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"] if c in factors.columns]
        merged = merged.merge(
            factors[["date"] + factor_cols], left_on="date", right_on="date", how="left"
        )

        beta, residuals = self._ols(merged["fwd_return"], merged[factor_cols])
        merged["residual"] = residuals

        summary = {
            "ic": merged[["alpha", "fwd_return"]].corr().iloc[0, 1],
            "t_stat_residual": beta.get("intercept", 0) / merged["residual"].std()
            if not merged.empty
            else 0,
            "n_obs": len(merged),
        }
        print("Evaluation summary", summary)
        return merged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate alpha via factor regressions")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    parser.add_argument("--horizon-days", type=int, default=1)
    parser.add_argument("--factor-family", default="fama_french_5")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start) if args.start else datetime.utcnow() - timedelta(days=10)
    end = datetime.fromisoformat(args.end) if args.end else datetime.utcnow()
    config = EvaluationConfig(horizon_days=args.horizon_days, factor_family=args.factor_family)
    evaluator = AlphaEvaluator(args.warehouse, config)
    evaluator.evaluate(start=start, end=end)


if __name__ == "__main__":
    main()
