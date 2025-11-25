from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class PortfolioConfig:
    max_weight: float = 0.05
    min_weight: float = -0.05
    turnover_limit: Optional[float] = 0.5
    rebalance_freq: str = "D"
    cash_symbol: str = "CASH"


class PortfolioOptimizer:
    def __init__(self, warehouse_root: str, config: PortfolioConfig) -> None:
        self.warehouse_root = Path(warehouse_root)
        self.config = config

    def _load_alpha(self, date: datetime) -> pd.DataFrame:
        alpha_path = self.warehouse_root / "alpha" / f"dt={date:%Y-%m-%d}" / "alpha.parquet"
        if not alpha_path.exists():
            raise FileNotFoundError(f"Alpha file not found at {alpha_path}")
        return pd.read_parquet(alpha_path)

    def form_portfolio(
        self, date: datetime, cov_matrix: pd.DataFrame, prev_weights: Optional[pd.Series] = None
    ) -> pd.Series:
        alpha_df = self._load_alpha(date)
        alphas = alpha_df.set_index("symbol")["alpha"]
        symbols = alphas.index

        cov = cov_matrix.loc[symbols, symbols]
        inv_cov = np.linalg.pinv(cov.values)
        raw_weights = inv_cov @ alphas.values
        raw_weights = raw_weights / np.sum(np.abs(raw_weights))

        weights = pd.Series(raw_weights, index=symbols)
        weights = weights.clip(lower=self.config.min_weight, upper=self.config.max_weight)

        if prev_weights is not None and self.config.turnover_limit is not None:
            turnover = np.abs(weights - prev_weights.reindex(symbols).fillna(0)).sum()
            if turnover > self.config.turnover_limit:
                scale = self.config.turnover_limit / turnover
                weights = prev_weights.reindex(symbols).fillna(0) + scale * (weights - prev_weights.reindex(symbols).fillna(0))

        weights[self.config.cash_symbol] = 1 - weights.sum()
        return weights

    def sample_backtest_config(self) -> Dict[str, str]:
        return {
            "rebalance_freq": self.config.rebalance_freq,
            "max_weight": self.config.max_weight,
            "min_weight": self.config.min_weight,
            "turnover_limit": self.config.turnover_limit,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Form a simple portfolio from alpha signals")
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--date", required=False)
    parser.add_argument("--max-weight", type=float, default=0.05)
    parser.add_argument("--min-weight", type=float, default=-0.05)
    parser.add_argument("--turnover", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = datetime.fromisoformat(args.date) if args.date else datetime.utcnow()
    config = PortfolioConfig(
        max_weight=args.max_weight,
        min_weight=args.min_weight,
        turnover_limit=args.turnover,
    )
    optimizer = PortfolioOptimizer(args.warehouse, config)

    # Dummy covariance for illustration
    alpha_df = optimizer._load_alpha(date)
    symbols = alpha_df["symbol"].tolist()
    cov_matrix = pd.DataFrame(np.eye(len(symbols)) * 0.05, index=symbols, columns=symbols)

    weights = optimizer.form_portfolio(date=date, cov_matrix=cov_matrix)
    print(weights)


if __name__ == "__main__":
    main()
