from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


@dataclass
class PricePoint:
    symbol: str
    date: datetime
    close: float


class MarketData:
    """Lightweight helpers for market data and factor residuals."""

    def fetch_prices(self, symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        records: List[PricePoint] = []
        for symbol in symbols:
            for i in range((end_dt.date() - start_dt.date()).days + 1):
                day = start_dt + timedelta(days=i)
                records.append(PricePoint(symbol=symbol.upper(), date=day, close=100 + i))
        df = pd.DataFrame([r.__dict__ for r in records])
        df["date"] = pd.to_datetime(df["date"])
        return df

    @staticmethod
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        prices = prices.sort_values(["symbol", "date"])
        prices["return"] = prices.groupby("symbol")["close"].pct_change()
        return prices.dropna(subset=["return"]).reset_index(drop=True)

    @staticmethod
    def residualize(returns: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
        merged = returns.merge(factors, on="date", how="left")
        factor_cols = [c for c in factors.columns if c != "date"]
        residuals: List[dict] = []
        for symbol, group in merged.groupby("symbol"):
            X = group[factor_cols].to_numpy()
            y = group["return"].to_numpy()
            if X.size == 0 or y.size == 0:
                continue
            X = np.column_stack([np.ones(len(X)), X])
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ coeffs
            resid = y - y_hat
            residuals.extend(
                {"symbol": symbol, "date": d, "residual": r} for d, r in zip(group["date"], resid)
            )
        return pd.DataFrame(residuals)

    @staticmethod
    def idiosyncratic_risk(residuals: pd.DataFrame, window: Optional[int] = None) -> pd.DataFrame:
        df = residuals.copy()
        df = df.sort_values(["symbol", "date"])
        if window:
            risk = df.groupby("symbol")["residual"].rolling(window).std().reset_index(level=0, drop=True)
            df["idio_risk"] = risk
        else:
            df["idio_risk"] = df.groupby("symbol")["residual"].transform("std")
        return df.dropna(subset=["idio_risk"]).reset_index(drop=True)
