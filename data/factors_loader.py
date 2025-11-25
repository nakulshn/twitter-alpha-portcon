from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List

import pandas as pd


@dataclass
class FactorRow:
    date: datetime
    mkt: float
    smb: float
    hml: float


class FactorsLoader:
    """Stubbed factor loader (e.g., Fama-French)."""

    def load_factors(self, start: str, end: str) -> pd.DataFrame:
        start_dt = datetime.fromisoformat(start)
        end_dt = datetime.fromisoformat(end)
        rows: List[FactorRow] = []
        for i in range((end_dt.date() - start_dt.date()).days + 1):
            day = start_dt + timedelta(days=i)
            rows.append(FactorRow(date=day, mkt=0.01, smb=0.002, hml=-0.001))
        df = pd.DataFrame([r.__dict__ for r in rows])
        df["date"] = pd.to_datetime(df["date"])
        return df
