from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Universe:
    tickers: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # store uppercase tickers for consistency
        self.tickers = [t.upper() for t in self.tickers]

    def index_map(self) -> Dict[str, int]:
        return {ticker: idx for idx, ticker in enumerate(self.tickers)}


@dataclass
class AlphaVector:
    universe: Universe
    values: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.values = np.zeros(len(self.universe.tickers))

    def from_dict(self, alpha_dict: Dict[str, float]) -> "AlphaVector":
        for ticker, value in alpha_dict.items():
            t = ticker.upper()
            if t not in self.universe.index_map():
                continue
            self.values[self.universe.index_map()[t]] = float(value)
        return self

    def to_dict(self) -> Dict[str, float]:
        return {ticker: float(self.values[idx]) for ticker, idx in self.universe.index_map().items()}

    def clip(self, lower: Optional[float] = None, upper: Optional[float] = None) -> "AlphaVector":
        if lower is not None:
            self.values = np.maximum(self.values, lower)
        if upper is not None:
            self.values = np.minimum(self.values, upper)
        return self

    def normalize(self) -> "AlphaVector":
        norm = np.linalg.norm(self.values)
        if norm > 0:
            self.values = self.values / norm
        return self

    def as_vector(self) -> np.ndarray:
        return self.values

    def nonzero(self) -> Dict[str, float]:
        return {t: v for t, v in self.to_dict().items() if v != 0.0}
