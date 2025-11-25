from __future__ import annotations

import csv
import dataclasses
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


@dataclasses.dataclass
class ExperimentRecord:
    run_id: str
    config_path: str
    stage: str
    started_at: datetime
    finished_at: Optional[datetime]
    prompt_hash: Optional[str]
    model: Optional[str]
    model_version: Optional[str]
    metrics: Dict[str, Any]


class ExperimentLogger:
    """Lightweight experiment logger writing to CSV for reproducibility."""

    def __init__(self, log_path: str = "experiments.csv") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: ExperimentRecord) -> None:
        is_new = not self.log_path.exists()
        fieldnames = [
            "run_id",
            "config_path",
            "stage",
            "started_at",
            "finished_at",
            "prompt_hash",
            "model",
            "model_version",
            "metrics",
        ]
        with self.log_path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if is_new:
                writer.writeheader()
            writer.writerow(
                {
                    "run_id": record.run_id,
                    "config_path": record.config_path,
                    "stage": record.stage,
                    "started_at": record.started_at.isoformat(),
                    "finished_at": record.finished_at.isoformat()
                    if record.finished_at
                    else None,
                    "prompt_hash": record.prompt_hash,
                    "model": record.model,
                    "model_version": record.model_version,
                    "metrics": record.metrics,
                }
            )
