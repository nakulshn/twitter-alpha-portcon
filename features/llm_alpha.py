from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class LLMOutput:
    symbol: str
    tweet_id: str
    created_at_utc: datetime
    sentiment: float
    relevance: float
    topic: str
    safety_flags: List[str]
    model: str
    model_version: str
    prompt_hash: str
    params: Dict[str, Any]
    extracted_at: datetime


class PromptTemplate:
    def __init__(self, template: str) -> None:
        self.template = template

    def format(self, tweet_text: str, symbol: str) -> str:
        return self.template.format(text=tweet_text, symbol=symbol)

    def hash(self) -> str:
        return hashlib.sha256(self.template.encode()).hexdigest()


class LLMAlphaExtractor:
    def __init__(
        self,
        warehouse_root: str,
        model: str,
        model_version: str,
        prompt_template: PromptTemplate,
        cache_dir: Optional[str] = None,
        rate_limit_per_minute: int = 60,
    ) -> None:
        self.warehouse_root = Path(warehouse_root)
        self.model = model
        self.model_version = model_version
        self.prompt_template = prompt_template
        self.cache_dir = Path(cache_dir) if cache_dir else self.warehouse_root / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_per_minute = rate_limit_per_minute
        self._last_call_timestamp = 0.0

    def _load_raw_tweets(self, date: datetime, symbols: Iterable[str]) -> pd.DataFrame:
        raw_dir = self.warehouse_root / "twitter" / "raw" / f"{date:%Y}" / f"{date:%m}" / f"{date:%d}"
        parts = list(raw_dir.glob("*.parquet"))
        if not parts:
            raise FileNotFoundError(f"No raw tweets for {date.date()} at {raw_dir}")
        frames = [pd.read_parquet(p) for p in parts]
        df = pd.concat(frames, ignore_index=True)
        df = df[df["symbol"].isin([s.upper() for s in symbols])]
        return df

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        now = time.time()
        min_interval = 60.0 / self.rate_limit_per_minute
        if now - self._last_call_timestamp < min_interval:
            time.sleep(min_interval - (now - self._last_call_timestamp))
        self._last_call_timestamp = time.time()

        # Placeholder LLM response
        return {
            "sentiment": 0.2,
            "relevance": 0.8,
            "topic": "earnings",
            "safety_flags": [],
        }

    def _cache_path(self, tweet_id: str) -> Path:
        return self.cache_dir / f"{tweet_id}.json"

    def _maybe_from_cache(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        path = self._cache_path(tweet_id)
        if path.exists():
            return json.loads(path.read_text())
        return None

    def _write_cache(self, tweet_id: str, response: Dict[str, Any]) -> None:
        self._cache_path(tweet_id).write_text(json.dumps(response))

    def extract(self, date: datetime, symbols: Iterable[str]) -> Path:
        df = self._load_raw_tweets(date, symbols)
        outputs: List[LLMOutput] = []
        prompt_hash = self.prompt_template.hash()

        for _, row in df.iterrows():
            cached = self._maybe_from_cache(row["tweet_id"])
            if cached:
                llm_resp = cached
            else:
                prompt = self.prompt_template.format(row["text"], row["symbol"])
                llm_resp = self._call_llm(prompt)
                self._write_cache(row["tweet_id"], llm_resp)

            outputs.append(
                LLMOutput(
                    symbol=row["symbol"],
                    tweet_id=row["tweet_id"],
                    created_at_utc=pd.to_datetime(row["created_at_utc"]),
                    sentiment=float(llm_resp["sentiment"]),
                    relevance=float(llm_resp["relevance"]),
                    topic=str(llm_resp.get("topic", "unknown")),
                    safety_flags=list(llm_resp.get("safety_flags", [])),
                    model=self.model,
                    model_version=self.model_version,
                    prompt_hash=prompt_hash,
                    params={"rate_limit": self.rate_limit_per_minute},
                    extracted_at=datetime.now(timezone.utc),
                )
            )

        features_dir = self.warehouse_root / "twitter" / "features" / self.model_version / f"dt={date:%Y-%m-%d}"
        features_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        for symbol, group in pd.DataFrame([dataclasses.asdict(o) for o in outputs]).groupby("symbol"):
            symbol_dir = features_dir / f"symbol={symbol}"
            symbol_dir.mkdir(parents=True, exist_ok=True)
            file_path = symbol_dir / f"part-{uuid.uuid4().hex}.parquet"
            group.to_parquet(file_path, index=False)
            output_paths.append(file_path)

        manifest = features_dir / "manifest.json"
        manifest_record = {
            "model": self.model,
            "model_version": self.model_version,
            "prompt_hash": prompt_hash,
            "date": f"{date:%Y-%m-%d}",
            "symbols": list(symbols),
            "files": [str(p) for p in output_paths],
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        manifest.write_text(json.dumps(manifest_record, indent=2))
        return features_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract LLM alpha features from tweets")
    parser.add_argument("--config", required=False)
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--model", default="gpt-5.1-codex-max")
    parser.add_argument("--model-version", default="v1")
    parser.add_argument("--date", required=False)
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "TSLA"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    date = datetime.fromisoformat(args.date) if args.date else datetime.now(timezone.utc)
    template = PromptTemplate(
        template=(
            "You are an equity analyst. Rate sentiment (-1 to 1) and relevance (0 to 1) "
            "for the ticker {symbol} given the tweet: {text}"
        )
    )
    extractor = LLMAlphaExtractor(
        warehouse_root=args.warehouse,
        model=args.model,
        model_version=args.model_version,
        prompt_template=template,
        rate_limit_per_minute=120,
    )
    extractor.extract(date=date, symbols=args.symbols)


if __name__ == "__main__":
    main()
