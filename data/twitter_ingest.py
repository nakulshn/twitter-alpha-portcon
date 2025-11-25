from __future__ import annotations

import argparse
import dataclasses
import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd


@dataclass
class TweetRecord:
    symbol: str
    tweet_id: str
    user_id: str
    username: str
    created_at_utc: datetime
    text: str
    language: str
    like_count: int
    retweet_count: int
    reply_count: int
    quote_count: int
    is_retweet: bool
    source: str
    ingested_at: datetime
    raw_json: Dict[str, Any]


class TwitterIngestor:
    """Pull Twitter/X content and store normalized Parquet files.

    Replace `_fetch_raw_tweets` with API calls (search/archives). Partitioned parquet
    files make downstream backtests efficient.
    """

    def __init__(self, warehouse_root: str) -> None:
        self.warehouse_root = Path(warehouse_root)

    def ingest(
        self,
        symbols: List[str],
        start: datetime,
        end: datetime,
        language: Optional[str] = None,
        include_retweets: bool = False,
        spam_filter: Optional[Iterable[str]] = None,
        source: str = "api",
        cashtag_prefix: str = "$",
    ) -> Path:
        raw_dir = self.warehouse_root / "twitter" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        tweets: List[TweetRecord] = []
        for symbol in symbols:
            raw_jsons = self._fetch_raw_tweets(symbol, start, end, cashtag_prefix)
            for obj in raw_jsons:
                record = self._normalize(
                    obj,
                    symbol=symbol,
                    language=language,
                    include_retweets=include_retweets,
                    spam_filter=spam_filter,
                    source=source,
                )
                if record:
                    tweets.append(record)

        if not tweets:
            raise ValueError("No tweets collected for given filters")

        df = pd.DataFrame([dataclasses.asdict(t) for t in tweets])
        df["created_date"] = pd.to_datetime(df["created_at_utc"]).dt.date

        partitions = df.groupby("created_date")
        output_paths: List[Path] = []
        for created_date, group in partitions:
            date_dir = raw_dir / f"{created_date:%Y}" / f"{created_date:%m}" / f"{created_date:%d}"
            date_dir.mkdir(parents=True, exist_ok=True)
            file_path = date_dir / f"part-{uuid.uuid4().hex}.parquet"
            group.drop(columns=["created_date"]).to_parquet(file_path, index=False)
            output_paths.append(file_path)

        manifest = raw_dir / "manifest.json"
        manifest_record = {
            "symbols": symbols,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "language": language,
            "include_retweets": include_retweets,
            "source": source,
            "spam_filter": list(spam_filter) if spam_filter else None,
            "files": [str(p) for p in output_paths],
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        existing = []
        if manifest.exists():
            existing = json.loads(manifest.read_text())
        existing.append(manifest_record)
        manifest.write_text(json.dumps(existing, indent=2))
        return raw_dir

    def _fetch_raw_tweets(
        self, symbol: str, start: datetime, end: datetime, cashtag_prefix: str
    ) -> List[Dict[str, Any]]:
        """Stub to fetch tweets. Replace with Twitter API/archives.

        This uses deterministic mock data for demonstration.
        """

        return [
            {
                "id": f"{symbol}-{int(start.timestamp())}",
                "user": {"id": "123", "username": "quantfan"},
                "created_at": start.isoformat(),
                "text": f"{cashtag_prefix}{symbol} looks interesting around earnings",
                "lang": "en",
                "public_metrics": {
                    "like_count": 10,
                    "retweet_count": 1,
                    "reply_count": 0,
                    "quote_count": 0,
                },
                "referenced_tweets": [],
                "source": "archive",
            },
            {
                "id": f"{symbol}-{int((start + timedelta(hours=1)).timestamp())}",
                "user": {"id": "124", "username": "spammy"},
                "created_at": (start + timedelta(hours=1)).isoformat(),
                "text": f"Check out {cashtag_prefix}{symbol}!!!",
                "lang": "en",
                "public_metrics": {
                    "like_count": 0,
                    "retweet_count": 0,
                    "reply_count": 0,
                    "quote_count": 0,
                },
                "referenced_tweets": [],
                "source": "archive",
            },
        ]

    def _normalize(
        self,
        raw: Dict[str, Any],
        symbol: str,
        language: Optional[str],
        include_retweets: bool,
        spam_filter: Optional[Iterable[str]],
        source: str,
    ) -> Optional[TweetRecord]:
        lang = raw.get("lang")
        if language and lang and lang != language:
            return None

        is_retweet = any(rt.get("type") == "retweeted" for rt in raw.get("referenced_tweets", []))
        if is_retweet and not include_retweets:
            return None

        text = raw.get("text", "")
        if spam_filter and any(token.lower() in text.lower() for token in spam_filter):
            return None

        metrics = raw.get("public_metrics", {})
        return TweetRecord(
            symbol=symbol.upper(),
            tweet_id=str(raw.get("id")),
            user_id=str(raw.get("user", {}).get("id")),
            username=raw.get("user", {}).get("username", ""),
            created_at_utc=datetime.fromisoformat(raw.get("created_at")).replace(tzinfo=timezone.utc),
            text=text,
            language=lang or "unknown",
            like_count=int(metrics.get("like_count", 0)),
            retweet_count=int(metrics.get("retweet_count", 0)),
            reply_count=int(metrics.get("reply_count", 0)),
            quote_count=int(metrics.get("quote_count", 0)),
            is_retweet=is_retweet,
            source=source,
            ingested_at=datetime.now(timezone.utc),
            raw_json=raw,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest Twitter/X data into parquet")
    parser.add_argument("--config", required=False, help="Path to ingestion config")
    parser.add_argument("--symbols", nargs="+", default=["AAPL", "TSLA"])
    parser.add_argument("--start", required=False)
    parser.add_argument("--end", required=False)
    parser.add_argument("--warehouse", default="data/warehouse")
    parser.add_argument("--language", default=None)
    parser.add_argument("--include-retweets", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.start) if args.start else datetime.now(timezone.utc) - timedelta(days=1)
    end = datetime.fromisoformat(args.end) if args.end else datetime.now(timezone.utc)

    ingestor = TwitterIngestor(args.warehouse)
    ingestor.ingest(
        symbols=args.symbols,
        start=start,
        end=end,
        language=args.language,
        include_retweets=args.include_retweets,
        spam_filter={"spammy"},
        source="api",
    )


if __name__ == "__main__":
    main()
