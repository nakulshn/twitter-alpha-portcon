from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional


@dataclass
class Tweet:
    handle: str
    tweet_id: str
    created_at: datetime
    text: str
    is_reply: bool = False


class TwitterClient:
    """Minimal Twitter/X surface area for research.

    Replace the stubbed data with calls to your preferred API (official v2/v1.1,
    Academic Research, or third-party archives).
    """

    def fetch_timeline(
        self,
        handle: str,
        start_dt: Optional[str] = None,
        end_dt: Optional[str] = None,
        include_replies: bool = True,
    ) -> List[Tweet]:
        start = self._parse_dt(start_dt) if start_dt else None
        end = self._parse_dt(end_dt) if end_dt else None

        sample = [
            Tweet(
                handle=handle,
                tweet_id=f"{handle}-1",
                created_at=datetime(2024, 1, 1, 12, 0, tzinfo=timezone.utc),
                text="Sample tweet about $AAPL earnings",
            ),
            Tweet(
                handle=handle,
                tweet_id=f"{handle}-2",
                created_at=datetime(2024, 1, 1, 13, 0, tzinfo=timezone.utc),
                text="Reply with more context on $MSFT guidance",
                is_reply=True,
            ),
        ]

        def in_window(tweet: Tweet) -> bool:
            if start and tweet.created_at < start:
                return False
            if end and tweet.created_at > end:
                return False
            if not include_replies and tweet.is_reply:
                return False
            return True

        return [t for t in sample if in_window(t)]

    def fetch_followers(self, handle: str) -> List[Dict[str, str]]:
        return [
            {"handle": f"follower_{i}", "user_id": str(1000 + i)}
            for i in range(3)
        ]

    def fetch_following(self, handle: str) -> List[Dict[str, str]]:
        return [
            {"handle": f"following_{i}", "user_id": str(2000 + i)}
            for i in range(2)
        ]

    @staticmethod
    def _parse_dt(value: str) -> datetime:
        dt = datetime.fromisoformat(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)


def demo() -> None:
    client = TwitterClient()
    timeline = client.fetch_timeline(handle="quant", start_dt="2024-01-01T11:00:00+00:00", end_dt="2024-01-01T13:30:00+00:00")
    followers = client.fetch_followers(handle="quant")
    following = client.fetch_following(handle="quant")
    print("Timeline:", timeline)
    print("Followers:", followers)
    print("Following:", following)


if __name__ == "__main__":
    demo()
