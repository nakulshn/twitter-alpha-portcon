from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal, Optional

from openai import OpenAI


@dataclass
class AccountPrediction:
    username: str
    is_investing_account: bool
    reason: str


@dataclass
class SentimentPrediction:
    ticker: str
    stance: Literal["bullish", "bearish", "neutral"]
    reason: str
    tweet_id: Optional[str] = None
    created_at: Optional[str] = None
    text: Optional[str] = None


class LlmClassifier:
    """Thin wrapper for the OpenAI client used for both prompts.

    The class keeps prompts consistent and ensures structured responses that are
    easy to post-process.
    """

    def __init__(self, *, model: str = "gpt-4o-mini", client: Optional[OpenAI] = None) -> None:
        self.model = model
        self.client = client or OpenAI()

    def _parse_json(self, content: str) -> dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError as err:
            raise ValueError(f"Expected JSON from model, received: {content}") from err

    def classify_account(self, username: str, description: str) -> AccountPrediction:
        prompt = (
            "You label Twitter bios as investing accounts or not. "
            "Return JSON with fields is_investing_account (boolean) and reason (string)."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Username: {username}\nBio: {description}\nJSON:",
            },
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0)
        content = response.choices[0].message.content.strip()
        data = self._parse_json(content)
        return AccountPrediction(
            username=username,
            is_investing_account=bool(data.get("is_investing_account")),
            reason=str(data.get("reason", "")),
        )

    def classify_sentiment(self, text: str, ticker: str) -> SentimentPrediction:
        prompt = (
            "You judge whether a tweet is bullish or bearish on a ticker. "
            "Return JSON with fields ticker (string), stance ('bullish'/'bearish'/'neutral'), reason (string)."
        )
        messages = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Ticker: {ticker}\nTweet: {text}\nJSON:",
            },
        ]
        response = self.client.chat.completions.create(model=self.model, messages=messages, temperature=0)
        content = response.choices[0].message.content.strip()
        data = self._parse_json(content)
        stance = data.get("stance", "neutral")
        if stance not in {"bullish", "bearish", "neutral"}:
            stance = "neutral"
        return SentimentPrediction(
            ticker=ticker,
            stance=stance,  # type: ignore[arg-type]
            reason=str(data.get("reason", "")),
        )


__all__ = ["AccountPrediction", "SentimentPrediction", "LlmClassifier"]
