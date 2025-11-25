from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Iterable, List, MutableMapping, Set

from .llm import LlmClassifier, SentimentPrediction
from .twitter_client import Tweet, TwitterClient, TwitterUser

logger = logging.getLogger(__name__)


@dataclass
class TickerSignal:
    ticker: str
    entries: List[SentimentPrediction] = field(default_factory=list)


@dataclass
class AccountSignals:
    username: str
    user_id: str
    tickers: MutableMapping[str, TickerSignal] = field(default_factory=dict)

    def add_prediction(self, prediction: SentimentPrediction) -> None:
        if prediction.ticker not in self.tickers:
            self.tickers[prediction.ticker] = TickerSignal(ticker=prediction.ticker)
        self.tickers[prediction.ticker].entries.append(prediction)


class AlphaPipeline:
    ticker_pattern = re.compile(r"\$[A-Za-z]{1,5}")

    def __init__(
        self,
        twitter_client: TwitterClient,
        classifier: LlmClassifier,
        *,
        follow_pages: int = 1,
        tweet_pages: int = 2,
    ) -> None:
        self.twitter_client = twitter_client
        self.classifier = classifier
        self.follow_pages = follow_pages
        self.tweet_pages = tweet_pages

    def discover_investing_accounts(self, seeds: Iterable[str]) -> List[TwitterUser]:
        seen_users: Set[str] = set()
        investing_users: List[TwitterUser] = []

        for seed in seeds:
            user_id = self.twitter_client.get_user_id(seed)
            if not user_id:
                logger.warning("Skipping seed %s; unable to resolve id.", seed)
                continue

            following = self.twitter_client.get_user_following(
                user_id, max_pages=self.follow_pages
            )
            for user in following:
                if user.id in seen_users:
                    continue
                seen_users.add(user.id)
                prediction = self.classifier.classify_account(user.username, user.description)
                if prediction.is_investing_account:
                    investing_users.append(user)

        return investing_users

    def collect_account_signals(self, users: Iterable[TwitterUser]) -> List[AccountSignals]:
        results: List[AccountSignals] = []
        for user in users:
            tweets = self.twitter_client.get_recent_tweets(
                user.id, max_pages=self.tweet_pages, exclude_retweets=True
            )
            account = AccountSignals(username=user.username, user_id=user.id)
            for tweet in tweets:
                tickers = self.extract_tickers(tweet)
                for ticker in tickers:
                    prediction = self.classifier.classify_sentiment(tweet.text, ticker)
                    prediction.tweet_id = tweet.id
                    prediction.created_at = tweet.created_at
                    prediction.text = tweet.text
                    account.add_prediction(prediction)
            if account.tickers:
                results.append(account)
        return results

    def extract_tickers(self, tweet: Tweet) -> List[str]:
        return [ticker.lstrip("$").upper() for ticker in self.ticker_pattern.findall(tweet.text)]


__all__ = ["AlphaPipeline", "AccountSignals", "TickerSignal"]
