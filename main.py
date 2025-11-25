from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from src.twitter_alpha.llm import LlmClassifier
from src.twitter_alpha.pipeline import AlphaPipeline
from src.twitter_alpha.twitter_client import TwitterClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover investing voices and extract bullish/bearish calls from their timelines."
    )
    parser.add_argument(
        "seeds",
        nargs="+",
        help="Seed Twitter handles (without @) that the pipeline should expand from.",
    )
    parser.add_argument(
        "--output",
        default="signals.json",
        help="Where to write the aggregated ticker signals (JSON).",
    )
    parser.add_argument(
        "--follow-pages",
        type=int,
        default=1,
        help="How many pages of followings to retrieve per seed (1000 users per page).",
    )
    parser.add_argument(
        "--tweet-pages",
        type=int,
        default=2,
        help="How many pages of tweets to analyze per investing account (100 tweets per page).",
    )
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="OpenAI model name used for account and sentiment classification.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise SystemExit("TWITTER_BEARER_TOKEN is required for Twitter API access.")

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for LLM classification.")

    twitter_client = TwitterClient(bearer_token)
    classifier = LlmClassifier(model=args.openai_model)
    pipeline = AlphaPipeline(
        twitter_client=twitter_client,
        classifier=classifier,
        follow_pages=args.follow_pages,
        tweet_pages=args.tweet_pages,
    )

    logger.info("Discovering investing accounts from %s seeds", len(args.seeds))
    investing_users = pipeline.discover_investing_accounts(args.seeds)
    logger.info("Identified %s investing accounts", len(investing_users))

    logger.info("Collecting ticker sentiments from investing accounts")
    signals = pipeline.collect_account_signals(investing_users)
    logger.info("Aggregated signals for %s accounts", len(signals))

    serializable = []
    for account in signals:
        serializable.append(
            {
                "username": account.username,
                "user_id": account.user_id,
                "tickers": {
                    ticker: [
                        {
                            "stance": pred.stance,
                            "reason": pred.reason,
                            "date": pred.created_at,
                            "tweet_id": pred.tweet_id,
                            "text": pred.text,
                        }
                        for pred in ticker_signal.entries
                    ]
                    for ticker, ticker_signal in account.tickers.items()
                },
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    logger.info("Wrote results to %s", args.output)


if __name__ == "__main__":
    main()
