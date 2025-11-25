# Twitter Alpha Discovery

This repository contains a small pipeline that expands from a set of seed Twitter/X accounts, finds other investing-focused accounts they follow, and then scans those accounts' timelines for ticker mentions with bullish/bearish sentiment labeling via an LLM.

## Prerequisites

* Python 3.11+
* Twitter/X API v2 bearer token (`TWITTER_BEARER_TOKEN` environment variable)
* OpenAI API key (`OPENAI_API_KEY` environment variable)

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

Run the CLI with one or more seed handles (without the `@` symbol). The tool will:

1. Resolve each seed to a user id.
2. Fetch accounts the seed follows (configurable via `--follow-pages`).
3. Use an LLM to keep only accounts that look like investing profiles.
4. Pull recent tweets from those investing accounts and classify each ticker mention as bullish, bearish, or neutral.

```bash
python main.py buysidealch3mist investor_handle --follow-pages 1 --tweet-pages 2 --output signals.json
```

The output is a JSON document with an entry per investing account, mapping tickers to the list of dated stances pulled from their tweets/replies. Example:

```json
[
  {
    "username": "example_account",
    "user_id": "123",
    "tickers": {
      "AAPL": [
        {
          "stance": "bullish",
          "reason": "Calls out strong earnings",
          "date": "2024-05-20T10:15:00.000Z",
          "tweet_id": "1234567890",
          "text": "$AAPL to the moon"
        }
      ]
    }
  }
]
```

## Configuration flags

* `--follow-pages`: Number of pages of followings to retrieve per seed (1000 users per page). Defaults to 1.
* `--tweet-pages`: Number of tweet pages to analyze per investing account (100 tweets per page). Defaults to 2.
* `--openai-model`: Model name for OpenAI classification. Defaults to `gpt-4o-mini`.
* `--output`: Path to the JSON file to write. Defaults to `signals.json`.

## Notes

* The script relies on live Twitter and OpenAI APIs. API quotas and network availability can affect runtime.
* Tweets are scanned for cashtags using the `$TICKER` pattern (1-5 letters). Mentions outside that pattern are ignored.
* The LLM prompts are defined in `src/twitter_alpha/llm.py` and can be adjusted if you need different output fields.
