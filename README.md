# Twitter Alpha Portfolio Research Scaffold

This repository sketches a lightweight research pipeline for building tradable equity signals from Twitter/X data enriched with LLM-derived features. It focuses on reproducible storage layouts, ingestion/preprocessing stubs, factor-aware alpha construction, and portfolio prototyping.

## Repository layout

```
configs/                 # YAML/JSON configuration examples for data, LLMs, and backtests
alpha/                   # Alpha construction utilities (aggregation, alignment)
data/                    # Data ingestion and loaders for Twitter, market, and factors
features/                # LLM-based feature extraction for tweets
portfolio/               # Simple portfolio construction/optimization prototype
research/                # Research scripts for evaluating alpha/factor regressions
utils/                   # Helper utilities (experiment logging, storage helpers)
```

## Data model and storage design

The pipeline uses columnar Parquet files stored locally (or on S3/warehouse) with hive-style partitioning for efficient backtests:

- **Twitter raw**: `data/warehouse/twitter/raw/{yyyy}/{mm}/{dd}/part-*.parquet`
  - Columns: `symbol`, `tweet_id`, `user_id`, `username`, `created_at_utc`, `text`, `language`, `like_count`, `retweet_count`, `reply_count`, `quote_count`, `is_retweet`, `source`, `ingested_at`, `raw_json`.
  - Partitioning: by `created_at_utc` date; optionally by `symbol` for heavy cashtag filtering.
  - Metadata: ingestion params (api source, filters) stored in file metadata or a sidecar manifest.

- **Twitter features (LLM)**: `data/warehouse/twitter/features/{model_version}/dt={yyyy-mm-dd}/symbol={SYM}/part-*.parquet`
  - Columns: `symbol`, `tweet_id`, `created_at_utc`, `sentiment`, `relevance`, `topic`, `safety_flags`, `model`, `model_version`, `prompt_hash`, `params`, `extracted_at`.
  - Partitioning: by date and symbol; include model version to keep history.

- **Market prices**: `data/warehouse/market/prices/dt={yyyy-mm-dd}/symbol={SYM}/part-*.parquet`
  - Columns: `symbol`, `date`, `open`, `high`, `low`, `close`, `adj_close`, `volume`, `currency`, `corporate_action_flag`, `source`.

- **Corporate actions**: `data/warehouse/market/corp_actions/dt={yyyy-mm-dd}/symbol={SYM}/part-*.parquet`
  - Columns: `symbol`, `ex_date`, `action_type`, `ratio`, `cash_amount`, `notes`, `source`.

- **Factors**: `data/warehouse/factors/{family}/dt={yyyy-mm-dd}/part-*.parquet`
  - Columns: `date`, `mkt_rf`, `smb`, `hml`, `rmw`, `cma`, `mom`, `rf`, `source`, `updated_at`.

## Getting started

1. Create and activate a Python environment (3.10+ recommended) with `pandas`, `pyarrow`, and optional `statsmodels` for regression.
2. Configure API keys for Twitter/X and any market data providers as environment variables (`TWITTER_BEARER_TOKEN`, etc.).
3. Edit configuration files under `configs/` to point to your storage (local path or `s3://...`) and provider settings.
4. Run ingestion, feature extraction, and alpha construction via the provided scripts (examples below).

```bash
python data/twitter_ingest.py --config configs/ingestion.yaml --start 2024-01-01 --end 2024-01-02
python features/llm_alpha.py --config configs/llm.yaml --date 2024-01-02
python alpha/construct_alpha.py --config configs/backtest.yaml --date 2024-01-02
python research/alpha_evaluation.py --config configs/backtest.yaml
```

## Privacy and rate limits

- Respect Twitter/X terms of service, privacy, and user deletion rules. Store only data allowed by the license and purge upon request.
- Implement rate limiting and backoff with the Twitter API; cached archives can reduce API usage.
- Keep LLM prompts and outputs compliant with provider usage policies and avoid sending unnecessary PII.

## Experiment logging

The `utils/experiment.py` helper records runs (config paths, prompt hashes, metrics) to a lightweight CSV or SQLite log for reproducibility.

## Notes

This scaffold uses placeholder implementations to illustrate data flow. Replace stubs with production-grade connectors, authentication, and monitoring for a live research stack.
