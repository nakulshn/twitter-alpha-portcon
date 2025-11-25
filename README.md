# Twitter Alpha Research Setup (minimal)

This repository provides minimal, swappable building blocks for exploring Twitter/X-driven equity research. Each module is intentionally small so you can replace the stubs with real data providers and models.

## Key pieces

- **Twitter access (`data/twitter_ingest.py`)**: fetch a user's tweets/replies (text + timestamps) and their follower/following lists. API calls are stubbed so you can drop in your preferred Twitter client.
- **LLM bridge (`features/llm_alpha.py`)**: a generic `LLMClient` that submits text and returns model output, with optional caching and rate limiting hooks.
- **Alpha container (`alpha/construct_alpha.py`)**: an `AlphaVector` that tracks a universe of tickers and stores alphas as a dict or vector; normalization and clipping are optional.
- **Market + factors (`data/market_loader.py`, `data/factors_loader.py`)**: helpers to pull basic price data, compute returns, residualize against factors, and estimate idiosyncratic risk. Factor loading is stubbed via `FactorsLoader`.

## Usage snippets

```python
from data.twitter_ingest import TwitterClient
from features.llm_alpha import LLMClient
from alpha.construct_alpha import AlphaVector, Universe
from data.market_loader import MarketData
from data.factors_loader import FactorsLoader

twitter = TwitterClient()
timeline = twitter.fetch_timeline(handle="user", start_dt="2024-01-01", end_dt="2024-01-02", include_replies=True)
followers = twitter.fetch_followers(handle="user")

llm = LLMClient(model="stub-llm", rate_limit_per_minute=60)
response = llm.query("How does AAPL look ahead of earnings?")

universe = Universe(["AAPL", "MSFT"])
alpha = AlphaVector(universe).from_dict({"AAPL": 0.1, "MSFT": -0.05})

prices = MarketData().fetch_prices(symbols=universe.tickers, start="2024-01-01", end="2024-01-05")
returns = MarketData.compute_returns(prices)
factors = FactorsLoader().load_factors(start="2024-01-01", end="2024-01-05")
residuals = MarketData.residualize(returns, factors)
idio_risk = MarketData.idiosyncratic_risk(residuals)
```

Replace the stubbed data-fetching pieces with your preferred APIs when you're ready to run full experiments.
