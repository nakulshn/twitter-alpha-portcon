from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class LLMResponse:
    prompt: str
    output: str
    model: str
    timestamp: float


class LLMClient:
    """Tiny wrapper for calling an LLM endpoint.

    Swap `_call_model` for your provider (OpenAI, Anthropic, local model). A
    simple JSON file cache is included to avoid duplicate calls during research.
    """

    def __init__(self, model: str, cache_dir: Optional[str] = None, rate_limit_per_minute: Optional[int] = None) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir) if cache_dir else Path(".cache_llm")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.rate_limit_per_minute = rate_limit_per_minute
        self._last_call = 0.0

    def query(self, prompt: str) -> LLMResponse:
        cached = self._maybe_read_cache(prompt)
        if cached:
            return cached

        self._respect_rate_limit()
        output = self._call_model(prompt)
        response = LLMResponse(prompt=prompt, output=output, model=self.model, timestamp=time.time())
        self._write_cache(prompt, response)
        return response

    def _call_model(self, prompt: str) -> str:
        # Replace with a real API call
        return f"Stub response for: {prompt[:80]}..."

    def _respect_rate_limit(self) -> None:
        if not self.rate_limit_per_minute:
            return
        min_interval = 60.0 / self.rate_limit_per_minute
        since_last = time.time() - self._last_call
        if since_last < min_interval:
            time.sleep(min_interval - since_last)
        self._last_call = time.time()

    def _cache_path(self, prompt: str) -> Path:
        key = str(abs(hash(prompt)))
        return self.cache_dir / f"{key}.json"

    def _maybe_read_cache(self, prompt: str) -> Optional[LLMResponse]:
        path = self._cache_path(prompt)
        if path.exists():
            data = json.loads(path.read_text())
            return LLMResponse(**data)
        return None

    def _write_cache(self, prompt: str, response: LLMResponse) -> None:
        path = self._cache_path(prompt)
        path.write_text(json.dumps(response.__dict__))


def demo() -> None:
    client = LLMClient(model="stub-llm", rate_limit_per_minute=30)
    resp = client.query("Summarize the sentiment on AAPL earnings")
    print(resp)


if __name__ == "__main__":
    demo()
