from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class TwitterUser:
    id: str
    username: str
    name: str
    description: str


@dataclass
class Tweet:
    id: str
    text: str
    created_at: str


class TwitterClient:
    """Lightweight wrapper around the Twitter API v2.

    This client only uses the endpoints needed for the alpha discovery workflow:
    * Resolve usernames to user ids.
    * List accounts a user follows.
    * Fetch recent tweets for a user.

    All calls require a bearer token (App-only auth). Pass it via the
    TWITTER_BEARER_TOKEN environment variable or directly to the constructor.
    """

    api_base = "https://api.twitter.com/2"

    def __init__(self, bearer_token: str) -> None:
        self.bearer_token = bearer_token

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.bearer_token}"}

    def get_user_id(self, username: str) -> Optional[str]:
        url = f"{self.api_base}/users/by/username/{username}"
        params = {"user.fields": "id"}
        response = requests.get(url, headers=self._headers(), params=params, timeout=30)
        if not response.ok:
            logger.warning("Failed to lookup %s: %s", username, response.text)
            return None
        data = response.json().get("data")
        return data.get("id") if data else None

    def get_user_following(
        self, user_id: str, *, max_pages: int = 1, page_size: int = 1000
    ) -> List[TwitterUser]:
        url = f"{self.api_base}/users/{user_id}/following"
        params = {
            "user.fields": "description,name,username",
            "max_results": page_size,
        }
        users: List[TwitterUser] = []
        page_count = 0
        pagination_token: Optional[str] = None

        while page_count < max_pages:
            if pagination_token:
                params["pagination_token"] = pagination_token
            response = requests.get(url, headers=self._headers(), params=params, timeout=30)
            if not response.ok:
                logger.warning("Failed to fetch following for %s: %s", user_id, response.text)
                break
            payload = response.json()
            for user in payload.get("data", []):
                users.append(
                    TwitterUser(
                        id=user["id"],
                        username=user["username"],
                        name=user.get("name", ""),
                        description=user.get("description", ""),
                    )
                )

            meta = payload.get("meta", {})
            pagination_token = meta.get("next_token")
            page_count += 1
            if not pagination_token:
                break

        return users

    def get_recent_tweets(
        self,
        user_id: str,
        *,
        max_pages: int = 1,
        page_size: int = 100,
        exclude_retweets: bool = True,
    ) -> List[Tweet]:
        url = f"{self.api_base}/users/{user_id}/tweets"
        params = {
            "tweet.fields": "created_at",
            "max_results": page_size,
        }
        if exclude_retweets:
            params["exclude"] = "retweets"

        tweets: List[Tweet] = []
        page_count = 0
        pagination_token: Optional[str] = None

        while page_count < max_pages:
            if pagination_token:
                params["pagination_token"] = pagination_token
            response = requests.get(url, headers=self._headers(), params=params, timeout=30)
            if not response.ok:
                logger.warning("Failed to fetch tweets for %s: %s", user_id, response.text)
                break

            payload = response.json()
            for item in payload.get("data", []):
                tweets.append(
                    Tweet(
                        id=item["id"],
                        text=item.get("text", ""),
                        created_at=item.get("created_at", ""),
                    )
                )

            meta = payload.get("meta", {})
            pagination_token = meta.get("next_token")
            page_count += 1
            if not pagination_token:
                break

        return tweets


__all__ = ["TwitterClient", "TwitterUser", "Tweet"]
