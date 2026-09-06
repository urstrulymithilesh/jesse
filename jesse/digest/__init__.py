"""Proactive daily learning — he reads pre-set sources on a schedule.

Split like everything else here: `feeds` fetches and parses (pure I/O + text), `store`
keeps what came back, `parse` decides deterministically when he is asking about it.
Nothing in this package touches the LLM: feed items are stored as the source wrote
them, so there is no step at which he can invent an article.
"""

from jesse.digest.feeds import FeedError, Item, fetch_feed, strip_html
from jesse.digest.parse import asks_whats_new
from jesse.digest.store import DigestStore

__all__ = ["FeedError", "Item", "fetch_feed", "strip_html", "asks_whats_new",
           "DigestStore"]
