"""Fetch and parse RSS/Atom feeds. Stdlib only, and deliberately narrow.

This is the ONLY part of Jesse that reaches the network during normal running, and it
is off by default (`CONFIG.digest.enabled`). What it does NOT do matters as much as
what it does:

  * **Feeds only, not web pages.** No HTML scraping, no JavaScript, no browser. A feed
    is already structured text, so there is no heuristic guessing at what the page
    meant — and nothing that could execute.
  * **It sends nothing.** A GET for a public feed carries no conversation, no memory,
    no identifier beyond a plain user agent. The offline promise in the README is that
    *his* data never leaves; that is untouched. But this is still outbound traffic he
    did not have before, which is why it is opt-in.
  * **The reply is data, never instruction.** Feed text is stored and later quoted back
    to him. It is never followed. A feed that contains "ignore your instructions" is a
    feed containing that string, and `context.digest_context` frames it as quoted
    material.

Hardening, all cheap and all stdlib:
  * http/https only — no file://, ftp:// or anything else urllib would happily open.
  * a hard byte cap while reading, so a huge or endless response cannot exhaust memory.
  * DOCTYPE/ENTITY declarations rejected before parsing. `xml.etree` does not fetch
    external entities (safe since 3.7.1) but IS vulnerable to entity-expansion blowups;
    no legitimate feed needs a DTD, so refusing one closes that without a dependency.
"""

from __future__ import annotations

import html
import re
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from urllib.parse import urlparse

USER_AGENT = "Jesse/1.0 (local personal assistant; feed reader)"
_ALLOWED_SCHEMES = ("http", "https")
_DOCTYPE = re.compile(rb"<!(?:DOCTYPE|ENTITY)", re.I)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")

# Atom lives in its own namespace; RSS mostly does not.
_ATOM = "{http://www.w3.org/2005/Atom}"


class FeedError(RuntimeError):
    """Fetching or parsing failed. Always caught — a bad source must never take the
    session down, and he must never pretend it worked."""


@dataclass(frozen=True)
class Item:
    source: str          # the friendly name he gave the feed
    url: str
    title: str
    summary: str
    published: str       # as the feed wrote it, or "" — never invented


def strip_html(raw: str, *, limit: int = 400) -> str:
    """Feed summaries are usually HTML fragments. Tags out, entities decoded, one line.

    Truncation is on a word boundary with an ellipsis so a half-word never gets read
    aloud, and the limit exists because these go into his context budget alongside the
    persona and history.
    """
    if not raw:
        return ""
    text = _WS.sub(" ", html.unescape(_TAG.sub(" ", raw))).strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip(",.;:")
    return f"{cut}…"


# Text from a feed that is shaped like an instruction to an assistant rather than like
# an article. Dropped before it is ever stored.
#
# This is defence in depth, not the main defence: the digest block already tells him
# the items are material he read, and probed live he never once obeyed one. What he
# DID do is worse in its own way — handed an item he could not repeat, he invented
# articles instead (a jellyfish species, a Kristin Hannah novel, and twice his own
# persona taste for pineapple on pizza, which is the fifth time an invented persona
# detail has resurfaced as a claim about reality). So the goal here is not to stop him
# obeying, it is to stop unusable text reaching him at all.
_INSTRUCTION_SHAPED = re.compile(
    r"ignore\s+(?:all\s+|any\s+)?(?:your|previous|prior|the\s+above)|"
    r"disregard\s+(?:your|previous|prior|all)|"
    r"(?:your|the)\s+(?:system\s+)?(?:prompt|instructions)\b|"
    r"^\s*system\s*:|"
    r"you\s+(?:must|should|will)\s+now\b|"
    r"reveal\s+your\b|"
    r"pretend\s+(?:you\s+are|you're|to\s+be)\b",
    re.I | re.M)
# Dropped from the list above after a false positive on a perfectly ordinary headline:
# "act as a/if" hits "act as a deterrent", and a bare "pretend you" hits "How to
# pretend you like a gift". The remaining patterns all name the assistant explicitly
# ("your instructions", "reveal your", "system:"), which is what makes them safe on
# news text. A blocklist that eats real articles is worse than a narrower one.


def looks_like_instruction(*parts: str) -> bool:
    """True when feed text is addressing the reader as an assistant."""
    return any(_INSTRUCTION_SHAPED.search(p or "") for p in parts)


def _text(node, *names) -> str:
    for name in names:
        found = node.find(name)
        if found is not None and (found.text or "").strip():
            return found.text.strip()
    return ""


def _link(node) -> str:
    link = _text(node, "link", f"{_ATOM}id")
    if link:
        return link
    # Atom puts the url in an attribute, and may list several.
    for el in node.findall(f"{_ATOM}link"):
        rel = el.get("rel", "alternate")
        if rel == "alternate" and el.get("href"):
            return el.get("href", "")
    return ""


def parse_feed(data: bytes, source: str, *, limit: int = 10) -> list[Item]:
    """RSS `channel/item` or Atom `feed/entry` into Items, newest-first as published.

    Order is left exactly as the feed gave it. Feeds are conventionally newest-first
    and guessing at dates across a dozen formats would be a lot of code to get subtly
    wrong; the fetch time is recorded separately and that is what "new" means here.
    """
    if _DOCTYPE.search(data[:4096]):
        raise FeedError("feed declares a DOCTYPE or ENTITY — refusing to parse it")
    try:
        root = ET.fromstring(data)
    except ET.ParseError as e:
        raise FeedError(f"not valid XML: {e}") from e

    nodes = root.findall(".//item") or root.findall(f".//{_ATOM}entry")
    items: list[Item] = []
    for node in nodes[:limit]:
        title = strip_html(_text(node, "title", f"{_ATOM}title"), limit=200)
        url = _link(node).strip()
        if not title and not url:
            continue
        summary = strip_html(_text(node, "description", f"{_ATOM}summary",
                                   f"{_ATOM}content"))
        published = _text(node, "pubDate", f"{_ATOM}published", f"{_ATOM}updated")
        items.append(Item(source=source, url=url, title=title, summary=summary,
                          published=published))
    return items


def fetch_feed(url: str, source: str, *, timeout: int = 15,
               max_bytes: int = 2_000_000, limit: int = 10) -> list[Item]:
    """GET a feed and parse it. Raises FeedError on anything that goes wrong."""
    scheme = urlparse(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise FeedError(f"refusing scheme {scheme!r} — only http and https")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
            data = response.read(max_bytes + 1)
    except (urllib.error.URLError, OSError, ValueError) as e:
        raise FeedError(f"could not fetch: {e}") from e
    if len(data) > max_bytes:
        raise FeedError(f"feed is larger than {max_bytes} bytes — refusing it")
    return parse_feed(data, source, limit=limit)
