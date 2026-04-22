"""Heuristic query classifier for forced tool use.

This module answers a single question: given a user message, does it likely
need a fresh web search to be answered correctly?

We use this to *force* the model to call `web_search` even when its natural
generation path would skip the tool (e.g. "whos the present president" where
"present" doesn't appear in the tool-use SFT training distribution).

Two outputs:
    needs_web_search(text) -> (bool, rewritten_query)
    needs_calculator(text) -> (bool, expression)

Keep the rules simple and fast. False positives are cheap (we pay a Tavily
call); false negatives are the real cost (user gets stale training-data).
"""
from __future__ import annotations

import re
from typing import Tuple

# ---------------------------------------------------------------------------
# Web-search triggers
# ---------------------------------------------------------------------------

# Time words that almost always indicate the user wants fresh info
_TIME_WORDS = r"""
(?:
  current | currently | now | today | tonight | tomorrow | yesterday |
  present | this\ (?:week|month|year|morning|afternoon|evening) |
  latest | recent | recently | upcoming | right\ now | as\ of |
  20\d{2} | this\ very\ (?:moment|second|minute)
)
"""

# Role / position words that change over time
_POSITION_WORDS = r"""
(?:
  president | vice[-\s]?president | prime\ minister | chancellor | governor |
  senator | congressman | congresswoman | representative | mayor |
  ceo | cto | cfo | coo | chairman | chief\s+executive |
  chief\s+justice | speaker | foreign\s+minister | attorney\s+general |
  pope | monarch | king | queen | emperor | dictator | leader
)
"""

# Topic categories that are inherently time-sensitive
_CATEGORY_PATTERNS = [
    # weather / climate
    r"\b(?:weather|temperature|forecast|humidity|rainfall|snowfall|wind\s+speed)\b",
    # finance / markets
    r"\b(?:stock\s+price|share\s+price|market\s+cap|exchange\s+rate|forex|crypto(?:currency)?\s+price)\b",
    r"\b(?:nasdaq|s&p\s*500|dow\s+jones|nifty|sensex|ftse|nikkei)\b",
    r"\bprice\s+of\s+(?:gold|silver|oil|bitcoin|ethereum)\b",
    # news / events
    r"\b(?:breaking\s+news|headlines?|latest\s+news|news\s+(?:today|now))\b",
    r"\bwhat(?:'|\s+i)s\s+happening\s+(?:in|with|at)\b",
    # sports scores
    r"\b(?:score|result|winner|loser|champion|finalist)\b.*(?:match|game|tournament|series|cup|open|championship)",
    r"\b(?:ipl|world\s+cup|super\s+bowl|nba|nfl|nhl|olympics?|wimbledon|us\s+open|australian\s+open|french\s+open)\b",
    # time of day / where
    r"\bwhat\s+time\s+is\s+it\b",
    r"\btime\s+in\s+[A-Z][a-z]+\b",
    # people (often changes) — VIPs, CEOs, recent releases
    r"\bwho\s+(?:is|'s|runs|leads|owns|founded|heads)\b",
    # "is X still alive" / "did X die"
    r"\bis\s+\w+\s+(?:still\s+)?(?:alive|dead)\b",
    # recent product / release
    r"\b(?:latest|newest|recent|upcoming)\s+(?:version|release|model|iphone|android|macbook|tesla|game|album|movie|film|book)\b",
    # recent statistics that change
    r"\b(?:population|number\s+of)\s+\w+\s+(?:of|in)\b",
    # explicit search instructions
    r"\b(?:search|look\s+up|google|find\s+out|check)\s+(?:online|on\s+the\s+web|on\s+google)\b",
    r"\buse\s+(?:the\s+)?web[_\s]?search\b",
]

# Standalone keyword patterns that, combined with any position/role, trigger search
_KEYWORD_GATE = re.compile(
    rf"""
    \b{_TIME_WORDS}\b          # at least one time word
  | \b(?:who|what|where|when)\b.*\b{_POSITION_WORDS}\b   # who/what + position
  | \b{_POSITION_WORDS}\b.*\b(?:of|in|for|at)\s+[A-Z]    # position + proper noun (of America, in India)
    """,
    re.IGNORECASE | re.VERBOSE,
)

_CATEGORY_REGEXES = [re.compile(p, re.IGNORECASE) for p in _CATEGORY_PATTERNS]


def needs_web_search(text: str) -> Tuple[bool, str]:
    """Classify whether a user query likely needs a live web search.

    Returns (needs, rewritten_query). The rewritten_query strips filler and
    reformulates for better Tavily results (e.g. "whos the present president" ->
    "who is the current president of the United States 2026").
    """
    if not text or not isinstance(text, str):
        return False, ""

    stripped = text.strip()
    if len(stripped) < 3:
        return False, ""

    # Any category pattern hit
    for rx in _CATEGORY_REGEXES:
        if rx.search(stripped):
            return True, _rewrite_query(stripped)

    # Keyword gate (time words or position + specifier)
    if _KEYWORD_GATE.search(stripped):
        return True, _rewrite_query(stripped)

    return False, ""


def _rewrite_query(text: str) -> str:
    """Clean up the query for Tavily — expand contractions, normalize 'present'->'current',
    strip filler, add a year anchor."""
    q = text.strip().rstrip("?.!")
    # contractions
    q = re.sub(r"\bwho(?:'| i)s\b", "who is", q, flags=re.IGNORECASE)
    q = re.sub(r"\bwhat(?:'| i)s\b", "what is", q, flags=re.IGNORECASE)
    q = re.sub(r"\bwhere(?:'| i)s\b", "where is", q, flags=re.IGNORECASE)
    q = re.sub(r"\bwhen(?:'| i)s\b", "when is", q, flags=re.IGNORECASE)
    q = re.sub(r"\bits\b", "it is", q, flags=re.IGNORECASE)
    # strip quantifiers the model tends to hallucinate
    q = re.sub(r"\b(please|kindly|could\s+you|can\s+you)\b\s*", "", q, flags=re.IGNORECASE)
    # "present X" -> "current X" (better Tavily results)
    q = re.sub(r"\bpresent\b", "current", q, flags=re.IGNORECASE)
    # collapse whitespace
    q = re.sub(r"\s+", " ", q).strip()
    # anchor to the current year if no year is already present
    if not re.search(r"\b20\d{2}\b", q):
        q = q + " 2026"
    return q


# ---------------------------------------------------------------------------
# Calculator triggers (cheap, local)
# ---------------------------------------------------------------------------

_CALC_RX = re.compile(
    r"""
    \b(?:calculate|compute|what\s+is|what's)\b.*?
    (?:
      \d[\d,\.\s]*\s*[+\-\*/x×÷]\s*\d               # basic arithmetic
    | \d+\s*%\s+(?:of|tip|tax|discount)\s+\d        # percentage
    | \b(?:emi|cagr|compound\s+interest|tip|discount|percent(?:age)?)\b.*\d
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)


def needs_calculator(text: str) -> Tuple[bool, str]:
    if not text:
        return False, ""
    m = _CALC_RX.search(text)
    if not m:
        return False, ""
    return True, text.strip()
