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

# ---------------------------------------------------------------------------
# Negative veto — queries about the model itself or its creator should NOT
# trigger a web search. The model's SFT training has the correct grounded
# identity answers (samosaChaat, Manmohan Sharma, socials, etc.). Sending
# these to Tavily returns irrelevant results (Tyler the Creator, Waaree CFO, etc.)
# ---------------------------------------------------------------------------
_IDENTITY_VETO_PATTERNS = [
    # self-referential questions directed at the model
    r"\bwho\s+are\s+you\b",
    r"\bwhat\s+are\s+you\b",
    r"\bwho\s+are\s+you\s+really\b",
    r"\bwhat(?:'|\s+i)s\s+your\s+name\b",
    r"\bintroduce\s+yourself\b",
    r"\btell\s+me\s+about\s+(?:yourself|you|you\s+first)\b",
    r"\btell\s+me\s+(?:more\s+)?about\s+yourself\b",
    r"\babout\s+you(?:rself)?\s*[?!.]*\s*$",
    r"\bdescribe\s+yourself\b",
    r"\bwhat(?:'|\s+i)s\s+your\s+(?:story|backstory|purpose|origin|deal|role|job|gig|mission|goal|objective)\b",
    r"\bwhat\s+do\s+you\s+(?:do|know\s+about\s+yourself|stand\s+for)\b",
    # creator / maker / trainer questions
    r"\bwho\s+(?:is\s+)?(?:made|created|built|trained|developed|designed|coded|programmed|engineered|authored|invented|produced|fine[-\s]?tuned|wrote)\s+you\b",
    r"\bwho(?:'|\s+i)s\s+(?:your|ur)\s+(?:creator|creater|maker|author|developer|designer|engineer|architect|founder|builder|programmer|trainer|daddy|mom|parent|boss|owner|father|mother|papa|mama)\b",
    r"\bwho\s+(?:brought|gave)\s+you\s+(?:to\s+life|into\s+being|into\s+existence)\b",
    r"\bwhere\s+(?:did\s+)?you\s+come\s+from\b",
    r"\bhow\s+(?:did|were)\s+you\s+(?:come|born|made|created|built)\b",
    # competitor/provenance questions (identity confusion attacks)
    r"\bare\s+you\s+(?:chatgpt|gpt[-\s]?\d|claude|gemini|bard|llama|mistral|perplexity|copilot|sonnet|opus|haiku|grok)\b",
    r"\bare\s+you\s+(?:made|created|built|owned|developed|trained)\s+by\s+(?:openai|anthropic|google|meta|microsoft|deepmind|x\.ai|xai)\b",
    r"\bwhich\s+(?:company|organization|team|ai|model)\s+(?:made|created|built|trained|owns|are\s+you)\s*(?:you)?\b",
    r"\bare\s+you\s+(?:an?\s+)?(?:ai|chat\s*bot|assistant|language\s+model|llm|robot)\b",
    # samosaChaat / creator name references
    r"\bsamosachaat\b",
    r"\bwho(?:'|\s+i)s\s+manmohan(?:\s+sharma)?\b",
    r"\bwho\s+is\s+manmohan\b",
    r"\bmanmohan\s+sharma\b",
    r"\btell\s+me\s+about\s+manmohan\b",
    r"\bwhat(?:'|\s+i)s\s+(?:manmohan|your\s+creator)(?:'s)?\s+(?:github|linkedin|twitter|x|website|email|socials?)\b",
    # model meta-questions
    r"\bhow\s+(?:many|much)\s+parameters?\b",
    r"\bwhat\s+(?:model|version|size|architecture|type|kind)\s+(?:are\s+you|of\s+(?:ai|model)\s+are\s+you)\b",
    r"\bwhat(?:'|\s+i)s\s+your\s+(?:model|version|size|architecture|context\s+(?:size|length|window))\b",
    r"\bare\s+you\s+(?:open[-\s]?source|open\s+weight|closed\s+source|proprietary)\b",
    r"\bwhere\s+(?:can\s+i\s+)?(?:find|download|get)\s+your\s+(?:weights|code|source|github|repo)\b",
    r"\bwhat\s+hardware\s+(?:were|are)\s+you\s+(?:trained|running)\b",
    r"\bhow\s+(?:were|are)\s+you\s+trained\b",
    r"\bwhen\s+were\s+you\s+(?:trained|released|built|made|created)\b",
    r"\bwhat(?:'|\s+i)s\s+your\s+(?:training|knowledge)\s+(?:data|cut[-\s]?off|cutoff)\b",
    r"\bwhat\s+data\s+(?:were|are)\s+you\s+trained\s+on\b",
    # capability / tooling questions (not factual queries)
    r"\bwhat\s+(?:tools|abilities|capabilities|languages|skills|features)\s+(?:do\s+)?you\s+(?:have|support|speak|offer|provide)\b",
    r"\bwhat\s+(?:can|could)\s+you\s+(?:do|help\s+(?:me\s+)?with)\b",
    r"\bcan\s+you\s+(?:search|do|use|access|help)\b",
    r"\bwhat\s+are\s+you\s+(?:good\s+at|capable\s+of|able\s+to\s+do)\b",
    r"\bhow\s+do\s+you\s+work\b",
    # greetings & social
    r"^(?:hi|hello|hey|yo|sup|greetings|namaste|good\s+(?:morning|afternoon|evening|night))\b",
    r"\bhow\s+are\s+you\b",
    r"\bwhat(?:'|\s+i)s\s+up\b",
    r"\bnice\s+to\s+meet\s+you\b",
    # general small talk / thanks
    r"^\s*(?:thanks?|thank\s+you|thx|ty|ok|okay|cool|nice|great|awesome|bye|goodbye)\s*[!.?]*\s*$",
    # writing / reasoning / coding tasks (answered by the model, not the web)
    r"\bwrite\s+(?:a|an|me)\s+(?:poem|haiku|limerick|story|essay|letter|email|code|function|script|query|sql|song|joke)\b",
    r"\bexplain\s+(?:what|how|why)\s+(?:is\s+)?(?:recursion|gradient\s+descent|backprop|attention|a\s+transformer|machine\s+learning|neural\s+network|rope|softmax)\b",
    r"\bsolve\b.*=",  # math equations
]
_IDENTITY_VETO_REGEXES = [re.compile(p, re.IGNORECASE) for p in _IDENTITY_VETO_PATTERNS]


def _is_identity_or_meta(text: str) -> bool:
    for rx in _IDENTITY_VETO_REGEXES:
        if rx.search(text):
            return True
    return False


def needs_web_search(text: str) -> Tuple[bool, str]:
    """Classify whether a user query likely needs a live web search.

    Returns (needs, rewritten_query). The rewritten_query strips filler and
    reformulates for better Tavily results (e.g. "whos the present president" ->
    "who is the current president of the United States 2026").

    Identity / meta / greeting / writing-task queries are vetoed — the model's
    SFT training has the correct grounded answer.
    """
    if not text or not isinstance(text, str):
        return False, ""

    stripped = text.strip()
    if len(stripped) < 3:
        return False, ""

    # Veto: identity / self-referential / meta / greeting / writing tasks
    if _is_identity_or_meta(stripped):
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
