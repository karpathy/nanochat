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
    r"\bwho\s+(?:r|ru|are)\s+(?:u|you)\b",
    r"\bwhat\s+(?:r|ru|are)\s+(?:u|you)\b",
    r"\bwho\s+(?:r|ru|are)\s+(?:u|you)\s+really\b",
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


# ---------------------------------------------------------------------------
# Context-aware classifier — resolves pronouns (him/her/it/they/this/that)
# against the conversation history so "tell me more about him" after a turn
# about Narendra Modi becomes "tell me more about Narendra Modi 2026".
# ---------------------------------------------------------------------------

# Follow-up phrasings that obviously depend on prior context — these should
# trigger search ONLY if we can resolve the subject from history.
_FOLLOWUP_PATTERNS = re.compile(
    r"""
    ^\s*(?:
        tell\s+me\s+more(?:\s+about\s+(?:him|her|it|them|this|that))?
      | more\s+about\s+(?:him|her|it|them)
      | what\s+(?:else|more)\s+about\s+(?:him|her|it|them)
      | (?:and|what\s+about)\s+(?:him|her|it|them)
      | anything\s+else\s+(?:about\s+(?:him|her|it|them))?
      | what(?:'| i)s\s+(?:his|her|their|its)\s+\w+
      | (?:his|her|their|its)\s+\w+\s*\?*
      | (?:what|how)\s+about\s+(?:his|her|their|its)\s+\w+
    )\s*[?.!]*\s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)

_PRONOUN_RX = re.compile(r"\b(him|her|it|them|this|that|he|she|they|his|hers|their|its)\b", re.IGNORECASE)

# Extract proper-noun phrases (one-or-more Capitalized tokens in a row)
_PROPER_NOUN_RX = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b")

# Words that look like Proper Nouns but are sentence-starters or function words
# (we drop these when extracting entities)
_NON_ENTITY_WORDS = {
    "I", "He", "She", "They", "It", "We", "You", "This", "That", "These", "Those",
    "The", "A", "An", "And", "Or", "But", "So", "As", "If", "When", "Where", "Why",
    "How", "What", "Who", "Which", "Whose", "Whom", "My", "Your", "His", "Her", "Its",
    "Their", "Our", "Is", "Are", "Was", "Were", "Be", "Been", "Being", "Have", "Has",
    "Had", "Do", "Does", "Did", "Will", "Would", "Should", "Could", "Can", "May",
    "Might", "Must", "Shall", "Answer", "Question", "Hello", "Hi", "Hey", "Yes", "No",
    "Okay", "Ok", "Thanks", "Thank", "Please", "Sorry", "Sure", "Maybe", "Perhaps",
    "Of", "In", "On", "At", "For", "With", "From", "To", "Into", "About", "Like",
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
}


def _clean_entity(cand: str) -> str:
    """Strip leading/trailing function words from an entity phrase."""
    toks = cand.split()
    # drop leading/trailing function tokens
    while toks and toks[0] in _NON_ENTITY_WORDS:
        toks = toks[1:]
    while toks and toks[-1] in _NON_ENTITY_WORDS:
        toks = toks[:-1]
    return " ".join(toks)


def _extract_entity_from_text(text: str) -> str:
    """First non-filter capitalized phrase in the text (most likely the subject)."""
    if not text:
        return ""
    for cand in _PROPER_NOUN_RX.findall(text):
        cleaned = _clean_entity(cand)
        if not cleaned:
            continue
        # reject single-token entities that are common filler words
        if " " not in cleaned and cleaned in _NON_ENTITY_WORDS:
            continue
        # reject very short all-caps acronyms like "AI", "I", "US" unless they're >= 3 chars mixed case
        if len(cleaned) < 3:
            continue
        return cleaned
    return ""


def _pick_subject_from_history(messages: list) -> str:
    """Pick the most likely subject from conversation history.

    Strategy: check the most recent USER message (excluding the current one)
    for a proper-noun phrase. This is usually the topic the user named.
    If no user-named entity, fall back to the first entity in the most
    recent ASSISTANT message.
    """
    # walk user messages first, most recent to oldest
    user_msgs = [m.get("content", "") for m in messages if m.get("role") == "user"]
    for c in reversed(user_msgs):
        if not c:
            continue
        entity = _extract_entity_from_text(c)
        if entity:
            return entity
    # fall back to assistant messages (take first entity, which is usually the subject)
    asst_msgs = [m.get("content", "") for m in messages if m.get("role") == "assistant"]
    for c in reversed(asst_msgs):
        entity = _extract_entity_from_text(c)
        if entity:
            return entity
    return ""


def _resolve_pronouns(query: str, entity: str) -> str:
    """If query contains pronouns AND we have a subject entity, replace
    pronouns with that entity. Otherwise return the query unchanged."""
    if not _PRONOUN_RX.search(query):
        return query
    if not entity:
        return query
    def _sub(m: re.Match) -> str:
        tok = m.group(1).lower()
        if tok in ("his", "hers", "their", "its"):
            return f"{entity}'s"
        return entity
    return _PRONOUN_RX.sub(_sub, query)


def needs_web_search_contextual(
    messages: list[dict],
    last_user_override: str | None = None,
) -> Tuple[bool, str]:
    """Context-aware classifier. Takes the full `messages` list (last entry is
    the current user turn), resolves pronouns against prior turns, then runs
    the normal classifier. Returns (needs, rewritten_with_context).

    `last_user_override` lets serve.py pass a pre-cleaned user text (e.g. with
    the system-prompt prefix already stripped).
    """
    if not messages:
        return False, ""
    # find latest user message
    last_user = last_user_override
    if last_user is None:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                last_user = msg.get("content", "")
                break
    if not last_user:
        return False, ""

    # Veto first — identity / meta / greeting etc. never need web search even
    # if they contain pronouns.
    if _is_identity_or_meta(last_user.strip()):
        return False, ""

    # Skip the current turn for subject extraction
    prior = messages[:-1] if messages and messages[-1].get("role") == "user" else messages

    # Also veto if the PRIOR conversation was about identity / the model itself
    # — even a pronoun follow-up shouldn't hit Tavily in that case.
    for m in prior[-3:]:
        if m.get("role") == "user" and _is_identity_or_meta(m.get("content", "").strip()):
            return False, ""

    entity = _pick_subject_from_history(prior[-6:])  # last 6 turns window
    resolved = _resolve_pronouns(last_user, entity)

    # Explicit follow-up phrasing ("tell me more about him"): trigger search
    # on the resolved query only if we actually substituted an entity.
    is_followup = _FOLLOWUP_PATTERNS.search(last_user) is not None
    if is_followup and resolved != last_user:
        return True, _rewrite_query(resolved)

    # Otherwise run the normal classifier on the (possibly resolved) query.
    return needs_web_search(resolved)


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

_BARE_EXPR_RX = re.compile(
    r"(-?\d[\d,\.]*\s*[+\-*/×÷]\s*-?\d[\d,\.]*(?:\s*[+\-*/×÷]\s*-?\d[\d,\.]*)*)"
)
_PERCENT_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:%|percent)\s+(?:of|tip|tax|discount|off)\s+(?:on\s+)?\$?(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_VERBAL_RX = re.compile(
    r"(\d+(?:\.\d+)?)\s+(plus|minus|times|divided\s+by|multiplied\s+by|over)\s+(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_WORD_OP = {
    "plus": "+", "minus": "-", "times": "*",
    "multiplied by": "*", "divided by": "/", "over": "/",
}


def _normalize_expr(expr: str) -> str:
    e = expr.replace(",", "").replace("×", "*").replace("÷", "/")
    e = re.sub(r"\s+", "", e)  # strip all internal whitespace
    return e


def needs_calculator(text: str) -> Tuple[bool, str]:
    """Return (True, expression) if the text contains arithmetic that the
    calculator tool should execute. `expression` is passed as-is to the
    sandboxed evaluator (accepts +-*/ on numbers, plus helpers like
    percent(base,rate), emi(p,r,n), cagr(s,e,y))."""
    if not text:
        return False, ""
    # 1. percentage phrasing
    m = _PERCENT_RX.search(text)
    if m:
        return True, f"percent({m.group(2)},{m.group(1)})"
    # 2. verbal arithmetic
    m = _VERBAL_RX.search(text)
    if m:
        op = _WORD_OP[m.group(2).lower().replace("  ", " ").strip()]
        return True, f"{m.group(1)}{op}{m.group(3)}"
    # 3. bare arithmetic expression
    m = _BARE_EXPR_RX.search(text)
    if m:
        return True, _normalize_expr(m.group(1))
    return False, ""


