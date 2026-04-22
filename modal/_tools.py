"""
Shared tool definitions for nanochat.

The current tokenizer only has python/output special tokens. To preserve
checkpoint compatibility, we reuse those tokens as a generic tool-call and
tool-result channel. Legacy "python" calculator payloads still work.
"""

from __future__ import annotations

import ast
import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import requests

TOOL_CALL_START = "<|python_start|>"
TOOL_CALL_END = "<|python_end|>"
TOOL_RESULT_START = "<|output_start|>"
TOOL_RESULT_END = "<|output_end|>"
MAX_TOOL_PAYLOAD_CHARS = 4096

DEFAULT_TOOL_SCHEMA = [
    {
        "name": "calculator",
        "description": "Deterministic scientific calculator for exact arithmetic and common finance formulas.",
        "arguments": {
            "expression": "String expression using numbers, operators, and supported functions.",
        },
    },
    {
        "name": "web_search",
        "description": "Search and fetch web content. Requires a search backend and optionally a page fetch client.",
        "arguments": {
            "query": "Search query string.",
            "top_k": "Maximum number of results to return.",
            "urls": "Optional explicit URLs to fetch instead of searching.",
        },
    },
]


def _compact_json(data: Any) -> str:
    return json.dumps(data, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


@dataclass
class ToolInvocation:
    tool_name: str
    arguments: dict[str, Any]
    raw_text: str = ""


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_payload(self) -> str:
        return _compact_json(
            {
                "tool": self.tool_name,
                "success": self.success,
                "output": self.output,
                "error": self.error,
                "metadata": self.metadata,
            }
        )


class BaseTool:
    name: str

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        raise NotImplementedError


class ToolRegistry:
    def __init__(self, tools: list[BaseTool] | tuple[BaseTool, ...]):
        self._tools = {tool.name: tool for tool in tools}

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self._tools.get(tool_name)
        if tool is None:
            return ToolResult(tool_name=tool_name, success=False, error=f"Unknown tool: {tool_name}")
        try:
            return tool.run(arguments)
        except Exception as exc:  # defensive: tool failures should become model-visible outputs
            return ToolResult(tool_name=tool_name, success=False, error=str(exc))

    def schema(self) -> list[dict[str, Any]]:
        return [item for item in DEFAULT_TOOL_SCHEMA if item["name"] in self._tools]


def serialize_tool_call(tool_name: str, arguments: dict[str, Any] | None = None) -> str:
    payload = {
        "tool": tool_name,
        "arguments": arguments or {},
    }
    text = _compact_json(payload)
    return text[:MAX_TOOL_PAYLOAD_CHARS]


def serialize_tool_result(
    tool_name: str,
    output: Any = None,
    *,
    success: bool = True,
    error: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    return ToolResult(
        tool_name=tool_name,
        success=success,
        output=output,
        error=error,
        metadata=metadata or {},
    ).to_payload()[:MAX_TOOL_PAYLOAD_CHARS]


def parse_tool_call_payload(text: str) -> ToolInvocation:
    stripped = text.strip()
    if not stripped:
        return ToolInvocation(tool_name="calculator", arguments={"expression": ""}, raw_text=text)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return ToolInvocation(tool_name="calculator", arguments={"expression": stripped}, raw_text=text)
    if isinstance(payload, dict):
        tool_name = payload.get("tool") or payload.get("tool_name") or payload.get("name")
        arguments = payload.get("arguments") or payload.get("args") or {}
        if isinstance(tool_name, str) and isinstance(arguments, dict):
            return ToolInvocation(tool_name=tool_name, arguments=arguments, raw_text=text)
    return ToolInvocation(tool_name="calculator", arguments={"expression": stripped}, raw_text=text)


def parse_tool_result_payload(text: str) -> ToolResult | None:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    tool_name = payload.get("tool")
    if not isinstance(tool_name, str):
        return None
    return ToolResult(
        tool_name=tool_name,
        success=bool(payload.get("success", True)),
        output=payload.get("output"),
        error=payload.get("error"),
        metadata=payload.get("metadata") or {},
    )


def _percent(value: float, rate: float) -> float:
    return value * rate / 100.0


def _percent_change(old: float, new: float) -> float:
    if old == 0:
        raise ValueError("percent_change old value cannot be zero")
    return ((new - old) / old) * 100.0


def _cagr(start: float, end: float, years: float) -> float:
    if start <= 0 or end <= 0 or years <= 0:
        raise ValueError("cagr inputs must be positive")
    return ((end / start) ** (1.0 / years) - 1.0) * 100.0


def _simple_interest(principal: float, annual_rate: float, years: float) -> float:
    return principal * annual_rate / 100.0 * years


def _compound_interest(principal: float, annual_rate: float, periods_per_year: float, years: float) -> float:
    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")
    return principal * (1.0 + annual_rate / 100.0 / periods_per_year) ** (periods_per_year * years)


def _emi(principal: float, annual_rate: float, months: float) -> float:
    if months <= 0:
        raise ValueError("months must be positive")
    monthly_rate = annual_rate / 100.0 / 12.0
    if monthly_rate == 0:
        return principal / months
    growth = (1.0 + monthly_rate) ** months
    return principal * monthly_rate * growth / (growth - 1.0)


ALLOWED_BINOPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.Pow: lambda a, b: a ** b,
    ast.Mod: lambda a, b: a % b,
}
ALLOWED_UNARYOPS = {
    ast.UAdd: lambda a: a,
    ast.USub: lambda a: -a,
}
ALLOWED_NAMES = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}
ALLOWED_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "degrees": math.degrees,
    "radians": math.radians,
    "percent": _percent,
    "percent_change": _percent_change,
    "cagr": _cagr,
    "simple_interest": _simple_interest,
    "compound_interest": _compound_interest,
    "emi": _emi,
}


class _SafeMathEvaluator:
    def __init__(self, expression: str):
        self.expression = expression
        self.node_count = 0

    def eval(self) -> float:
        if len(self.expression) > 512:
            raise ValueError("expression too long")
        parsed = ast.parse(self.expression, mode="eval")
        return self._visit(parsed.body)

    def _visit(self, node: ast.AST) -> Any:
        self.node_count += 1
        if self.node_count > 128:
            raise ValueError("expression too complex")

        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"unsupported constant: {node.value!r}")
        if isinstance(node, ast.Num):  # pragma: no cover - py<3.8 compatibility
            return node.n
        if isinstance(node, ast.BinOp):
            op = ALLOWED_BINOPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported operator: {type(node.op).__name__}")
            return op(self._visit(node.left), self._visit(node.right))
        if isinstance(node, ast.UnaryOp):
            op = ALLOWED_UNARYOPS.get(type(node.op))
            if op is None:
                raise ValueError(f"unsupported unary operator: {type(node.op).__name__}")
            return op(self._visit(node.operand))
        if isinstance(node, ast.Name):
            if node.id not in ALLOWED_NAMES:
                raise ValueError(f"unknown symbol: {node.id}")
            return ALLOWED_NAMES[node.id]
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("only direct function calls are allowed")
            fn = ALLOWED_FUNCTIONS.get(node.func.id)
            if fn is None:
                raise ValueError(f"unsupported function: {node.func.id}")
            if node.keywords:
                raise ValueError("keyword arguments are not supported")
            args = [self._visit(arg) for arg in node.args]
            return fn(*args)
        raise ValueError(f"unsupported expression node: {type(node).__name__}")


def _normalize_numeric_output(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("result is not finite")
        return float(f"{value:.12g}")
    return value


class CalculatorTool(BaseTool):
    name = "calculator"

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        expression = str(arguments.get("expression", "")).strip()
        if not expression:
            return ToolResult(tool_name=self.name, success=False, error="Missing expression")
        value = _SafeMathEvaluator(expression).eval()
        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"expression": expression, "value": _normalize_numeric_output(value)},
        )


@dataclass
class SearchHit:
    url: str
    title: str = ""
    snippet: str = ""


class SearchBackend(Protocol):
    def search(self, query: str, top_k: int) -> list[SearchHit]:
        raise NotImplementedError


class MockSearchBackend:
    def __init__(self, canned_results: dict[str, list[dict[str, str]]] | None = None):
        self.canned_results = canned_results or {
            "browser rendering markdown": [
                {
                    "url": "https://developers.cloudflare.com/browser-rendering/rest-api/markdown-endpoint/",
                    "title": "Cloudflare markdown endpoint",
                    "snippet": "Extract markdown from a webpage using Cloudflare Browser Rendering.",
                }
            ],
            "nanochat gpt2 speedrun": [
                {
                    "url": "https://github.com/karpathy/nanochat",
                    "title": "karpathy/nanochat",
                    "snippet": "Minimal LLM training harness with pretraining, SFT, RL, and chat UI.",
                }
            ],
        }

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        normalized = query.strip().lower()
        rows = self.canned_results.get(normalized, [])
        return [SearchHit(**row) for row in rows[:top_k]]

class TavilySearchBackend:
    """LLM-optimized web search via Tavily. Falls back silently on errors."""
    def __init__(self, api_key: str | None = None, timeout: float = 15.0):
        self.api_key = api_key or os.environ.get('TAVILY_API_KEY')
        if not self.api_key:
            raise ValueError('TavilySearchBackend requires TAVILY_API_KEY')
        self.timeout = timeout

    def search(self, query: str, top_k: int) -> list[SearchHit]:
        import requests
        try:
            r = requests.post(
                'https://api.tavily.com/search',
                json={
                    'api_key': self.api_key,
                    'query': query,
                    'max_results': max(1, min(int(top_k), 8)),
                    'include_answer': False,
                    'include_raw_content': False,
                    'search_depth': 'basic',
                },
                timeout=self.timeout,
            )
            r.raise_for_status()
            data = r.json()
        except Exception:
            return []
        return [
            SearchHit(
                url=h.get('url', ''),
                title=h.get('title', ''),
                snippet=h.get('content', ''),
            )
            for h in data.get('results', [])[:top_k]
        ]



class CloudflareBrowserRenderingClient:
    def __init__(
        self,
        *,
        api_token: str | None = None,
        account_id: str | None = None,
        base_url: str = "https://api.cloudflare.com/client/v4",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.api_token = api_token or os.environ.get("CLOUDFLARE_API_TOKEN")
        self.account_id = account_id or os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        if not self.api_token or not self.account_id:
            raise ValueError("Cloudflare Browser Rendering requires CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            }
        )

    def _post(self, endpoint: str, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}/accounts/{self.account_id}/browser-rendering/{endpoint}"
        last_error = None
        for attempt in range(1, self.max_retries + 1):
            response = self.session.post(url, json=body, timeout=self.timeout)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                sleep_seconds = float(retry_after) if retry_after else float(attempt)
                last_error = RuntimeError(f"Cloudflare Browser Rendering rate limited on {endpoint}")
                time.sleep(min(sleep_seconds, 5.0))
                continue
            response.raise_for_status()
            payload = response.json()
            if not payload.get("success", False):
                errors = payload.get("errors", [])
                last_error = RuntimeError(f"Cloudflare Browser Rendering request failed: {errors}")
                break
            return payload.get("result")
        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Cloudflare Browser Rendering request failed for {endpoint}")

    def markdown(self, url: str, **options: Any) -> str:
        body = {"url": url}
        body.update(options)
        return self._post("markdown", body)

    def links(self, url: str, **options: Any) -> list[str]:
        body = {"url": url}
        body.update(options)
        return self._post("links", body)

    def json_extract(self, url: str, *, prompt: str | None = None, schema: dict[str, Any] | None = None, **options: Any) -> dict[str, Any]:
        body: dict[str, Any] = {"url": url}
        if prompt is not None:
            body["prompt"] = prompt
        if schema is not None:
            body["schema"] = schema
        body.update(options)
        return self._post("json", body)


class WebSearchTool(BaseTool):
    name = "web_search"

    def __init__(
        self,
        *,
        search_backend: SearchBackend | None = None,
        fetch_client: CloudflareBrowserRenderingClient | None = None,
        max_results: int = 3,
    ):
        self.search_backend = search_backend
        self.fetch_client = fetch_client
        self.max_results = max_results

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query", "")).strip()
        requested_urls = arguments.get("urls") or []
        if isinstance(requested_urls, str):
            requested_urls = [requested_urls]
        top_k = int(arguments.get("top_k", self.max_results) or self.max_results)
        top_k = max(1, min(top_k, 8))

        if not query and not requested_urls:
            return ToolResult(tool_name=self.name, success=False, error="Missing query or urls")

        hits: list[SearchHit]
        if requested_urls:
            hits = [SearchHit(url=str(url)) for url in requested_urls[:top_k]]
        else:
            if self.search_backend is None:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="No search backend configured. Cloudflare Browser Rendering can fetch pages but does not provide public web search by itself.",
                )
            hits = self.search_backend.search(query, top_k)

        results = []
        for hit in hits[:top_k]:
            entry: dict[str, Any] = {"url": hit.url}
            if hit.title:
                entry["title"] = hit.title
            if hit.snippet:
                entry["snippet"] = hit.snippet
            if self.fetch_client is not None:
                try:
                    markdown = self.fetch_client.markdown(hit.url)
                    links = self.fetch_client.links(hit.url)
                    entry["markdown"] = markdown[:4000]
                    entry["links"] = links[:10]
                except Exception as exc:
                    entry["fetch_error"] = str(exc)
            results.append(entry)

        return ToolResult(
            tool_name=self.name,
            success=True,
            output={"query": query, "results": results},
            metadata={
                "search_backend": type(self.search_backend).__name__ if self.search_backend is not None else None,
                "fetch_backend": type(self.fetch_client).__name__ if self.fetch_client is not None else None,
                "num_results": len(results),
            },
        )


def build_default_tool_registry(
    *,
    cloudflare_token: str | None = None,
    cloudflare_account_id: str | None = None,
    search_backend: SearchBackend | None = None,
) -> ToolRegistry:
    fetch_client = None
    if cloudflare_token or os.environ.get("CLOUDFLARE_API_TOKEN"):
        try:
            fetch_client = CloudflareBrowserRenderingClient(
                api_token=cloudflare_token,
                account_id=cloudflare_account_id,
            )
        except Exception:
            fetch_client = None
    if search_backend is None:
        if os.environ.get('TAVILY_API_KEY'):
            try:
                search_backend = TavilySearchBackend()
            except Exception:
                search_backend = MockSearchBackend()
        else:
            search_backend = MockSearchBackend()
    registry = ToolRegistry(
        [
            CalculatorTool(),
            WebSearchTool(
                search_backend=search_backend,
                fetch_client=fetch_client,
            ),
        ]
    )
    return registry
