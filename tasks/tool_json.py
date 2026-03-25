"""
Local JSONL task for tool-use evaluation and lightweight RL reward shaping.

Each line should be a JSON object with:
{
  "conversation": {"messages": [...]},
  "checks": {
    "must_call": "calculator",
    "must_not_call": ["web_search"],
    "answer_contains": ["42"],
    "citation_required": false
  }
}
"""

import json
import os
import re

from nanochat.tools import TOOL_CALL_END, TOOL_CALL_START, parse_tool_call_payload
from tasks.common import Task


TOOL_BLOCK_RE = re.compile(re.escape(TOOL_CALL_START) + r"(.*?)" + re.escape(TOOL_CALL_END), re.DOTALL)


class ToolJSON(Task):
    def __init__(self, filepath, split="eval", **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.split = split
        self.rows = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Tool JSONL dataset not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if "conversation" not in row:
                    raise ValueError(f"Row missing conversation field: {row}")
                row.setdefault("checks", {})
                self.rows.append(row)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.rows)

    def get_example(self, index):
        row = self.rows[index]
        conversation = dict(row["conversation"])
        conversation["checks"] = row.get("checks", {})
        return conversation

    def _tool_calls(self, assistant_response):
        calls = []
        for payload in TOOL_BLOCK_RE.findall(assistant_response):
            invocation = parse_tool_call_payload(payload)
            calls.append(invocation.tool_name)
        return calls

    def evaluate(self, conversation, assistant_response):
        checks = conversation.get("checks", {})
        score = self.reward(conversation, assistant_response)
        return int(score >= 0.999)

    def reward(self, conversation, assistant_response):
        checks = conversation.get("checks", {})
        total = 0.0
        passed = 0.0
        tool_calls = self._tool_calls(assistant_response)

        must_call = checks.get("must_call")
        if must_call:
            total += 1.0
            passed += float(must_call in tool_calls)

        for tool_name in checks.get("must_not_call", []):
            total += 1.0
            passed += float(tool_name not in tool_calls)

        for needle in checks.get("answer_contains", []):
            total += 1.0
            passed += float(needle in assistant_response)

        answer_regex = checks.get("answer_regex")
        if answer_regex:
            total += 1.0
            passed += float(re.search(answer_regex, assistant_response) is not None)

        if checks.get("citation_required", False):
            total += 1.0
            passed += float(("http://" in assistant_response) or ("https://" in assistant_response))

        if total == 0.0:
            return 0.0
        return passed / total
