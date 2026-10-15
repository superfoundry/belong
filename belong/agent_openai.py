"""Personal agent powered by OpenAI (used by the two-person demo).

Same job, same instructions and the same draft checks as the Claude agent
(agent_llm.py): the model talks with the person and fills in the listing through the
`update_listing` tool, and every tool call goes through `build_draft` before it touches
the draft. Only the API plumbing differs.

Calls the OpenAI Chat Completions API directly with the standard library, so no extra
package is needed. Configuration (environment variables, or a .env file in the project
folder, which the server loads):

  OPENAI_API_KEY   required
  OPENAI_MODEL     optional, defaults to DEFAULT_MODEL
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

from .agent_llm import (MAX_TOOL_ROUNDS, SYSTEM_PROMPT, UPDATE_LISTING_TOOL, AgentUnavailable, _summary,
                        build_draft, honest)

API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-5"

OPENAI_TOOL = {
    "type": "function",
    "function": {
        "name": UPDATE_LISTING_TOOL["name"],
        "description": UPDATE_LISTING_TOOL["description"],
        "parameters": UPDATE_LISTING_TOOL["input_schema"],
        "strict": True,
    },
}


def _http_post(url: str, payload: dict, headers: dict, timeout: float = 90.0) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


class OpenAIAgent:
    """One instance serves every phone; each phone passes in its own conversation."""

    def __init__(self, api_key: str | None = None, model: str | None = None, post=_http_post):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise AgentUnavailable("OPENAI_API_KEY isn't set")
        self.model = model or os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL
        self._post = post

    def respond(self, conversation: list, user_text: str, first_name: str, age_band: str) -> tuple:
        """Append the person's message, run the agent, and return (reply, draft or None).
        `conversation` is this person's message history; it is only ever appended to,
        except that a turn which fails part-way is removed again."""
        start = len(conversation)
        conversation.append({"role": "user", "content": user_text})
        try:
            return self._run(conversation, first_name, age_band)
        except AgentUnavailable:
            del conversation[start:]
            raise

    def _run(self, conversation: list, first_name: str, age_band: str) -> tuple:
        system = SYSTEM_PROMPT + f"\nThe person you work for: first name {first_name}, age band {age_band}."
        replies, draft, attempted = [], None, False
        for _ in range(MAX_TOOL_ROUNDS):
            choice = self._call(system, conversation)["choices"][0]
            message = choice.get("message") or {}
            if choice.get("finish_reason") == "content_filter" or message.get("refusal"):
                conversation.append({"role": "assistant", "content": "I can't help with that one."})
                return "Sorry, I can't help with that. Tell me about the kind of person you'd like to meet?", draft
            tool_calls = message.get("tool_calls") or []
            entry = {"role": "assistant", "content": message.get("content") or ""}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            conversation.append(entry)
            if (message.get("content") or "").strip():
                replies.append(message["content"].strip())
            if not tool_calls:
                break
            for call in tool_calls:
                fn = call.get("function") or {}
                if fn.get("name") != "update_listing":
                    result = f"ERROR: unknown tool {fn.get('name')}"
                else:
                    attempted = True
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except json.JSONDecodeError:
                        args = None
                    if not isinstance(args, dict):
                        result = "ERROR: the arguments weren't valid JSON. Call update_listing again."
                    else:
                        new_draft, problems, _ = build_draft(args, first_name, age_band)
                        if problems:
                            result = ("ERROR: Not saved. Fix these and call update_listing again:\n- "
                                      + "\n- ".join(problems))
                        else:
                            draft = new_draft
                            result = _summary(new_draft)
                conversation.append({"role": "tool", "tool_call_id": call.get("id"), "content": result})
        reply = "\n\n".join(replies).strip() or ("I've drafted your listing. Take a look in the Listing tab."
                                                  if draft else "Sorry, could you say that another way?")
        return honest(reply, attempted, draft is not None), draft

    def _call(self, system: str, conversation: list) -> dict:
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}] + conversation,
            "tools": [OPENAI_TOOL],
        }
        if self.model.startswith(("gpt-5", "o")):
            payload["reasoning_effort"] = "low"  # a phone chat: quick, short turns
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            return self._post(API_URL, payload, headers)
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = json.loads(e.read()).get("error", {}).get("message", "")
            except Exception:
                pass
            if e.code == 401:
                raise AgentUnavailable("the OpenAI key was rejected (check OPENAI_API_KEY)") from e
            if e.code == 404:
                raise AgentUnavailable(f"OpenAI model '{self.model}' isn't available to this key (set OPENAI_MODEL)") from e
            if e.code == 429:
                raise AgentUnavailable("OpenAI rate limit or quota reached") from e
            raise AgentUnavailable(f"OpenAI returned an error ({e.code}{': ' + detail[:120] if detail else ''})") from e
        except (urllib.error.URLError, TimeoutError) as e:
            raise AgentUnavailable("couldn't reach OpenAI (network)") from e
