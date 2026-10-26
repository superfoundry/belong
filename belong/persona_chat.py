"""Replies from synthetic people in the demo (demo only: not part of matching or the Warden).

When someone writes to a synthetic person they're connected with, an AI model plays that person
(SPEC §8.1 synthetic people, brought to life for the demo). It is given only what that person would
know: their own shown details and request, what the introduction showed them about the other person,
and the conversation. Never anyone's hidden details, so it can't let one slip. It replies only after
a real person matches with them (a hello) or writes to them, so nothing is spent on matches between
simulated people. Without a model, a short built-in reply is used.
"""
from __future__ import annotations

import json
import re

from .schema import value_strings

SKIP = {"first_name", "seeking", "offering"}  # the name is said separately; phrases read badly as facts

SYSTEM = """You are {name}, a person using Belong., a service that introduces two people only when each is \
what the other is looking for. You and {their} both said yes to the introduction, and now you're \
chatting.
About you (what you chose to show): {about_me}
What you asked Belong. for: {request}
What Belong. showed you about {their}: {about_them}
Why you were introduced: {reason}

Reply as {name} would in a chat app: one to three short sentences, friendly and natural, in the \
language {their} writes in (English until they've written). Stay on what you both came for. Don't invent private details (an address, \
a phone number, a workplace, a surname); if meeting up comes up, suggest a public place. You are a \
simulated person in the Belong. demo: if asked whether you're real or an AI, say so plainly.
Also decide whether to react to {their}'s latest message with one of these: {reactions}. React only \
when one clearly fits (a joke, good news, thanks, a surprise); most messages get no reaction.
Answer with JSON only: {{"reply": "...", "react": ""}} (react is "" or one of the reactions)."""


def _facts(fields: dict) -> str:
    parts = []
    for key, value in fields.items():
        if key in SKIP or value in (None, "", [], ()):
            continue
        parts.append(f"{key.replace('_', ' ')}: {', '.join(value_strings(value))}")
    return "; ".join(parts) or "nothing beyond your name"


def context(name: str, my_shown: dict, request: str, their_view: dict) -> dict:
    """Everything the model may know. my_shown must be the person's SHOWN fields only, and their_view
    the introduction view they received (the other person's shown fields and the Warden's reason)."""
    fields = their_view.get("fields", {})
    return {"name": name, "their": fields.get("first_name") or "them", "about_me": _facts(my_shown),
            "request": request, "about_them": _facts(fields), "reason": their_view.get("reason", "")}


def reply(llm, ctx: dict, thread: list, me: str, reactions, rng) -> tuple:
    """(reply text, reaction or "") for the synthetic person `me`, given the conversation so far."""
    if llm is not None and hasattr(llm, "chat_json"):
        system = SYSTEM.format(reactions=" ".join(reactions), **ctx)
        messages = [{"role": "assistant" if m["from"] == me else "user", "content": m["text"]} for m in thread]
        if not messages:  # they've just matched: you write first
            messages = [{"role": "user", "content": "(You've both just said yes. Write the first message: "
                                                    "say hello and mention what you're both looking for.)"}]
        schema = {"type": "object", "additionalProperties": False, "required": ["reply", "react"],
                  "properties": {"reply": {"type": "string"}, "react": {"type": "string", "enum": [""] + list(reactions)}}}
        out = llm.chat_json(system, messages, schema)
        text = str(out.get("reply") or "").strip()[:1000]
        react = out.get("react") if out.get("react") in reactions else ""
        if text:
            return text, react
    return fallback(thread, me, reactions, rng)


def fallback(thread: list, me: str, reactions, rng) -> tuple:
    """A short built-in reply when no model is configured (offline demo and tests)."""
    if not thread:
        return rng.choice(["Hi! Glad we matched.", "Hello! Nice to be introduced.",
                           "Hi there, great to match with you!"]), ""
    last = next((m["text"] for m in reversed(thread) if m["from"] != me), "")
    low = last.lower()
    if "?" in last:
        text = rng.choice(["Good question! Weekends usually work best for me. How about you?",
                           "I think so, yes. What would suit you?", "Happy to. When are you free?"])
    else:
        text = rng.choice(["Nice to meet you too!", "That sounds great.", "Ha, same here.",
                           "Sounds good to me. When works for you?", "Yes, I'd like that."])
    cues = [("🙏", r"\bthank|\bthanks\b"), ("😂", r"\bhaha|\blol\b"), ("😮", r"\bwow\b|\breally\?"),
            ("❤️", r"\blove\b"), ("👍", r"\bgreat\b|\bperfect\b|\bawesome\b")]
    react = next((e for e, pat in cues if e in reactions and re.search(pat, low)), "")
    return text, react


def parse_json(text: str) -> dict:
    """The first JSON object in a model's text answer (Claude path), or {} if there isn't one."""
    found = re.search(r"\{.*\}", text or "", flags=re.S)
    try:
        return json.loads(found.group(0)) if found else {}
    except ValueError:
        return {}
