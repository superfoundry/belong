"""Personal agent powered by Claude (used by the two-person demo).

Each person gets their own conversation with their own agent. The agent talks with
the person, then fills in their listing by calling the `update_listing` tool. Every
tool call is checked here before it touches the draft:

  - the description must clearly say what the person wants (so matching can read it),
  - it must not target a specific person (the Warden's own rule, checked early),
  - sensitive details must not be written into the free-text description,
  - every value must come from the allowed options, and sensitive fields stay hidden
    (only the person can choose to show one, in the listing editor).

Problems go back to the agent as a tool error so it can fix them and try again.

This agent works for the person. It runs outside matching and only ever sees its own
person's conversation. The person still reviews the listing before anything is
submitted, and the Warden checks it again at entry.

Needs the `anthropic` package and Claude credentials (ANTHROPIC_API_KEY or an
`ant auth login` profile). Without them the demo falls back to the rule-based agent.
"""
from __future__ import annotations

import re

from . import synth as vocab
from .agent import _BY_KIND, _COMMON, _KIND_NAMES, _SENSITIVE_VOCAB, ALL_MONTHS, LABELS, field_options
from .matching.intent import REMOTE_KINDS, parse_intent
from .schema import FIELD_REGISTRY, FREE_TEXT_MAX, HIDDEN, SHOWN
from .warden import specific_person_problems

MODEL = "claude-opus-5-5"
MAX_TOOL_ROUNDS = 5
KINDS = list(_KIND_NAMES)
EDITABLE_KEYS = [k for k in FIELD_REGISTRY if k not in ("first_name", "age_band")]

UPDATE_LISTING_TOOL = {
    "name": "update_listing",
    "description": (
        "Create or replace the person's draft listing. Call it once you know what kind of match they "
        "want and where, and again whenever they change something. Include every field you want in the "
        "draft (it replaces the previous one). The person reviews the draft before anything is submitted."
    ),
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "kind": {"type": "string", "enum": KINDS, "description": "The kind of match."},
            "intent_text": {
                "type": "string",
                "description": "What they're looking for, one or two plain sentences in the first person. "
                               "Never shown to anyone; matching reads it.",
            },
            "exactness": {
                "type": "number",
                "description": "0.3 = cast a wide net, 0.6 = balanced, 0.9 = exactly what they said.",
            },
            "fields": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "enum": EDITABLE_KEYS},
                        "values": {"type": "array", "items": {"type": "string"},
                                   "description": "One value, or several for interests / kinks."},
                        "visibility": {"type": "string", "enum": [SHOWN, HIDDEN]},
                    },
                    "required": ["key", "values", "visibility"],
                    "additionalProperties": False,
                },
            },
            "suggestions": {
                "type": "array",
                "description": "Two to four things that would most improve this particular profile, "
                               "by common sense for what they asked (not a fixed checklist), and not "
                               "already on it. Shown to the person only, as buttons.",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {"type": "string", "description": "A few words, e.g. 'The problem', "
                                                                    "'Your budget', 'Your interests'."},
                        "ask": {"type": "string", "description": "The plain start of a sentence the person "
                                                                  "could finish, e.g. 'The problem is'. No "
                                                                  "examples or brackets."},
                    },
                    "required": ["label", "ask"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["kind", "intent_text", "exactness", "fields", "suggestions"],
        "additionalProperties": False,
    },
}


class AgentUnavailable(Exception):
    """Claude can't be reached (no SDK, no credentials, network, rate limit...)."""


# Fields whose values must come from a fixed list (matching or the leak checks rely on
# them). Everything else is short free text.
STRICT_KEYS = {"gender", "schedule", "level", "pace", "native_language", "learning_language", "role",
               "move_in", "diet", "smoking", "children", "income_band", "religion", "politics", "health", "kinks"}
ROLE_OPTIONS = ["technical", "business", "tutor", "student"]
GENERIC_INCOME = ["lower income", "middle income", "higher income"]


def _options_catalogue() -> str:
    lines = []
    generic = field_options(None, None)
    for key in EDITABLE_KEYS:
        spec = FIELD_REGISTRY[key]
        tag = "sensitive, always hidden unless the person shows it themselves" if spec.sensitive else \
            f"default {spec.default}"
        if key in STRICT_KEYS:
            opts = ROLE_OPTIONS if key == "role" else ALL_MONTHS if key == "move_in" else generic.get(key) or []
            if key == "income_band":
                opts = sorted({b for bands in vocab.INCOME.values() for b in bands}) + GENERIC_INCOME
            lines.append(f"- {key} ({tag}): {', '.join(opts)}")
        else:
            examples = {"interests": "e.g. fishing, jazz, hiking", "subject": "e.g. maths, guitar"}.get(key, "")
            lines.append(f"- {key} ({tag}): free text{(', ' + examples) if examples else ''}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""You are a personal agent in Belong. That is a service that introduces two people \
only when each is what the other is looking for: dating, or any practical need (a climbing \
partner, a roommate, a guitarist for a band, a chess partner, a tutor, a cofounder...). You work only for \
the person you're talking to. You're chatting with them on their phone, so keep replies short \
and warm: two or three sentences, no headings or bullet lists. Talk in plain words: say "your \
description", "what you're looking for", "what you offer"; never mention tools, intent_text, \
kinds, fields or other internal names.

Your job: understand what they're looking for, then draft their listing with the \
update_listing tool. A listing has a short description of what they want, plus details, each \
either shown (a match can see it) or hidden (used only to improve matching, never shown to \
anyone). Belong. never reveals hidden details.

How to work:
- Pick the kind that genuinely fits. Use a named kind (dating, climbing, running, language, \
tutoring, cofounder, roommate) only when it really is that. For anything else, use kind "other". \
Never squeeze a request into a kind it doesn't fit, and never tell the person Belong. can't help \
with something just because it isn't a named kind.
- For every kind except dating, set two short noun phrases (under 120 characters, starting with "a"/"an"), \
written as "[activity] [role]": "seeking" = who they're looking for; "offering" = who they are, \
from the other person's point of view. Work "offering" out yourself; never ask what they "offer" or \
"bring". Use these role words so the right two people line up: teacher / student (someone who'd \
teach / someone who wants to learn), partner (doing something together as equals), band / \
guitarist / drummer etc. Examples: "teach me to fish" -> seeking "a fishing teacher", offering "a \
fishing student"; someone happy to teach fishing -> seeking "a fishing student", offering "a \
fishing teacher"; a band leader -> seeking "a ska guitarist", offering "a ska band"; a guitarist \
looking for a band -> seeking "a ska band", offering "a ska guitarist"; "someone to play chess with" \
-> seeking and offering "a chess partner". Keep "offering" to a short role (a few words, like "an AC \
repair customer"): times, places, pay and format go in their own fields, never in "offering". Two people are introduced only when each one's "seeking" \
fits the other's "offering". If the person is open to more than one kind of \
person ("either one", "both", "maybe a tutor"), don't make them choose: list every option in \
"seeking", joined with " or ", each one complete (e.g. "a German tutor or a German language \
exchange partner"), and do the same in "offering" (e.g. "a German student or an English and \
Japanese language exchange partner"). Make both phrases specific: name the main activities (e.g. "a chess \
and French conversation partner", not "a hangout buddy"), and update them whenever the person \
adds something that changes who they're looking for or who they are. Also set the matching \
structured fields when they apply (e.g. native_language for a fluent speaker). Write intent_text in plain words without "date", "relationship" or \
"partner" unless it's dating (e.g. "Looking for someone to teach me to fish in Ireland"). \
Formal paid lessons in a school subject are kind "tutoring"; "show me how to..." between people \
is kind "other".
- Location: Belong. works anywhere in the world. Set "country" (in English, e.g. "Ireland") and, \
if they have one, "city" (e.g. "Galway"). If they're happy anywhere in a country or region, set \
only the country. Never list example cities or suggest they're limited to certain places.
- Fit the questions to the request. For paid or practical help (a tutor, a job, lessons, a \
service, a room), set "pay" with what they offer or charge (e.g. "¥4,000 per hour") and "format" \
("In person", "Online" or "In person or online"), and ask about those, the subject or skill, and \
when; don't ask about hobbies or interests. Ask about interests only for social requests \
(dating, friends, activity partners).
- Ask at most one or two questions at a time, and only about things you really need. When they \
answer "either", "both" or "not sure", treat every option as fine and move on; never ask them to \
pick between options they've said they're open to. You need \
the kind of match and where (country, and city if relevant) before drafting. For dating you also need their gender, who they want to meet (a man, a woman, or \
anyone) and an age range. For a language exchange: their native language and the one they're \
learning. For tutoring: the subject and whether they teach or want lessons. For a cofounder: \
whether they're the technical or business side.
- Save early and often. On the person's very first message, ALWAYS call update_listing before you \
reply, even if you only know one thing (e.g. "I'm offering piano lessons" is enough to save them as a \
piano teacher looking for students). Call update_listing as soon as you learn anything, with everything you're sure of (pick the closest kind, or "other"), then keep asking \
about what's still missing. The profile on their screen updates every time you save, so never \
wait until you know everything. Each save replaces the draft, so always include what's already \
on the profile (shown to you below) plus what's new. Never invent details they didn't give.
- With each save, suggest the two to four things that would most help this particular person find \
the right match, using common sense for what they asked: for a repair, the problem and when; for \
dating, interests; for a band, the music they play. Don't suggest what doesn't fit the request \
(no interests for a plumber), and drop a suggestion once it's on the profile.
- Write intent_text in plain first-person words that name the kind of match, e.g. "Looking for a \
climbing partner in Osaka on weekends", "Looking for a long-term relationship with a woman in \
Lisbon, aged 30 to 39", "I tutor maths in Kyoto and I'm looking for new students", "Need a \
physics tutor in Berlin", "Looking for a technical cofounder for a climate startup", "Native \
German speaker learning Japanese, looking for a language exchange (online is fine)", "Looking \
for a roommate in London, moving in November", "Running buddy in Tokyo for easy weekend runs".
- Never put sensitive things (religion, health, politics, income, intimacy) into intent_text. \
If they mention one, offer to add it as a hidden detail instead, and explain it can improve \
matches but is never shown.
- Sensitive details always start hidden. If they want to show one, tell them they can switch it \
in the listing editor themselves.
- Belong. never helps find a specific person (names, handles, someone's workplace or looks). If \
asked, explain kindly that Belong. only introduces people who are also looking.
- Where a field lists allowed values below, use one of them; fields marked "free text" take any \
short wording. Their first name and age band come from their ID check; don't set or change them.
- Only say you've added or changed something after update_listing replied "Draft saved". If it \
replied with problems you couldn't fix, say so plainly. Never claim a change that wasn't saved.
- What's private: the description (intent_text) is NEVER shown to anyone, not even to matches; \
it's only read to find matches. Hidden details are never shown either. A match sees only the \
details marked shown, plus a short reason written from details you both show. Never tell the \
person a match will see their description.
- After drafting, say in one or two sentences what a match will see, in natural words (e.g. "a \
match will see that you're in Ireland, love fishing, and want a fishing teacher; your description \
stays private"), and that they can review it on their profile before it goes live. Describe \
their role naturally ("you're the teacher"), never as "offering a teacher".

Allowed values:
{_options_catalogue()}
"""


def _clean_value(key: str, values: list, city) -> tuple:
    """Return (value or None, error or None) for one field."""
    spec = FIELD_REGISTRY[key]
    values = [" ".join(str(v).split()) for v in values if str(v).strip()]
    if not values:
        return None, None
    if key in STRICT_KEYS:
        options = ROLE_OPTIONS if key == "role" else ALL_MONTHS if key == "move_in" else \
            field_options(None, None).get(key) or []
        if key == "income_band":
            options = sorted({b for bands in vocab.INCOME.values() for b in bands}) + GENERIC_INCOME
        lookup = {o.lower(): o for o in options}
        bad = [v for v in values if v.lower() not in lookup]
        if bad:
            return None, f"{key}: {', '.join(bad)} isn't an allowed value. Allowed: {', '.join(options)}"
        values = [lookup[v.lower()] for v in values]
    elif any(len(v) > FREE_TEXT_MAX for v in values):
        return None, f"{key} must be under {FREE_TEXT_MAX} characters"
    if spec.multi:
        return tuple(dict.fromkeys(values))[:10], None
    if len(values) > 1:
        return None, f"{key} takes a single value"
    return values[0], None


def _suggestions(raw) -> list:
    """The agent's suggestions for what to add next, as short plain strings (they're shown only
    to the person, as buttons that start a message)."""
    out = []
    for s in raw if isinstance(raw, list) else []:
        if not isinstance(s, dict):
            continue
        label = " ".join(str(s.get("label", "")).split())[:40]
        ask = re.sub(r"\s*\([^)]*\)", "", str(s.get("ask", "")))  # no "(e.g. ...)" hints in their message box
        ask = " ".join(ask.split()).rstrip(":").rstrip()[:80]
        if label and label.lower() not in {o["label"].lower() for o in out}:
            out.append({"label": label, "ask": (ask + " ") if ask else ""})
    return out[:4]


def build_draft(args: dict, first_name: str, age_band: str) -> tuple:
    """Check one update_listing call. Returns (draft or None, problems, notes)."""
    kind = args.get("kind")
    intent = " ".join(str(args.get("intent_text", "")).split())[:500]
    try:
        exactness = min(1.0, max(0.0, float(args.get("exactness", 0.6))))
    except (TypeError, ValueError):
        exactness = 0.6
    # problems: wrong values or privacy issues; the draft is not saved.
    # missing:  things not known yet; the draft IS saved (a work in progress) and the agent asks next.
    problems, notes, missing = [], [], []
    if kind not in KINDS:
        problems.append(f"kind must be one of {', '.join(KINDS)}")
        return None, problems, notes

    fields_in = {}
    for f in args.get("fields") or []:
        if f.get("key") in EDITABLE_KEYS:
            fields_in[f["key"]] = f
    city_values = (fields_in.get("city") or {}).get("values") or []
    city = " ".join(str(city_values[0]).split()) if city_values else None

    values, visibility = {"first_name": first_name, "age_band": age_band}, {}
    for key, f in fields_in.items():
        value, err = _clean_value(key, f.get("values") or [], city)
        if err:
            problems.append(err)
        elif value is not None:
            values[key] = value
            vis = f.get("visibility") if f.get("visibility") in (SHOWN, HIDDEN) else FIELD_REGISTRY[key].default
            if FIELD_REGISTRY[key].sensitive and vis == SHOWN:
                vis = HIDDEN
                notes.append(f"{key} was set to hidden: only the person can choose to show a sensitive detail.")
            visibility[key] = vis

    # Can matching read the description?
    parsed = parse_intent(intent)
    open_ended = [k for k in ("seeking", "offering") if k in values]
    if not intent:
        missing.append("what they're looking for (intent_text)")
    elif kind == "dating":
        if open_ended:
            problems.append("seeking/offering aren't used for dating; remove them.")
        if parsed.kind != "dating":
            missing.append("a description that clearly says this is dating, e.g. 'Looking for a relationship with "
                           "a woman in Lisbon, aged 30 to 39'")
        else:
            if not parsed.seeking:
                missing.append("who they want to meet ('a man', 'a woman' or 'anyone', in intent_text)")
            if parsed.age_range is None:
                missing.append("an age range (e.g. 'aged 30 to 39' in intent_text)")
            if "gender" not in values:
                missing.append("their gender (shown for dating matches)")
            elif visibility.get("gender") == HIDDEN:
                visibility["gender"] = SHOWN
                notes.append("gender was set to shown: dating matches need it.")
    else:
        if parsed.kind == "dating":
            problems.append("intent_text reads like dating. Rephrase it without words like 'date', 'relationship' "
                            "or 'partner'.")
        for k in ("seeking", "offering"):
            if k not in values:
                missing.append(f"a '{k}' phrase (a short noun phrase, e.g. 'a fishing teacher')")
            else:
                visibility[k] = SHOWN  # matching needs both phrases to find the other person
    known = next((c for c in vocab.CITIES if city and city.lower() == c.lower()), None)
    if "country" not in values and known:
        values["country"] = vocab.COUNTRY_NAMES[vocab.CITIES[known][0]]  # a city we know: fill in its country
        visibility["country"] = SHOWN
    if kind not in REMOTE_KINDS or not parsed.remote_ok:
        if "country" not in values:
            missing.append("where they are: country, and city if they have one")

    for text in [intent] + [values[k] for k in ("seeking", "offering") if k in values]:
        for problem in specific_person_problems(text):
            problems.append(f"Belong. can't search for a specific person; '{text}' contains {problem}.")
    low = intent.lower()
    for key, words in _SENSITIVE_VOCAB.items():
        if any(re.search(r"\b" + re.escape(w.lower()) + r"\b", low) for w in words):
            problems.append(f"Don't put {LABELS[key].lower()} in intent_text; add it as a hidden '{key}' field instead.")

    if problems:
        return None, problems, notes

    options = field_options(kind, city)
    # The kind's usual fields, the two phrases for every non-dating listing, and anything else
    # the agent set (e.g. native_language on an open-ended listing). Nothing set is dropped.
    keys = list(dict.fromkeys(_COMMON + _BY_KIND.get(kind, []) + ([] if kind == "dating" else ["seeking", "offering"])
                              + [k for k in values if k in FIELD_REGISTRY]))
    fields = []
    for key in keys:
        spec = FIELD_REGISTRY[key]
        value = values.get(key)
        fields.append({
            "key": key, "label": LABELS[key],
            "value": list(value) if isinstance(value, tuple) else value,
            "visibility": visibility.get(key, spec.default), "sensitive": spec.sensitive, "multi": spec.multi,
            "enabled": (key in values) if spec.sensitive else True,
            "locked": key == "age_band", "options": options.get(key, []),
        })
    where = values.get("city") or values.get("country")
    understood = f"Looking for {_KIND_NAMES[kind]}" + (f" in {where}" if where else "")
    draft = {"intent_text": intent, "exactness": exactness, "kind": kind, "fields": fields,
             "understood": understood, "notes": notes, "missing": missing, "complete": not missing, "by": "claude",
             "suggestions": _suggestions(args.get("suggestions"))}
    return draft, [], notes


UNSAVED_NOTE = ("(I wasn't able to save that change to your listing. Could you say it another way, or edit it "
                "yourself with Edit details?)")


def honest(reply: str, attempted: bool, saved: bool) -> str:
    """Never let a reply suggest a change was made when no draft was actually saved."""
    if attempted and not saved:
        return f"{reply}\n\n{UNSAVED_NOTE}".strip()
    return reply


def profile_so_far(draft) -> str:
    """What's on the person's profile right now, for the agent to build on."""
    if not draft:
        return "The profile is empty so far."
    parts = [f"kind: {draft.get('kind') or 'not set'}", f"intent_text: {draft.get('intent_text') or '(empty)'}"]
    for f in draft.get("fields", []):
        value = f.get("value")
        if f.get("enabled") and value not in (None, "", []) and f["key"] not in ("first_name", "age_band"):
            shown = ", ".join(value) if isinstance(value, list) else value
            parts.append(f"{f['key']} ({f.get('visibility')}): {shown}")
    return "The profile so far (keep all of it when you save, unless they change something): " + "; ".join(parts) + "."


def _summary(draft: dict) -> str:
    shown = [f"{f['key']}={f['value']}" for f in draft["fields"] if f["value"] and f["enabled"] and f["visibility"] == SHOWN]
    hidden = [f["key"] for f in draft["fields"] if f["value"] and f["enabled"] and f["visibility"] == HIDDEN]
    out = (f"Draft saved. A match will see only these details: {', '.join(shown) or 'nothing'}. "
           f"Hidden (never shown): {', '.join(hidden) or 'nothing'}. The description itself is never shown "
           f"to anyone, including matches. If their listing is already live, the change goes live once they tap "
           f"'Review changes' and save.")
    if draft["notes"]:
        out += " Notes: " + " ".join(draft["notes"])
    if draft.get("missing"):
        out += " Still missing (ask about these next, one or two at a time): " + "; ".join(draft["missing"]) + "."
    return out


class ClaudeAgent:
    """One instance serves every phone; each phone passes in its own conversation."""

    def __init__(self, client=None, model: str = MODEL):
        if client is None:
            try:
                import anthropic
            except ImportError as e:
                raise AgentUnavailable("the 'anthropic' package isn't installed") from e
            client = anthropic.Anthropic()
        self.client = client
        self.model = model

    def respond(self, conversation: list, user_text: str, first_name: str, age_band: str, current=None) -> tuple:
        """Append the person's message, run the agent, and return (reply, draft or None).
        `conversation` is this person's message history; it is only ever appended to.
        `current` is the profile draft as it stands (so the agent builds on it)."""
        start = len(conversation)
        conversation.append({"role": "user", "content": user_text})
        try:
            return self._run(conversation, first_name, age_band, current)
        except AgentUnavailable:
            del conversation[start:]  # drop the unfinished turn so a later retry starts clean
            raise

    def _run(self, conversation: list, first_name: str, age_band: str, current=None) -> tuple:
        system = (SYSTEM_PROMPT + f"\nThe person you work for: first name {first_name}, age band {age_band}.\n"
                  + profile_so_far(current))
        replies, draft, attempted = [], None, False
        for _ in range(MAX_TOOL_ROUNDS):
            response = self._call(system, conversation)
            if response.stop_reason == "refusal":
                conversation.append({"role": "assistant", "content": "I can't help with that one."})
                return "Sorry, I can't help with that. Tell me about the kind of person you'd like to meet?", draft
            conversation.append({"role": "assistant", "content": response.content})
            replies += [b.text for b in response.content if b.type == "text" and b.text.strip()]
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if response.stop_reason != "tool_use" or not tool_uses:
                break
            results = []
            for block in tool_uses:
                if block.name != "update_listing":
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": f"Unknown tool {block.name}"})
                    continue
                attempted = True
                new_draft, problems, _ = build_draft(dict(block.input), first_name, age_band)
                if problems:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": "Not saved. Fix these and call update_listing again:\n- " +
                                               "\n- ".join(problems)})
                else:
                    draft = new_draft
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": _summary(new_draft)})
            conversation.append({"role": "user", "content": results})
        reply = "\n\n".join(replies).strip() or ("I've drafted your listing. Take a look at your profile."
                                                  if draft else "Sorry, could you say that another way?")
        return honest(reply, attempted, draft is not None), draft

    def _call(self, system: str, conversation: list):
        request = dict(
            model=self.model,
            max_tokens=16000,
            system=system,
            messages=conversation,
            tools=[UPDATE_LISTING_TOOL],
            output_config={"effort": "low"},  # a phone chat: quick, short turns
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # if a request is declined, the API retries it on a fallback model
        )
        try:
            import anthropic
        except ImportError:  # a test stub client is in use
            return self.client.beta.messages.create(**request)
        try:
            return self.client.beta.messages.create(**request)
        except anthropic.AuthenticationError as e:
            raise AgentUnavailable("Claude credentials were rejected (check ANTHROPIC_API_KEY)") from e
        except anthropic.PermissionDeniedError as e:
            raise AgentUnavailable("these Claude credentials can't use this model") from e
        except anthropic.RateLimitError as e:
            raise AgentUnavailable("Claude is rate-limited right now; try again shortly") from e
        except anthropic.APIConnectionError as e:
            raise AgentUnavailable("couldn't reach Claude (network)") from e
        except anthropic.APIStatusError as e:
            raise AgentUnavailable(f"Claude returned an error ({e.status_code})") from e
