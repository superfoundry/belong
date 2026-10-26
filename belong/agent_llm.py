"""Personal agent powered by Claude (used by the two-person demo).

Each person gets their own conversation with their own agent. The agent talks with
the person, then fills in their listing by calling the `update_listing` tool. Every
tool call is checked here before it touches the draft:

  - the description must clearly say what the person wants (so matching can read it),
  - it must not target a specific person (the Warden's own rule, checked early),
  - sensitive details must not be written into the free-text description,
  - every value must come from the allowed options, and sensitive fields stay hidden
    (only the person can choose to show one, in the listing editor),
  - what the agent understood is recorded on the listing as shown details the person can check:
    the kind of match it chose ("kind") and, for dating, who they'd like to meet and those people's
    ages in years ("wants_gender", "wants_age"), in the fixed forms the Warden accepts. Matching uses
    them before anything in the description (TRUST.md #33).

Problems go back to the agent as a tool error so it can fix them and try again. When the kind of match
is unclear, the agent asks one short question before saving (SYSTEM_PROMPT), and an answer of "both"
saves two listings in one turn (turn_drafts).

This agent works for the person. It runs outside matching and only ever sees its own
person's conversation. The person still reviews the listing before anything is
submitted, and the Warden checks it again at entry.

Needs the `anthropic` package and Claude credentials (ANTHROPIC_API_KEY or an
`ant auth login` profile). Without them the demo falls back to the rule-based agent.
"""
from __future__ import annotations

import re

from . import synth as vocab
from .agent import (_BY_KIND, _COMMON, _KIND_NAMES, _SENSITIVE_VOCAB, ALL_MONTHS, DATING_ONLY, LABELS, carry_over,
                    field_options, fill_gaps, mentions_sensitive, wants_text)
from .matching.intent import ages_text, default_age_range, parse_intent
from .schema import (AGE_MAX, AGE_MIN, GENDERS, HEADLINE_MAX, FIELD_REGISTRY, FREE_TEXT_MAX, HIDDEN, SHOWN,
                     age_span_choice, meeting_format, travel_choice)
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
            "kind": {"type": "string", "enum": KINDS,
                     "description": "The kind of match: what they want to meet someone for, never a hobby they "
                                    "mention. It's recorded on their listing (the 'kind' detail) and matching uses "
                                    "it. Don't save while it's unclear: ask first."},
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
                                   "description": "One value, or several for interests / kinks / wants_gender. "
                                                  "Fields with allowed values (e.g. format: In person, Online, In "
                                                  "person or online; travel: Same city, Up to 30 km, Up to 100 km) "
                                                  "take one of them. Dating only: wants_gender takes woman, man "
                                                  "and/or nonbinary (all three for anyone); wants_age takes one "
                                                  "age range in years written lo-hi, e.g. 30-39 (18 to 99)."},
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
            "separate_listing": {
                "type": "boolean",
                "description": "true only when the person wants this as an additional, separate listing "
                               "alongside the one shown to you (a different search: 'I'm also looking for a "
                               "guitar teacher', 'make another listing for...'). false when it updates the "
                               "current listing.",
            },
        },
        "required": ["kind", "intent_text", "exactness", "fields", "suggestions", "separate_listing"],
        "additionalProperties": False,
    },
}


class AgentUnavailable(Exception):
    """Claude can't be reached (no SDK, no credentials, network, rate limit...)."""


# Fields whose values must come from a fixed list (matching or the leak checks rely on
# them). Everything else is short free text.
STRICT_KEYS = {"gender", "schedule", "level", "pace", "native_language", "learning_language", "role",
               "move_in", "diet", "smoking", "children", "income_band", "religion", "politics", "health", "kinks",
               "format", "travel", "kind", "wants_gender"}
# Fixed-choice fields matching reads in its own words: other wordings ("online only", "up to 30km")
# are put into the fixed wording before they're checked.
_NORMALISE = {"format": meeting_format, "travel": travel_choice}
ROLE_OPTIONS = ["technical", "business", "tutor", "student"]
GENERIC_INCOME = ["lower income", "middle income", "higher income"]


def _options_catalogue() -> str:
    lines = []
    generic = field_options(None, None)
    for key in EDITABLE_KEYS:
        spec = FIELD_REGISTRY[key]
        tag = "sensitive, always hidden unless the person shows it themselves" if spec.sensitive else \
            f"default {spec.default}"
        if key == "kind":
            lines.append(f"- kind ({tag}): set for you from the kind you choose; list it in fields only to keep "
                         f"the person's own choice of hidden")
        elif key == "wants_age":
            lines.append(f"- wants_age ({tag}, dating only): an age range in years, written lo-hi, from {AGE_MIN} "
                         f"to {AGE_MAX}, e.g. 30-39")
        elif key in STRICT_KEYS:
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
with something just because it isn't a named kind. The kind is what they want to meet someone FOR, \
never a hobby or interest they happen to mention: "looking for a relationship; I enjoy hiking and \
chess" is dating, and hiking and chess go in "interests". The kind you choose is saved on their \
listing and is what matching uses, so their description never has to repeat it.
- When it's unclear, ask before saving. If a message could mean more than one kind of match, don't \
guess. Words that plainly mean romance ("a relationship", "a date", "a girlfriend") make it dating, \
and activities they only say they enjoy are interests. But wanting to meet a man or a woman alongside \
an activity they mention could be a date or a partner for that activity, and "a partner" with nothing \
saying whether it's romantic or for something to do together could be either. Then ask ONE short \
question and wait for the answer before calling update_listing, e.g. "Are you looking for a date, or \
a [activity] partner? Or both? I can make two separate listings." Never save a guessed kind while \
it's unclear. If they say both, save one listing for each kind, in the same turn: first the current \
listing (separate_listing false), then the other (separate_listing true), each with its own kind, \
headline and details.
- For every kind except dating, set two short noun phrases (under 120 characters, starting with "a"/"an"), \
written as "[activity] [role]": "seeking" = who they're looking for; "offering" = who they are, \
from the other person's point of view. Work "offering" out yourself; never ask what they "offer" or \
"bring". Use these role words so the right two people line up: teacher / student (someone who'd \
teach / someone who wants to learn), partner (doing something together as equals), band / \
guitarist / drummer etc. Examples: "teach me to fish" -> seeking "a fishing teacher", offering "a \
fishing student"; someone happy to teach fishing -> seeking "a fishing student", offering "a \
fishing teacher"; a band leader -> seeking "a ska guitarist", offering "a ska band"; a guitarist \
looking for a band -> seeking "a ska band", offering "a ska guitarist"; "someone to play chess with" \
-> seeking and offering "a chess partner". Those two phrases are for matching only and can sound \
stiff; people never see them as titles. Also set "headline": the listing in plain, natural words, as \
the person would say it, starting with an -ing word, under 60 characters: "Selling 100 cardboard \
boxes", "Looking for free cardboard boxes", "Teaching beginner guitar", "Looking for someone to fix \
a leaking tap", "Starting a punk band and need a drummer". Leave out the place, times and price: they \
have their own fields. It's the listing's title, so update it \
whenever the request changes. Also set "topics": two to five broader skills or areas the \
listing is about, so related requests can be found even when the words differ (someone wanting a bass teacher: \
"bass guitar", "music theory"; a guitarist who teaches: "guitar", "bass guitar"; a leaking tap: \
"plumbing", "home repair"). Specific enough to mean something: never just \
"sports", "lessons", "help" or "people", and never anything personal. Keep "offering" to a short role (a few words, like "an AC \
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
- How they meet: for every kind except dating, set "format": "In person", "Online" or "In person or \
online". Two people are matched online only if both are happy online; then where they live doesn't \
matter. Dating is always in person. For anything in person, set "travel" when they say how far \
they'd go to meet: "Same city" (the default when they don't say), "Up to 30 km" or "Up to 100 km" \
(pick the furthest that isn't further than they said). Two people in different cities meet only if \
both would travel that far.
- Fit the questions to the request. For paid or practical help (a tutor, a job, lessons, a \
service, a room), set "pay" with what they offer or charge (e.g. "¥4,000 per hour") and "format", \
and ask about those, the subject or skill, and \
when; don't ask about hobbies or interests. Ask about interests only for social requests \
(dating, friends, activity partners).
- Ask at most one or two questions at a time, and only about things you really need. When they \
answer "either", "both" or "not sure", treat every option as fine and move on; never ask them to \
pick between options they've said they're open to. You need \
the kind of match and where (country, and city if relevant) before drafting. For dating, always \
record three things: their own gender ("gender"), who they'd like to meet ("wants_gender": woman, man \
and/or nonbinary; all three for anyone) and those people's ages in years ("wants_age", written lo-hi, \
e.g. "30-39"). Work the ages out from whatever they said, using their age band (below): "around my \
age" is about one age band either side of theirs (age band 35-39: "30-49"), "early 30s" is "30-33", \
"late 20s to mid 30s" is "27-36", "a bit younger" is the band below theirs and their own, "40 or \
older" is "40-99". Ask only for what they gave nothing about; until they give ages they're open to \
people one age band either side of their own. Their gender: ask plainly if you don't know it; it's \
shown by default, because that's how people looking for a woman or a man find them. Offer the usual \
choice: they can keep it hidden, but a hidden gender can't help others find them (a hidden detail never \
decides a match), so then only people open to anyone can match them and they may get fewer matches. \
Never pick hidden for them. For a language exchange: their native language and the one they're \
learning. For tutoring: the subject and whether they teach or want lessons. For a cofounder: \
whether they're the technical or business side.
- Save early and often (unless the kind is unclear: then ask first, as above). As soon as a message says \
anything about what the person is looking for or offering, call update_listing before you reply, even if \
it's the first message and you only know one thing \
(e.g. "I'm offering piano lessons" is enough to save them as a piano teacher looking for students). A message \
that says nothing about that ("hi", "connect me", "how does this work?") is not a request: just answer it and \
ask what they're looking for; never save a draft from it. Call update_listing as soon as you learn anything, with everything you're sure of (the kind that fits, or "other" when it's clearly none of the named ones), then keep asking \
about what's still missing. The profile on their screen updates every time you save, so never \
wait until you know everything. Each save replaces the draft, so always include what's already \
on the profile (shown to you below) plus what's new. Never invent details they didn't give.
- A person can have several listings, each its own search. If they ask for something different from the \
listing shown to you (another kind of person or purpose, e.g. a guitar teacher while their listing is a \
climbing partner), call update_listing with separate_listing true and a complete new listing (its own kind, \
headline, phrases and place; don't carry over details that belong to the other search). It becomes a new \
listing and the current one stays as it is. Otherwise keep separate_listing false.
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
- Never put sensitive things (religion, health, politics, income, intimacy, sexual orientation or \
gender identity) into intent_text, or into any detail that is shown (seeking, offering, headline, \
topics, interests, or a place): "a Muslim partner" is not an activity. If they mention one, offer to \
add it as a hidden detail instead, and explain it can improve matches but is never shown.
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
    if key == "wants_age":
        # Years, in the fixed form the Warden accepts ("33 to 42" is put into "33-42" first). Relative
        # wording ("around my age") must be worked out by the agent, never guessed here.
        fixed = [age_span_choice(v) for v in values]
        if len(values) > 1 or fixed[0] is None:
            return None, (f"wants_age: {', '.join(values)} isn't an age range in years. Give one range written lo-hi, "
                          f"from {AGE_MIN} to {AGE_MAX} (e.g. 30-39); work out wording like 'around my age' from "
                          f"their age band")
        return fixed[0], None
    if key in STRICT_KEYS:
        if key in _NORMALISE:
            values = [_NORMALISE[key](v) or v for v in values]
        options = ROLE_OPTIONS if key == "role" else ALL_MONTHS if key == "move_in" else \
            field_options(None, None).get(key) or []
        if key == "income_band":
            options = sorted({b for bands in vocab.INCOME.values() for b in bands}) + GENERIC_INCOME
        lookup = {o.lower(): o for o in options}
        bad = [v for v in values if v.lower() not in lookup]
        if bad:
            return None, f"{key}: {', '.join(bad)} isn't an allowed value. Allowed: {', '.join(options)}"
        values = [lookup[v.lower()] for v in values]
        if key == "wants_gender":
            values = [g for g in GENDERS if g in values]
    elif any(len(v) > FREE_TEXT_MAX for v in values):
        return None, f"{key} must be under {FREE_TEXT_MAX} characters"
    if spec.multi:
        return tuple(dict.fromkeys(values))[:10], None
    if len(values) > 1:
        return None, f"{key} takes a single value"
    return values[0], None


# A headline starts with an -ing word ("Selling...", "Looking for..."), so the Warden can quote it
# in a sentence: "Someone is selling 100 cardboard boxes."
_ING = re.compile(r"^[A-Za-z]+ing\b")


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


def build_draft(args: dict, first_name: str, age_band: str, current=None) -> tuple:
    """Check one update_listing call. Returns (draft or None, problems, notes).

    current: the profile this save replaces (None for a new listing), with "choices": the person's own
    choices of shown or hidden for this listing. What the call leaves out is kept from it with the shown or
    hidden it had, and every detail the call gives no value and visibility of its own keeps the person's
    choice (agent.carry_over), BEFORE anything below is checked or summarised: what's missing, whether a
    hidden kind still reads as dating, and what the agent is told a match will see are all about the draft
    the person will actually see."""
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
    chose = {k for k, f in fields_in.items() if f.get("visibility") in (SHOWN, HIDDEN)}  # a visibility of its own
    for key, f in fields_in.items():
        if key == "kind":  # its value is the kind chosen above; listed only to keep the person's choice of hidden
            if key in chose:
                visibility["kind"] = f["visibility"]
            continue
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

    # The kind the agent chose is what it understood: it goes on the listing as the "kind" detail (shown
    # unless the person chose hidden), and matching uses it before anything in the description.
    values["kind"] = kind
    if kind != "dating":
        for key in DATING_ONLY:
            if values.pop(key, None) is not None:
                visibility.pop(key, None)
                notes.append(f"{key} was left out: it's only used for dating.")
    # The details the agent chose shown or hidden for on purpose: a value it kept, with a visibility of its
    # own (the tool always sends one, even with no value: that chooses nothing). Every other visibility below
    # is a default, and a default never overrides the person's own choice (carry_over).
    listed = sorted(k for k in chose if k == "kind" or k in values)
    own = [k for k in values if k not in ("first_name", "age_band", "kind")]  # what this call wrote

    # Problems with what this call wrote. Nothing is saved until they're fixed.
    if kind == "dating" and any(k in values for k in ("seeking", "offering")):
        problems.append("seeking/offering aren't used for dating; remove them.")
    if kind != "dating" and "headline" in values and (len(values["headline"]) > HEADLINE_MAX
                                                      or not _ING.match(values["headline"])):
        problems.append(f"headline must be under {HEADLINE_MAX} characters and start with an -ing word, "
                        "e.g. 'Selling 100 cardboard boxes', 'Looking for free cardboard boxes'.")
    for text in [intent] + [values[k] for k in ("seeking", "offering") if k in values]:
        for problem in specific_person_problems(text):
            problems.append(f"Belong. can't search for a specific person; '{text}' contains {problem}.")
    low = intent.lower()
    for key, words in _SENSITIVE_VOCAB.items():
        if any(re.search(r"\b" + re.escape(w.lower()) + r"\b", low) for w in words):
            problems.append(f"Don't put {LABELS[key].lower()} in intent_text; add it as a hidden '{key}' field instead.")

    options = field_options(kind, city)
    # The kind's usual fields, the two phrases for every non-dating listing, and anything else
    # the agent set (e.g. native_language on an open-ended listing). Nothing set is dropped.
    keys = list(dict.fromkeys(["kind"] + _COMMON + _BY_KIND.get(kind, [])
                              + ([] if kind == "dating" else ["headline", "seeking", "offering", "topics"])
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
    draft = {"intent_text": intent, "exactness": exactness, "kind": kind, "fields": fields, "listed": listed}
    # The listing as it will be: what's already on the profile, and the person's choices of shown or hidden.
    carry_over(draft, current, (current or {}).get("choices"))
    by = {f["key"]: f for f in draft["fields"]}
    if kind != "dating":
        # Matching needs both phrases to find the other person, topics spot related requests and the
        # headline is the listing's title: on a listing that isn't dating, they're always shown.
        for key in ("seeking", "offering", "topics", "headline"):
            if key in by and by[key]["value"] not in (None, "", []):
                by[key]["visibility"] = SHOWN
        if "topics" in by and isinstance(by["topics"]["value"], list):
            by["topics"]["value"] = by["topics"]["value"][:5]
    known = next((c for c in vocab.CITIES if c.lower() == str(by["city"]["value"] or "").lower()), None) \
        if "city" in by else None
    if known and by.get("country", {}).get("value") in (None, "", []):
        # A city we know: fill in its country, shown unless the person chose to hide it.
        fill_gaps(draft, {"country": vocab.COUNTRY_NAMES[vocab.CITIES[known][0]]},
                  {"country": ((current or {}).get("choices") or {}).get("country", SHOWN)})
        by = {f["key"]: f for f in draft["fields"]}
    merged = {k: (tuple(f["value"]) if isinstance(f["value"], list) else f["value"]) for k, f in by.items()
              if f.get("enabled", True) and f["value"] not in (None, "", [])}
    shown = lambda key: key in merged and by[key]["visibility"] == SHOWN

    # A shown detail in free text never carries a sensitive word (a religion, an orientation, a health
    # condition): matches would see it. It belongs in the matching sensitive detail, hidden, or nowhere.
    # Checked on what this call wrote (the person's own edits are theirs), as it will be shown.
    for key in own:
        spec = FIELD_REGISTRY.get(key)
        if spec is None or spec.sensitive or key in STRICT_KEYS or not shown(key):
            continue
        value = values[key]
        if any(mentions_sensitive(v) for v in (value if isinstance(value, tuple) else (value,))):
            problems.append(f"'{key}' is shown, so it must not say anything sensitive (religion, health, politics, "
                            f"sexuality or intimacy). Take that out; if it matters, add it as a hidden sensitive field.")
    # What matching reads from the description: only what the shown details don't say (with the headline,
    # if it's shown; it always is on a non-dating listing).
    parsed = parse_intent(intent, merged.get("headline", "") if shown("headline") else "")
    if kind != "dating" and parsed.kind == "dating":
        problems.append("intent_text (with the headline) reads like dating. Rephrase it without words like "
                        "'date', 'relationship', 'love' or 'partner'.")
    if problems:
        return None, problems, notes

    if not intent:
        missing.append("what they're looking for (intent_text)")
    elif kind == "dating":
        if not shown("kind") and parsed.kind != "dating":
            # Only a hidden kind leaves it to the description, and this one doesn't read as dating.
            missing.append("the kind is hidden, so matching reads the description, and it doesn't clearly say "
                           "this is dating: keep 'kind' shown, or say so in intent_text")
        if "wants_gender" not in merged:
            missing.append("who they want to meet: set wants_gender (woman, man and/or nonbinary; all three for "
                           "anyone)")
        if "wants_age" not in merged:
            missing.append(f"the ages they'd like, in years: set wants_age (e.g. 30-39); work out wording like "
                           f"'around my age' or 'early 30s' from their age band {age_band}, and ask only if they "
                           f"said nothing. Until then they're open to {ages_text(default_age_range(age_band))}, one age "
                           f"band either side of their own")
        for key in DATING_ONLY:
            if key in merged and not shown(key):
                # Their choice stands. A hidden detail never decides a match, so the description is read instead.
                notes.append(f"{key} is hidden: a hidden detail never decides a match, so matching reads this from "
                             f"the description instead. Make sure they know, and that the description says it.")
        if "gender" not in merged:
            missing.append("their gender (needed so people looking for a woman or a man can find them; they may "
                           "keep it hidden, but then only people open to anyone can match them)")
        elif not shown("gender"):
            # Their choice stands: nobody is made to show their gender. A hidden detail never decides a
            # match, so the person should know what it costs.
            notes.append("gender is hidden: a hidden detail never decides a match, so people looking for a woman "
                         "or a man won't be matched with them, only people open to anyone. Make sure they know "
                         "they may get fewer matches and can switch it to shown.")
    else:
        for k in ("seeking", "offering"):
            if k not in merged:
                missing.append(f"a '{k}' phrase (a short noun phrase, e.g. 'a fishing teacher')")
        if "headline" not in merged:
            missing.append("a headline (e.g. 'Selling 100 cardboard boxes')")
    # A listing that can meet online can be matched without a place; one that only meets in person
    # can't. Dating is always in person. Without a format, the description decides (as in matching).
    fmt = meeting_format(merged.get("format"))
    online = kind != "dating" and (fmt in ("Online", "In person or online") or (fmt is None and parsed.remote_ok))
    if not online and "country" not in merged:
        missing.append("where they are: country, and city if they have one")

    where = merged.get("city") or merged.get("country")
    understood = f"Looking for {_KIND_NAMES[kind]}" + (f" in {where}" if where else "")
    if kind == "dating" and (merged.get("wants_gender") or merged.get("wants_age")):
        understood += ", with " + wants_text(merged.get("wants_gender"), merged.get("wants_age"))
    draft.update({"understood": understood, "notes": notes, "missing": missing, "complete": not missing, "by": "claude",
                  "suggestions": _suggestions(args.get("suggestions")), "separate": args.get("separate_listing") is True})
    return draft, [], notes


UNSAVED_NOTE = ("(I wasn't able to save that change to your listing. Could you say it another way, or edit it "
                "yourself with Edit details?)")


def honest(reply: str, attempted: bool, saved: bool) -> str:
    """Never let a reply suggest a change was made when no draft was actually saved."""
    if attempted and not saved:
        return f"{reply}\n\n{UNSAVED_NOTE}".strip()
    return reply


def turn_drafts(saved: list):
    """Everything one turn saved, as one draft to hand back: the current listing's latest draft, then
    each separate listing's latest (one per kind), in that order. The last is returned, carrying the
    others under "earlier" for whoever applies them in order (a "both" answer makes two listings, each
    with its own kind). None if nothing was saved."""
    current = [d for d in saved if not d.get("separate")][-1:]
    separate = {}
    for d in saved:
        if d.get("separate"):
            separate[d.get("kind")] = d
    ordered = current + list(separate.values())
    if not ordered:
        return None
    last = {**ordered[-1]}
    if len(ordered) > 1:
        last["earlier"] = ordered[:-1]
    return last


def builds_on(args: dict, saved: list, current):
    """The profile a save replaces: nothing for a separate (new) listing; else this turn's last save for
    the current listing, or the profile as it was, each with the person's choices of shown or hidden."""
    if args.get("separate_listing") is True:
        return None
    mine = [d for d in saved if not d.get("separate")]
    if not mine:
        return current
    return {**mine[-1], "choices": (current or {}).get("choices") or {}}


def profile_so_far(draft) -> str:
    """What's on the person's profile right now, for the agent to build on, with the details the person
    chose to keep hidden (so a save keeps them hidden, even ones not on this draft)."""
    hidden = sorted(k for k, v in ((draft or {}).get("choices") or {}).items()
                    if v == HIDDEN and k not in ("first_name", "age_band"))
    kept = (" The person chose to keep these hidden; keep them hidden unless they ask to show one: "
            + ", ".join(hidden) + ".") if hidden else ""
    if not draft or not (draft.get("fields") or draft.get("intent_text")):
        return "The profile is empty so far." + kept
    kind_field = next((f for f in draft.get("fields", []) if f.get("key") == "kind" and f.get("value")), None)
    hidden_kind = " (hidden: keep it hidden by listing kind as hidden in fields)" if kind_field and \
        kind_field.get("visibility") == HIDDEN else ""
    parts = [f"kind: {draft.get('kind') or 'not set'}{hidden_kind}",
             f"intent_text: {draft.get('intent_text') or '(empty)'}"]
    for f in draft.get("fields", []):
        value = f.get("value")
        if f.get("enabled") and value not in (None, "", []) and f["key"] not in ("first_name", "age_band", "kind"):
            shown = ", ".join(value) if isinstance(value, list) else value
            parts.append(f"{f['key']} ({f.get('visibility')}): {shown}")
    return "The profile so far (keep all of it when you save, unless they change something): " + "; ".join(parts) + "." + kept


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
        replies, draft, attempted, saved = [], None, False, []
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
                args = dict(block.input)
                new_draft, problems, _ = build_draft(args, first_name, age_band, builds_on(args, saved, current))
                if problems:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True,
                                    "content": "Not saved. Fix these and call update_listing again:\n- " +
                                               "\n- ".join(problems)})
                else:
                    saved.append(new_draft)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": _summary(new_draft)})
            conversation.append({"role": "user", "content": results})
            draft = turn_drafts(saved)
        reply = "\n\n".join(replies).strip() or ("I've drafted your listing. Take a look at your profile."
                                                  if draft else "Sorry, could you say that another way?")
        return honest(reply, attempted, draft is not None), draft

    def chat_json(self, system: str, messages: list, schema: dict) -> dict:
        """One plain reply as JSON (used for simulated people's chat replies). Claude answers in text
        and the JSON object is read out of it; `schema` is described in the system prompt."""
        from .persona_chat import parse_json
        convo = list(messages) or [{"role": "user", "content": "(say hello)"}]
        try:
            response = self.client.messages.create(model=self.model, max_tokens=1000, system=system,
                                                   messages=convo, output_config={"effort": "low"})
        except Exception as e:  # the SDK's own errors (credentials, rate limits, network)
            raise AgentUnavailable(f"couldn't get a reply from Claude ({type(e).__name__})") from e
        return parse_json("".join(b.text for b in response.content if getattr(b, "type", "") == "text"))

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
