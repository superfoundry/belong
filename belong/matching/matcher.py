"""Matchmaker agent (SPEC §3, §5, §6, §8.3).

Open source, and still treated as hostile: it needs no trust, because it can only hand
the Warden the objects defined in contract.py, and the Warden re-checks every one.

Pluggable: anything with `match(board) -> MatchResult` is a Matchmaker.
  - OfflineMatcher: rules + TF-IDF, no API key, no network (this file).
  - LLMMatcher (hybrid.py): the same rules narrow and keep everything hard and exact, and an injected judge
    (a model, outside this package) judges the fuzzy part of each pair that passes them. It reuses the steps
    below (read, decide, finish), so a pair the judge doesn't answer is decided exactly as here.

How OfflineMatcher decides, in order:
  1. Read what each listing asks for (understood, intent.read_listing): what the person's agent
     recorded on it comes first (the shown kind of match, dating or other, and for dating the
     shown genders and ages wanted), and the description and shown headline only for what that
     doesn't say (words are read only to tell dating from the rest). A listing that shows its
     "looking for" / "you are" phrases and no kind is "other". Missing pieces become fixed-menu
     follow-up questions; until answered, something not said is not a strict preference (no age
     range: one age band either side of your own).
  2. Narrow (narrow.py), top down: how they meet first (online only if both allow it, then
     place doesn't matter), then where (in person: cities within both people's travel
     distance), then what (dating with dating, anything else by a shared word in what one looks
     for and the other is). Each listing gets a shortlist.
  3. Both must be looking: keep a pair only if A's intent is met by B *and* B's by A,
     judged from shown fields only: dating by its own rules, anything else by its phrases, and
     always every wish A shows (its "them" and "both" details) met by B's own shown detail of
     the same name (details_ok). There is no list of activities, details or values here: each
     detail brings its own name, values and scale, written by the person's agent.
  4. Never the only reason: the pair needs at least one reason shown by both (where they can
     meet, the open details they have in common as schema.detail_reasons works them out, their
     phrases), and a shown-only score above the threshold.
  5. Hidden fields may only raise the score, and only values that at least k other people share.
  6. Pick the best pairs, at most `max_per_person` each, fairly: a little lower for a listing
     that already had introductions in the last day, a little higher for one updated in the
     last day. Ranking only: no pair skips a check above (TRUST.md #32).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, replace

from ..schema import (AGE_BANDS, FOLLOWUP_MENU, FREE_TEXT_MAX, HIDDEN, K_CROWD, STRICT_EXACTNESS, Listing, canonical,
                      comparable, detail_reasons, satisfies, says_nothing, split_key)
from .contract import FollowUp, MatchResult, Matcher, Proposal  # noqa: F401 (re-exported)
from .intent import GENDERS, ParsedIntent, default_age_range, parse_intent, read_listing  # noqa: F401
from .marketboard import Marketboard
from .narrow import ONLINE, ShownView, how_they_meet, shortlists, within_reach
from .text import cosine, letter_words

_PHRASE_STOP = {"a", "an", "the", "my", "our", "your", "for", "to", "of", "and", "or", "in", "on", "with",
                "who", "someone", "somebody", "person", "people", "some", "new", "join", "find", "looking",
                "can", "could", "also", "that", "which", "will", "would", "me", "us", "them", "is", "are", "be",
                "help"}


# Role words that mean the same thing (after the suffix-stripping below).
_SYNONYMS = {
    "mentor": "teach", "coach": "teach", "instructor": "teach", "tutor": "teach", "guide": "teach",
    "learn": "student", "begin": "student", "apprentice": "student", "pupil": "student", "novice": "student",
    "partn": "partner", "buddy": "partner", "friend": "partner", "companion": "partner", "pal": "partner",
}


# A close match is judged as if its person had cast a wide net.
CLOSE_EXACTNESS = 0.2

ROLE_WORDS = {"teach", "student", "partner"}
# Purely social wording: someone who just wants company is open to any activity.
SOCIAL_WORDS = {"social", "hangout", "hang", "out", "meet", "meetup", "company", "chat", "casual", "fun", "time"}
# With a wide net, these roles may stand in for each other when the activity overlaps:
# a learner happy to practise with a peer, a peer happy to help a learner.
_ROLE_RELAX = {"teach": {"partner"}, "partner": {"student", "teach"}, "student": {"partner"}}


def phrase_tokens(text: str) -> list:
    """Content words of a short phrase, in any script (text.letter_words: compared the way every value
    is, so "Café" and "cafe" agree, and a phrase in Japanese or Cyrillic has words too), crudely
    normalised on Latin words so 'painter', 'painters' and 'paint' agree, as do 'planner' and 'plan'."""
    found = _TOKENS.get(text)
    if found is None:  # the same few phrases come up for every pair they're in
        if len(_TOKENS) > 50_000:
            _TOKENS.clear()
        found = _TOKENS[text] = tuple(_phrase_tokens(text))
    return list(found)


_TOKENS: dict = {}  # phrase -> its tokens


def _phrase_tokens(text: str) -> list:
    out = []
    for w in letter_words(text):
        if w in _PHRASE_STOP or len(w) < 2:
            continue
        if w.isascii():  # English endings, on Latin words only
            for suffix in ("ists", "ist", "ers", "er", "ing", "s"):
                if len(w) > len(suffix) + 2 and w.endswith(suffix):
                    w = w[: -len(suffix)]
                    break
            if len(w) > 3 and w[-1] == w[-2]:
                w = w[:-1]  # plann -> plan, beginn -> begin
        w = _SYNONYMS.get(w, w)
        out.append(w)
    return out


_ALTERNATIVE = re.compile(r"\s+or\s+|\s*;\s*", re.I)


def alternatives(text: str) -> list:
    """Split "a pottery teacher or a pottery club partner" into the options the person is
    open to, each as a token list. Short options borrow what they leave out from the others:
    "a pottery or weaving teacher" -> [pottery ... teacher], [weaving ... teacher]; "a pottery
    teacher or club partner" -> the second also gets "pottery"."""
    found = _OPTIONS.get(text)
    if found is None:
        if len(_OPTIONS) > 50_000:
            _OPTIONS.clear()
        found = _OPTIONS[text] = tuple(map(tuple, _alternatives(text)))
    return [list(option) for option in found]


_OPTIONS: dict = {}  # phrase -> its options


def _alternatives(text: str) -> list:
    parts = [list(dict.fromkeys(phrase_tokens(p))) for p in _ALTERNATIVE.split(text or "")]
    parts = [p for p in parts if p]
    if len(parts) < 2:
        return parts
    roles = [t for p in parts for t in p if t in ROLE_WORDS] or [parts[-1][-1]]
    activity = [t for p in parts for t in p if t not in ROLE_WORDS]
    out = []
    for p in parts:
        has_role = any(t in ROLE_WORDS for t in p) or p[-1] == roles[-1]
        has_activity = any(t not in ROLE_WORDS for t in p)
        if not has_activity:
            p = activity + p
        if not has_role and len(p) <= 2:
            p = p + [roles[-1]]  # "a pottery" (or weaving teacher) -> "a pottery teacher"
        out.append(list(dict.fromkeys(p)))
    return out


def _content(tokens) -> set:
    """The words of a phrase that say what it is about: not a role word, not a purely social one."""
    return {t for t in tokens if t not in ROLE_WORDS and t not in SOCIAL_WORDS}


def _about(option, head: str) -> set:
    """What an option is about besides its head (_head): its content words other than the head. The
    head is the role ("partner" in "a kayaking partner", "keeper" in "a wicket keeper", and so in any
    script, whatever the word), so two options with the same head are compared on these."""
    return _content(option) - {head}


def _lacks(option: list, own_offer) -> set:
    """What an option asks for that the person isn't themselves: its content words that aren't in
    their own "you are" phrase. "a second violinist" from someone who is "a first violinist" lacks
    "second"; "a kayaking partner" from a kayaking partner lacks nothing."""
    return _content(option) - set(phrase_tokens(str(own_offer or "")))


def phrase_satisfies(wanted: str, offered: str, exactness: float, also_have=(), own_offer=None) -> bool:
    """Does `offered` give what `wanted` asks for? True if any option the person is open to
    is met by any option the other person is (see alternatives). With `own_offer` (the
    person's own "you are"), the option must also be one whose words they lack (_lacks) are
    all in what the other person is: two people who are the same thing and both look for its
    counterpart are never each other's answer."""
    theirs = set(phrase_tokens(str(offered or "")))
    for w in alternatives(wanted):
        if own_offer is not None and not _lacks(w, own_offer) <= theirs:
            continue
        if any(_satisfies(w, h, exactness, also_have) for h in alternatives(offered)):
            return True
    return False


def _satisfies(want: list, have_list: list, exactness: float, also_have=()) -> bool:
    """One option against one option. The main noun, its head ('painter' in 'a portrait painter',
    'partner' in 'a kayaking partner'), must be the same, and then the head is the role and what the
    option is about is the rest (_about): more than half of those words must be in the other's phrase.
    A wide net asks for at least half of them, in that phrase or in what the other shows about
    themselves, and still for at least half of all the option's words in the phrase. At the default
    exactness and above, two options that each have a content word the other lacks never meet, whatever
    they share and wherever the head is (a role noun first, "compagno di ...", is read the same way): only
    the same option, or one that narrows the other, meets. So a head in common alone ("partner",
    "teacher", "keeper", or the same noun in any script) never meets an option that says what it is
    about, and at the default exactness neither does one or more describing words in common when another
    differs ("a winter birding partner" and "a winter stargazing partner"). An option that is only its
    head (with role or social words: "a keeper", "a partner", "a hangout partner") is compared on the
    head."""
    if not want or not have_list:
        return False
    head, offered_head = _head(want), _head(have_list)
    # What the other person shows about themselves (their shown "me" details, about()) counts too.
    have = set(have_list) | {t for text in also_have for t in phrase_tokens(text)}
    if head == offered_head or (head in have_list and head in ROLE_WORDS):
        # Same role: the phrases themselves must agree. The wanted noun has to be what they are
        # ("a CV reviewer"), not just a word describing something else ("a CV review client").
        phrase = set(have_list)
        share = sum(1 for w in want if w in phrase) / len(want)
        about_it = _about(want, head)
        if exactness < 0.5:
            # A wide net: at least half of all its words in their phrase, as ever, and at least half of
            # what it is about in their phrase or in what they show about themselves.
            if share < 0.5:
                return False
            if about_it:
                return len(about_it & have) / len(about_it) >= 0.5
        elif about_it:
            # Each saying something the other doesn't is two different things, whatever the word order or
            # script (a role noun that comes first, as in "partenaire de ..."): only the same option, or one
            # that narrows the other, meets.
            if about_it - phrase and _content(have_list) - {head, offered_head} - set(want):
                return False
            return len(about_it & phrase) / len(about_it) > 0.5
        elif share <= 0.5:
            return False
        content = _content(want)
        return not content or bool(content & _content(have))
    # Wide net: a partner can stand in for a teacher (and so on) if they share what it's about.
    if exactness < 0.5 and offered_head in _ROLE_RELAX.get(head, ()):
        activity = [w for w in want if w not in ROLE_WORDS]
        if all(w in SOCIAL_WORDS for w in activity):
            return True  # "a hangout partner": anyone up for company qualifies
        return any(w in have for w in activity if w not in SOCIAL_WORDS)
    return False


def _head(tokens: list) -> str:
    """The main noun of a phrase: its last role word if it has one ("a pottery teacher who can
    also teach weaving" -> teacher), otherwise its last word ("a brass band" -> band)."""
    roles = [t for t in tokens if t in ROLE_WORDS]
    return roles[-1] if roles else tokens[-1]


# Umbrella words that say nothing about the activity itself ("winter sports", "music lessons").
# They never count as something two listings share.
_UMBRELLA = {"sport", "lesson", "class", "classe", "service", "job", "work", "activity", "activitie", "hobby",
             "hobbie", "skill", "thing", "general", "other", "beginner", "basic", "advanced", "help"}


def _related(a: str, b: str) -> bool:
    """The same word or stem: "repair" / "repairman", "snow" / "snowboard"."""
    if a in _UMBRELLA or b in _UMBRELLA:
        return False
    short, long_ = sorted((a, b), key=len)
    return a == b or (len(a) >= 5 and len(b) >= 5 and a[:5] == b[:5]) or (len(short) >= 4 and long_.startswith(short))


def _topic_words(topics) -> set:
    return {t for topic in (topics or ()) for t in phrase_tokens(str(topic))
            if t not in ROLE_WORDS and t not in SOCIAL_WORDS}


def nearly(wanted, offered, wanted_topics=(), offered_topics=(), own_offer=None) -> bool:
    """Is `offered` nearly what `wanted` asks for? Roles must line up (a coach is never nearly a
    student), and the two must share an activity: a word or stem in their phrases, or one of the
    broader topics each person's agent wrote for the listing (someone wanting a watercolour
    teacher and someone teaching oil painting might both list "painting"). With `own_offer` (the
    wanting person's own "you are"), every word they lack (_lacks: what they look for and aren't)
    besides its main noun, which is what may differ in a near match, must be the same word or stem
    as something the other person is or lists, as for an exact match: someone looking for "a
    second violinist" who is "a first violinist" is never nearly met by another first violinist.
    There is no built-in list of activities; shown fields only."""
    if not wanted or not offered:
        return False
    tw, th = _topic_words(wanted_topics), _topic_words(offered_topics)
    for w in alternatives(wanted):
        lacks = set() if own_offer is None else _lacks(w, own_offer) - _UMBRELLA - {_head(w)}
        for h in alternatives(offered):
            if any(t in ROLE_WORDS for t in w) and any(t in ROLE_WORDS for t in h) and _head(w) != _head(h):
                continue
            act_w, act_h = _content(w), _content(h)
            if not all(any(a == b or _related(a, b) for b in act_h | th) for a in lacks):
                continue
            if any(_related(a, b) for a in act_w | tw for b in act_h | th):
                return True
    return False


def looking_for(view) -> bool:
    """Is this person looking for something (a service, a thing) rather than offering it? Read from
    the headline their agent wrote ("Looking for...", "Need..."), else from casting themselves as the
    one being served (a student, a customer, a buyer)."""
    headline = str(view.get("headline") or "").lower()
    if headline.startswith(("looking for", "need", "searching for", "seeking", "wanting", "hoping")):
        return True
    if headline:
        return False
    offering = set(phrase_tokens(str(view.get("offering") or "")))
    return bool(offering & {"student", "customer", "client", "buyer", "recipient", "patient", "tenant"})


def phrase_overlap(wanted: str, offered: str) -> float:
    """Best share of wanted words found in the offered phrase, over the options (0..1)."""
    best = 0.0
    for want in alternatives(wanted):
        for have in alternatives(offered):
            best = max(best, sum(1 for w in want if w in set(have)) / len(want))
    return best


def about(view) -> list:
    """What someone shows about themselves beyond their phrases: the values of their shown "me" and
    "both" details (never a wish about the other person, and "any" is no value of its own)."""
    return list(view.own_values())


def has_phrases(view) -> bool:
    return view.get("seeking") is not None and view.get("offering") is not None


def _index_keys(tokens) -> set:
    """Narrowing's keys for some words: each word's first four letters (a shorter word whole). Two
    words that are the same, or the same stem by _related (one starts with the other, or both
    share their first five letters), always share a key."""
    return {t[:4] for t in tokens}


def index_words(view) -> tuple:
    """What narrowing indexes a listing under (narrow.shortlists, step 3), read with the same
    tokenisation the decisions below use, so any two listings that intent_met or nearly() could
    accept share at least one key:
      want: the words of each option in what it's looking for ("seeking"), and its topics;
      have: the words of what it is ("offering"), its topics, and what it shows about itself (about():
            the values of its shown "me" and "both" details).
    Two listings meet when one's want shares a key with the other's have.

    Why that is enough: phrase_satisfies needs a word the option is about (not a role or social
    word, and with the same head on both sides not the head either: _about) in the offered phrase or,
    for a wide net, in what they show about themselves; an option that is only its head needs that
    head in the offered phrase, unless the option is about nothing but its role; nearly() needs a word
    or stem in common between want and topics on one side and offer and topics on the other. So role words (teacher, student, partner)
    are left out of `want`: shared role words alone never meet an option that says what it is about.
    An option with nothing but social words besides its role ("a hangout partner") can be met
    without any shared word, so such a listing says so (`open`) and is compared with every phrase
    listing in reach. Open details' wishes (details_ok) and what an option lacks (_lacks) are an AND
    on top of these, so they only ever remove pairs: every pair a decision accepts still shares a key."""
    topics = view.get("topics")
    key = (view.get("seeking"), view.get("offering"), topics if isinstance(topics, (str, tuple)) else None,
           tuple(map(str, about(view))))
    out = _INDEX_WORDS.get(key)
    if out is None:  # many listings share their wording, and the board is read again every run
        if len(_INDEX_WORDS) > 50_000:
            _INDEX_WORDS.clear()
        out = _INDEX_WORDS[key] = _index_words(view)
    return out


_INDEX_WORDS: dict = {}  # shown wording -> index_words(), so each wording is read once


def _index_words(view) -> tuple:
    want, is_open = set(_topic_words(view.get("topics"))), False
    for option in alternatives(str(view.get("seeking") or "")):
        rest = [t for t in option if t not in ROLE_WORDS]
        want.update(rest)
        if len(rest) < len(option) and all(t in SOCIAL_WORDS for t in rest):
            is_open = True  # a role and nothing it's about: shared role words alone may meet it
    have = set(phrase_tokens(str(view.get("offering") or ""))) | _topic_words(view.get("topics"))
    have |= {t for text in about(view) for t in phrase_tokens(str(text))
             if t not in ROLE_WORDS and t not in SOCIAL_WORDS}
    return _index_keys(want), _index_keys(have), is_open


def effective_kind(parsed: ParsedIntent, view: ShownView) -> ParsedIntent:
    """A listing that shows its "looking for" and "you are" phrases is open-ended ("other") unless it
    shows the kind "dating", whatever its description's words suggest: the phrases say what it is for."""
    if not has_phrases(view) or parsed.kind == "other" or view.get("kind") == "dating":
        return parsed
    return read_listing(view.intent_text, lambda k: "other" if k == "kind" else view.get(k))


def understood(view: ShownView) -> ParsedIntent:
    """What a listing asks for, as every decision and narrowing reads it: the shown kind, and for
    dating the shown wants_gender / wants_age, first; its description only for what they don't say
    (intent.read_listing). Shown fields only."""
    return effective_kind(read_listing(view.intent_text, view.get), view)


# A wide net (exactness under 0.5) also accepts a step next to the ones it asked for, on the identical
# scale (schema.satisfies widens only there).
WIDE_NET_STEPS = 1


def details_ok(av: ShownView, bv: ShownView) -> bool:
    """Are A's shown wishes met by B? For each of A's shown "them" and "both" details, B's own shown
    detail of that name ("me" or "both"). A wish that says nothing (schema.says_nothing: "any", or every
    step of its scale) is always met, so it never asks B to show that detail. One B doesn't show, or
    that can't be read against the wish (another way of comparing, a step not on the wish's scale, or
    for a strict A an own "any", which says nothing: schema.comparable), meets it only if A said
    "ideally" (not required) and isn't strict: a required wish never meets a detail it can't compare,
    so a detail named or scaled differently costs a match but never makes a wrong one. Otherwise B's
    value must satisfy the wish (schema.satisfies): a shared value, an own "any" (A not strict), or one
    of the wished steps, or for a wide net a step next to one on the identical scale. Shown fields only."""
    strict = av.exactness >= STRICT_EXACTNESS
    for want in av.wishes():
        if says_nothing(want):
            continue
        theirs = bv.own(want.name)
        if theirs is None or not comparable(want, theirs, strict):
            if want.required or strict:
                return False
            continue
        if not satisfies(want, theirs, WIDE_NET_STEPS if av.exactness < 0.5 else 0):
            return False
    return True


def phrases_ok(av: ShownView, bv: ShownView) -> bool:
    """Is what A is looking for what B is? Both show their phrases, and B's "you are" meets A's
    "looking for" (phrase_satisfies, with what B shows about themselves, and with A's own "you
    are" so that what A looks for and isn't must be what B is)."""
    return has_phrases(av) and has_phrases(bv) and phrase_satisfies(
        av.get("seeking"), bv.get("offering"), av.exactness, about(bv), own_offer=av.get("offering"))


def _dating_met(ap: ParsedIntent, av: ShownView, bv: ShownView) -> bool:
    """Dating's own rules: B's shown gender is one A wants, and B's shown age band is in A's range."""
    gender, band = bv.get("gender"), bv.get("age_band")
    # Someone open to anyone doesn't need to know your gender; anyone else needs it shown (a hidden
    # gender never decides a match, so it can't count for them).
    anyone = set(GENDERS) <= set(ap.seeking)
    if band not in AGE_BANDS or not (anyone or gender in ap.seeking):
        return False
    # No age range said: not a strict preference (as with any detail they didn't say), so one age band
    # either side of their own shown band (default_age_range in intent.py).
    lo, hi = ap.age_range or default_age_range(av.get("age_band"))
    slack = 1 if av.exactness < 0.5 else 0
    return lo - slack <= AGE_BANDS.index(band) <= hi + slack


def intent_met(ap: ParsedIntent, av: ShownView, bp: ParsedIntent, bv: ShownView) -> bool:
    """Is A's intent met by B? The same kind of match; for dating its own rules, for anything else
    A's phrases (phrases_ok); and always every wish A shows (details_ok). Reads shown fields only
    (ShownView)."""
    if ap.kind is None or ap.kind != bp.kind:
        return False
    met = _dating_met(ap, av, bv) if ap.kind == "dating" else phrases_ok(av, bv)
    return met and details_ok(av, bv)


def detail_refs(av: ShownView, bv: ShownView) -> tuple:
    """The open details two listings have in common, as the reason refs the Warden works out itself
    (schema.detail_reasons, from both people's shown details; strict if either listing is)."""
    return detail_reasons(av.fields, bv.fields, max(av.exactness, bv.exactness) >= STRICT_EXACTNESS)


def reason_fields(kind: str, av: ShownView, bv: ShownView, online: bool = False, refs=None) -> tuple:
    """What genuinely connects two people, shown by both: where they can meet, the open details they
    have in common (refs, exactly detail_refs: the Warden accepts no subset), and for anything but
    dating their phrases. `online`: they meet online, so if they aren't near each other, how they
    meet is the reason ("format")."""
    keys = []
    place = within_reach(av, bv)
    if place:
        keys.append(place)
    elif online and av.get("format") is not None and bv.get("format") is not None:
        keys.append("format")
    keys += detail_refs(av, bv) if refs is None else refs
    if kind != "dating" and has_phrases(av) and has_phrases(bv):
        keys += ["seeking", "offering"]
        if av.get("headline") is not None and bv.get("headline") is not None:
            keys.append("headline")  # the reason says who they are in their own words
    return tuple(keys)


def shown_threshold(ex_a: float, ex_b: float) -> float:
    return 0.05 + 0.15 * max(ex_a, ex_b)


# Fair ranking (step 6). Small next to the differences between good and poor matches, so they only
# break near-ties: a listing introduced to many people today gives way to one nobody has seen yet,
# and a listing someone just updated is looked at first.
LOAD_PENALTY = 0.01  # per introduction either listing had in the last 24 hours
FRESH_BOOST = 0.02  # per listing updated within a day of the newest update on the board

_STAMP = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:[T ](\d{2}):(\d{2})(?::(\d{2}))?)?")


def _minutes(stamp):
    """Minutes since 1970 for an ISO time ("2026-10-12T09:00:00Z"; treated as UTC), or None. Plain
    arithmetic: matching has no clock and needs none."""
    m = _STAMP.match(str(stamp or ""))
    if not m:
        return None
    y, mo, d, h, mi, s = (int(x or 0) for x in m.groups())
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return None
    y -= mo <= 2  # days from the civil calendar (March-based years)
    era, yoe = divmod(y, 400)
    doy = (153 * (mo + (-3 if mo > 2 else 9)) + 2) // 5 + d - 1
    days = era * 146097 + yoe * 365 + yoe // 4 - yoe // 100 + doy - 719468
    return days * 1440 + h * 60 + mi + s / 60


_TIMES: dict = {}  # time text -> minutes (most listings share a few update times)


def recently_updated(listings) -> set:
    """Listings updated within a day of the newest update on the board (no wall clock)."""
    times = {}
    for l in listings:
        stamp = str(l.updated_at)
        t = _TIMES.get(stamp, _TIMES)
        if t is _TIMES:
            if len(_TIMES) > 100_000:
                _TIMES.clear()
            t = _TIMES[stamp] = _minutes(stamp)
        times[l.person_id] = t
    known = [t for t in times.values() if t is not None]
    if not known:
        return set()
    newest = max(known)
    return {pid for pid, t in times.items() if t is not None and newest - t < 24 * 60}


# The shown score's term for open details in common: full at DETAILS_FULL of them.
DETAILS_WEIGHT, DETAILS_FULL = 0.5, 2


def _after_yes(listing: Listing, other: ShownView) -> ShownView:
    """How a listing would read after its person says yes to "Would that work for you?" about the
    other listing: what the other looks for joins what it offers, and what the other is joins what it
    looks for (as the person's agent writes it, agent.also_offer). Only its phrases change: none of
    its details, so a wish it shows still has to be met."""
    add = {"offering": other.get("seeking"), "seeking": other.get("offering")}
    fields = []
    for f in listing.fields:
        extra = add.get(f.key)
        both = f"{f.value} or {extra}"
        if isinstance(extra, str) and extra.lower() not in str(f.value).lower() and len(both) <= FREE_TEXT_MAX:
            f = replace(f, value=both)
        fields.append(f)
    return ShownView(replace(listing, fields=tuple(fields)))


@dataclass
class Reading:
    """One run's reading of the board (OfflineMatcher.read): the active listings, their shown views and what each
    asks for, the age-range questions, each focus listing's shortlist, the pairs to decide (sorted, a < b, never one
    person's two listings) and the crowd counts. Internal to matching."""
    listings: dict
    views: dict
    parsed: dict
    followups: list
    lists: dict
    index: object
    pairs: list
    crowd: object


class OfflineMatcher:
    def __init__(self, k_crowd: int = K_CROWD, max_per_person: int = 3, shortlist_size: int = 40,
                 hidden_bonus: float = 0.04, hidden_cap: float = 0.12, word_index: bool = True):
        self.k_crowd = k_crowd
        self.max_per_person = max_per_person
        self.shortlist_size = shortlist_size
        self.hidden_bonus = hidden_bonus
        self.hidden_cap = hidden_cap
        # Narrow phrase listings by the words they share (step 3). False: one catch-all group, as
        # before the index, kept so the two can be compared (tools/compare_narrowing.py).
        self.word_index = word_index

    # -- scoring ---------------------------------------------------------------

    def shown_score(self, index, av: ShownView, bv: ShownView, refs=None) -> float:
        """How well two listings fit, from shown fields only: how alike they read, how many open
        details they have in common (detail_refs, capped so piling on details can't raise rank), and
        for phrase listings how much of what each wants the other's phrase covers."""
        text = cosine(index.embed(av.intent_text), index.embed(bv.intent_text))
        refs = detail_refs(av, bv) if refs is None else refs
        score = 0.5 * text + DETAILS_WEIGHT * min(len(refs), DETAILS_FULL) / DETAILS_FULL
        if has_phrases(av) and has_phrases(bv):
            # How much of what each wants the other's "you are" phrase covers (shown fields).
            score += 0.25 * (phrase_overlap(av.get("seeking"), bv.get("offering")) +
                             phrase_overlap(bv.get("seeking"), av.get("offering"))) / 2
        return score

    def hidden_overlap(self, a: Listing, b: Listing, crowd) -> tuple:
        """Keys both listings have, hidden on at least one, where they share a value (compared
        canonically) that k other people have too: crowd counts different people per (key, canonical
        value) (Marketboard.value_counts), and the pair themselves are two of them. Never a wish about
        the other person ("them"). This is the only place the Matchmaker reads hidden values."""
        used = []
        for fa in a.fields:
            fb = b.get(fa.key)
            if fb is None or split_key(fa.key)[0] == "them" or HIDDEN not in (fa.visibility, fb.visibility):
                continue
            shared = {canonical(v) for v in fa.strings()} & {canonical(v) for v in fb.strings()}
            if any(crowd[(fa.key, s)] >= self.k_crowd + 2 for s in shared):
                used.append(fa.key)
        return tuple(sorted(used))

    # -- main ------------------------------------------------------------------

    def match(self, board: Marketboard, focus=None, load=None) -> MatchResult:
        """focus: only find matches and questions for these listings (against everyone), e.g. the
        demo's two phones, instead of for every listing on the board.
        load: {listing id: introductions it got in the last 24 hours}, for fair ranking (step 6).
        It only changes the order of pairs that already qualify. In a focused run, the focus
        listings' own load is left out.

        In three steps that the hybrid Matchmaker (hybrid.LLMMatcher) shares: read the board (read), decide
        each shortlisted pair (decide), then pick and ask (finish)."""
        run = self.read(board, focus)
        candidates, also_offer = [], {}
        for a, b in run.pairs:
            proposal, offers = self.decide(run, a, b)
            if proposal is not None:
                candidates.append(proposal)
            for pid, fu in offers:
                also_offer.setdefault(pid, fu)  # one per person, the first pair's
        return self.finish(board, run, candidates, also_offer, focus, load)

    def read(self, board: Marketboard, focus=None) -> "Reading":
        """Steps 1 and 2: what each active listing asks for, read from shown fields, its shortlist, and the pairs
        to decide (never two listings of one person), with the age-range questions to ask."""
        listings = {l.person_id: l for l in board.listings() if l.active}  # paused listings aren't matched
        views = {pid: ShownView(l) for pid, l in listings.items()}
        # What each listing asks for: its shown kind and wants first, then its description and shown
        # headline (never a hidden field). Narrowing groups by this same kind.
        parsed = {pid: understood(views[pid]) for pid in listings}

        followups = []
        for pid, p in parsed.items():
            if focus is not None and pid not in focus:
                continue
            if not listings[pid].notify_questions:
                continue  # only people who opted in are asked questions
            if p.kind == "dating" and p.age_range is None:
                followups.append(FollowUp(pid, "age_range"))  # to narrow the default (one band either side)

        lists, index = shortlists(parsed, views, self.shortlist_size, focus,
                                  words=index_words if self.word_index else None)
        # A person's own listings are never paired with each other.
        pairs = {tuple(sorted((a, b))) for a, cands in lists.items() for b in cands
                 if listings[a].owner != listings[b].owner}
        return Reading(listings, views, parsed, followups, lists, index, sorted(pairs), board.value_counts())

    def decide(self, run: "Reading", a: str, b: str) -> tuple:
        """Steps 3 to 5 for one shortlisted pair (a < b): (its Proposal or None, the "also offer" questions it
        suggests, as (listing id, FollowUp) in the order they are offered)."""
        listings, views, parsed = run.listings, run.views, run.parsed
        av, bv = views[a], views[b]
        a_met = intent_met(parsed[a], av, parsed[b], bv)
        b_met = intent_met(parsed[b], bv, parsed[a], av)
        close = ""
        if not (a_met and b_met):
            # A close match: one side fully meets the other's intent, and the other is only
            # nearly met (it would be met with a wide net, which may also look one step either
            # side of a wished step). Only for someone who opted in. Either way every wish both
            # people show still holds: a required one is never ignored.
            for near, far in ((a, b), (b, a)):
                far_met = b_met if far == b else a_met
                if not (far_met and listings[near].notify_close):
                    continue
                wide = ShownView(replace(listings[near], exactness=CLOSE_EXACTNESS))
                nv, fv = views[near], views[far]
                if intent_met(parsed[near], wide, parsed[far], fv) or (
                        # a related activity: if the other person takes questions, asking
                        # them (below) is better, since a yes makes a match that fits both
                        not listings[far].notify_questions
                        and nearly(nv.get("seeking"), fv.get("offering"), nv.get("topics"), fv.get("topics"),
                                   nv.get("offering"))
                        and details_ok(wide, fv)):
                    close = near
            if not close:
                return None, self._offers(run, a, b)  # both must be looking
        # Narrowing pairs only listings of the same kind of match, so either one's kind will do.
        refs = detail_refs(av, bv)
        reasons = reason_fields(parsed[a].kind, av, bv, how_they_meet(parsed[a], av, parsed[b], bv) == ONLINE, refs)
        if not reasons:
            return None, []  # never the only reason: needs a shown reason
        s = self.shown_score(run.index, av, bv, refs)
        threshold = shown_threshold(CLOSE_EXACTNESS if close == a else av.exactness,
                                    CLOSE_EXACTNESS if close == b else bv.exactness)
        if s < threshold:
            return None, []  # never the only reason: shown fields must justify it
        used, bonus = self.hidden_part(run, a, b)
        return Proposal(a, b, a_met, b_met, round(s + bonus, 4), round(s, 4), round(threshold, 4), reasons, used,
                        close), []

    def hidden_part(self, run: "Reading", a: str, b: str) -> tuple:
        """Step 5: the hidden fields that raise a pair's score (hidden_overlap, crowd-gated) and the bonus they give."""
        used = self.hidden_overlap(run.listings[a], run.listings[b], run.crowd)
        return used, min(self.hidden_cap, self.hidden_bonus * len(used))

    def _offers(self, run: "Reading", a: str, b: str) -> list:
        """Nearly: what one offers is close to what the other wants. One person is asked at a time, whoever is
        looking for something first ("Someone is offering oil painting lessons. Would that work for you?"); the
        other only if they say no. A yes makes the match."""
        listings, views, parsed = run.listings, run.views, run.parsed
        av_, bv_ = views[a], views[b]
        la, lb = listings[a], listings[b]
        # Cheap checks first: someone must be open to a question about the other.
        possible = (la.notify_questions and lb.notify_close) or (lb.notify_questions and la.notify_close)
        near = possible and has_phrases(av_) and has_phrases(bv_) and how_they_meet(
            parsed[a], av_, parsed[b], bv_) and (
            nearly(av_.get("seeking"), bv_.get("offering"), av_.get("topics"), bv_.get("topics"),
                   av_.get("offering"))
            or nearly(bv_.get("seeking"), av_.get("offering"), bv_.get("topics"), av_.get("topics"),
                      bv_.get("offering")))
        if not near:
            return []
        order = sorted((a, b), key=lambda pid: not looking_for(views[pid]))
        # Each question needs the asked person's opt-in to questions and the other's to
        # close matches (their headline is quoted, never who they are), and is asked only
        # if a yes would make a match both ways: a yes only widens the asked person's
        # phrases, so a wish either of them shows must already hold.
        other_of = {order[0]: order[1], order[1]: order[0]}
        allowed = [p for p in order if listings[p].notify_questions
                   and listings[other_of[p]].notify_close
                   and self._yes_would_match(p, other_of[p], listings, views, parsed)]
        if not allowed:
            return []
        first = allowed[0]
        other = b if first == a else a
        offers = [(first, FollowUp(first, "also_offer", about=other))]
        if len(allowed) > 1:
            offers.append((other, FollowUp(other, "also_offer", about=first, after=first)))
        return offers

    def finish(self, board: Marketboard, run: "Reading", candidates: list, also_offer: dict, focus=None,
               load=None) -> MatchResult:
        """Step 6 and the questions: the best pairs among the candidates (every one already passed every check
        above), and the questions to ask (age range, "also offer", and "include close matches?")."""
        listings, followups = run.listings, list(run.followups)
        has_match = {pid for p in candidates for pid in (p.person_a, p.person_b)}
        for pid, fu in sorted(also_offer.items()):
            if pid not in has_match and fu.about not in has_match:
                followups.append(fu)
        # No match yet and close matches off: ask whether to include them (the Warden asks only once).
        # A yes turns on close matches only; questions keep their own choice. Close matches already look a little wider.
        for pid in sorted(listings):
            if focus is not None and pid not in focus:
                continue
            if not listings[pid].notify_close and pid not in has_match and pid not in also_offer:
                followups.append(FollowUp(pid, "broaden"))

        # Exact matches first; a close match only fills a spare slot, at most one per listing.
        # Close matches are ranked on shown details, never a hidden one (fair ranking, TRUST.md #32,
        # may reorder them), so a hidden detail never decides which close match someone hears about.
        # Fair ranking (only among pairs that passed every check above): a little lower for each
        # introduction either listing had in the last day, a little higher for a listing updated in
        # the last day.
        fresh = recently_updated(board.listings())
        load = load or {}
        if focus is not None:
            # A focused run (the demo's phones): a focus listing's own load is the same for every one
            # of its candidates, so it would change nothing about whom it meets, except to push two
            # focus listings down for each other. Only the other side's load counts.
            load = {pid: n for pid, n in load.items() if pid not in focus}

        def rank(p: Proposal) -> float:
            base = p.shown_score if p.close else p.score
            return (base - LOAD_PENALTY * (load.get(p.person_a, 0) + load.get(p.person_b, 0))
                    + FRESH_BOOST * ((p.person_a in fresh) + (p.person_b in fresh)))

        chosen, count, close_sent = [], {}, set()
        for prop in sorted(candidates, key=lambda p: (bool(p.close), -rank(p), p.person_a, p.person_b)):
            if prop.close and prop.close in close_sent:
                continue
            if count.get(prop.person_a, 0) < self.max_per_person and count.get(prop.person_b, 0) < self.max_per_person:
                if prop.close:
                    close_sent.add(prop.close)
                chosen.append(prop)
                count[prop.person_a] = count.get(prop.person_a, 0) + 1
                count[prop.person_b] = count.get(prop.person_b, 0) + 1

        assert all(f.question_id in FOLLOWUP_MENU for f in followups)
        return MatchResult(proposals=chosen, followups=followups)

    @staticmethod
    def _yes_would_match(asked: str, other: str, listings: dict, views: dict, parsed: dict) -> bool:
        """Would a yes from `asked` to "Would that work for you?" about `other` make a match both ways?"""
        after = _after_yes(listings[asked], views[other])
        return (intent_met(parsed[asked], after, parsed[other], views[other])
                and intent_met(parsed[other], views[other], parsed[asked], after))

