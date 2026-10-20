"""Matchmaker agent (SPEC §3, §5, §6, §8.3).

Open source, and still treated as hostile: it needs no trust, because it can only hand
the Warden the objects defined in contract.py, and the Warden re-checks every one.

Pluggable: anything with `match(board) -> MatchResult` is a Matchmaker.
  - OfflineMatcher: rules + TF-IDF, no API key, no network.
  - LLMMatcher: placeholder for a model-based matcher (Claude API on fake data first,
    then a self-hosted model inside the enclave).

How OfflineMatcher decides, in order:
  1. Read each intent into structure (intent.py). Missing pieces become fixed-menu
     follow-up questions.
  2. Narrow: shortlist candidates per person (narrow.py).
  3. Both must be looking: keep a pair only if A's intent is met by B *and* B's by A,
     judged from shown fields only.
  4. Never the only reason: the pair needs at least one reason field shown by both,
     and a shown-only score above the threshold.
  5. Hidden fields may only raise the score, and only values at least k people share.
  6. Pick the best pairs, at most `max_per_person` each.
"""
from __future__ import annotations

import re
from dataclasses import replace

from ..schema import AGE_BANDS, FOLLOWUP_MENU, HIDDEN, Listing
from .contract import FollowUp, MatchResult, Matcher, Proposal  # noqa: F401 (re-exported)
from .intent import ParsedIntent, parse_intent
from .marketboard import Marketboard
from .narrow import ShownView, same_place, shortlists
from .text import cosine

LEVEL_SCALES = {"level": ["beginner", "intermediate", "advanced"], "pace": ["easy", "moderate", "fast"]}


def schedules_overlap(x, y) -> bool:
    return x is not None and y is not None and (x == y or "flexible" in (x, y))


_PHRASE_STOP = {"a", "an", "the", "my", "our", "your", "for", "to", "of", "and", "or", "in", "on", "with",
                "who", "someone", "somebody", "person", "people", "some", "new", "join", "find", "looking",
                "can", "could", "also", "that", "which", "will", "would", "me", "us", "them", "is", "are", "be",
                "help", "speaking", "speak"}


# Role words that mean the same thing (after the suffix-stripping below).
_SYNONYMS = {
    "mentor": "teach", "coach": "teach", "instructor": "teach", "tutor": "teach", "guide": "teach",
    "learn": "student", "begin": "student", "apprentice": "student", "pupil": "student", "novice": "student",
    "partn": "partner", "buddy": "partner", "friend": "partner", "companion": "partner", "pal": "partner",
}


LANGUAGE_NAMES = {"english", "french", "spanish", "german", "japanese", "portuguese", "italian", "chinese",
                  "mandarin", "cantonese", "korean", "irish", "arabic", "hindi", "dutch", "russian", "polish",
                  "turkish", "greek", "swedish"}
# A close match is judged as if its person had cast a wide net.
CLOSE_EXACTNESS = 0.2

ROLE_WORDS = {"teach", "student", "partner"}
# Purely social wording: someone who just wants company is open to any activity.
SOCIAL_WORDS = {"social", "hangout", "hang", "out", "meet", "meetup", "company", "chat", "casual", "fun", "time"}
# With a wide net, these roles may stand in for each other when the activity overlaps:
# a learner happy to practise with a peer, a peer happy to help a learner.
_ROLE_RELAX = {"teach": {"partner"}, "partner": {"student", "teach"}, "student": {"partner"}}


def phrase_tokens(text: str) -> list:
    """Content words of a short phrase, crudely normalised so 'guitarist', 'guitarists'
    and 'guitar' agree, as do 'drummer' and 'drum'."""
    out = []
    for w in re.findall(r"[a-z]+", text.lower()):
        if w in _PHRASE_STOP or len(w) < 2:
            continue
        for suffix in ("ists", "ist", "ers", "er", "ing", "s"):
            if len(w) > len(suffix) + 2 and w.endswith(suffix):
                w = w[: -len(suffix)]
                break
        if len(w) > 3 and w[-1] == w[-2]:
            w = w[:-1]  # drumm -> drum, beginn -> begin
        w = _SYNONYMS.get(w, w)
        out.append(w)
        if w in LANGUAGE_NAMES:
            out.append("language")  # "French" is also a language
    return out


_ALTERNATIVE = re.compile(r"\s+or\s+|\s*;\s*", re.I)


def alternatives(text: str) -> list:
    """Split "a German tutor or a German language exchange partner" into the options the
    person is open to, each as a token list. Short options borrow what they leave out from
    the others: "a German or Spanish tutor" -> [german ... tutor], [spanish ... tutor];
    "a German tutor or exchange partner" -> the second also gets "german"."""
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
            p = p + [roles[-1]]  # "a German" (or Spanish tutor) -> "a German tutor"
        out.append(list(dict.fromkeys(p)))
    return out


def phrase_satisfies(wanted: str, offered: str, exactness: float, also_have=()) -> bool:
    """Does `offered` give what `wanted` asks for? True if any option the person is open to
    is met by any option the other person is (see alternatives)."""
    return any(_satisfies(w, h, exactness, also_have)
               for w in alternatives(wanted) for h in alternatives(offered))


def _satisfies(want: list, have_list: list, exactness: float, also_have=()) -> bool:
    """One option against one option. The main noun (e.g. 'guitarist' in 'a ska guitarist')
    must match, and more than half of the wanted words must appear (at least half for a
    wide net)."""
    if not want or not have_list:
        return False
    head, offered_head = _head(want), _head(have_list)
    if _language_clash(want, have_list):
        return False  # a German tutor is no help to someone learning Spanish
    if head in have_list:
        # Same role: the phrases themselves must agree.
        phrase = set(have_list)
        share = sum(1 for w in want if w in phrase) / len(want)
        return share >= 0.5 if exactness < 0.5 else share > 0.5
    # Wide net: a partner can stand in for a teacher (and so on) if they share an activity.
    # Here what the other person shows about themselves (interests, languages) counts too.
    if exactness < 0.5 and offered_head in _ROLE_RELAX.get(head, ()):
        have = set(have_list) | {t for text in also_have for t in phrase_tokens(text)}
        if _language_clash(want, have_list):
            return False
        activity = [w for w in want if w not in ROLE_WORDS]
        if all(w in SOCIAL_WORDS for w in activity):
            return True  # "a hangout partner": anyone up for company qualifies
        return any(w in have for w in activity if w not in SOCIAL_WORDS)
    return False


def _language_clash(want: list, have: list) -> bool:
    """Both name specific languages, and none of them agree."""
    want_langs, have_langs = set(want) & LANGUAGE_NAMES, set(have) & LANGUAGE_NAMES
    return bool(want_langs and have_langs and not want_langs & have_langs)


def _head(tokens: list) -> str:
    """The main noun of a phrase: its last role word if it has one ("a French teacher who can
    also teach chess" -> teacher), otherwise its last word ("a ska band" -> band)."""
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


def nearly(wanted, offered, wanted_topics=(), offered_topics=()) -> bool:
    """Is `offered` nearly what `wanted` asks for? Roles must line up (a coach is never nearly a
    student), and the two must share an activity: a word or stem in their phrases, or one of the
    broader topics each person's agent wrote for the listing (someone wanting an ice hockey
    coach and a figure skating coach might both list "ice skating"). There is no built-in list
    of activities; shown fields only."""
    if not wanted or not offered:
        return False
    tw, th = _topic_words(wanted_topics), _topic_words(offered_topics)
    for w in alternatives(wanted):
        for h in alternatives(offered):
            if _language_clash(w, h):
                continue
            if any(t in ROLE_WORDS for t in w) and any(t in ROLE_WORDS for t in h) and _head(w) != _head(h):
                continue
            act_w = {t for t in w if t not in ROLE_WORDS and t not in SOCIAL_WORDS}
            act_h = {t for t in h if t not in ROLE_WORDS and t not in SOCIAL_WORDS}
            if any(_related(a, b) for a in act_w | tw for b in act_h | th):
                return True
    return False


def phrase_overlap(wanted: str, offered: str) -> float:
    """Best share of wanted words found in the offered phrase, over the options (0..1)."""
    best = 0.0
    for want in alternatives(wanted):
        for have in alternatives(offered):
            best = max(best, sum(1 for w in want if w in set(have)) / len(want))
    return best


def about(view) -> list:
    """What someone shows about themselves beyond their phrases (interests, languages, subject)."""
    out = []
    for key in ("interests", "native_language", "learning_language", "subject"):
        value = view.get(key)
        if value:
            out.extend([value] if isinstance(value, str) else list(value))
    return out


def has_phrases(view) -> bool:
    return view.get("seeking") is not None and view.get("offering") is not None


def phrases_apply(ap: ParsedIntent, av, bp: ParsedIntent, bv) -> bool:
    """Any two non-dating listings that both carry "looking for" / "you are" phrases can be
    compared by those phrases, whatever kind each was filed under."""
    return has_phrases(av) and has_phrases(bv) and "dating" not in (ap.kind, bp.kind)


def effective_kind(parsed: ParsedIntent, view: ShownView) -> ParsedIntent:
    """A listing with 'seeking' + 'offering' phrases and no recognisable named kind is an
    open-ended ("other") listing."""
    if parsed.kind is None and has_phrases(view):
        return replace(parsed, kind="other")
    return parsed


def intent_met(ap: ParsedIntent, av: ShownView, bp: ParsedIntent, bv: ShownView) -> bool:
    """Is A's intent met by B? Reads shown fields only (ShownView)."""
    if phrases_apply(ap, av, bp, bv):
        if phrase_satisfies(av.get("seeking"), bv.get("offering"), av.exactness, about(bv)):
            return True
        if "other" in (ap.kind, bp.kind) or ap.kind != bp.kind:
            return False  # no structured rule to fall back on
    if ap.kind is None or ap.kind != bp.kind or ap.kind == "other":
        return False
    ex = av.exactness
    kind = ap.kind
    if kind == "dating":
        gender, band = bv.get("gender"), bv.get("age_band")
        if gender is None or band is None or ap.age_range is None or gender not in ap.seeking:
            return False
        slack = 1 if ex < 0.5 else 0
        lo, hi = ap.age_range
        return lo - slack <= AGE_BANDS.index(band) <= hi + slack
    if kind == "language":
        native = bv.get("native_language")
        return native is not None and native == av.get("learning_language")
    if kind == "cofounder":
        return ap.seeks_role is not None and bv.get("role") == ap.seeks_role
    if kind == "tutoring":
        subject = bv.get("subject")
        return (ap.seeks_role is not None and bv.get("role") == ap.seeks_role
                and subject is not None and subject == av.get("subject"))
    if kind in ("climbing", "running"):
        key = "level" if kind == "climbing" else "pace"
        mine, theirs = av.get(key), bv.get(key)
        if mine is None or theirs is None:
            return ex < 0.5  # a wide net tolerates an unknown level
        tolerance = 0 if ex >= 0.8 else 1
        if abs(LEVEL_SCALES[key].index(mine) - LEVEL_SCALES[key].index(theirs)) > tolerance:
            return False
        if ex >= 0.8 and not schedules_overlap(av.get("schedule"), bv.get("schedule")):
            return False
        return True
    if kind == "roommate":
        return True
    return False


def reason_fields(kind: str, av: ShownView, bv: ShownView) -> tuple:
    """Fields shown by both people that genuinely connect them."""
    keys = []
    place = same_place(av, bv)
    if place:
        keys.append(place)
    if schedules_overlap(av.get("schedule"), bv.get("schedule")):
        keys.append("schedule")
    a_int, b_int = av.get("interests"), bv.get("interests")
    if a_int and b_int and set(a_int) & set(b_int):
        keys.append("interests")
    both = lambda k: av.get(k) is not None and bv.get(k) is not None
    if kind == "climbing" and both("level"):
        keys.append("level")
    if kind == "running" and both("pace"):
        keys.append("pace")
    if kind == "language" and both("native_language") and both("learning_language"):
        keys += ["native_language", "learning_language"]
    if kind == "tutoring" and both("subject") and both("role"):
        keys += ["subject", "role"]
    if kind == "cofounder" and both("role"):
        keys.append("role")
    if kind == "roommate" and both("move_in") and av.get("move_in") == bv.get("move_in"):
        keys.append("move_in")
    if both("seeking") and both("offering") and kind != "dating":
        keys += ["seeking", "offering"]
        if both("headline"):
            keys.append("headline")  # the reason says who they are in their own words
    return tuple(keys)


def shown_threshold(ex_a: float, ex_b: float) -> float:
    return 0.05 + 0.15 * max(ex_a, ex_b)


class OfflineMatcher:
    def __init__(self, k_crowd: int = 20, max_per_person: int = 3, shortlist_size: int = 40,
                 hidden_bonus: float = 0.04, hidden_cap: float = 0.12):
        self.k_crowd = k_crowd
        self.max_per_person = max_per_person
        self.shortlist_size = shortlist_size
        self.hidden_bonus = hidden_bonus
        self.hidden_cap = hidden_cap

    # -- scoring ---------------------------------------------------------------

    def shown_score(self, index, av: ShownView, bv: ShownView) -> float:
        text = cosine(index.embed(av.intent_text), index.embed(bv.intent_text))
        a_int, b_int = set(av.get("interests") or ()), set(bv.get("interests") or ())
        jaccard = len(a_int & b_int) / len(a_int | b_int) if a_int | b_int else 0.0
        sched = 1.0 if schedules_overlap(av.get("schedule"), bv.get("schedule")) else 0.0
        score = 0.5 * text + 0.3 * jaccard + 0.2 * sched
        if has_phrases(av) and has_phrases(bv):
            # How much of what each wants the other's "you are" phrase covers (shown fields).
            score += 0.25 * (phrase_overlap(av.get("seeking"), bv.get("offering")) +
                             phrase_overlap(bv.get("seeking"), av.get("offering"))) / 2
        return score

    def hidden_overlap(self, a: Listing, b: Listing, crowd) -> tuple:
        """Hidden keys where A and B share a value that at least k people have.
        This is the only place the Matchmaker reads hidden values."""
        used = []
        for fa in a.fields:
            fb = b.get(fa.key)
            if fb is None or HIDDEN not in (fa.visibility, fb.visibility):
                continue
            shared = set(fa.strings()) & set(fb.strings())
            if any(crowd[(fa.key, s)] >= self.k_crowd for s in shared):
                used.append(fa.key)
        return tuple(sorted(used))

    # -- main ------------------------------------------------------------------

    def match(self, board: Marketboard) -> MatchResult:
        listings = {l.person_id: l for l in board.listings()}
        views = {pid: ShownView(l) for pid, l in listings.items()}
        parsed = {pid: effective_kind(parse_intent(l.intent_text), views[pid]) for pid, l in listings.items()}

        followups = []
        for pid, p in parsed.items():
            if not listings[pid].notify_questions:
                continue  # only people who opted in are asked questions
            v = views[pid]
            if p.kind == "dating" and p.age_range is None:
                followups.append(FollowUp(pid, "age_range"))
            elif p.kind in ("climbing", "running") and v.exactness >= 0.8 and v.get("schedule") is None:
                followups.append(FollowUp(pid, "schedule"))

        lists, index = shortlists(parsed, views, self.shortlist_size)
        # A person's own listings are never paired with each other.
        pairs = {tuple(sorted((a, b))) for a, cands in lists.items() for b in cands
                 if listings[a].owner != listings[b].owner}
        crowd = board.value_counts()

        candidates, has_match, also_offer = [], set(), {}
        for a, b in sorted(pairs):
            av, bv = views[a], views[b]
            a_met = intent_met(parsed[a], av, parsed[b], bv)
            b_met = intent_met(parsed[b], bv, parsed[a], av)
            close = ""
            if not (a_met and b_met):
                # A close match: one side fully meets the other's intent, and the other is only
                # nearly met (it would be met with a wide net). Only for someone who opted in.
                for near, far in ((a, b), (b, a)):
                    far_met = b_met if far == b else a_met
                    if far_met and listings[near].notify_close and (intent_met(
                            parsed[near], ShownView(replace(listings[near], exactness=CLOSE_EXACTNESS)),
                            parsed[far], views[far]) or (
                            # a related activity: if the other person takes questions, asking
                            # them (below) is better, since a yes makes a match that fits both
                            not listings[far].notify_questions
                            and nearly(views[near].get("seeking"), views[far].get("offering"),
                                       views[near].get("topics"), views[far].get("topics")))):
                        close = near
                if not close:
                    # Nearly: one person offers something close to what the other wants, and
                    # would welcome them. Ask if they could help with that too, without saying who.
                    for wants, could in ((a, b), (b, a)):
                        wants_met, could_met = (a_met, b_met) if wants == a else (b_met, a_met)
                        wv, cv = views[wants], views[could]
                        if (not wants_met and listings[wants].notify_close and listings[could].notify_questions
                                and has_phrases(wv) and has_phrases(cv) and same_place(wv, cv)
                                and nearly(wv.get("seeking"), cv.get("offering"), wv.get("topics"), cv.get("topics"))
                                and (could_met or nearly(cv.get("seeking"), wv.get("offering"),
                                                         cv.get("topics"), wv.get("topics")))):
                            also_offer.setdefault(could, wants)
                    continue  # both must be looking
            reasons = reason_fields(parsed[a].kind, av, bv)
            if not reasons:
                continue  # never the only reason: needs a shown reason
            s = self.shown_score(index, av, bv)
            threshold = shown_threshold(CLOSE_EXACTNESS if close == a else av.exactness,
                                        CLOSE_EXACTNESS if close == b else bv.exactness)
            if s < threshold:
                continue  # never the only reason: shown fields must justify it
            has_match.update((a, b))
            used = self.hidden_overlap(listings[a], listings[b], crowd)
            bonus = min(self.hidden_cap, self.hidden_bonus * len(used))
            candidates.append(Proposal(a, b, a_met, b_met, round(s + bonus, 4), round(s, 4),
                                       round(threshold, 4), reasons, used, close))

        for could, wants in sorted(also_offer.items()):
            if could not in has_match and could not in {f.person_id for f in followups}:
                followups.append(FollowUp(could, "also_offer", about=wants))
        asked = {f.person_id for f in followups}
        for pid, v in views.items():
            if (pid not in has_match and pid not in asked and parsed[pid].kind and v.exactness >= 0.5
                    and listings[pid].notify_questions):
                followups.append(FollowUp(pid, "wider_net"))

        # Exact matches first; a close match only fills a spare slot, at most one per listing.
        # Close matches are ranked on shown details alone, so a hidden detail never decides which
        # close match someone hears about.
        chosen, count, close_sent = [], {}, set()
        for prop in sorted(candidates, key=lambda p: (bool(p.close), -(p.shown_score if p.close else p.score),
                                                      p.person_a, p.person_b)):
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


class LLMMatcher:
    """Placeholder (SPEC §8.3). Same interface as OfflineMatcher.

    Plan: use the Claude API on synthetic data only, to test whether a model reads
    free-text intents better than the rules. For real people the model must run
    inside the enclave (self-hosted), because matching has no network access. It must still
    return only Proposal/FollowUp objects, so the Warden's exit checks are unchanged.
    """

    def match(self, board: Marketboard) -> MatchResult:
        raise NotImplementedError("LLMMatcher is planned; use OfflineMatcher for now.")
