"""Two-person demo (proof of concept for the front end).

Two people, A and B, join a service already holding synthetic listings (5,000 when served;
BELONG_PEOPLE changes it). Each person
can hold several listings at once (say, one looking for a tutor and one looking for
love). Each listing has its own conversation with the person's agent, its own draft and
its own introductions, and is matched on its own; a person is never introduced to their
own other listing.

Everything Belong. receives or sends is recorded in a data log for the presenter:

  in       Belong. received something (KYC facts, a listing, a decision, consents)
  out      Belong. sent something to a person, through the Warden (exact payload logged)
  blocked  the Warden refused to let something in or out
  inside   private matching activity (presenter view only; never sent to anyone)

("The room" in code comments is internal shorthand for private matching; it is never
shown to users.)

Privacy rule for this module: `phone_state(slot)` must contain only what that person
may know. Room internals go only into `room_state()`, the presenter's panel.
"""
from __future__ import annotations

import copy
import hashlib
import os
import random
import re
import threading
from datetime import datetime, timezone
from dataclasses import replace

from . import agent
from .agent_llm import AgentUnavailable, ClaudeAgent
from .agent_openai import OpenAIAgent
from .leaks import leak_report
from .matching.marketboard import Marketboard
from .matching.contract import FollowUp, Proposal
from .matching.matcher import OfflineMatcher
from .schema import AGE_BANDS, FIELD_REGISTRY, FOLLOWUP_MENU, HIDDEN, SHOWN, Field, Listing, Person, age_band_index, value_to_json
from .synth import generate
from .warden import Clock, Warden

SLOTS = ("A", "B")
K_CROWD = 20
APPROACH_LIMIT = 15  # introductions per listing per day (the Warden's limit)
PER_RUN = 3  # at most this many new introductions per listing in one Matchmaker run, so nobody gets them all at once
LOOK_PAST = 1000  # how far down each listing's ranking the Matchmaker looks, past people already introduced
SHORTLIST = 400  # candidates per phone listing; only the two phones are matched each run, so it can be generous
MAX_LISTINGS = 5
MESSAGE_MAX = 1000  # characters in one message to a match
REACTIONS = ("👍", "❤️", "😂", "😮", "🙏")  # the only reactions a message can get


# Words for the person being served; a listing looking for one is named by what it offers.
_CLIENT_WORDS = {"student", "students", "customer", "customers", "client", "clients", "learner", "learners",
                 "beginner", "beginners", "pupil", "pupils", "patient", "patients", "tenant", "tenants"}
# Listings without a "who you're looking for" phrase are named by their kind.
KIND_TITLES = {"dating": "Dating", "climbing": "Climbing partner", "running": "Running buddy",
               "language": "Language exchange", "roommate": "Roommate", "cofounder": "Cofounder", "tutoring": "Tutoring"}


def _join(items: list) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


class DemoError(ValueError):
    """A request the demo can't carry out (shown to the user as a message)."""


class _Listing:
    """One of a person's listings: its own chat with their agent, draft and introductions."""

    def __init__(self, slot: str, n: int):
        self.n = n
        self.id = f"demo{slot}-{n}"  # the listing's id in matching
        self.chat = []  # [{"from": "agent"|"me", "text": str}]
        self.agent_conversation = []  # the agent's own history for this listing
        self.agent_busy = False
        self.draft = None
        self.entry = None  # {"accepted": bool, "problems": [...]}
        self.followups = []  # pending [{"question_id", "question_text"}]
        self.intro_ids = []
        self.my_consents = {}  # intro_id -> set(keys)
        self.reveal_results = {}  # intro_id -> [{"key", "values"}]
        # Both off until the person says yes to "Want to include close matches?" (or uses the gear).
        self.prefs = {"notify_close": False, "notify_questions": False}
        self.asked = set()  # (question_id, about) already answered: never asked again


def _current(attr):
    """A _Slot attribute that reads and writes the current listing's value."""
    return property(lambda self: getattr(self.listing(), attr),
                    lambda self, value: setattr(self.listing(), attr, value))


class _Slot:
    """One phone: one verified person, any number of listings."""

    def __init__(self, slot: str):
        self.slot = slot
        self.person_id = f"demo{slot}"  # the person (owner of the listings)
        self.person = None
        self.name = None
        self.listings: list[_Listing] = []
        self.current = None  # id of the listing being edited
        self.receipts = []

    def listing(self, listing_id=None) -> _Listing:
        listing_id = listing_id or self.current
        found = next((l for l in self.listings if l.id == listing_id), None)
        if found is None:
            raise DemoError("No such listing.")
        return found

    # The current listing's state, for code (and tests) that work with one listing at a time.
    chat = _current("chat")
    draft = _current("draft")
    entry = _current("entry")
    followups = _current("followups")
    intro_ids = _current("intro_ids")
    my_consents = _current("my_consents")
    reveal_results = _current("reveal_results")
    agent_conversation = _current("agent_conversation")
    agent_busy = _current("agent_busy")


class DemoSession:
    def __init__(self, seed: int = 7, agent="auto", people: int = 300):
        """agent: "auto" picks a model-powered agent when one is configured (else the rule-based
        agent), None forces the rule-based agent (tests), or pass any object with respond().
        people: how many synthetic people share the board with the two phones."""
        self.lock = threading.Lock()
        self.people = people
        self._agent_choice = agent
        self.generation = 0
        # People who use Belong. through their own AI assistant (the connector). Their matches are
        # told, because what they write to that person passes through the assistant's provider.
        self.assistant_slots: set = set()
        self.run_inline = False  # tests: run agent work directly instead of on a thread
        self.reset(seed)

    def _make_agent(self):
        """Pick the personal agent. BELONG_AGENT=openai|claude|rules forces one; otherwise
        OpenAI if OPENAI_API_KEY is set, then Claude if configured, else the rule-based agent."""
        if self._agent_choice is None:
            return None, "Rule-based agent (offline)"
        if self._agent_choice != "auto":
            return self._agent_choice, "Model-powered agent"
        wanted = os.environ.get("BELONG_AGENT", "auto").strip().lower()
        if wanted == "rules":
            return None, "Rule-based agent (BELONG_AGENT=rules)"
        reasons = []
        if wanted in ("auto", "openai"):
            try:
                llm = OpenAIAgent()
                return llm, f"OpenAI agent ({llm.model})"
            except AgentUnavailable as e:
                reasons.append(str(e))
        if wanted in ("auto", "claude"):
            try:
                llm = ClaudeAgent()
                return llm, f"Claude agent ({llm.model})"
            except AgentUnavailable as e:
                reasons.append(str(e))
            except Exception as e:  # e.g. the SDK found no credentials
                reasons.append(f"Claude not configured ({type(e).__name__})")
        return None, "Rule-based agent: " + "; ".join(reasons)

    # -- setup -----------------------------------------------------------------

    def reset(self, seed: int = 7) -> None:
        self.seed = seed
        self.generation += 1
        self.llm, self.agent_status = self._make_agent()
        self.rng = random.Random(seed)
        self.personas = {p.person.id: p for p in generate(self.people, seed)}
        self.clock = Clock()
        self.warden = Warden(self.clock, key=hashlib.sha256(f"demo-audit-{seed}".encode()).digest(),
                             approach_limit_per_day=APPROACH_LIMIT)
        self.board = Marketboard()
        for p in self.personas.values():
            self.warden.admit(p.person, agent.draft_listing(p, self.clock.now()), self.board)
        self._seed_receipts = len(self.warden.receipts)
        # The Matchmaker doesn't remember who was introduced, so it ranks everyone; the demo skips pairs
        # already introduced and sends the Warden at most PER_RUN new ones per listing each run.
        self.matcher = OfflineMatcher(k_crowd=K_CROWD, max_per_person=LOOK_PAST, shortlist_size=SHORTLIST)
        self.slots = {s: _Slot(s) for s in SLOTS}
        self.events = []
        self.intros = {}  # intro_id -> Introduction
        self.declined = set()  # (listing_id, about): said no to "Would that work for you?" about that listing
        self.intro_reasons = {}  # intro_id -> the reason fields it was made with
        self.threads = {}  # intro_id -> [{"from": listing_id, "text"}]: messages between two connected people
        self.introduced_pairs = set()
        self.decisions = {}  # (intro_id, listing_id) -> bool
        self.consents = {}  # (intro_id, listing_id) -> set
        self.connections = set()
        self.outputs = {"followups": [], "reveals": []}
        self.last_proposals = []
        self.last_run = None  # demo time the Matchmaker last ran (it runs on its own when something changes)
        self.changes = 0  # goes up with every action anyone takes (the server counts them)
        # Held back by the Warden: not sent again until something changes, so a stuck pair doesn't
        # add receipts every few seconds. pair -> (versions, failed checks, when)
        self.held_back = {}
        self.held_questions = {}  # (listing, question, about) -> listing version it was held back at
        self.at_limit = {}  # listing -> when a pair with it hit the daily introduction limit
        self._event("inside", "Matching service started",
                    f"{len(self.board.listings())} synthetic listings, all accepted by the Warden's entry check.")

    def _event(self, kind: str, title: str, text: str = "", payload=None, receipts=None, slots=()):
        self.events.append({"n": len(self.events), "kind": kind, "title": title, "text": text,
                            "payload": payload, "receipts": receipts or [], "slots": list(slots),
                            "time": self.clock.now()})

    def _receipts_since(self, start: int, slot=None) -> list:
        new = [r.to_dict() for r in self.warden.receipts[start:]]
        for s in ([slot] if slot else []):
            self.slots[s].receipts += new
        return new

    def _slot(self, slot: str) -> _Slot:
        if slot not in self.slots:
            raise DemoError("unknown phone")
        return self.slots[slot]

    def _verified(self, slot: str) -> _Slot:
        st = self._slot(slot)
        if not st.person:
            raise DemoError("Verify your identity first.")
        return st

    def _owner_of(self, listing_id: str):
        """(slot, listing) for a demo listing id, or (None, None) for a synthetic person."""
        for s, st in self.slots.items():
            for l in st.listings:
                if l.id == listing_id:
                    return s, l
        return None, None

    def _slot_of(self, listing_id: str):
        return self._owner_of(listing_id)[0]

    def _name(self, listing_id: str) -> str:
        if listing_id in self.board:
            f = self.board.listing(listing_id).get("first_name")
            if f:
                return f.value
        return listing_id

    @classmethod
    def _live_title(cls, listing) -> str:
        """The title of what is actually live (what matching reads), not of a draft."""
        draft = {"kind": None, "intent_text": listing.intent_text,
                 "fields": [{"key": f.key, "value": value_to_json(f.value)} for f in listing.fields]}
        return cls._title(draft, listing)

    @staticmethod
    def _live_values(listing) -> dict:
        return {"intent_text": [listing.intent_text, None], **{f.key: [value_to_json(f.value), f.visibility] for f in listing.fields}}

    @staticmethod
    def _draft_values(draft) -> dict:
        out = {"intent_text": [(draft or {}).get("intent_text", ""), None]}
        for f in (draft or {}).get("fields", []):
            v = f.get("value")
            if f.get("enabled", True) and v not in (None, "", []):
                out[f["key"]] = [list(v) if isinstance(v, (list, tuple)) else v, f.get("visibility")]
        return out

    @staticmethod
    def _sync_draft(mine, listing) -> None:
        """After a save, the draft is exactly what went live (including edits made by hand)."""
        if not mine.draft:  # a listing made without the agent: start its draft from what's live
            mine.draft = {"intent_text": "", "exactness": listing.exactness, "kind": None, "fields": [], "understood": "",
                          "notes": [], "missing": [], "complete": False, "by": "sync"}
        live = {f.key: f for f in listing.fields}
        fields = []
        for f in mine.draft["fields"]:
            g = live.get(f["key"])
            if g is not None:
                f = {**f, "value": value_to_json(g.value), "visibility": g.visibility, "enabled": True}
            elif f.get("value") not in (None, "", []):
                f = {**f, "value": None}
            fields.append(f)
        known = {f["key"] for f in fields}
        for key, g in live.items():
            if key not in known:
                fields.append({"key": key, "label": agent.LABELS.get(key, key), "value": value_to_json(g.value),
                               "visibility": g.visibility, "sensitive": g.sensitive, "multi": FIELD_REGISTRY[key].multi,
                               "enabled": True, "locked": key == "age_band", "options": []})
        mine.draft = {**mine.draft, "intent_text": listing.intent_text, "exactness": listing.exactness, "fields": fields}

    def _unsaved(self, mine) -> bool:
        """A live listing whose draft has changes that aren't live yet."""
        if mine.id not in self.board or not mine.draft:
            return False
        return self._draft_values(mine.draft) != self._live_values(self.board.listing(mine.id))

    @staticmethod
    def _what_changed(old, new) -> list:
        """Plain words for what an update changed, for the agent to tell the person."""
        words = {"seeking": "who you're looking for", "offering": "what you offer", "headline": "your headline",
                 "topics": "related topics", "intent_text": "your description", "city": "your city",
                 "country": "your country", "first_name": "your name"}
        before, after = (DemoSession._live_values(x) for x in (old, new))
        out = []
        for key in dict.fromkeys(list(after) + list(before)):
            if before.get(key) == after.get(key):
                continue
            label = words.get(key) or agent.LABELS.get(key, key).lower()
            if key in before and key in after and before[key][0] == after[key][0]:
                out.append(f"{label} is now {after[key][1]}")
            elif key not in after:
                out.append(f"removed {label}")
            else:
                out.append(label)
        return out

    @staticmethod
    def _title(draft, board_listing=None) -> str:
        """A short name for a listing, from what it's looking for."""
        fields = {f["key"]: f["value"] for f in (draft or {}).get("fields", []) if f.get("value")}
        if fields.get("headline"):
            return str(fields["headline"])[:60]  # the person's own words
        seeking, offering = fields.get("seeking"), fields.get("offering")
        if seeking and offering and _CLIENT_WORDS & set(re.findall(r"[a-z]+", str(seeking).lower())):
            return str(offering)[:60]  # a teacher looking for students: name it by what they offer
        if seeking:
            return str(seeking)[:60]
        kind = (draft or {}).get("kind") or ""
        if kind in KIND_TITLES:
            return KIND_TITLES[kind]
        text = (draft or {}).get("intent_text") or (board_listing.intent_text if board_listing else "")
        return (text[:57] + "…") if len(text) > 60 else (text or "New listing")

    # -- 1. identity check (simulated KYC) ---------------------------------------

    def verify(self, slot: str, name: str, age) -> None:
        st = self._slot(slot)
        name = (name or "").strip()[:30]
        try:
            age = int(age)
        except (TypeError, ValueError):
            raise DemoError("Please enter your age as a number.")
        if not name:
            raise DemoError("Please enter your first name.")
        if not 18 <= age <= 99:
            raise DemoError("You must be 18 or older.")
        band = AGE_BANDS[age_band_index(age)]
        st.person = Person(id=st.person_id, verified=True, age_band=band, created_at=self.clock.now())
        st.name = name
        self._event("in", f"Identity check for phone {slot}",
                    "The KYC provider saw the ID document. Belong. received only these two facts:",
                    payload={"verified": True, "age_band": band}, slots=[slot])

    def update_profile(self, slot: str, name: str) -> None:
        """Change the first name shown on all of this person's listings. Live listings are
        re-checked by the Warden. The age band comes from the ID check and can't change."""
        st = self._verified(slot)
        name = (name or "").strip()[:30]
        if not name:
            raise DemoError("Please enter your first name.")
        st.name = name
        for mine in st.listings:
            if mine.draft:
                mine.draft = {**mine.draft, "fields": [{**f, "value": name} if f["key"] == "first_name" else f
                                                       for f in mine.draft["fields"]]}
            if mine.id in self.board:
                current = self.board.listing(mine.id)
                fields = tuple(replace(f, value=name) if f.key == "first_name" else f for f in current.fields)
                start = len(self.warden.receipts)
                result = self.warden.admit(st.person, replace(current, fields=fields, updated_at=self.clock.now(),
                                                              version=current.version + 1), self.board)
                if result.accepted:
                    self._refresh_intros(mine.id)
                self._event("in" if result.accepted else "blocked", f"{name} updated their profile",
                            "Re-checked by the Warden like any update." if result.accepted else "; ".join(result.problems),
                            receipts=self._receipts_since(start, slot), slots=[slot])

    def _refresh_intros(self, listing_id: str) -> None:
        """After a listing changes, the Warden reissues its introductions so the other person
        sees its current shown details (a new name, say) and nothing since hidden."""
        for intro_id, intro in list(self.intros.items()):
            if listing_id not in (intro.person_a, intro.person_b):
                continue
            start = len(self.warden.receipts)
            new = self.warden.refresh_introduction(intro_id, self.intro_reasons.get(intro_id, []), self.board)
            receipts = self._receipts_since(start)
            for s in (self._slot_of(intro.person_a), self._slot_of(intro.person_b)):
                if s:
                    self.slots[s].receipts += receipts
            if new is not None:
                self.intros[intro_id] = new

    # -- listings --------------------------------------------------------------------

    @staticmethod
    def _clean_prefs(prefs) -> dict:
        prefs = prefs if isinstance(prefs, dict) else {}
        return {k: prefs.get(k) is True for k in ("notify_close", "notify_questions")}

    def new_listing(self, slot: str) -> str:
        """Start another listing (e.g. a tutor search alongside dating). It gets its own chat.
        Close matches and questions start off; Belong. offers them once if no match turns up."""
        st = self._verified(slot)
        if len(st.listings) >= MAX_LISTINGS and not any(self._untouched(l) for l in st.listings):
            raise DemoError(f"You can have up to {MAX_LISTINGS} listings.")
        # An untouched draft is reused rather than piling up empty listings.
        empty = next((l for l in st.listings if self._untouched(l)), None)
        if empty is not None:
            st.current = empty.id
            return empty.id
        listing = _Listing(slot, max((l.n for l in st.listings), default=0) + 1)
        listing.chat = [{"from": "agent", "text": f"Hi {st.name}, I'm your agent. I work only for you, and nothing is "
                                                  "shared without your OK. Tell me what you're looking for, in your own words."
                         if not st.listings else "What else are you looking for? I'll keep this listing separate "
                                                 "from your others, with its own details."}]
        st.listings.append(listing)
        st.current = listing.id
        return listing.id

    def set_active(self, slot: str, listing_id: str, active) -> None:
        """Pause or resume a live listing. Paused, it isn't matched: no new introductions or
        questions (the Matchmaker skips it and the Warden refuses them). Existing matches stay."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        if mine.id not in self.board:
            raise DemoError("Only a live listing can be paused.")
        current = self.board.listing(mine.id)
        updated = replace(current, active=active is True, updated_at=self.clock.now(), version=current.version + 1)
        start = len(self.warden.receipts)
        result = self.warden.admit(st.person, updated, self.board)
        if result.accepted and not updated.active:
            mine.followups = []
        self._event("in" if result.accepted else "blocked", f"{st.name} {'resumed' if updated.active else 'paused'} a listing",
                    "Paused listings aren't matched." if not updated.active else "Matching again.",
                    receipts=self._receipts_since(start, slot), slots=[slot])

    def delete_listing(self, slot: str, listing_id: str) -> None:
        """Delete a listing: off the board (the Warden withdraws it) and gone from the person's list."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        start = len(self.warden.receipts)
        if mine.id in self.board and not self.warden.withdraw(st.person, mine.id, self.board):
            raise DemoError("That listing couldn't be deleted.")
        st.listings.remove(mine)
        st.current = st.listings[-1].id if st.listings else None
        self._event("in", f"{st.name} deleted a listing", "It's off the board and won't be matched again.",
                    receipts=self._receipts_since(start, slot), slots=[slot])

    def set_prefs(self, slot: str, listing_id: str, prefs) -> None:
        """Change a listing's notification choices. A live listing is re-checked with them."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        mine.prefs = self._clean_prefs(prefs)
        if mine.id in self.board:
            current = self.board.listing(mine.id)
            updated = replace(current, notify_close=mine.prefs["notify_close"],
                              notify_questions=mine.prefs["notify_questions"],
                              updated_at=self.clock.now(), version=current.version + 1)
            start = len(self.warden.receipts)
            result = self.warden.admit(st.person, updated, self.board)
            if result.accepted:
                self._refresh_intros(mine.id)
            self._event("in" if result.accepted else "blocked", f"{st.name} changed their notification choices",
                        f"Close matches {'on' if mine.prefs['notify_close'] else 'off'}, questions "
                        f"{'on' if mine.prefs['notify_questions'] else 'off'}.",
                        receipts=self._receipts_since(start, slot), slots=[slot])
        self._drop_moot_questions(mine)

    @staticmethod
    def _drop_moot_questions(mine: _Listing) -> None:
        """Questions that no longer apply: any, once questions are off; "include close matches?" once
        they're on. That one question stays while both are off: it's how a person turns them on."""
        def still_applies(q):
            return not mine.prefs["notify_close"] if q["question_id"] == "broaden" else mine.prefs["notify_questions"]
        mine.followups = [q for q in mine.followups if still_applies(q)]

    def _untouched(self, listing: _Listing) -> bool:
        return listing.draft is None and listing.id not in self.board and not any(m["from"] == "me" for m in listing.chat)

    def _listing_for(self, st: _Slot, listing_id=None) -> _Listing:
        """The listing a message is about; the first message creates one if there is none yet."""
        if not listing_id and not st.listings:
            self.new_listing(st.slot)
        return st.listing(listing_id)

    def open_listing(self, slot: str, listing_id: str) -> None:
        st = self._verified(slot)
        st.current = st.listing(listing_id).id

    # -- 2. talking to the agent -------------------------------------------------

    def prefill(self, slot: str, text: str, listing_id=None) -> None:
        """Instant profile update as soon as a message is sent, before the agent replies:
        facts stated plainly enough to be sure of go into empty fields (never overwriting)."""
        st = self._verified(slot)
        listing = self._listing_for(st, listing_id)
        facts = agent.quick_facts((text or "")[:2000])
        if not facts:
            return
        if listing.draft is None:
            listing.draft = agent.empty_draft(st.name, st.person.age_band)
        else:
            listing.draft = copy.deepcopy(listing.draft)
        agent.fill_gaps(listing.draft, facts)
        if not listing.draft.get("intent_text"):
            # Until the agent writes one, their own words describe what they want (never shown).
            listing.draft["intent_text"] = " ".join((text or "").split())[:500]

    def agent_message(self, slot: str, text: str, listing_id=None) -> None:
        """The person writes to their agent about one listing. Called WITHOUT the session lock:
        a model's reply can take a few seconds, and the other phone must keep working meanwhile."""
        with self.lock:
            st = self._verified(slot)
            listing = self._listing_for(st, listing_id)
            text = (text or "").strip()[:2000]
            if not text:
                raise DemoError("Say what you're looking for.")
            if listing.agent_busy:
                raise DemoError("Your agent is still replying.")
            listing.chat.append({"from": "me", "text": text})
            listing.agent_busy = True
            generation, llm = self.generation, self.llm
            name, band, conversation = st.name, st.person.age_band, listing.agent_conversation
            current = copy.deepcopy(listing.draft)

        reply = draft = None
        if llm is not None:
            try:
                reply, draft = llm.respond(conversation, text, name, band, current)
            except AgentUnavailable as e:
                with self.lock:
                    if self.llm is llm:
                        self.llm, self.agent_status = None, f"Rule-based agent: {e}"

        with self.lock:
            if generation != self.generation:
                return  # the demo was reset while the agent was thinking
            listing.agent_busy = False
            if reply is None:
                self._rules_reply(st, listing, text)
                return
            if draft and draft.get("separate") and (listing.id in self.board or listing.draft):
                # The agent started a separate search: a new listing. The current one stays as it
                # was, and the conversation carries on in the new one.
                try:
                    new = st.listing(self.new_listing(slot))
                except DemoError as e:
                    listing.chat.append({"from": "agent", "text": f"I couldn't start another listing: {e}"})
                    return
                new.agent_conversation = list(conversation)
                new.chat = [{"from": "me", "text": text}, {"from": "agent", "text": reply, "draft_updated": True}]
                new.draft = draft
                listing.chat.append({"from": "agent", "text": "I've started a separate listing for that, so this one "
                                                              "stays as it is.", "moved_to": new.id})
                return
            if draft:
                # Keep anything already on the profile that the agent's draft left empty.
                if listing.draft:
                    agent.fill_gaps(draft, {f["key"]: f["value"] for f in listing.draft["fields"]
                                            if f.get("enabled") and f["value"] not in (None, "", [])
                                            and not f.get("sensitive") and f["key"] not in ("first_name", "age_band")})
                listing.draft = draft
            # The tick on the phone comes from what actually happened, not from what the agent says.
            listing.chat.append({"from": "agent", "text": reply, "draft_updated": bool(draft)})

    def _rules_reply(self, st: _Slot, listing: _Listing, text: str) -> None:
        """The offline, rule-based agent: drafts from this one message."""
        draft = agent.draft_from_message(text, st.name, st.person.age_band)
        listing.draft = draft
        hidden = sum(1 for f in draft["fields"] if f["value"] and f["enabled"] and f["visibility"] == HIDDEN)
        reply = f"Here's what I understood: {draft['understood']}. I've drafted your listing"
        reply += f", with {hidden} detail(s) hidden." if hidden else "."
        reply += " Review it: you decide what's shown and what stays hidden."
        listing.chat.append({"from": "agent", "text": reply})
        for note in draft["notes"]:
            listing.chat.append({"from": "agent", "text": note})

    def options(self, slot: str, kind, city) -> dict:
        return agent.field_options(kind, city)

    # -- 3. entering the room -----------------------------------------------------

    def submit(self, slot: str, body: dict, listing_id=None) -> None:
        st = self._verified(slot)
        mine = st.listing(listing_id)
        listing = self._listing_from_json(st, mine, body)
        before = self.board.listing(mine.id) if mine.id in self.board else None
        start = len(self.warden.receipts)
        result = self.warden.admit(st.person, listing, self.board)
        receipts = self._receipts_since(start, slot)
        mine.entry = {"accepted": result.accepted, "problems": result.problems}
        n_hidden = sum(1 for f in listing.fields if f.visibility == HIDDEN)
        title = self._title(mine.draft, listing)
        if result.accepted:
            mine.followups = []
            self._refresh_intros(mine.id)
            self._sync_draft(mine, listing)
            self._event("in", f"{st.name}'s listing accepted: {title}",
                        f"{len(listing.fields)} details ({n_hidden} hidden) passed the Warden's entry check.",
                        receipts=receipts, slots=[slot])
            if before is None:
                text = (f"Your listing is live: \u201c{self._live_title(listing)}\u201d. I'll tell you when Belong. finds "
                        "someone who is also looking for you.")
            else:
                changed = self._what_changed(before, listing)
                text = ("Saved, and live now. What changed: " + _join(changed) + "."
                        if changed else "Nothing changed, so your listing stays as it was.")
            mine.chat.append({"from": "agent", "text": text, "live": True})
        else:
            self._event("blocked", f"{st.name}'s listing was refused",
                        "; ".join(result.problems), receipts=receipts, slots=[slot])

    def _listing_from_json(self, st: _Slot, mine: _Listing, body: dict) -> Listing:
        fields, confirmed = [], []
        for f in body.get("fields", []):
            key, value, vis = f.get("key"), f.get("value"), f.get("visibility")
            spec = FIELD_REGISTRY.get(key)
            if spec is None or value in (None, "", []):
                continue
            if spec.multi:
                value = tuple(str(x) for x in (value if isinstance(value, list) else [value]))
            elif not isinstance(value, str):
                value = str(value)
            if spec.sensitive and vis == SHOWN and f.get("confirmed"):
                confirmed.append(key)
            fields.append(Field(key=key, value=value, visibility=vis, sensitive=spec.sensitive))
        version = self.board.listing(mine.id).version + 1 if mine.id in self.board else 1
        try:
            exactness = float(body.get("exactness", 0.6))
        except (TypeError, ValueError):
            exactness = -1.0  # the Warden's schema check will refuse it
        return Listing(person_id=mine.id, intent_text=str(body.get("intent_text", "")),
                       exactness=exactness, fields=tuple(fields), updated_at=self.clock.now(),
                       version=version, confirmed_shown=tuple(confirmed), owner_id=st.person_id,
                       notify_close=mine.prefs["notify_close"], notify_questions=mine.prefs["notify_questions"],
                       active=self.board.listing(mine.id).active if mine.id in self.board else True)

    # -- 4. the Matchmaker runs ---------------------------------------------------

    def _demo_listing_ids(self) -> set:
        return {l.id for st in self.slots.values() for l in st.listings if l.id in self.board}

    def run_matchmaker(self, quiet: bool = False) -> bool:
        """quiet: the automatic run. It only leaves a trace in the presenter's log when something
        happens: a new introduction or question, or one the Warden held back. Returns whether it
        sent anything (then there may be more waiting for the next run)."""
        demo_ids = self._demo_listing_ids()
        if not demo_ids:
            raise DemoError("Neither phone has a listing yet.")
        events_before, sent_before = len(self.events), (len(self.intros), len(self.outputs["followups"]))
        self.clock.advance(minutes=10) if quiet else self.clock.advance(hours=1)
        self.last_run = self.clock.now()
        result = self.matcher.match(self.board, focus=demo_ids)  # only the phones' matches, not everyone's
        mine = [p for p in result.proposals if {p.person_a, p.person_b} & demo_ids]
        self.last_proposals = mine
        # New pairs only, a few per listing at a time, in the Matchmaker's own order.
        fresh, per = [], {}
        for p in mine:
            if frozenset((p.person_a, p.person_b)) in self.introduced_pairs or self._still_held(p):
                continue
            if per.get(p.person_a, 0) >= PER_RUN or per.get(p.person_b, 0) >= PER_RUN:
                continue
            per[p.person_a], per[p.person_b] = per.get(p.person_a, 0) + 1, per.get(p.person_b, 0) + 1
            fresh.append(p)
        self._event("inside", "Matchmaker ran",
                    f"Read all {len(self.board.listings())} listings, including hidden details. "
                    f"Proposed {len(result.proposals)} pairs; {len(fresh)} new ones involve the demo phones. "
                    f"Nothing it read can be sent to anyone except through the Warden's checks.",
                    payload=[self._proposal_summary(p) for p in fresh])
        noted = len(self.events)

        for fu in result.followups:
            slot, listing = self._owner_of(fu.person_id)
            if slot is None or any(q["question_id"] == fu.question_id for q in listing.followups):
                continue
            if (fu.question_id, fu.about) in listing.asked:
                continue  # answered before: not asked again
            # One person at a time: the second is asked only after the first said no. (A synthetic
            # person who'd be asked first declines, as they do in the simulation.)
            if fu.after and self._slot_of(fu.after) is not None and (fu.after, fu.person_id) not in self.declined:
                continue
            key = (fu.person_id, fu.question_id, fu.about)
            if self.held_questions.get(key) == self.board.listing(fu.person_id).version:
                continue  # held back before, and nothing has changed since
            self._exit_followup(fu, slot, listing)

        for p in fresh:
            pair = frozenset((p.person_a, p.person_b))
            if any(self._within_hour(self.at_limit.get(x)) for x in (p.person_a, p.person_b)):
                continue  # at the daily limit: wait an hour of demo time before trying again
            start = len(self.warden.receipts)
            intro = self.warden.exit_introduction(p, self.board)
            owners = [o for o in (self._owner_of(p.person_a), self._owner_of(p.person_b)) if o[0]]
            slots = [s for s, _ in owners]
            receipts = self._receipts_since(start, None)
            for s in slots:
                self.slots[s].receipts += receipts
            if intro is None:
                failed = [r["check"] for r in receipts if r["result"] == "fail"]
                self.held_back[pair] = (self._versions(p), tuple(failed), self.clock.now())
                if "approach_limit" in failed:
                    self.at_limit.update({p.person_a: self.clock.now(), p.person_b: self.clock.now()})
                self._event("blocked", f"Introduction {self._name(p.person_a)} + {self._name(p.person_b)} held back",
                            f"Failed: {', '.join(failed)}", receipts=receipts, slots=slots)
                continue
            self.held_back.pop(pair, None)
            self.introduced_pairs.add(pair)
            self.intros[intro.intro_id] = intro
            self.intro_reasons[intro.intro_id] = list(p.reason_fields)
            for _, listing in owners:
                listing.intro_ids.append(intro.intro_id)
                # "No matches yet" no longer applies.
                listing.followups = [q for q in listing.followups if q["question_id"] != "broaden"]
                # A one-way close match is only news to its recipient; the other person isn't told.
                if not intro.one_way_to or intro.one_way_to == listing.id:
                    self._tell(listing, "Belong. found someone for this listing. Go back to Belong. to see them.")
            if intro.one_way_to:
                to_a = intro.one_way_to == intro.person_a
                self._event("out", f"Close match for {self._name(intro.one_way_to)}",
                            f"Sent only to {self._name(intro.one_way_to)}, who opted in. "
                            f"{self._name(intro.person_b if to_a else intro.person_a)} isn't told unless they say yes:",
                            payload=[{"to": self._name(intro.one_way_to), "view": intro.a_view if to_a else intro.b_view}],
                            receipts=receipts, slots=slots)
                if self._slot_of(intro.one_way_to) is None:
                    # A synthetic recipient decides for themselves (simulated).
                    self._record(intro.intro_id, intro.one_way_to, self.rng.random() < 0.6)
                continue
            self._event("out", f"Introduction: {self._name(p.person_a)} + {self._name(p.person_b)}",
                        "This is everything sent out about this match:",
                        payload=[{"to": self._name(intro.person_a), "view": intro.a_view},
                                 {"to": self._name(intro.person_b), "view": intro.b_view}],
                        receipts=receipts, slots=slots)
        sent = (len(self.intros), len(self.outputs["followups"])) != sent_before
        if quiet and len(self.events) == noted:
            del self.events[events_before:]  # nothing happened: no trace in the log
            return sent
        if not mine and not any(l.followups for st in self.slots.values() for l in st.listings):
            self._event("inside", "No match yet for the demo phones",
                        "Nobody on the board meets their intent and is also looking for them.")
        return sent

    def _versions(self, p: Proposal) -> tuple:
        return tuple(self.board.listing(x).version if x in self.board else -1 for x in (p.person_a, p.person_b))

    def _still_held(self, p: Proposal) -> bool:
        """Held back before and nothing changed: neither listing was updated, and (for the daily
        introduction limit) less than an hour of demo time has passed, so it isn't sent again yet."""
        held = self.held_back.get(frozenset((p.person_a, p.person_b)))
        if held is None or held[0] != self._versions(p):
            return False
        return self._within_hour(held[2]) if "approach_limit" in held[1] else True

    def _within_hour(self, when) -> bool:
        if not when:
            return False
        from datetime import datetime, timedelta
        now = datetime.fromisoformat(self.clock.now().rstrip("Z"))
        return now - datetime.fromisoformat(when.rstrip("Z")) < timedelta(hours=1)

    @staticmethod
    def _tell(listing: _Listing, text: str) -> None:
        """A line from the person's own agent in that listing's chat (not sent to the model)."""
        listing.chat.append({"from": "agent", "text": text, "notice": True})

    def _exit_followup(self, fu: FollowUp, slot: str, listing: _Listing) -> None:
        start = len(self.warden.receipts)
        out = self.warden.exit_followup(fu, self.board)
        receipts = self._receipts_since(start, slot)
        name = self.slots[slot].name
        if out is None:
            failed = [r["check"] for r in receipts if r["result"] == "fail"]
            self.held_questions[(fu.person_id, fu.question_id, fu.about)] = self.board.listing(fu.person_id).version
            self._event("blocked", f"Question to {name} held back", f"Failed: {', '.join(failed)}",
                        receipts=receipts, slots=[slot])
            return
        d = out.to_dict()
        self.outputs["followups"].append(d)
        listing.followups.append({"question_id": d["question_id"], "question_text": d["question_text"],
                                  "offer_add": d["offer_add"], "seek_add": d["seek_add"], "pay": d["pay"], "about": fu.about})
        self._tell(listing, "Belong. has a question for this listing. Go back to Belong. to answer it.")
        self._event("out", f"Question to {name}", "A fixed-menu question, worded by the Warden:",
                    payload={"question": d["question_text"]}, receipts=receipts, slots=[slot])

    def _proposal_summary(self, p: Proposal) -> dict:
        return {"pair": f"{self._name(p.person_a)} + {self._name(p.person_b)}",
                "shown_score": p.shown_score, "threshold": p.shown_threshold, "score": p.score,
                "reason_fields": list(p.reason_fields), "hidden_fields_that_raised_it": list(p.used_hidden_fields)}

    def answer_followup(self, slot: str, question_id: str, answer: dict, listing_id=None) -> None:
        st = self._verified(slot)
        mine = st.listing(listing_id)
        if not any(q["question_id"] == question_id for q in mine.followups):
            raise DemoError("No such question.")
        listing = self.board.listing(mine.id)
        text, exactness, fields = listing.intent_text, listing.exactness, list(listing.fields)
        if question_id == "age_range":
            lo, hi = int(answer.get("min", 18)), int(answer.get("max", 99))
            if not 18 <= lo <= hi <= 99:
                raise DemoError("Pick an age range from 18 up.")
            text += f" Aged {lo} or older." if hi >= 99 else f" Aged {lo} to {hi}."
        elif question_id == "schedule":
            sched = answer.get("schedule")
            fields = [f for f in fields if f.key != "schedule"]
            fields.append(Field("schedule", str(sched), SHOWN, False))
        question = next(q for q in mine.followups if q["question_id"] == question_id)
        mine.followups = [q for q in mine.followups if q["question_id"] != question_id]
        mine.asked.add((question_id, question.get("about", "")))
        if question_id == "broaden":
            # A yes is the opt-in to both: close matches, and follow-up questions that could find one.
            if answer.get("yes"):
                self.set_prefs(slot, mine.id, {"notify_close": True, "notify_questions": True})
                mine.chat.append({"from": "agent", "text": "You said yes, so you'll hear about close matches too. "
                                                           "They're one-way: the other person isn't told. Belong. may "
                                                           "also ask you a short question that could find you a match."})
            return
        if question_id == "also_offer" and not answer.get("yes"):
            self.declined.add((mine.id, question.get("about", "")))  # now the other person may be asked
            return  # nothing changes for this listing, and nobody is told who said no
        updated = replace(listing, intent_text=text, exactness=exactness, fields=tuple(fields),
                          updated_at=self.clock.now(), version=listing.version + 1)
        if question_id == "also_offer":
            old_pay = updated.get("pay").value if updated.get("pay") else None
            updated = agent.also_offer(updated, question["offer_add"], self.clock.now(), question["seek_add"],
                                       question.get("pay", ""))
            new_pay = updated.get("pay").value if updated.get("pay") else None
            note = "You said yes, so I've updated your listing to match" + (
                f": the price is now {new_pay}." if new_pay and new_pay != old_pay else ".")
            if self.llm is not None:
                # The wording has to match too ("Selling..." isn't right once it's free). The agent
                # rewords it first, and the whole update goes live in one step, before matching sees it.
                mine.agent_busy = True
                self._spawn(lambda: self._reword_and_save(slot, mine.id, updated, question["question_text"], note))
                return
            mine.chat.append({"from": "agent", "text": note})
        self._save_answer(st, mine, updated)

    def _spawn(self, fn) -> None:
        """Run slow agent work off the session lock (tests set run_inline to run it directly)."""
        if self.run_inline:
            fn()
        else:
            threading.Thread(target=fn, daemon=True).start()

    def _reword_and_save(self, slot: str, listing_id: str, updated, question_text: str, note: str) -> None:
        """After a yes that changes the terms, the person's agent rewords the headline and description to
        match what they agreed to; then the whole update is checked by the Warden and goes live."""
        with self.lock:
            st, generation, llm = self._slot(slot), self.generation, self.llm
            mine = st.listing(listing_id)
            name, band, conversation = st.name, st.person.age_band, mine.agent_conversation
            self._sync_draft(mine, updated)  # the agent sees the listing as it will be
            current = copy.deepcopy(mine.draft)
        ask = (f'Belong. asked me: "{question_text}" and I said yes. Update my listing so its headline and '
               "description describe what I agreed to now. Keep everything else as it is, and keep "
               "separate_listing false.")
        draft = None
        try:
            _, draft = llm.respond(conversation, ask, name, band, current)
        except AgentUnavailable:
            pass
        with self.lock:
            if generation != self.generation or listing_id not in {l.id for l in st.listings}:
                return
            mine.agent_busy = False
            if draft and not draft.get("separate"):
                words = {f["key"]: f["value"] for f in draft["fields"] if f.get("value")}
                fields = tuple(replace(f, value=words["headline"]) if f.key == "headline" and words.get("headline") else f
                               for f in updated.fields)
                if words.get("headline") and not updated.get("headline"):
                    fields += (Field("headline", words["headline"], SHOWN, False),)
                updated = replace(updated, fields=fields, intent_text=draft.get("intent_text") or updated.intent_text)
                if updated.get("headline"):
                    note += f" It now reads “{updated.get('headline').value}”."
            mine.chat.append({"from": "agent", "text": note})
            self._save_answer(st, mine, updated)

    def _save_answer(self, st, mine, updated) -> None:
        start = len(self.warden.receipts)
        result = self.warden.admit(st.person, updated, self.board)
        receipts = self._receipts_since(start, st.slot)
        if result.accepted:
            self._refresh_intros(mine.id)
            self._sync_draft(mine, updated)  # the profile shows what's now live (a new price included)
        self._event("in" if result.accepted else "blocked", f"{st.name}'s answer updated their listing",
                    "Re-checked by the Warden like any update." if result.accepted else "; ".join(result.problems),
                    receipts=receipts, slots=[st.slot])

    # -- 5. decisions --------------------------------------------------------------

    def _listing_with_intro(self, st: _Slot, intro_id: str) -> _Listing:
        found = next((l for l in st.listings if intro_id in l.intro_ids), None)
        if found is None:
            raise DemoError("No such introduction.")
        return found

    def decide(self, slot: str, intro_id: str, yes: bool) -> None:
        st = self._verified(slot)
        mine = self._listing_with_intro(st, intro_id)
        if (intro_id, mine.id) in self.decisions:
            raise DemoError("You've already answered.")
        self._record(intro_id, mine.id, bool(yes))
        intro = self.intros[intro_id]
        other = intro.person_b if intro.person_a == mine.id else intro.person_a
        if yes and self._slot_of(other) is None and (intro_id, other) not in self.decisions:
            # A synthetic counterpart answers for themselves (simulated).
            self._record(intro_id, other, self.rng.random() < 0.8)
        if self.warden.is_connected(intro_id):
            self.connections.add(frozenset((intro.person_a, intro.person_b)))
            slots = [s for s in (self._slot_of(intro.person_a), self._slot_of(intro.person_b)) if s]
            self._event("out", f"{self._name(intro.person_a)} and {self._name(intro.person_b)} connected",
                        "Both said yes, so each can now be put in touch.", slots=slots)

    def _record(self, intro_id: str, listing_id: str, yes: bool) -> None:
        self.decisions[(intro_id, listing_id)] = yes
        self.warden.record_decision(intro_id, listing_id, yes)
        self._event("in", f"{self._name(listing_id)} said {'yes' if yes else 'no'}",
                    f"to introduction {intro_id}.", slots=[s for s in [self._slot_of(listing_id)] if s])

    # -- 6. mutual reveal ----------------------------------------------------------

    def reveal(self, slot: str, intro_id: str, keys: list) -> None:
        st = self._verified(slot)
        mine = self._listing_with_intro(st, intro_id)
        if not self.warden.is_connected(intro_id):
            raise DemoError("You can only compare details with someone you're connected with.")
        if intro_id in mine.my_consents:
            raise DemoError("You've already chosen.")
        own_hidden = {f.key for f in self.board.listing(mine.id).fields if f.visibility == HIDDEN}
        consent = {k for k in keys if k in own_hidden}
        mine.my_consents[intro_id] = consent
        self.consents[(intro_id, mine.id)] = consent
        self._event("in", f"{st.name} chose details to compare",
                    f"Picked {len(consent)} of {len(own_hidden)} private details (which ones stays private).",
                    slots=[slot])
        intro = self.intros[intro_id]
        other = intro.person_b if intro.person_a == mine.id else intro.person_a
        if self._slot_of(other) is None and (intro_id, other) not in self.consents:
            listing = self.board.listing(other)
            self.consents[(intro_id, other)] = {f.key for f in listing.fields
                                                if f.visibility == HIDDEN and self.rng.random() < 0.7}
        if (intro_id, intro.person_a) in self.consents and (intro_id, intro.person_b) in self.consents:
            self._settle_reveal(intro)

    def _settle_reveal(self, intro) -> None:
        start = len(self.warden.receipts)
        reveals = self.warden.exit_reveal(intro.intro_id, self.consents[(intro.intro_id, intro.person_a)],
                                          self.consents[(intro.intro_id, intro.person_b)], self.board)
        owners = [o for o in (self._owner_of(intro.person_a), self._owner_of(intro.person_b)) if o[0]]
        receipts = self._receipts_since(start)
        for s, _ in owners:
            self.slots[s].receipts += receipts
        results = [r.to_dict() for r in reveals]
        self.outputs["reveals"] += results
        for _, listing in owners:
            listing.reveal_results[intro.intro_id] = [{"key": r["key"], "values": r["values"]} for r in results]
        names = f"{self._name(intro.person_a)} and {self._name(intro.person_b)}"
        slots = [s for s, _ in owners]
        if results:
            self._event("out", f"{names} have more in common", "Both have it, and both chose to compare it:",
                        payload=[{"detail": agent.LABELS[r["key"]], "value": ", ".join(r["values"])} for r in results],
                        receipts=receipts, slots=slots)
        else:
            self._event("out", f"Compared details for {names}: nothing to show",
                        "Nothing they both have and both chose. Neither learns which.", receipts=receipts,
                        slots=slots)

    # -- 7. talking to a match -------------------------------------------------------

    def send_message(self, slot: str, intro_id: str, text: str) -> None:
        """A message to someone you're connected with. Only after both said yes (the Warden's
        record of decisions), delivered only to the two of them, and never seen by matching:
        threads are not on the Marketboard and nothing in matching can read them."""
        st = self._verified(slot)
        mine = self._listing_with_intro(st, intro_id)
        if not self.warden.is_connected(intro_id):
            raise DemoError("You can message someone once you've both said yes.")
        text = (text or "").strip()[:MESSAGE_MAX]
        if not text:
            raise DemoError("Write a message first.")
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.threads.setdefault(intro_id, []).append({"from": mine.id, "text": text, "at": at, "reactions": {}})
        intro = self.intros[intro_id]
        other = intro.person_b if intro.person_a == mine.id else intro.person_a
        # The presenter sees that a message was delivered, never what it says.
        self._event("out", f"{st.name} sent {self._name(other)} a message",
                    "Delivered only to the two of them. Matching never reads messages.",
                    slots=[s for s in (slot, self._slot_of(other)) if s])

    def react(self, slot: str, intro_id: str, index, emoji: str) -> None:
        """React to a message in a conversation with a match: one of a fixed few, one per
        person per message; choosing the same one again takes it back."""
        st = self._verified(slot)
        mine = self._listing_with_intro(st, intro_id)
        if not self.warden.is_connected(intro_id):
            raise DemoError("You can react once you've both said yes.")
        if emoji not in REACTIONS:
            raise DemoError("That reaction isn't available.")
        thread = self.threads.get(intro_id, [])
        if not isinstance(index, int) or not 0 <= index < len(thread):
            raise DemoError("No such message.")
        if thread[index]["from"] == mine.id:
            raise DemoError("You can react to their messages, not your own.")
        reactions = thread[index]["reactions"]
        if reactions.get(mine.id) == emoji:
            del reactions[mine.id]
        else:
            reactions[mine.id] = emoji

    # -- presenter: a compromised Matchmaker ---------------------------------------

    def attack(self) -> None:
        """Pretend the Matchmaker has been tampered with and tries to sneak data out.
        The Warden's fixed rules must stop each attempt."""
        ids = sorted(self._demo_listing_ids())
        if not ids:
            raise DemoError("Give at least one phone a listing first.")
        a_id = ids[0]
        a_slot, _ = self._owner_of(a_id)
        a_name = self.slots[a_slot].name
        owner = self.board.listing(a_id).owner
        b_id = next((i for i in ids if self.board.listing(i).owner != owner), None) or \
            next(l.person_id for l in self.board.listings() if l.owner != owner)
        listing = self.board.listing(a_id)
        hidden = [f.key for f in listing.fields if f.visibility == HIDDEN] or ["religion"]
        target = hidden[0]
        self._event("inside", "Simulating a compromised Matchmaker",
                    "It now tries several ways to get private details out.")

        attempts = [
            (f"Use {a_name}'s hidden '{agent.LABELS.get(target, target)}' as the reason for a match",
             lambda: self.warden.exit_introduction(
                 Proposal(a_id, b_id, True, True, 1.0, 1.0, 0.1, ("city", target), (target,)), self.board)),
            (f"Introduce {a_name} to someone who never asked for them",
             lambda: self.warden.exit_introduction(
                 Proposal(a_id, b_id, True, False, 1.0, 1.0, 0.1, ("city",), ()), self.board)),
            (f"Ask {a_name} an off-menu question to fish for details",
             lambda: self.warden.exit_followup(FollowUp(a_id, "what_is_your_religion"), self.board)),
        ]
        own_other = [i for i in ids if i != a_id and self.board.listing(i).owner == owner]
        if own_other:
            attempts.append((f"Introduce {a_name} to their own other listing",
                             lambda: self.warden.exit_introduction(
                                 Proposal(a_id, own_other[0], True, True, 1.0, 1.0, 0.1, ("city",), ()), self.board)))
        for title, attempt in attempts:
            start = len(self.warden.receipts)
            out = attempt()
            receipts = self._receipts_since(start)
            if out is not None:  # must never happen; make it loud if it does
                self._event("out", f"ATTACK GOT THROUGH: {title}", "This is a bug.", receipts=receipts)
            else:
                failed = [r["check"] for r in receipts if r["result"] == "fail"]
                self._event("blocked", f"Blocked: {title}", f"Stopped by the Warden's '{failed[-1]}' check.",
                            receipts=receipts)

    # -- views ---------------------------------------------------------------------

    def _listing_state(self, mine: _Listing) -> dict:
        listed = mine.id in self.board
        intros = []
        for intro_id in mine.intro_ids:
            intro = self.intros[intro_id]
            me_a = intro.person_a == mine.id
            other = intro.person_b if me_a else intro.person_a
            my_answer = self.decisions.get((intro_id, mine.id))
            theirs = self.decisions.get((intro_id, other))
            if intro.one_way_to and intro.one_way_to != mine.id and theirs is not True:
                continue  # a close match for the other person: you see nothing unless they say yes
            if my_answer is False:
                continue  # you said no: the match disappears from your list
            if my_answer is None:
                status = "pending"
            elif theirs is False:
                status = "closed"
            elif theirs is None:
                status = "waiting"
            else:
                status = "connected"
            item = {"intro_id": intro_id, "view": intro.a_view if me_a else intro.b_view, "status": status,
                    "reveal_chosen": intro_id in mine.my_consents,
                    "reveal_result": mine.reveal_results.get(intro_id),
                    "close": intro.one_way_to == mine.id}
            if status == "connected":
                item["thread"] = [{"mine": m["from"] == mine.id, "text": m["text"], "at": m["at"],
                                   "reactions": [{"emoji": e, "mine": who == mine.id} for who, e in m["reactions"].items()]}
                                  for m in self.threads.get(intro_id, [])]
                item["simulated"] = self._slot_of(other) is None  # a synthetic person can't reply
                item["their_assistant"] = self._slot_of(other) in self.assistant_slots
            intros.append(item)
        hidden_keys = []
        if listed:
            l = self.board.listing(mine.id)
            hidden_keys = [{"key": f.key, "label": agent.LABELS[f.key]} for f in l.fields if f.visibility == HIDDEN]
        return {"id": mine.id, "n": mine.n,
                "title": self._live_title(self.board.listing(mine.id)) if listed else self._title(mine.draft),
                "unsaved": self._unsaved(mine),
                "active": self.board.listing(mine.id).active if listed else True,
                "kind": (mine.draft or {}).get("kind"), "listed": listed, "draft": mine.draft, "chat": mine.chat,
                "agent_busy": mine.agent_busy, "entry": mine.entry,
                "intros": intros, "hidden_keys": hidden_keys, "prefs": mine.prefs,
                "followups": [{"question_id": q["question_id"], "question_text": q["question_text"]}
                              for q in mine.followups]}

    def phone_state(self, slot: str) -> dict:
        """Only what this person may know."""
        st = self._slot(slot)
        listings = [self._listing_state(l) for l in st.listings]
        current = next((l for l in listings if l["id"] == st.current), None) or \
            {"id": None, "listed": False, "draft": None, "chat": [], "agent_busy": False, "entry": None,
             "followups": [], "intros": [], "hidden_keys": []}
        return {"slot": slot, "verified": st.person is not None, "name": st.name,
                "age_band": st.person.age_band if st.person else None,
                "listings": listings, "current": st.current, "receipts": st.receipts[-60:],
                # The current listing, for clients that show one at a time.
                "chat": current["chat"], "draft": current["draft"], "entry": current["entry"],
                "listed": current["listed"], "followups": current["followups"], "agent_busy": current["agent_busy"],
                "hidden_keys": current["hidden_keys"],
                "intros": [i for l in listings for i in l["intros"]]}

    def room_state(self) -> dict:
        """Presenter view. Shows inside-the-room data for the demo listings only."""
        counts = self.board.value_counts()
        inside = []
        for st in self.slots.values():
            for mine in st.listings:
                if mine.id not in self.board:
                    continue
                l = self.board.listing(mine.id)
                rows = []
                for f in l.fields:
                    row = {"key": f.key, "label": agent.LABELS[f.key], "value": ", ".join(f.strings()),
                           "visibility": f.visibility}
                    if f.visibility == HIDDEN:
                        best = max(counts[(f.key, s)] for s in f.strings())
                        row["crowd"] = best
                        row["usable"] = best >= K_CROWD
                    rows.append(row)
                inside.append({"slot": st.slot, "name": st.name, "title": self._live_title(l),
                               "intent_text": l.intent_text, "fields": rows})
        intro_dicts = [i.to_dict() for i in self.intros.values()]
        report = leak_report(self.board, intro_dicts, self.outputs["followups"], self.outputs["reveals"],
                             [r.to_dict() for r in self.warden.receipts[self._seed_receipts:]],
                             self.last_proposals, self.connections, self.consents, K_CROWD, APPROACH_LIMIT)
        return {"listings": len(self.board.listings()), "k": K_CROWD, "inside": inside,
                "events": self.events[-150:], "leak_report": {"ok": report["ok"], "findings": report["findings"],
                                                               "checked": report["checked"]},
                "followup_menu": FOLLOWUP_MENU, "agent_status": self.agent_status,
                "matchmaker_last_run": self.last_run}

    def state(self) -> dict:
        return {"phones": {s: self.phone_state(s) for s in SLOTS}, "room": self.room_state()}
