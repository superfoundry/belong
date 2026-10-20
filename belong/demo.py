"""Two-person demo (proof of concept for the front end).

Two people, A and B, join a service already holding 300 synthetic listings. Each person
can hold several listings at once (say, one looking for a tutor and one looking for
love). Each listing has its own conversation with the person's agent, its own draft and
its own introductions, and is matched on its own; a person is never introduced to their
own other listing.

Everything Belong receives or sends is recorded in a data log for the presenter:

  in       Belong received something (KYC facts, a listing, a decision, consents)
  out      Belong sent something to a person, through the Warden (exact payload logged)
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
import threading

from . import agent
from .agent_llm import AgentUnavailable, ClaudeAgent
from .agent_openai import OpenAIAgent
from .leaks import leak_report
from .matching.marketboard import Marketboard
from .matching.contract import FollowUp, Proposal
from .matching.matcher import OfflineMatcher
from .schema import AGE_BANDS, FIELD_REGISTRY, FOLLOWUP_MENU, HIDDEN, SHOWN, Field, Listing, Person, age_band_index
from .synth import generate
from .warden import Clock, Warden

SLOTS = ("A", "B")
K_CROWD = 20
APPROACH_LIMIT = 3
MAX_LISTINGS = 5


# Listings without a "who you're looking for" phrase are named by their kind.
KIND_TITLES = {"dating": "Dating", "climbing": "Climbing partner", "running": "Running buddy",
               "language": "Language exchange", "roommate": "Roommate", "cofounder": "Cofounder", "tutoring": "Tutoring"}


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
    def __init__(self, seed: int = 7, agent="auto"):
        """agent: "auto" picks a model-powered agent when one is configured (else the rule-based
        agent), None forces the rule-based agent (tests), or pass any object with respond()."""
        self.lock = threading.Lock()
        self._agent_choice = agent
        self.generation = 0
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
        self.personas = {p.person.id: p for p in generate(300, seed)}
        self.clock = Clock()
        self.warden = Warden(self.clock, key=hashlib.sha256(f"demo-audit-{seed}".encode()).digest(),
                             approach_limit_per_week=APPROACH_LIMIT)
        self.board = Marketboard()
        for p in self.personas.values():
            self.warden.admit(p.person, agent.draft_listing(p, self.clock.now()), self.board)
        self._seed_receipts = len(self.warden.receipts)
        self.matcher = OfflineMatcher(k_crowd=K_CROWD, max_per_person=APPROACH_LIMIT)
        self.slots = {s: _Slot(s) for s in SLOTS}
        self.events = []
        self.intros = {}  # intro_id -> Introduction
        self.introduced_pairs = set()
        self.decisions = {}  # (intro_id, listing_id) -> bool
        self.consents = {}  # (intro_id, listing_id) -> set
        self.connections = set()
        self.outputs = {"followups": [], "reveals": []}
        self.last_proposals = []
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

    @staticmethod
    def _title(draft, board_listing=None) -> str:
        """A short name for a listing, from what it's looking for."""
        fields = {f["key"]: f["value"] for f in (draft or {}).get("fields", []) if f.get("value")}
        if fields.get("seeking"):
            return str(fields["seeking"])[:60]
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
        if not st.listings:
            first = _Listing(slot, 1)
            first.chat = [{"from": "agent", "text": f"Hi {name}, I'm your agent. I work only for you, and nothing "
                                                    f"is shared without your OK. Tell me what you're looking for, in your own words."}]
            st.listings.append(first)
            st.current = first.id
        self._event("in", f"Identity check for phone {slot}",
                    "The KYC provider saw the ID document. Belong received only these two facts:",
                    payload={"verified": True, "age_band": band}, slots=[slot])

    # -- listings --------------------------------------------------------------------

    def new_listing(self, slot: str) -> str:
        """Start another listing (e.g. a tutor search alongside dating). It gets its own chat."""
        st = self._verified(slot)
        if len(st.listings) >= MAX_LISTINGS:
            raise DemoError(f"You can have up to {MAX_LISTINGS} listings.")
        listing = _Listing(slot, max(l.n for l in st.listings) + 1)
        listing.chat = [{"from": "agent", "text": "What else are you looking for? I'll keep this listing separate "
                                                  "from your others, with its own details."}]
        st.listings.append(listing)
        st.current = listing.id
        return listing.id

    def open_listing(self, slot: str, listing_id: str) -> None:
        st = self._verified(slot)
        st.current = st.listing(listing_id).id

    # -- 2. talking to the agent -------------------------------------------------

    def prefill(self, slot: str, text: str, listing_id=None) -> None:
        """Instant profile update as soon as a message is sent, before the agent replies:
        facts stated plainly enough to be sure of go into empty fields (never overwriting)."""
        st = self._verified(slot)
        listing = st.listing(listing_id)
        facts = agent.quick_facts((text or "")[:2000])
        if not facts:
            return
        if listing.draft is None:
            listing.draft = agent.empty_draft(st.name, st.person.age_band)
        else:
            listing.draft = copy.deepcopy(listing.draft)
        agent.fill_gaps(listing.draft, facts)

    def agent_message(self, slot: str, text: str, listing_id=None) -> None:
        """The person writes to their agent about one listing. Called WITHOUT the session lock:
        a model's reply can take a few seconds, and the other phone must keep working meanwhile."""
        with self.lock:
            st = self._verified(slot)
            listing = st.listing(listing_id)
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
        start = len(self.warden.receipts)
        result = self.warden.admit(st.person, listing, self.board)
        receipts = self._receipts_since(start, slot)
        mine.entry = {"accepted": result.accepted, "problems": result.problems}
        n_hidden = sum(1 for f in listing.fields if f.visibility == HIDDEN)
        title = self._title(mine.draft, listing)
        if result.accepted:
            mine.followups = []
            self._event("in", f"{st.name}'s listing accepted: {title}",
                        f"{len(listing.fields)} details ({n_hidden} hidden) passed the Warden's entry check.",
                        receipts=receipts, slots=[slot])
            mine.chat.append({"from": "agent", "text": "You're all set. I'll tell you when Belong finds someone "
                                                       "who is also looking for you.", "live": True})
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
                       version=version, confirmed_shown=tuple(confirmed), owner_id=st.person_id)

    # -- 4. the Matchmaker runs ---------------------------------------------------

    def _demo_listing_ids(self) -> set:
        return {l.id for st in self.slots.values() for l in st.listings if l.id in self.board}

    def run_matchmaker(self) -> None:
        demo_ids = self._demo_listing_ids()
        if not demo_ids:
            raise DemoError("Neither phone has a listing yet.")
        self.clock.advance(hours=1)
        result = self.matcher.match(self.board)
        mine = [p for p in result.proposals if {p.person_a, p.person_b} & demo_ids]
        self.last_proposals = mine
        self._event("inside", "Matchmaker ran",
                    f"Read all {len(self.board.listings())} listings, including hidden details. "
                    f"Proposed {len(result.proposals)} pairs; {len(mine)} involve the demo phones. "
                    f"Nothing it read can be sent to anyone except through the Warden's checks.",
                    payload=[self._proposal_summary(p) for p in mine])

        for fu in result.followups:
            slot, listing = self._owner_of(fu.person_id)
            if slot is None or any(q["question_id"] == fu.question_id for q in listing.followups):
                continue
            self._exit_followup(fu, slot, listing)

        for p in mine:
            pair = frozenset((p.person_a, p.person_b))
            if pair in self.introduced_pairs:
                continue
            start = len(self.warden.receipts)
            intro = self.warden.exit_introduction(p, self.board)
            owners = [o for o in (self._owner_of(p.person_a), self._owner_of(p.person_b)) if o[0]]
            slots = [s for s, _ in owners]
            receipts = self._receipts_since(start, None)
            for s in slots:
                self.slots[s].receipts += receipts
            if intro is None:
                failed = [r["check"] for r in receipts if r["result"] == "fail"]
                self._event("blocked", f"Introduction {self._name(p.person_a)} + {self._name(p.person_b)} held back",
                            f"Failed: {', '.join(failed)}", receipts=receipts, slots=slots)
                continue
            self.introduced_pairs.add(pair)
            self.intros[intro.intro_id] = intro
            for _, listing in owners:
                listing.intro_ids.append(intro.intro_id)
            self._event("out", f"Introduction: {self._name(p.person_a)} + {self._name(p.person_b)}",
                        "This is everything sent out about this match:",
                        payload=[{"to": self._name(intro.person_a), "view": intro.a_view},
                                 {"to": self._name(intro.person_b), "view": intro.b_view}],
                        receipts=receipts, slots=slots)
        if not mine and not any(l.followups for st in self.slots.values() for l in st.listings):
            self._event("inside", "No match yet for the demo phones",
                        "Nobody on the board meets their intent and is also looking for them.")

    def _exit_followup(self, fu: FollowUp, slot: str, listing: _Listing) -> None:
        start = len(self.warden.receipts)
        out = self.warden.exit_followup(fu, self.board)
        receipts = self._receipts_since(start, slot)
        name = self.slots[slot].name
        if out is None:
            self._event("blocked", f"Question to {name} held back",
                        "Not on the fixed menu.", receipts=receipts, slots=[slot])
            return
        d = out.to_dict()
        self.outputs["followups"].append(d)
        listing.followups.append({"question_id": d["question_id"], "question_text": d["question_text"]})
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
        elif question_id == "wider_net" and answer.get("yes"):
            exactness = 0.3
        mine.followups = [q for q in mine.followups if q["question_id"] != question_id]
        updated = Listing(mine.id, text, exactness, tuple(fields), self.clock.now(),
                          listing.version + 1, listing.confirmed_shown, owner_id=st.person_id)
        if mine.draft:
            mine.draft = {**mine.draft, "intent_text": text, "exactness": exactness}
        start = len(self.warden.receipts)
        result = self.warden.admit(st.person, updated, self.board)
        receipts = self._receipts_since(start, slot)
        self._event("in" if result.accepted else "blocked", f"{st.name}'s answer updated their listing",
                    "Re-checked by the Warden like any update." if result.accepted else "; ".join(result.problems),
                    receipts=receipts, slots=[slot])

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
            intros.append({"intro_id": intro_id, "view": intro.a_view if me_a else intro.b_view, "status": status,
                           "reveal_chosen": intro_id in mine.my_consents,
                           "reveal_result": mine.reveal_results.get(intro_id)})
        hidden_keys = []
        if listed:
            l = self.board.listing(mine.id)
            hidden_keys = [{"key": f.key, "label": agent.LABELS[f.key]} for f in l.fields if f.visibility == HIDDEN]
        return {"id": mine.id, "n": mine.n, "title": self._title(mine.draft, self.board.listing(mine.id) if listed else None),
                "kind": (mine.draft or {}).get("kind"), "listed": listed, "draft": mine.draft, "chat": mine.chat,
                "agent_busy": mine.agent_busy, "entry": mine.entry, "followups": mine.followups,
                "intros": intros, "hidden_keys": hidden_keys}

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
                inside.append({"slot": st.slot, "name": st.name, "title": self._title(mine.draft, l),
                               "intent_text": l.intent_text, "fields": rows})
        intro_dicts = [i.to_dict() for i in self.intros.values()]
        report = leak_report(self.board, intro_dicts, self.outputs["followups"], self.outputs["reveals"],
                             [r.to_dict() for r in self.warden.receipts[self._seed_receipts:]],
                             self.last_proposals, self.connections, self.consents, K_CROWD, APPROACH_LIMIT)
        return {"listings": len(self.board.listings()), "k": K_CROWD, "inside": inside,
                "events": self.events[-150:], "leak_report": {"ok": report["ok"], "findings": report["findings"],
                                                               "checked": report["checked"]},
                "followup_menu": FOLLOWUP_MENU, "agent_status": self.agent_status}

    def state(self) -> dict:
        return {"phones": {s: self.phone_state(s) for s in SLOTS}, "room": self.room_state()}
