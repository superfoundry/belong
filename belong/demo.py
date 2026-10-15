"""Two-person demo (proof of concept for the front end).

Two people, A and B, join a service already holding 300 synthetic listings. Each
person's phone talks to their own personal agent. Everything Belong receives or sends
is recorded in a data log for the presenter:

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


class DemoError(ValueError):
    """A request the demo can't carry out (shown to the user as a message)."""


class _Slot:
    def __init__(self, slot: str):
        self.slot = slot
        self.person_id = f"demo{slot}"
        self.person = None
        self.name = None
        self.chat = []  # [{"from": "agent"|"me", "text": str}]
        self.draft = None
        self.entry = None  # {"accepted": bool, "problems": [...]}
        self.followups = []  # pending [{"question_id", "question_text"}]
        self.intro_ids = []
        self.my_consents = {}  # intro_id -> set(keys)
        self.reveal_results = {}  # intro_id -> [{"key", "values"}]
        self.receipts = []
        self.agent_conversation = []  # this person's own conversation with their agent
        self.agent_busy = False


class DemoSession:
    def __init__(self, seed: int = 7, agent="auto"):
        """agent: "auto" uses Claude when available (else the rule-based agent), None forces
        the rule-based agent (tests), or pass any object with ClaudeAgent's respond()."""
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
        self.decisions = {}  # (intro_id, person_id) -> bool
        self.consents = {}  # (intro_id, person_id) -> set
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

    def _slot_of(self, person_id: str):
        return next((s for s, st in self.slots.items() if st.person_id == person_id), None)

    def _name(self, person_id: str) -> str:
        if person_id in self.board:
            f = self.board.listing(person_id).get("first_name")
            if f:
                return f.value
        return person_id

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
        st.chat = [{"from": "agent", "text": f"Hi {name}, I'm your agent. I work only for you, and nothing "
                                             f"is shared without your OK. Tell me what you're looking for, in your own words."}]
        self._event("in", f"Identity check for phone {slot}",
                    "The KYC provider saw the ID document. Belong received only these two facts:",
                    payload={"verified": True, "age_band": band}, slots=[slot])

    # -- 2. talking to the agent -------------------------------------------------

    def agent_message(self, slot: str, text: str) -> None:
        """The person writes to their agent. Called WITHOUT the session lock: a Claude reply
        can take a few seconds, and the other phone must keep working meanwhile."""
        with self.lock:
            st = self._slot(slot)
            if not st.person:
                raise DemoError("Verify your identity first.")
            text = (text or "").strip()[:2000]
            if not text:
                raise DemoError("Say what you're looking for.")
            if st.agent_busy:
                raise DemoError("Your agent is still replying.")
            st.chat.append({"from": "me", "text": text})
            st.agent_busy = True
            generation, llm = self.generation, self.llm
            name, band, conversation = st.name, st.person.age_band, st.agent_conversation

        reply = draft = None
        if llm is not None:
            try:
                reply, draft = llm.respond(conversation, text, name, band)
            except AgentUnavailable as e:
                with self.lock:
                    if self.llm is llm:
                        self.llm, self.agent_status = None, f"Rule-based agent: {e}"

        with self.lock:
            if generation != self.generation:
                return  # the demo was reset while the agent was thinking
            st.agent_busy = False
            if reply is None:
                self._rules_reply(st, text)
                return
            if draft:
                st.draft = draft
            # The tick on the phone comes from what actually happened, not from what the agent says.
            st.chat.append({"from": "agent", "text": reply, "draft_updated": bool(draft)})

    def _rules_reply(self, st: _Slot, text: str) -> None:
        """The offline, rule-based agent: drafts from this one message."""
        draft = agent.draft_from_message(text, st.name, st.person.age_band)
        st.draft = draft
        hidden = sum(1 for f in draft["fields"] if f["value"] and f["enabled"] and f["visibility"] == HIDDEN)
        reply = f"Here's what I understood: {draft['understood']}. I've drafted your listing"
        reply += f", with {hidden} detail(s) hidden." if hidden else "."
        reply += " Review it: you decide what's shown and what stays hidden."
        st.chat.append({"from": "agent", "text": reply})
        for note in draft["notes"]:
            st.chat.append({"from": "agent", "text": note})

    def options(self, slot: str, kind, city) -> dict:
        return agent.field_options(kind, city)

    # -- 3. entering the room -----------------------------------------------------

    def submit(self, slot: str, body: dict) -> None:
        st = self._slot(slot)
        if not st.person:
            raise DemoError("Verify your identity first.")
        listing = self._listing_from_json(st, body)
        start = len(self.warden.receipts)
        result = self.warden.admit(st.person, listing, self.board)
        receipts = self._receipts_since(start, slot)
        st.entry = {"accepted": result.accepted, "problems": result.problems}
        n_hidden = sum(1 for f in listing.fields if f.visibility == HIDDEN)
        if result.accepted:
            st.followups = []
            self._event("in", f"{st.name}'s listing accepted",
                        f"{len(listing.fields)} details ({n_hidden} hidden) passed the Warden's entry check.",
                        receipts=receipts, slots=[slot])
            st.chat.append({"from": "agent", "text": "You're all set. I'll tell you when Belong finds someone "
                                                     "who is also looking for you."})
        else:
            self._event("blocked", f"{st.name}'s listing was refused",
                        "; ".join(result.problems), receipts=receipts, slots=[slot])

    def _listing_from_json(self, st: _Slot, body: dict) -> Listing:
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
        version = self.board.listing(st.person_id).version + 1 if st.person_id in self.board else 1
        try:
            exactness = float(body.get("exactness", 0.6))
        except (TypeError, ValueError):
            exactness = -1.0  # the Warden's schema check will refuse it
        return Listing(person_id=st.person_id, intent_text=str(body.get("intent_text", "")),
                       exactness=exactness, fields=tuple(fields), updated_at=self.clock.now(),
                       version=version, confirmed_shown=tuple(confirmed))

    # -- 4. the Matchmaker runs ---------------------------------------------------

    def run_matchmaker(self) -> None:
        demo_ids = {st.person_id for st in self.slots.values() if st.person_id in self.board}
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
            slot = self._slot_of(fu.person_id)
            if slot is None or any(q["question_id"] == fu.question_id for q in self.slots[slot].followups):
                continue
            self._exit_followup(fu, slot)

        for p in mine:
            pair = frozenset((p.person_a, p.person_b))
            if pair in self.introduced_pairs:
                continue
            start = len(self.warden.receipts)
            intro = self.warden.exit_introduction(p, self.board)
            slots = [s for s in (self._slot_of(p.person_a), self._slot_of(p.person_b)) if s]
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
            for s in slots:
                self.slots[s].intro_ids.append(intro.intro_id)
            self._event("out", f"Introduction: {self._name(p.person_a)} + {self._name(p.person_b)}",
                        "This is everything sent out about this match:",
                        payload=[{"to": self._name(intro.person_a), "view": intro.a_view},
                                 {"to": self._name(intro.person_b), "view": intro.b_view}],
                        receipts=receipts, slots=slots)
        if not mine and not any(st.followups for st in self.slots.values()):
            self._event("inside", "No match yet for the demo phones",
                        "Nobody on the board meets their intent and is also looking for them.")

    def _exit_followup(self, fu: FollowUp, slot: str) -> None:
        start = len(self.warden.receipts)
        out = self.warden.exit_followup(fu, self.board)
        receipts = self._receipts_since(start, slot)
        if out is None:
            self._event("blocked", f"Question to {self.slots[slot].name} held back",
                        "Not on the fixed menu.", receipts=receipts, slots=[slot])
            return
        d = out.to_dict()
        self.outputs["followups"].append(d)
        self.slots[slot].followups.append({"question_id": d["question_id"], "question_text": d["question_text"]})
        self._event("out", f"Question to {self.slots[slot].name}", "A fixed-menu question, worded by the Warden:",
                    payload={"question": d["question_text"]}, receipts=receipts, slots=[slot])

    def _proposal_summary(self, p: Proposal) -> dict:
        return {"pair": f"{self._name(p.person_a)} + {self._name(p.person_b)}",
                "shown_score": p.shown_score, "threshold": p.shown_threshold, "score": p.score,
                "reason_fields": list(p.reason_fields), "hidden_fields_that_raised_it": list(p.used_hidden_fields)}

    def answer_followup(self, slot: str, question_id: str, answer: dict) -> None:
        st = self._slot(slot)
        if not any(q["question_id"] == question_id for q in st.followups):
            raise DemoError("No such question.")
        listing = self.board.listing(st.person_id)
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
        st.followups = [q for q in st.followups if q["question_id"] != question_id]
        updated = Listing(st.person_id, text, exactness, tuple(fields), self.clock.now(),
                          listing.version + 1, listing.confirmed_shown)
        if st.draft:
            st.draft = {**st.draft, "intent_text": text, "exactness": exactness}
        start = len(self.warden.receipts)
        result = self.warden.admit(st.person, updated, self.board)
        receipts = self._receipts_since(start, slot)
        self._event("in" if result.accepted else "blocked", f"{st.name}'s answer updated their listing",
                    "Re-checked by the Warden like any update." if result.accepted else "; ".join(result.problems),
                    receipts=receipts, slots=[slot])

    # -- 5. decisions --------------------------------------------------------------

    def decide(self, slot: str, intro_id: str, yes: bool) -> None:
        st = self._slot(slot)
        if intro_id not in st.intro_ids:
            raise DemoError("No such introduction.")
        if (intro_id, st.person_id) in self.decisions:
            raise DemoError("You've already answered.")
        self._record(intro_id, st.person_id, bool(yes))
        intro = self.intros[intro_id]
        other = intro.person_b if intro.person_a == st.person_id else intro.person_a
        if yes and self._slot_of(other) is None and (intro_id, other) not in self.decisions:
            # A synthetic counterpart answers for themselves (simulated).
            self._record(intro_id, other, self.rng.random() < 0.8)
        if self.warden.is_connected(intro_id):
            self.connections.add(frozenset((intro.person_a, intro.person_b)))
            slots = [s for s in (self._slot_of(intro.person_a), self._slot_of(intro.person_b)) if s]
            self._event("out", f"{self._name(intro.person_a)} and {self._name(intro.person_b)} connected",
                        "Both said yes, so each can now be put in touch.", slots=slots)

    def _record(self, intro_id: str, person_id: str, yes: bool) -> None:
        self.decisions[(intro_id, person_id)] = yes
        self.warden.record_decision(intro_id, person_id, yes)
        self._event("in", f"{self._name(person_id)} said {'yes' if yes else 'no'}",
                    f"to introduction {intro_id}.", slots=[s for s in [self._slot_of(person_id)] if s])

    # -- 6. mutual reveal ----------------------------------------------------------

    def reveal(self, slot: str, intro_id: str, keys: list) -> None:
        st = self._slot(slot)
        if intro_id not in st.intro_ids or not self.warden.is_connected(intro_id):
            raise DemoError("You can only reveal to someone you're connected with.")
        if intro_id in st.my_consents:
            raise DemoError("You've already chosen.")
        own_hidden = {f.key for f in self.board.listing(st.person_id).fields if f.visibility == HIDDEN}
        consent = {k for k in keys if k in own_hidden}
        st.my_consents[intro_id] = consent
        self.consents[(intro_id, st.person_id)] = consent
        self._event("in", f"{st.name} chose details to compare",
                    f"Picked {len(consent)} of {len(own_hidden)} private details (which ones stays private).",
                    slots=[slot])
        intro = self.intros[intro_id]
        other = intro.person_b if intro.person_a == st.person_id else intro.person_a
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
        slots = [s for s in (self._slot_of(intro.person_a), self._slot_of(intro.person_b)) if s]
        receipts = self._receipts_since(start)
        for s in slots:
            self.slots[s].receipts += receipts
        results = [r.to_dict() for r in reveals]
        self.outputs["reveals"] += results
        for s in slots:
            self.slots[s].reveal_results[intro.intro_id] = [{"key": r["key"], "values": r["values"]} for r in results]
        names = f"{self._name(intro.person_a)} and {self._name(intro.person_b)}"
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
        a = next((st for st in self.slots.values() if st.person_id in self.board), None)
        if a is None:
            raise DemoError("Give at least one phone a listing first.")
        b_id = next((st.person_id for st in self.slots.values() if st is not a and st.person_id in self.board), None)
        listing = self.board.listing(a.person_id)
        if b_id is None:
            b_id = next(l.person_id for l in self.board.listings() if l.person_id != a.person_id)
        hidden = [f.key for f in listing.fields if f.visibility == HIDDEN] or ["religion"]
        target = hidden[0]
        self._event("inside", "Simulating a compromised Matchmaker",
                    "It now tries three ways to get private details out.")

        attempts = [
            (f"Use {a.name}'s hidden '{agent.LABELS.get(target, target)}' as the reason for a match",
             lambda: self.warden.exit_introduction(
                 Proposal(a.person_id, b_id, True, True, 1.0, 1.0, 0.1, ("city", target), (target,)), self.board)),
            (f"Introduce {a.name} to someone who never asked for them",
             lambda: self.warden.exit_introduction(
                 Proposal(a.person_id, b_id, True, False, 1.0, 1.0, 0.1, ("city",), ()), self.board)),
            (f"Ask {a.name} an off-menu question to fish for details",
             lambda: self.warden.exit_followup(FollowUp(a.person_id, "what_is_your_religion"), self.board)),
        ]
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

    def phone_state(self, slot: str) -> dict:
        """Only what this person may know."""
        st = self._slot(slot)
        listed = st.person_id in self.board
        intros = []
        for intro_id in st.intro_ids:
            intro = self.intros[intro_id]
            me_a = intro.person_a == st.person_id
            other = intro.person_b if me_a else intro.person_a
            mine = self.decisions.get((intro_id, st.person_id))
            theirs = self.decisions.get((intro_id, other))
            if mine is False:
                continue  # you said no: the match disappears from your list
            if mine is None:
                status = "pending"
            elif theirs is False:
                status = "closed"
            elif theirs is None:
                status = "waiting"
            else:
                status = "connected"
            intros.append({"intro_id": intro_id, "view": intro.a_view if me_a else intro.b_view, "status": status,
                           "reveal_chosen": intro_id in st.my_consents,
                           "reveal_result": st.reveal_results.get(intro_id)})
        hidden_keys = []
        listing = None
        if listed:
            l = self.board.listing(st.person_id)
            hidden_keys = [{"key": f.key, "label": agent.LABELS[f.key]} for f in l.fields if f.visibility == HIDDEN]
            listing = {"intent_text": l.intent_text, "exactness": l.exactness}
        return {"slot": slot, "agent_busy": st.agent_busy, "verified": st.person is not None, "name": st.name,
                "age_band": st.person.age_band if st.person else None, "chat": st.chat, "draft": st.draft,
                "entry": st.entry, "listed": listed, "listing": listing, "followups": st.followups,
                "intros": intros, "hidden_keys": hidden_keys, "receipts": st.receipts[-60:]}

    def room_state(self) -> dict:
        """Presenter view. Shows inside-the-room data for the two demo listings only."""
        counts = self.board.value_counts()
        inside = []
        for st in self.slots.values():
            if st.person_id not in self.board:
                continue
            l = self.board.listing(st.person_id)
            rows = []
            for f in l.fields:
                row = {"key": f.key, "label": agent.LABELS[f.key], "value": ", ".join(f.strings()),
                       "visibility": f.visibility}
                if f.visibility == HIDDEN:
                    best = max(counts[(f.key, s)] for s in f.strings())
                    row["crowd"] = best
                    row["usable"] = best >= K_CROWD
                rows.append(row)
            inside.append({"slot": st.slot, "name": st.name, "intent_text": l.intent_text, "fields": rows})
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
