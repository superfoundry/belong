"""Two-person demo (proof of concept for the front end; SPEC §8, the web view).

Two people, A and B, join a service already holding synthetic listings (5,000 when served;
BELONG_PEOPLE changes it). Each person
can hold several listings at once (say, one looking for a tutor and one looking for
love). Each listing has its own conversation with the person's agent, its own draft and
its own introductions, and is matched on its own; a person is never introduced to their
own other listing.

The presenter can also take over a synthetic person (from the inspector, `take_over`): they
get a phone of their own, held to the same rules as A and B, their live listing is matched
like a phone's, and nothing is decided for them any more until they're handed back.

Everything Belong receives or sends is recorded in a data log for the presenter:

  in       Belong received something (KYC facts, a listing, a decision, consents)
  out      Belong sent something to a person, through the Warden (exact payload logged)
  blocked  the Warden refused to let something in or out
  inside   private matching activity (presenter view only; never sent to anyone)

("The room" in code comments is internal shorthand for private matching; it is never
shown to users.)

Privacy rule for this module: `phone_state(slot)` must contain only what that person
may know. Room internals go only into `room_state()`, the presenter's panel.

A change the person makes on their profile themselves (a detail's sheet closed, the editor done, Save draft) is kept
on their draft and told to their agent, which may bring only the parts that still say what they replaced in line, and
write the related topics again when what the listing is about changed, by one fixed rule (keep_change,
agent.keep_edit, agent.rereads; SPEC §3); without a model agent the phone offers that change instead.

A wish (an open detail about them or both) is a must only when the person said it (SPEC §5): whatever agent drafts a
listing, a wish saved as required is kept as "ideally" (agent.ground_wishes) unless what they typed themselves for that
listing says one of its values (_said: their messages in its chat, never a turn the app wrote; a new separate listing,
those from the message that led to this one's last save: _said_since), or their agent quotes their answer to its
question about it (the message they typed next after the latest save that kept it as ideally: the listing remembers
which wishes and when, ideally_since, and _answers gives what came after), or its draft already had it as a must; and
at exactness 0.8 or more, where every wish shuts people out, a listing with a wish that passes none of these is made
less exact (agent.ground_exactness). Each save of a model agent is checked against those words, for where it lands (they
go to the check, not into the model's prompt: it has their messages only as its own conversation), and the agent is
told (agent_llm.build_draft, typed_for); the draft of an agent that doesn't take them is held to the rule by its values
before it is kept, without being told (agent_message, _grounded). TRUST.md #45.

A person's account settings (set_settings; SPEC §3) only set each listing's own close matches and questions, which the
Warden enforces: where a new listing's questions start, and whether a listing with no match yet has its close matches
turned on by itself (_close_by_setting, by the Warden's own checks for asking) instead of being asked. Both start off.

Open details (SPEC §4) travel as draft rows like any other detail, keyed about:name: only a person's agent
writes what one is (its name, whose it is, how it compares, its steps); the phone changes its values and
whether it's shown, and "them" details are always shown. The shared list of details in use (SPEC §6) is
built by the Warden and handed only to agents; phones never get it.

The Matchmaker (SPEC §8.3; _make_matchmaker): the rules alone, or the hybrid (matching/hybrid.py) with the Matchmaker
AI judging the fuzzy part of each pair the rules allow, a mock of the enclosed model (OpenAI's API,
matchmaker_openai.py; synthetic and test profiles only). BELONG_MATCHMAKER=ai|rules picks one; otherwise the AI when
OPENAI_API_KEY is set. Only the phones' listings are matched (run_matchmaker's focus), so only their pairs reach the
AI; the rest of the board isn't matched in the demo. A model takes seconds to answer and the Matchmaker runs with the
session lock held, so its requests go out on a thread of their own (_JudgeOffLock) and their pairs wait for a later
run. Which Matchmaker runs, and what the AI did, is only for the presenter's panel (room_state); phones show nothing
of it but what their own agent runs on (phone_state's agent_kind, for Settings). The presenter's leak report audits
every request the AI was sent (leaks.py, A8; _judge_sent), and the model's own record of what went out.
"""
from __future__ import annotations

import copy
import hashlib
import inspect
import os
import json
import random
import re
import threading
import time
from collections import deque
from datetime import datetime, timezone
from dataclasses import replace

from . import agent, persona_chat
from .agent_llm import (EDIT_UNSAVED_NOTE, KEPT_IDEALLY, SINGLED_OUT_NOTE, AgentUnavailable, ClaudeAgent,
                        description_save_problems, detail_save_problems, edit_message, edit_note, listed_name_problems,
                        read_exactness)
from .agent_openai import OpenAIAgent
from .leaks import leak_report
from .matching.marketboard import Marketboard
from .matching.contract import FollowUp, Proposal
from .matching.hybrid import JudgeWaiting, LLMMatcher
from .matching.intent import parse_intent
from .matching.matcher import OfflineMatcher
from .matchmaker_openai import JudgeUnavailable, OpenAIJudge
from .schema import (AGE_BANDS, FIELD_REGISTRY, FOLLOWUP_MENU, HIDDEN, K_CROWD, MATCH_KINDS, MAX_LISTINGS_PER_OWNER,
                     SHOWN, Field, Listing, Person, age_band_index, age_span_choice, canonical, display_form,
                     is_detail_key, name_problems, split_key, value_to_json)
from .synth import draft_listing, generate  # only to build the demo's board of synthetic people
from .warden import OWN_WORDS, PLACE_KEYS, Clock, RefusedByWarden, Warden, kept_labels

SLOTS = ("A", "B")  # the two demo phones; a synthetic person taken over gets a phone too (take_over)
SLOT_ID = re.compile(r"[A-Za-z0-9]{1,16}")  # a taken-over phone's id: their person id, safe in URLs and HTML
APPROACH_LIMIT = 15  # introductions per listing per day (the Warden's limit)
GUESSED_PLACES = {"city", "country"}  # see _Listing.provisional
PER_RUN = 3  # at most this many new introductions per listing in one Matchmaker run, so nobody gets them all at once
LOOK_PAST = 1000  # how far down each listing's ranking the Matchmaker looks, past people already introduced
SHORTLIST = 400  # candidates per phone listing; only the two phones are matched each run, so it can be generous
MAX_LISTINGS = MAX_LISTINGS_PER_OWNER  # the Warden's own cap on live listings per person
CHOICES_MAX = 100  # shown or hidden choices remembered per listing (built-ins, plus open details not drafted yet)
MESSAGE_MAX = 1000  # characters in one message to a match
PERSONA_REPLIES = 30  # a simulated person replies at most this many times in one conversation
REACTIONS = ("👍", "❤️", "😂", "😮", "🙏")  # the only reactions a message can get
# What a listing's own agent says when Belong has news for it: (one, several), by kind (DemoSession._tell).
NOTICES = {
    "match": ("Belong found someone for this listing. Go back to Belong to see them.",
              "Belong found {n} people for this listing. Go back to Belong to see them."),
    "question": ("Belong has a question for this listing. Go back to Belong to answer it.",
                 "Belong has {n} questions for this listing. Go back to Belong to answer them."),
    # Once per listing at most (the Warden asks "include close matches?" only once), so one sentence either way.
    "close": ("No match yet, so I've included close matches for this listing, as your settings ask. They're one-way: "
              "they aren't told about you. You can turn them off with the gear on this listing.",) * 2,
}
# The person's account settings (the menu's Settings), both off until they turn one on. They only set a listing's own
# choices, which the Warden enforces: "questions" is where a new listing's questions start; "close" turns a listing's
# close matches on by itself, where Belong would otherwise ask whether to include them (DemoSession._close_by_setting).
SETTINGS = ("always_close", "always_questions")


# Words for the person being served; a listing looking for one is named by what it offers.
_CLIENT_WORDS = {"student", "students", "customer", "customers", "client", "clients", "learner", "learners",
                 "beginner", "beginners", "pupil", "pupils", "patient", "patients", "tenant", "tenants"}
# A dating listing without a headline is named by its kind; any other is named by its phrases or description.
KIND_TITLES = {"dating": "Dating"}


def _join(items: list) -> str:
    items = [str(i) for i in items]
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


class DemoError(ValueError):
    """A request the demo can't carry out (shown to the user as a message)."""


class _JudgeOffLock:
    """The Matchmaker AI's judge as the demo calls it. A model takes seconds to answer and the Matchmaker runs with the
    session lock held, so a request goes to the judge on a thread of its own and the run goes on without it (the
    judge raises hybrid.JudgeWaiting: its pairs wait). When the answer comes, `done` is called (the session counts a
    change, so the Matchmaker runs again soon) and the next run that sends the same request gets it at once. A request
    the judge failed on gets its failure instead (the hybrid decides those pairs by the rules) for FAILED_FOR seconds,
    and is asked again after that. With inline() true (tests), the judge is called directly.

    Every request passes screen(request) (the Warden's check, warden.exit_judge_request, against the board as it is
    then) just before it is handed to the judge: with the session lock held, on the thread, so a request that waited
    in the queue while someone hid a detail, paused or deleted a listing, or the demo was reset, never goes (it fails,
    and the rules decide its pairs). forget() drops everything queued or answered (on a reset). Whatever the judge
    raises, the thread always lets the request go and ends cleanly, so later requests are still asked."""

    QUEUED = 8  # requests waiting for the thread; the oldest goes when there are more (a later run asks again)
    KEPT = 64  # answers waiting for a run to read them
    FAILED_FOR = 60.0  # seconds a request the judge failed on is decided by the rules before it is asked again

    def __init__(self, judge, done, inline, screen=None, lock=None):
        self.judge, self._done, self._inline = judge, done, inline
        self.screen, self._session_lock = screen, lock
        self._lock = threading.Lock()
        self._answers = {}  # request (as text) -> (when, the judge's answer or what it failed with)
        self._queue = deque()  # (request as text, request) waiting for the thread
        self._busy = set()  # requests queued or being asked
        self._thread = None
        self.refused = 0  # requests the screen stopped

    def _screened(self, request) -> bool:
        try:
            allowed = self.screen is None or bool(self.screen(request))
        except Exception:
            allowed = False
        if not allowed:
            self.refused += 1
        return allowed

    def __call__(self, request):
        if self._inline():
            if not self._screened(request):  # (the session lock is the caller's)
                raise RefusedByWarden("refused by the Warden's check")
            return self.judge(request)
        key = json.dumps(request, ensure_ascii=False, sort_keys=True)
        with self._lock:
            if key in self._answers:
                when, got = self._answers[key]
                if not isinstance(got, BaseException):
                    del self._answers[key]  # read once: the matcher keeps the verdicts itself
                    return got
                if time.monotonic() - when < self.FAILED_FOR:
                    raise JudgeUnavailable(str(got) or "no answer")  # the rules decide these pairs meanwhile
                del self._answers[key]
            if key not in self._busy:
                self._busy.add(key)
                self._queue.append((key, request))
                while len(self._queue) > self.QUEUED:
                    self._busy.discard(self._queue.popleft()[0])
            if self._queue and self._thread is None:
                self._thread = threading.Thread(target=self._work, daemon=True)
                self._thread.start()
        raise JudgeWaiting()

    def busy(self) -> int:
        """Requests queued or being asked."""
        with self._lock:
            return len(self._busy)

    def forget(self) -> None:
        """Drop every request queued and every answer not yet read: they were built from a board that is gone."""
        with self._lock:
            for key, _ in self._queue:
                self._busy.discard(key)
            self._queue.clear()
            self._answers.clear()

    def _work(self) -> None:
        try:
            while True:
                with self._lock:
                    if not self._queue:
                        self._thread = None  # (in the same hold as the check, so no request is left unasked)
                        return
                    key, request = self._queue.popleft()
                stop = None
                try:
                    if self._session_lock is not None:
                        with self._session_lock:  # the board as it is now, not as it was when the request was made
                            allowed = self._screened(request)
                    else:
                        allowed = self._screened(request)
                    got = self.judge(request) if allowed else RefusedByWarden("refused by the Warden's check")
                except Exception as e:  # its failure is the answer: the rules decide those pairs
                    got = e
                except BaseException as e:  # (one that would end the thread: an answer too, then the thread ends)
                    got = stop = e
                with self._lock:
                    self._busy.discard(key)
                    self._answers[key] = (time.monotonic(), got)
                    while len(self._answers) > self.KEPT:
                        del self._answers[next(iter(self._answers))]
                try:
                    self._done()
                except Exception:
                    pass
                if stop is not None:
                    raise stop
        except BaseException:
            with self._lock:  # ending early: the next call starts another thread for what is left
                self._thread = None
            raise


class _Listing:
    """One of a person's listings: its own chat with their agent, draft and introductions."""

    def __init__(self, slot: str, n: int, listing_id: str = None):
        self.n = n
        self.id = listing_id or f"demo{slot}-{n}"  # the listing's id in matching
        # [{"from": "agent"|"me", "text": str}]; a notice also has "notice" (its kind) and "count"; "from": "change" is the
        # short line saying their agent was told about a change they made on their profile (keep_change).
        self.chat = []
        self.agent_conversation = []  # the agent's own history for this listing
        self.agent_busy = False
        self.draft = None
        # Whether the phone shows this listing's profile: from the first time it's live or its draft is ready, for
        # good (DemoSession._has_profile). Until then the phone shows only the chat with their agent.
        self.has_profile = False
        self.entry = None  # {"accepted": bool, "problems": [...]}
        self.followups = []  # pending [{"question_id", "question_text"}]
        self.intro_ids = []
        self.my_consents = {}  # intro_id -> set(keys)
        self.reveal_results = {}  # intro_id -> [{"key", "values"}]
        # Both off unless the person turns them on: with the gear, a yes to "Want to include close matches?" (close
        # matches only), or their account settings (DemoSession.new_listing, _close_by_setting).
        self.prefs = {"notify_close": False, "notify_questions": False}
        # The listing's choices were set on it (set_prefs): its own from then on, never reset by the account settings.
        self.own_prefs = False
        self.asked = set()  # (question_id, about) already answered: never asked again
        # The rule-based agent asked which kind of match this is: {"text": their message, "options": [...]}
        self.pending_kind = None
        # What the person typed for this listing before it had a chat of its own here (DemoSession._said): for a
        # separate listing their agent started, their messages before the one that started it, from the one that led to
        # the other's last save; for a synthetic person taken over, their description and messages, as the simulation
        # counts them.
        self.said_before = []
        # Each wish an agent's save kept as ideally because nothing they typed said it (agent.ground_wishes, and
        # agent.ground_exactness for a listing made less exact over it), and how many messages they had typed for this
        # listing at the latest such save (len(DemoSession._said)): only the message they type next can be their answer
        # to their agent's question whether it must be met (DemoSession._answers, agent.answered). Kept while the wish
        # is on the draft as ideally; the person's own listing state, never sent anywhere.
        self.ideally_since = {}
        # Where they are, as the instant first draft guessed it from keywords, until an agent decides. Places
        # are the guesses that can say something about someone (an origin or a family's country), so a model
        # agent isn't handed them: it reads their words itself, and they're dropped unless its draft has them.
        # (With a model agent there's no instant draft at all: see prefill.)
        self.provisional = set()
        # The person's own choices of shown or hidden for this listing's details, key -> visibility: what they
        # saved (_sync_draft) and what they set in the editor (set_visibility). Every agent draft keeps them
        # unless its agent chose otherwise on purpose (agent.keep_visibility), so hidden stays hidden.
        self.choices = {}
        # Without a model agent: what the phone offers to change too after a change the person made on their profile
        # themselves (DemoSession.keep_change, agent.offers), until they answer or the part changes.
        self.offers = []


def _current(attr):
    """A _Slot attribute that reads and writes the current listing's value."""
    return property(lambda self: getattr(self.listing(), attr),
                    lambda self, value: setattr(self.listing(), attr, value))


class _Slot:
    """One phone: one verified person, any number of listings."""

    def __init__(self, slot: str, person_id: str = None):
        self.slot = slot
        self.person_id = person_id or f"demo{slot}"  # the person (owner of the listings)
        self.taken_over = person_id is not None  # a synthetic person the presenter answers as
        self.person = None
        self.name = None
        self.listings: list[_Listing] = []
        self.current = None  # id of the listing being edited
        self.receipts = []
        self.last_n = 0  # the highest listing number this person has used, deleted listings included
        self.settings = {k: False for k in SETTINGS}  # account-wide, both off until they turn one on (set_settings)

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
    def __init__(self, seed: int = 7, agent="auto", people: int = 300, matchmaker="auto"):
        """agent: "auto" picks a model-powered agent when one is configured (else the rule-based
        agent), None forces the rule-based agent (tests), or pass any object with respond(conversation, text,
        first_name, age_band, current, in_use=...).
        people: how many synthetic people share the board with the two phones.
        matchmaker: "auto" picks the Matchmaker AI or the rules (_make_matchmaker), "ai" or "rules" (or None) forces
        one, or pass a judge (request -> answer, hybrid.LLMMatcher's; a stub in tests)."""
        self.lock = threading.Lock()
        self.people = people
        self._agent_choice = agent
        self._matchmaker_choice = matchmaker
        self._judge = None  # the Matchmaker AI's judge, off the lock; kept across resets (its usage, its answers)
        self.generation = 0
        # People who use Belong through their own AI assistant (the connector). Their matches are
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

    def _make_matchmaker(self) -> tuple:
        """(the Matchmaker, what the presenter's panel says about it). BELONG_MATCHMAKER=ai|rules forces one; otherwise
        the Matchmaker AI when OPENAI_API_KEY is set, else the rules. The AI is a mock of the enclosed model (OpenAI's
        API, or any server BELONG_MATCHMAKER_URL names), for synthetic and test profiles only: the rules still narrow
        and decide everything hard, and the Warden checks every proposal the same way. If it can't be set up, the
        rules run and the panel says why."""
        rules = OfflineMatcher(k_crowd=K_CROWD, max_per_person=LOOK_PAST, shortlist_size=SHORTLIST)
        choice = self._matchmaker_choice
        if choice is None or choice == "rules":
            return rules, "Matchmaker: rules"
        if callable(choice):
            if self._judge is None:
                self._judge = self._off_lock(choice)
            return LLMMatcher(self._judge, rules=rules), "Matchmaker AI: a stand-in judge (no model)"
        wanted = choice if choice == "ai" else os.environ.get("BELONG_MATCHMAKER", "auto").strip().lower()
        if wanted == "rules":
            return rules, "Matchmaker: rules (BELONG_MATCHMAKER=rules)"
        if wanted != "ai" and not os.environ.get("OPENAI_API_KEY"):
            return rules, "Matchmaker: rules"
        if self._judge is None or not isinstance(self._judge.judge, OpenAIJudge):
            try:
                self._judge = self._off_lock(OpenAIJudge())
            except JudgeUnavailable as e:
                return rules, f"Matchmaker: rules (the Matchmaker AI is off: {e})"
        return LLMMatcher(self._judge, rules=rules), (f"Matchmaker AI: {self._judge.judge.describe()} (a mock of the "
                                                      "enclosed model; synthetic and test profiles only)")

    def _off_lock(self, judge) -> _JudgeOffLock:
        """The judge as the demo calls it (off the session lock), behind the Warden's check (_judge_screen)."""
        return _JudgeOffLock(judge, self._judge_answered, lambda: self.run_inline, screen=self._judge_screen,
                             lock=self.lock)

    def _judge_screen(self, request) -> bool:
        """The Warden's check on a request to the Matchmaker AI (warden.exit_judge_request), just before it is handed
        to the model's client, against the board as it is now and with its labels from the Matchmaker's own record of
        what it kept: called with the session lock held. A request the current Matchmaker didn't keep, one built
        before a reset, or one no longer true of a listing in it (a detail hidden, a listing paused or deleted since)
        never goes."""
        if not isinstance(self.matcher, LLMMatcher):
            return False
        labels = kept_labels(request, self.matcher.payloads, self.matcher.labels)
        return self.warden.exit_judge_request(request, labels, self.board)

    def _judge_answered(self) -> None:
        """The Matchmaker AI answered a request (or failed on it), off the lock: a change, so the Matchmaker runs again
        soon and reads it."""
        with self.lock:
            self.changes += 1

    def matchmaker_state(self) -> dict:
        """For the presenter's panel only: whether the Matchmaker AI runs, the counts of its last run (how each pair
        was decided), requests still out, requests the Warden's check stopped, and the model's own record (model,
        response format, token usage, last error, whether it is resting after a failure)."""
        if not isinstance(self.matcher, LLMMatcher):
            return {"ai": False}
        how = {}
        for d in self.matcher.decisions.values():
            how[d["how"]] = how.get(d["how"], 0) + 1
        out = {"ai": True, "last_run": dict(self.matcher.last_run), "decided": how,
               "requests_out": self._judge.busy() if self._judge is not None else 0,
               "refused": self._judge.refused if self._judge is not None else 0}
        model = getattr(self._judge, "judge", None)
        if isinstance(model, OpenAIJudge):
            out.update(model.state())
        return out

    def details_in_use(self) -> dict:
        """The shared list of details in use, as the Warden builds it from the board (once a day): what every
        person's agent is given, the model agents as data and the offline agent as its only vocabulary. Phones
        never get it. Each list handed out is kept for the leak audit (which rebuilds it from the board and
        needs it to be exactly the same) and logged once, with the Warden's receipt for building it."""
        start = len(self.warden.receipts)
        built = self.warden.details_in_use(self.board)
        if not any(d["built_at"] == built.built_at and d["board_change"] == built.board_change
                   for d in self.outputs["details_in_use"]):
            self.outputs["details_in_use"].append(built.to_dict())
            self.in_use = vocabulary = built.vocabulary()
            self._event("out", "Details in use, for every person's agent",
                        f"Built by the Warden from shown details at least {K_CROWD} different people use, at most once "
                        "a day. Agents are given it as data, so the same thing is written the same way. Phones never get it.",
                        payload={section: len(items) for section, items in vocabulary.items()},
                        receipts=self._receipts_since(start))
        return built.vocabulary()

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
            self.warden.admit(p.person, draft_listing(p, self.clock.now()), self.board)
        self.events = []
        self.outputs = {"followups": [], "reveals": [], "details_in_use": []}
        self._event("inside", "Matching service started",
                    f"{len(self.board.listings())} synthetic listings, all accepted by the Warden's entry check.")
        self.details_in_use()  # today's shared list, built with the board (its receipt among the seed's)
        self._seed_receipts = len(self.warden.receipts)
        # The Matchmaker doesn't remember who was introduced, so it ranks everyone; the demo skips pairs
        # already introduced and sends the Warden at most PER_RUN new ones per listing each run. The rules alone or
        # with the Matchmaker AI (_make_matchmaker); either way the same Proposal contract and the same Warden checks.
        old = getattr(self, "matcher", None)
        if self._judge is not None:
            self._judge.forget()  # requests queued for the board that is going: never sent
        if isinstance(old, LLMMatcher):  # requests the last Matchmaker AI may still have going out (_judge_sent)
            self._wire_before = getattr(self, "_wire_before", set()) | {
                json.dumps(r, ensure_ascii=False, sort_keys=True) for r in old.payloads}
            if len(self._wire_before) > 10_000:
                self._wire_before = set()
        self.matcher, self.matchmaker_status = self._make_matchmaker()
        model = getattr(self._judge, "judge", None)
        self._wire_from = model.sent_dropped + len(model.sent) if isinstance(model, OpenAIJudge) else 0
        self.slots = {s: _Slot(s) for s in SLOTS}  # plus a phone for each synthetic person taken over
        self.handed_back = {}  # slot -> _Slot: taken over, then handed back (simulated again until taken over again)
        self.intros = {}  # intro_id -> Introduction
        self.declined = set()  # (listing_id, about): said no to "Would that work for you?" about that listing
        self.intro_reasons = {}  # intro_id -> the reason fields it was made with
        self.threads = {}  # intro_id -> [{"from": listing_id, "text"}]: messages between two connected people
        self.persona_typing = set()  # intro_ids where a simulated person is writing a reply
        self.persona_replies = {}  # intro_id -> how many times the simulated person has replied
        self.introduced_pairs = set()
        self.decisions = {}  # (intro_id, listing_id) -> bool
        self.consents = {}  # (intro_id, listing_id) -> set
        self.connections = set()
        self.last_proposals = []
        self.last_proposals_at = 0  # the board change the last proposals were made on
        self.last_run = None  # demo time the Matchmaker last ran (it runs on its own when something changes)
        self.changes = 0  # goes up with every action anyone takes (the server counts them)
        # Held back by the Warden: not sent again until something changes, so a stuck pair doesn't
        # add receipts every few seconds. pair -> (versions, failed checks, when)
        self.held_back = {}
        self.held_questions = {}  # (listing, question, about) -> listing version it was held back at
        self.at_limit = {}  # listing -> when a pair with it hit the daily introduction limit

    def _event(self, kind: str, title: str, text: str = "", payload=None, receipts=None, slots=()):
        self.events.append({"n": len(self.events), "kind": kind, "title": title, "text": text,
                            "payload": payload, "receipts": receipts or [], "slots": list(slots),
                            "time": self.clock.now()})

    def _receipts_since(self, start: int, slot=None) -> list:
        new = [r.to_dict() for r in self.warden.receipts[start:]]
        st = (self.slots.get(slot) or self.handed_back.get(slot)) if slot else None
        if st is not None:
            st.receipts += new
        return new

    def _slot(self, slot: str) -> _Slot:
        if not isinstance(slot, str) or slot not in self.slots:
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
        draft = {"kind": listing.get("kind").value if listing.get("kind") else None, "intent_text": listing.intent_text,
                 "fields": [{"key": f.key, "value": value_to_json(f.value)} for f in listing.fields]}
        return cls._title(draft, listing)

    @staticmethod
    def _live_values(listing) -> dict:
        """What's live, detail by detail: its value and shown or hidden, and for an open detail how it compares
        (a new scale with the same value is a change too)."""
        return {"intent_text": [listing.intent_text, None],
                **{f.key: [value_to_json(f.value), f.visibility] + ([f.compare, list(f.steps), f.required]
                                                                    if is_detail_key(f.key) else []) for f in listing.fields}}

    @staticmethod
    def _draft_values(draft) -> dict:
        """The same for a draft, with an open detail's values and steps in the form they go live in."""
        out = {"intent_text": [(draft or {}).get("intent_text", ""), None]}
        for f in (draft or {}).get("fields", []):
            v = f.get("value")
            if not f.get("enabled", True) or v in (None, "", []):
                continue
            if is_detail_key(f["key"]):
                live = agent.detail_field(f)
                out[f["key"]] = [list(live.value), f.get("visibility"), live.compare, list(live.steps), live.required]
            else:
                out[f["key"]] = [list(v) if isinstance(v, (list, tuple)) else v, f.get("visibility")]
        return out

    @staticmethod
    def _sync_draft(mine, listing) -> None:
        """After a save, the draft is exactly what went live (including edits made by hand)."""
        mine.provisional = set()  # what went live is theirs now, guesses included
        if not mine.draft:  # a listing made without the agent: start its draft from what's live
            mine.draft = {"intent_text": "", "exactness": listing.exactness, "kind": None, "fields": [], "understood": "",
                          "notes": [], "missing": [], "complete": False, "by": "sync"}
        live = {f.key: f for f in listing.fields}
        confirmed = set(listing.confirmed_shown)  # sensitive details they chose to show, already confirmed
        options = agent.field_options()
        fields = []
        for f in mine.draft["fields"]:
            g = live.get(f["key"])
            if g is not None and g.about:  # an open detail: as it went live, how it compares included
                f = agent.detail_row(g.about, g.name, g.compare, g.steps, g.value, g.visibility, g.required)
            elif g is not None:
                f = {**f, "value": value_to_json(g.value), "visibility": g.visibility, "enabled": True,
                     **({"confirmed": True} if g.key in confirmed else {})}
            elif is_detail_key(f["key"]):
                continue  # an open detail that didn't go live isn't kept as an empty row
            elif f.get("value") not in (None, "", []):
                f = {**f, "value": None}
            fields.append(f)
        known = {f["key"] for f in fields}
        for key, g in live.items():
            if key in known:
                continue
            if g.about:
                fields.append(agent.detail_row(g.about, g.name, g.compare, g.steps, g.value, g.visibility, g.required))
                continue
            fields.append({"key": key, "label": agent.LABELS.get(key, key), "value": value_to_json(g.value),
                           "visibility": g.visibility, "sensitive": g.sensitive, "multi": FIELD_REGISTRY[key].multi,
                           "enabled": True, "locked": key == "age_band", "options": options.get(key, []),
                           **({"confirmed": True} if key in confirmed else {})})
        mine.draft = {**mine.draft, "intent_text": listing.intent_text, "exactness": listing.exactness, "fields": fields}
        # What they saved is their choice; a wish is always shown, so nobody chose anything about it.
        mine.choices.update({f.key: f.visibility for f in listing.fields if f.about not in agent.WISHES})
        if live.get("kind") is not None:
            mine.draft["kind"] = live["kind"].value  # the kind of match that went live (it may have been edited)

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
        without_vis = lambda v: v[:1] + v[2:]  # [value, visibility, (how an open detail compares)]
        out = []
        for key in dict.fromkeys(list(after) + list(before)):
            if before.get(key) == after.get(key):
                continue
            label = words.get(key) or agent.label_for(key).lower()
            if key in before and key in after and without_vis(before[key]) == without_vis(after[key]):
                out.append(f"{label} is now {after[key][1]}")  # only shown or hidden changed
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
        if st.taken_over:
            raise DemoError("This person is already verified.")  # their age band is on their live listing
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
                    "The KYC provider saw the ID document. Belong received only these two facts:",
                    payload={"verified": True, "age_band": band}, slots=[slot])

    def update_profile(self, slot: str, name: str) -> None:
        """Change the first name shown on all of this person's listings. Live listings are
        re-checked by the Warden. The age band comes from the ID check and can't change. A live listing the
        Warden refuses keeps the name it shows, and the person is told why (DemoError, after the others are done):
        the phone never shows a change the Warden doesn't hold."""
        st = self._verified(slot)
        name = (name or "").strip()[:30]
        if not name:
            raise DemoError("Please enter your first name.")
        st.name = name
        refused = []
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
                else:
                    refused.append(result)
                self._event("in" if result.accepted else "blocked", f"{name} updated their profile",
                            "Re-checked by the Warden like any update." if result.accepted else "; ".join(result.problems),
                            receipts=self._receipts_since(start, slot), slots=[slot])
        if refused:
            raise DemoError(f"Your name is changed, but {len(refused)} live listing(s) still show the one before: "
                            + self._refusal(refused[0]))

    def _knows(self, intro, listing_id: str) -> bool:
        """Does this listing's person know of this introduction: yes, unless it's a close match for the other person
        that they haven't said yes to (one-way: until then nothing on this person's phone changes, not even the
        receipts of the Warden's checks on it). _listing_state shows it by the same rule."""
        return (not intro.one_way_to or intro.one_way_to == listing_id
                or self.decisions.get((intro.intro_id, intro.one_way_to)) is True)

    def _refresh_intros(self, listing_id: str) -> None:
        """After a listing changes, the Warden reissues its introductions so the other person
        sees its current shown details (a new name, say) and nothing since hidden. That includes an
        introduction whose other listing was deleted: the Warden's reduced reissue is stored, so no
        introduction kept here holds something since hidden, whoever left first."""
        for intro_id, intro in list(self.intros.items()):
            if listing_id not in (intro.person_a, intro.person_b):
                continue
            start = len(self.warden.receipts)
            new = self.warden.refresh_introduction(intro_id, self.intro_reasons.get(intro_id, []), self.board)
            receipts = self._receipts_since(start)
            for side in (intro.person_a, intro.person_b):
                s = self._slot_of(side)
                if s and self._knows(intro, side):
                    self.slots[s].receipts += receipts
            if new is not None:
                self.intros[intro_id] = new

    # -- listings --------------------------------------------------------------------

    @staticmethod
    def _clean_prefs(prefs) -> dict:
        prefs = prefs if isinstance(prefs, dict) else {}
        return {k: prefs.get(k) is True for k in ("notify_close", "notify_questions")}

    def set_settings(self, slot: str, settings) -> None:
        """The person's account settings, both off until they turn one on: "always allow questions" (a new listing's
        questions start on) and "always include close matches" (where Belong would ask whether to include them, it
        turns them on by itself and says so: _close_by_setting). They never change a listing's choices once it has
        any of its own (it was written to, or its choices were set on it), and the Warden enforces each listing's own
        choices whatever they say."""
        st = self._verified(slot)
        settings = settings if isinstance(settings, dict) else {}
        st.settings = {k: settings.get(k) is True for k in SETTINGS}
        for listing in st.listings:
            # Just started, nothing said yet and no choice set on it: it starts from the settings as they are now.
            if self._untouched(listing) and not listing.own_prefs:
                listing.prefs = self._starting_prefs(st)
        self._event("in", f"{st.name} changed their settings",
                    f"Always include close matches {'on' if st.settings['always_close'] else 'off'}, always allow "
                    f"questions {'on' if st.settings['always_questions'] else 'off'}. They set a listing's own choices, "
                    "which the Warden enforces.", slots=[slot])

    def new_listing(self, slot: str) -> str:
        """Start another listing (e.g. a tutor search alongside dating). It gets its own chat.
        Questions start as the person's settings say (off unless they chose always); close matches start off, and
        Belong offers them once if no match turns up (or, if their settings say always, turns them on then)."""
        st = self._verified(slot)
        if len(st.listings) >= MAX_LISTINGS and not any(self._untouched(l) for l in st.listings):
            raise DemoError(f"You can have up to {MAX_LISTINGS} listings.")
        # An untouched draft is reused rather than piling up empty listings.
        empty = next((l for l in st.listings if self._untouched(l)), None)
        if empty is not None:
            if not empty.own_prefs:
                empty.prefs = self._starting_prefs(st)
            st.current = empty.id
            return empty.id
        # Past every number this person has used, deleted listings included: a deleted listing's id never comes
        # back (the Warden refuses it, warden.listing_id_problem), so its introductions stay with it.
        n = max([st.last_n] + [l.n for l in st.listings]) + 1
        st.last_n = n
        listing = _Listing(slot, n, f"{st.person_id}-{n}" if st.taken_over else None)
        listing.chat = [{"from": "agent", "text": f"Hi {st.name}, I'm your agent. I work only for you, and nothing is "
                                                  "shared without your OK. Tell me what you're looking for, in your own words."
                         if not st.listings else "What else are you looking for? I'll keep this listing separate "
                                                 "from your others, with its own details."}]
        listing.prefs = self._starting_prefs(st)
        st.listings.append(listing)
        st.current = listing.id
        return listing.id

    @staticmethod
    def _starting_prefs(st: _Slot) -> dict:
        """A new listing's choices, from the person's settings: questions as they chose (off unless always); close
        matches always start off, so a listing looks for exact matches first (_close_by_setting turns them on later)."""
        return {"notify_close": False, "notify_questions": st.settings["always_questions"]}

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
                    ("Paused listings aren't matched." if not updated.active else "Matching again.") if result.accepted
                    else "; ".join(result.problems), receipts=self._receipts_since(start, slot), slots=[slot])
        if not result.accepted:
            raise DemoError(f"That listing couldn't be {'resumed' if updated.active else 'paused'}: "
                            + self._refusal(result))

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

    def set_prefs(self, slot: str, listing_id: str, prefs, why: str = "", start: int = None) -> None:
        """Change a listing's notification choices. A live listing is re-checked with them. why: for the presenter's
        log, when the change isn't the person's own tap (their settings did it); start: receipts from there on go
        with it (the Warden's check that let it happen)."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        chosen = self._clean_prefs(prefs)
        if mine.id in self.board:
            current = self.board.listing(mine.id)
            updated = replace(current, notify_close=chosen["notify_close"], notify_questions=chosen["notify_questions"],
                              updated_at=self.clock.now(), version=current.version + 1)
            start = len(self.warden.receipts) if start is None else start
            result = self.warden.admit(st.person, updated, self.board)
            if result.accepted:
                self._refresh_intros(mine.id)
            said = (f"Close matches {'on' if chosen['notify_close'] else 'off'}, questions "
                    f"{'on' if chosen['notify_questions'] else 'off'}.")
            self._event("in" if result.accepted else "blocked",
                        f"Close matches for {st.name}, by their settings" if why
                        else f"{st.name} changed their notification choices",
                        (f"{why} {said}" if why else said) if result.accepted else "; ".join(result.problems),
                        receipts=self._receipts_since(start, slot), slots=[slot])
            if not result.accepted:  # the phone keeps showing what the Warden holds
                raise DemoError("Your notification choices couldn't be changed: " + self._refusal(result))
        mine.prefs, mine.own_prefs = chosen, True
        self._drop_moot_questions(mine)

    @staticmethod
    def _refusal(result) -> str:
        """Why the Warden refused an update, for the person: its problems, and that nothing changed."""
        return "; ".join(result.problems or ["the Warden refused the update"]) + ". Nothing else changed."

    @staticmethod
    def _drop_moot_questions(mine: _Listing) -> None:
        """Questions that no longer apply: any, once questions are off; "include close matches?" once
        they're on. That one question stays while both are off: it's how a person turns them on."""
        def still_applies(q):
            return not mine.prefs["notify_close"] if q["question_id"] == "broaden" else mine.prefs["notify_questions"]
        mine.followups = [q for q in mine.followups if still_applies(q)]

    @staticmethod
    def _said(listing: _Listing) -> list:
        """What the person typed themselves for this listing, in order: what they typed for it before it had a chat of
        its own here (said_before), then their messages in its chat (only agent_message writes them), never a turn the
        app wrote for their agent (a change they made on their profile, the save check, a yes to a question). Derived
        each time from those two, never kept anywhere else."""
        return [*listing.said_before, *(m["text"] for m in listing.chat if m.get("from") == "me")]

    @staticmethod
    def _said_since(listing: _Listing) -> list:
        """What the person typed for a separate listing their agent starts from this one: their messages in this
        listing's chat from the one that led to its last save (all of them if it was never saved), so the request the
        new listing is for, even when one message asked for both and this listing was saved from it first, and never
        what came before, which was about this one. Only a save that answered a message of theirs counts: one from the
        turn the app writes after a change they made on their profile (its "change" line came after their last
        message) was led by none of theirs, so it never moves where their request starts."""
        def answered(i: int) -> bool:
            led = next((m.get("from") for m in reversed(listing.chat[:i]) if m.get("from") in ("me", "change")), None)
            return led != "change"

        last = max((i for i, m in enumerate(listing.chat) if m.get("draft_updated") and answered(i)), default=None)
        start = 0 if last is None else max((i for i, m in enumerate(listing.chat[:last]) if m.get("from") == "me"),
                                           default=last)
        return [m["text"] for m in listing.chat[start:] if m.get("from") == "me"]

    def _answers(self, listing: _Listing) -> dict:
        """For each wish a save kept as ideally on this listing (ideally_since), what the person typed for it since the
        latest such save: the first of it is their answer to their agent's question whether it must be met, and only
        there can their agent find it (agent.answered)."""
        said = self._said(listing)
        return {key: said[n:] for key, n in listing.ideally_since.items()}

    @staticmethod
    def _note_ideally(listing: _Listing, keys, n: int) -> None:
        """Remember each wish this save kept as ideally and had their agent ask about (keys: never one they were already
        asked about, agent.asked_about) as of n messages typed, the latest time (the message they type next is their
        answer), and forget each that is no longer on the draft as ideally (a must now, or taken off): nothing is
        waiting for an answer about it. A wish they were already asked about keeps when it was asked, so their answer
        stays the message they typed after that question."""
        for key in keys or ():
            listing.ideally_since[key] = n
        ideally = {f.get("key") for f in (listing.draft or {}).get("fields", [])
                   if split_key(f.get("key"))[0] in agent.WISHES and f.get("required") is False}
        listing.ideally_since = {k: v for k, v in listing.ideally_since.items() if k in ideally}

    def _untouched(self, listing: _Listing) -> bool:
        return (listing.draft is None and listing.id not in self.board and listing.pending_kind is None
                and not any(m["from"] == "me" for m in listing.chat))

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
        """Instant profile update as soon as a message is sent, before the rule-based agent replies: facts
        stated plainly enough to be sure of go into empty fields (never overwriting). With a model agent it
        does nothing: the model reads their words itself, and its save fills the profile (a keyword reading,
        such as "online preferred" as either way, is never put up first)."""
        st = self._verified(slot)
        listing = self._listing_for(st, listing_id)
        if self.llm is not None:
            return
        facts = agent.quick_facts((text or "")[:2000], agent.words_in_use(self.details_in_use()))
        if not facts:
            return
        if listing.draft is None:
            listing.draft = agent.empty_draft(st.name, st.person.age_band)
        else:
            listing.draft = copy.deepcopy(listing.draft)
        had = {f["key"] for f in listing.draft["fields"] if f.get("value") not in (None, "", [])}
        agent.fill_gaps(listing.draft, facts, listing.choices)  # a detail they hid stays hidden
        listing.provisional |= {f["key"] for f in listing.draft["fields"] if f["key"] in GUESSED_PLACES
                                and f.get("value") not in (None, "", []) and f["key"] not in had}
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
            listing.offers = []  # what the phone offered is let go: their agent builds on the draft as it stands
            generation, llm = self.generation, self.llm
            name, band, conversation = st.name, st.person.age_band, listing.agent_conversation
            # The profile as it stands, with the person's choices of shown or hidden, for the agent to build on.
            # What the instant first draft only guessed is left out: the model reads their words itself.
            current = ({**self._without_guesses(listing), "choices": dict(listing.choices)}
                       if listing.draft or listing.choices else None)
            chosen_before = dict(listing.choices)
            started = copy.deepcopy(listing.draft)  # the draft as the agent is given it
            in_use = self.details_in_use() if llm is not None else None
            # What they typed, this message last, for where each save lands: a wish is a must only if they said it, or
            # answered their agent's question about it after a save kept it as ideally. A separate save goes on a new
            # listing whenever this one has a draft or is live (below), and only then can the two differ
            # (agent_llm.typed_for).
            said = {"listing": self._said(listing), "separate": self._said_since(listing),
                    "answers": self._answers(listing)}
            knows = llm is not None and "said" in inspect.signature(llm.respond).parameters

        reply = draft = None
        if llm is not None:
            try:
                if knows:  # build_draft holds each save to the rule, and tells the agent
                    reply, draft = llm.respond(conversation, text, name, band, current, in_use=in_use, said=said)
                else:  # an agent of its own that doesn't know the rule: held to it below all the same
                    reply, draft = llm.respond(conversation, text, name, band, current, in_use=in_use)
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
                if listing.pending_kind is None:
                    listing.provisional = set()  # the offline agent drafted from the same words
                return
            # Everything the agent saved this turn, in order: usually one draft; two when the person
            # said "both" to "a date, or a partner for that activity?" (one listing for each kind).
            drafts = [*(draft or {}).get("earlier", []), draft] if draft else []
            updated, moved = False, []
            # A part of the draft that changed while it was replying (an answer that went live meanwhile) stays as it
            # is now: the agent worked from an older copy (agent.keep_meanwhile).
            ended = listing.draft if started is not None and listing.draft is not None and (
                agent.changes_made(started, listing.draft) or started.get("kind") != listing.draft.get("kind")) else None
            for d in drafts:
                # Which wishes its saves kept as ideally is remembered on the listing, never kept on the draft.
                kept_ideally = d.get(KEPT_IDEALLY, []) if knows else []
                d = {k: v for k, v in d.items() if k not in ("earlier", KEPT_IDEALLY)}
                if d.get("separate") and (listing.id in self.board or listing.draft):
                    # The agent started a separate search: a new listing. The current one stays as it
                    # was, and the conversation carries on in the new one.
                    try:
                        new = st.listing(self.new_listing(slot))
                    except DemoError as e:
                        listing.chat.append({"from": "agent", "text": f"I couldn't start another listing: {e}"})
                        continue
                    new.agent_conversation = list(conversation)
                    new.chat = [{"from": "me", "text": text}, {"from": "agent", "text": reply, "draft_updated": True}]
                    # What they typed for it: their messages from the one that led to this listing's last save (this
                    # one last).
                    new.said_before = said["separate"][:-1]
                    new.draft = d if knows else self._grounded(d, said["separate"])
                    self._note_ideally(new, kept_ideally, len(said["separate"]))
                    moved.append(new.id)
                    continue
                # Keep anything already on the profile that the agent's draft left empty, and the person's
                # own choice of shown or hidden for every detail the agent didn't set on purpose (build_draft
                # does this already; this also covers any agent that hands back a draft of its own).
                agent.carry_over(d, self._without_guesses(listing), listing.choices)
                if ended is not None:
                    d, _ = agent.keep_meanwhile(d, started, ended)
                # A choice of shown or hidden the person made while the agent was replying wins over the
                # agent's draft: hidden always, shown only for a detail that isn't sensitive.
                for key, vis in listing.choices.items():
                    if chosen_before.get(key) != vis:
                        for f in d.get("fields", []):
                            if f["key"] == key and (vis == HIDDEN or not f.get("sensitive")):
                                f["visibility"] = vis
                # A must only from what they typed, whatever agent wrote the draft (build_draft holds a model agent to
                # it, and tells it): a wish saved as required that they didn't say is kept as ideally, and a must the
                # draft already had stays.
                listing.draft = d if knows else self._grounded(d, said["listing"], started)
                self._note_ideally(listing, kept_ideally, len(said["listing"]))
                updated = True
            if updated:
                listing.provisional = set()  # the agent has decided them (kept or dropped)
            if moved and updated:
                st.current = listing.id  # both: the person stays on this listing, and the other is waiting
            # The tick on the phone comes from what actually happened, not from what the agent says.
            if updated or not moved:
                listing.chat.append({"from": "agent", "text": reply, "draft_updated": updated})
            for new_id in moved:
                listing.chat.append({"from": "agent", "text": "I've started a separate listing for that, so this one "
                                                              "stays as it is." if not updated else
                                     "I've started a separate listing for the other one, with its own details.",
                                     "moved_to": new_id})

    @staticmethod
    def _grounded(draft: dict, said: list, before=None) -> dict:
        """The draft of an agent that doesn't take what the person typed, with each wish saved as required that their
        typed words (said) don't say kept as ideally, and each must the draft it replaces (before) already had, as wide
        or wider, as it is (agent.ground_wishes); and at exactness 0.8 or more, where every wish shuts people out, made
        less exact if any wish passes neither (agent.ground_exactness). It only ever lowers. Such an agent quotes
        nothing, so no answer counts for it. Its exactness is read first as build_draft reads a model agent's (a number
        from 0 to 1, else the default: agent_llm.read_exactness), so every draft has one rule for it."""
        if isinstance(draft.get("fields"), list):
            words, was = agent.said_words(said), (before or {}).get("fields")
            draft["fields"], _ = agent.ground_wishes(draft["fields"], words, before=was)
            draft["exactness"], _ = agent.ground_exactness(read_exactness(draft.get("exactness")), draft["fields"],
                                                           words, before=was,
                                                           before_exactness=(before or {}).get("exactness"))
        return draft

    @staticmethod
    def _without_guesses(listing) -> dict:
        """The listing's draft without what the instant first draft guessed and no agent has confirmed."""
        draft = copy.deepcopy(listing.draft or {"fields": []})
        for f in draft.get("fields", []):
            if f["key"] in listing.provisional:
                f["value"] = [] if f.get("multi") else None
        return draft

    def _rules_reply(self, st: _Slot, listing: _Listing, text: str, before=()) -> None:
        """The offline, rule-based agent: builds on what was already said (_rules_draft). A message that asks for
        something of its own, other than the request the listing already has (agent.starts_another_request), is
        read as a new request: when it could mean more than one kind of match, it asks one short question first
        (agent.kind_question) and drafts from the person's answer: one listing per kind they choose ("both" makes
        two). Any other message adds to the request (a place, a free time, an answer to what it asked), with no
        question about the kind first. A request of its own instead of an answer leaves the question unanswered:
        the message it asked about is read first (before), so what that said about the person stays, and its
        request gives way to the new one by the same rule. Its only vocabulary is the shared list of details in use
        (agent.words_in_use)."""
        words = agent.words_in_use(self.details_in_use())
        pending = listing.pending_kind
        if pending is None:
            adds = (listing.draft or {}).get("kind") and not agent.starts_another_request(text, listing.draft, words)
            question = None if adds else agent.kind_question(text, words)
            if question:
                listing.pending_kind = {"text": text, "options": question["options"], "question": question["text"],
                                        "before": list(before)}
                listing.chat.append({"from": "agent", "text": question["text"]})
                return
            self._rules_draft(st, listing, text, words=words, before=before)
            return
        chosen = agent.chosen_kinds(text, pending["options"], words)
        if chosen is None and agent.says_what_they_want(text, words):
            # A kind that wasn't offered: a new request instead of an answer, so start from it, after the messages
            # the question was about.
            listing.pending_kind = None
            self._rules_reply(st, listing, text, before=[*pending.get("before", ()), pending["text"]])
            return
        if chosen is None:  # not an answer: ask again, briefly
            if len(pending["options"]) > 1:
                which = " or ".join(o["label"] for o in pending["options"])
                again = f"{which[:1].upper()}{which[1:]}? {agent.ASK_HINT}"
            else:
                again = pending.get("question") or "What would you like to meet someone for?"
            if any(o["kind"] is None for o in pending["options"]) and agent.activity_hint(words) not in again:
                again += " " + agent.activity_hint(words)  # only a listed activity can go on the listing
            listing.chat.append({"from": "agent", "text": f"Sorry, I didn't catch that. {again}"})
            return
        listing.pending_kind = None
        # Their first message, plus everything the answer says besides the choice ("a date, I'm a man, 30
        # to 40" adds "I'm a man, 30 to 40"). The kind they chose is passed on its own.
        details = agent.answer_details(text, pending["options"], chosen)
        message = f"{pending['text']} {details}" if details else pending["text"]
        for n, choice in enumerate(chosen):
            if choice["kind"] is None:  # "both", and the activity isn't named yet: ask, on its own listing
                target = self._another_listing(st, listing)
                if target is None:
                    return
                target.pending_kind = {"text": message, "options": [o for o in pending["options"] if o["kind"] is None],
                                       "question": agent.WHICH_ACTIVITY, "before": list(pending.get("before", ()))}
                target.chat.append({"from": "agent", "text": f"{agent.WHICH_ACTIVITY} I'll keep it as its own listing."})
                continue
            target = listing if n == 0 else self._another_listing(st, listing)
            if target is None:
                return
            self._rules_draft(st, target, message, kind=choice["kind"], activity=choice["activity"], words=words,
                              before=pending.get("before", ()))
        st.current = listing.id

    def _another_listing(self, st: _Slot, listing: _Listing):
        """A new listing for the person's second choice (the first stays on this one), or None if they
        already have as many as they can."""
        try:
            new = st.listing(self.new_listing(st.slot))
        except DemoError as e:
            listing.chat.append({"from": "agent", "text": f"I couldn't start another listing: {e}"})
            return None
        new.chat = [{"from": "agent", "text": "You said both, so this is your second listing, with its own details."}]
        listing.chat.append({"from": "agent", "text": "I've started a separate listing for the other one.",
                             "moved_to": new.id})
        return new

    def _rules_draft(self, st: _Slot, listing: _Listing, text: str, kind=None, activity=None, words=None,
                     before=()) -> None:
        """Draft from this message on top of the listing's draft as it stands (agent.draft_from_message's earlier: a
        message that adds to the request keeps everything it doesn't say, the person's own changes included; one that
        plainly starts a different request replaces it, keeping only what's about the person). What the instant
        first draft only guessed is left out of what it builds on: the agent reads their words itself, the messages
        it asked about and never drafted (before) first, in order. The Warden's rules for the description and other
        free text are run on the result (_free_text_problems, as Save draft runs them); once the profile is on screen,
        anything they'd refuse is said."""
        earlier = self._without_guesses(listing) if listing.draft else None
        for said in before:
            earlier = agent.draft_from_message(said, st.name, st.person.age_band, words=words, earlier=earlier)
        draft = agent.draft_from_message(text, st.name, st.person.age_band, kind=kind, activity=activity, words=words,
                                         earlier=earlier)
        # The open details it can't write stay (a model agent's, before it became unavailable, or the person's), each
        # as it was: a must among them was held to the rule that a must comes from the person's own words when it was
        # saved, or came from what went live (agent.ground_wishes).
        kept = agent.keep_details(draft, earlier if before else listing.draft)
        agent.keep_visibility(draft, listing.draft, listing.choices)  # a detail they hid stays hidden
        refused = self._refused_kept(draft, kept)
        if refused:
            names = " and ".join(f"'{split_key(k)[1]}'" for k in refused)
            why = [p for p in self._draft_problems(draft) if p not in self._draft_problems(draft, refused)]
            draft["notes"].append(f"I took out {names} from before: with what you've written now, "
                                  f"{'it' if len(refused) == 1 else 'they'} couldn't be saved. {' '.join(why)}")
            draft["fields"] = [f for f in draft["fields"] if f["key"] not in refused]
        listing.draft = draft
        hidden = sum(1 for f in draft["fields"] if f["value"] and f["enabled"] and f["visibility"] == HIDDEN)
        reply = f"Here's what I understood: {draft['understood']}."
        # Only a profile on their screen is pointed to (_has_profile). Before there is one, the notes that follow
        # ask for what it still needs.
        if self._has_profile(listing):
            reply += " I've drafted your listing" + (f", with {hidden} detail(s) hidden." if hidden else ".")
            reply += " Review it: you decide what's shown and what stays hidden."
            problems = self._free_text_problems(listing, draft["fields"], draft.get("intent_text", ""))
            if problems:  # their own words, which the Warden would refuse: said before they start matching
                draft["notes"].append("Before you start matching: " + " ".join(problems))
        listing.chat.append({"from": "agent", "text": reply, "draft_updated": True})
        for note in draft["notes"]:
            listing.chat.append({"from": "agent", "text": note})

    @staticmethod
    def _draft_problems(draft: dict, leave_out=()) -> list:
        """What the one pre-save check of open details (agent_llm.detail_save_problems) says about a draft's open
        details, read with its description and other free text, without the ones in leave_out."""
        fields = [f for f in draft["fields"] if f["key"] not in leave_out]
        rows = [f for f in fields if is_detail_key(f["key"])]
        own_words = [(f["key"], f.get("value")) for f in fields if f["key"] in OWN_WORDS and f.get("enabled", True)]
        return detail_save_problems(rows, draft.get("kind"), draft.get("intent_text", ""), own_words=own_words,
                                    for_person=True) if rows else []

    def _refused_kept(self, draft: dict, kept: list) -> list:
        """The open details kept from before (agent.keep_details) that the one pre-save check refuses under the new
        draft (its description, phrases and kind), until only the new draft's own problems are left (what it has with
        none of them kept): each time the one whose leaving out leaves the fewest problems; when no one alone leaves
        fewer (two or more each complete the same problem: a place in the description and two wishes about looks),
        all of them, then each put back that the check still passes with. So the rule-based agent never stores a
        draft every other save would refuse because of what it kept."""
        problems = lambda leave_out: len(self._draft_problems(draft, leave_out))
        own, out = problems(kept), []
        while (rest := [k for k in kept if k not in out]) and (now := problems(out)) > own:
            counts = {k: problems(out + [k]) for k in rest}
            best = min(rest, key=lambda k: counts[k])
            if counts[best] < now:
                out.append(best)
                continue
            group = list(rest)
            for k in rest:
                if problems(out + [g for g in group if g != k]) <= own:
                    group.remove(k)
            out += group
        return out

    def set_visibility(self, slot: str, listing_id, visibility, confirmed=None) -> None:
        """Shown or hidden, chosen in the editor before the person saves: remembered for this listing and put
        on its draft now, so the agent's next draft keeps it (the phone sends these before each message to the
        agent). A sensitive detail is made shown only with the confirmation the editor asks for; a hidden
        choice always counts, for an open detail too (about:name), even one the agent hasn't written yet, so
        it starts hidden if it does. What someone would like from the other person (a "them" detail) and what the
        request needs them both to share (a "both" detail) are always shown: hidden, a wish could never decide
        anything (a hidden choice remembered for one is dropped)."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        if not isinstance(visibility, dict):
            raise DemoError("Nothing to change.")
        confirmed = set(confirmed) if isinstance(confirmed, list) else set()
        if mine.draft:
            mine.draft = copy.deepcopy(mine.draft)
        fields = {f["key"]: f for f in (mine.draft or {}).get("fields", [])}
        for key in [k for k, v in mine.choices.items() if split_key(k)[0] in ("them", "both") and v == HIDDEN]:
            del mine.choices[key]  # a wish is always shown
        for key, vis in visibility.items():
            if not isinstance(key, str) or vis not in (SHOWN, HIDDEN) or key == "age_band":  # the age band is fixed
                continue
            spec = FIELD_REGISTRY.get(key)
            if spec is None:
                about, name = split_key(key)
                if not about or name_problems(name) or (about in ("them", "both") and vis == HIDDEN):
                    continue
                if key not in fields and key not in mine.choices and len(mine.choices) >= CHOICES_MAX:
                    continue
            sensitive = spec is not None and spec.sensitive
            if sensitive and vis == SHOWN and key not in confirmed:
                continue
            mine.choices[key] = vis
            if key in fields:
                fields[key]["visibility"] = vis
                if sensitive and vis == SHOWN:
                    fields[key]["confirmed"] = True

    def save_draft(self, slot: str, body: dict, listing_id=None) -> None:
        """Save draft: the person's own changes are kept on the listing's draft without it going live, so they
        outlast a reload, and a model agent is given them to build on (the only place they go). It is never
        checked, matched or sent to anyone else: nothing reaches the Warden, the Marketboard or the presenter's
        log. A detail already on the draft keeps its shown or hidden (saved as it's chosen, by set_visibility);
        a detail a new kind of match adds takes the phone's, except that one hidden on this listing before
        stays hidden and a sensitive one is shown only with their confirmation. A detail the phone put aside
        (a kind that doesn't have it) comes off the draft, and a hidden value comes back hidden. The name and
        age band come from the ID check. An open detail keeps what its agent wrote (its name, about, how it
        compares, its steps): the phone changes only its values, and clearing them takes it off (back hidden if
        it was). Every open detail on the draft goes through the one pre-save check an agent's save runs too
        (agent_llm.detail_save_problems: the Warden's entry rule for open details, and its specific-person rules
        over every detail read with the description), and the description and other free text through the Warden's
        specific-person rules read with this draft's shown city and country and the Warden's last shared list's
        places, with those a live listing was accepted with (Warden.kept_places), as the Warden reads them
        (agent_llm.description_save_problems, as a model agent's save runs them); one name, one meaning (the
        shared list's rule for what
        agents write) only to a detail whose definition this save changes (_list_problems), so a list that
        changed after the agent wrote a detail never stops the person changing a value: the agent's next save
        checks it against the list it is shown. A save that fails is refused and changes nothing. Refused while
        the agent is replying, whose draft would otherwise replace it."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        if mine.agent_busy:
            raise DemoError("Your agent is still replying. Save your draft once it has.")
        if not mine.draft or not isinstance(body, dict) or not isinstance(body.get("fields"), list):
            raise DemoError("Nothing to save yet.")
        before = {f["key"]: f for f in mine.draft.get("fields", [])}
        sent = [f for f in body["fields"] if isinstance(f, dict) and isinstance(f.get("key"), str)]
        said = {f["key"]: f.get("value") for f in sent}
        kind = said["kind"] if "kind" in said else mine.draft.get("kind")
        kind = kind if isinstance(kind, str) and kind in MATCH_KINDS else None
        options = agent.field_options()
        fields, changed, hide = [], set(), set()
        for f in sent:
            key = f["key"]
            spec = FIELD_REGISTRY.get(key)
            if any(g["key"] == key for g in fields) or not agent.carries(kind, key):
                continue
            old = copy.deepcopy(before.get(key))
            if spec is None:
                if not is_detail_key(key) or old is None:
                    continue  # an open detail only its agent writes: one not on the draft is ignored
                values = self._open_values(old, f.get("value"))
                if not values:  # cleared: off the draft, and back hidden if it was
                    if old.get("visibility") == HIDDEN:
                        hide.add(key)
                    changed.add(key)
                    continue
                if values != old.get("value"):
                    changed.add(key)
                fields.append({**old, "value": values, "enabled": True})
                continue
            if key in ("first_name", "age_band"):  # from the ID check: kept as it is
                fields += [old] if old else []
                continue
            if old is None:  # a row the kind they chose has (they changed the kind)
                vis = f.get("visibility") if f.get("visibility") in (SHOWN, HIDDEN) else mine.choices.get(key, spec.default)
                if mine.choices.get(key) == HIDDEN or (spec.sensitive and vis == SHOWN and not f.get("confirmed")):
                    vis = HIDDEN
                old = {"key": key, "label": agent.LABELS.get(key, key), "value": [] if spec.multi else None,
                       "visibility": vis, "sensitive": spec.sensitive, "multi": spec.multi, "enabled": not spec.sensitive,
                       "locked": False, "options": options.get(key, []),
                       **({"confirmed": True} if spec.sensitive and vis == SHOWN else {})}
            value = f.get("value")
            if value in (None, "", []):
                value = old["value"] if old.get("value") in (None, "", []) else [] if spec.multi else None
            elif spec.multi:
                value = [str(x)[:200] for x in (value if isinstance(value, list) else [value])][:20]
            else:
                value = str(value)[:500]
                if key == "wants_age":
                    value = age_span_choice(value) or value  # as typed by hand; the Warden checks the form at entry
            old.setdefault("options", [])
            enabled = bool(f.get("enabled")) if spec.sensitive else True
            if value != old.get("value") or enabled != old.get("enabled"):
                changed.add(key)
            fields.append({**old, "value": value, "enabled": enabled})
        for key, old in before.items():
            if any(g["key"] == key for g in fields) or key in changed:
                continue
            if key in ("first_name", "age_band") or is_detail_key(key):
                fields.append(copy.deepcopy(old))  # an open detail the phone's copy didn't have stays as it is
            elif old.get("visibility") == HIDDEN and old.get("value") not in (None, "", []):
                hide.add(key)  # put aside by the phone: off the draft, and a hidden value comes back hidden
        intent = body.get("intent_text", mine.draft.get("intent_text", ""))
        intent = str(intent if intent is not None else "")[:2000]
        problems = self._save_problems(mine, fields, kind, intent, before)
        if problems:
            raise DemoError("; ".join(problems))
        kept = read_exactness(mine.draft.get("exactness"))  # what's stored, always a number
        try:
            exactness = float(body.get("exactness", kept))
        except (TypeError, ValueError):
            exactness = kept
        mine.choices.update({key: HIDDEN for key in hide})
        mine.draft = {**copy.deepcopy(mine.draft), "fields": fields, "kind": kind, "intent_text": intent,
                      "exactness": exactness if 0.0 <= exactness <= 1.0 else kept}
        mine.provisional -= changed  # a guessed place they changed is theirs now; one they left stays a guess

    def _save_problems(self, mine: _Listing, fields: list, kind, intent: str, before: dict) -> list:
        """Every check a save from the phone runs: the one pre-save check of open details an agent's save runs too (the
        Warden's entry rules, and the specific-person rules over every detail read with the description and the draft's
        other free text), on every open detail; the description and other free text (_free_text_problems); and one
        name, one meaning for a detail whose definition changes (_list_problems). before: the stored draft's rows."""
        rows = [f for f in fields if is_detail_key(f["key"])]
        own_words = [(f["key"], f.get("value")) for f in fields if f["key"] in OWN_WORDS and f.get("enabled", True)]
        problems = detail_save_problems(rows, kind, intent, own_words=own_words, for_person=True)
        problems += self._free_text_problems(mine, fields, intent)
        problems += self._list_problems(rows, before, kind, self.details_in_use) if rows else []
        return problems

    # -- a change the person made on their profile themselves ---------------------------------------------------------

    def keep_change(self, slot: str, body: dict, listing_id=None) -> None:
        """A change the person made on their profile themselves, kept (a detail sheet closed, the editor done, or Save
        draft): saved as Save draft saves it (save_draft: every check, and refused while their agent is replying), then
        read part by part (agent.changes_made). When it replaced words that another part still says, or changed what
        the listing is about (nothing hidden) while it has topics, which are its agent's reading of that (agent.depends,
        agent.rereads: only adding something, or shown or hidden alone, replaces none), a model agent is told
        what changed, from what to what, in a turn of its own (_consistency_turn, off the session lock), and the chat
        shows a short line saying so; without one, the phone offers the same words changed in the other parts that
        still say them (agent.offers), one tap each or all (answer_offers). Nothing here reaches anyone else, the
        presenter's log or the Warden: only the person's draft and their own agent.

        A change made on a copy older than the draft (body["base"]: the draft the phone's copy was made from, when
        their agent changed it meanwhile) is made on the draft as it now is (_rebased, agent.rebase), and what it
        replaced is read from that copy, so it never brings back what the agent had just brought in line. What the
        phone already offered stays offered while its part is unchanged; a new change's offers build on it."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        before = copy.deepcopy(mine.draft)
        edited = {}
        if isinstance(body, dict) and isinstance(body.get("base"), dict) and mine.draft and not mine.agent_busy:
            body, edited = self._rebased(mine, body, body["base"])
        self.save_draft(slot, body, mine.id)
        changes = agent.changes_made(before, mine.draft)
        for c in changes:
            if c["key"] in edited:
                c["edited"] = edited[c["key"]]
        mine.offers = self._live_offers(mine)
        if self.llm is None:
            if agent.replaced_words(changes):  # offered on top of what's offered already (_offers)
                mine.offers = self._offers(mine, changes, mine.offers)
            return
        if not agent.depends(mine.draft, changes):
            return  # nothing else says what they replaced: nothing for their agent to bring in line
        mine.chat.append({"from": "change", "text": agent.change_line(changes)})
        mine.agent_busy = True
        self._spawn(lambda: self._consistency_turn(slot, mine.id, changes))

    @staticmethod
    def _body_value(body: dict, key: str):
        """A part's value in a draft as the phone sends it (save_draft's body), as agent.part_value reads a draft."""
        if key == agent.DESCRIPTION:
            return str(body.get("intent_text") or "") or None
        f = next((g for g in body.get("fields") or [] if isinstance(g, dict) and g.get("key") == key), None)
        if f is None or f.get("enabled", True) is False or f.get("value") in (None, "", []):
            return None
        v = f["value"]
        return [str(x) for x in v] if isinstance(v, (list, tuple)) else str(v)

    def _rebased(self, mine: _Listing, body: dict, base: dict, live: bool = False) -> tuple:
        """The phone's body with each part the person changed on an older copy (base) that the draft has changed since
        made on the draft as it now is (agent.rebase), and each part they didn't change as the draft has it (live: for
        a listing going live, one the draft has off or empty is left out, as the phone leaves it out). Exactness too:
        as the copy had it, it is the draft's as it now is (their agent may have lowered it meanwhile:
        agent.ground_exactness), and moved on the copy, it is theirs. Returns (body, {key: (the copy's value, theirs)})
        for the parts it made again, so keep_change reads what they replaced from what they saw."""
        body = {**body, "fields": [dict(f) for f in body.get("fields") or [] if isinstance(f, dict)]}
        if "exactness" in mine.draft and body.get("exactness") == base.get("exactness"):
            body["exactness"] = mine.draft["exactness"]
        edited = {}
        for key in [f.get("key") for f in body["fields"] if isinstance(f.get("key"), str)] + [agent.DESCRIPTION]:
            was, sent, now = self._body_value(base, key), self._body_value(body, key), agent.part_value(mine.draft, key)
            if agent._same(was, now) or agent._same(sent, now):
                continue
            row = agent._row(mine.draft, key) if key != agent.DESCRIPTION else None
            if agent._same(was, sent):  # not something they changed: as the draft has it
                if key == agent.DESCRIPTION:
                    body["intent_text"] = mine.draft.get("intent_text", "")
                    continue
                keep = row is not None and not (live and now is None)
                body["fields"] = [f if f.get("key") != key else {**f, "value": copy.deepcopy(row["value"]),
                                                                 "enabled": row.get("enabled", True)}
                                  for f in body["fields"] if f.get("key") != key or keep]
                continue
            value = agent.rebase(was, sent, now, agent._choices(mine.draft, key))
            edited[key] = (was, sent)
            if key == agent.DESCRIPTION:
                body["intent_text"] = value or ""
            else:
                body["fields"] = [f if f.get("key") != key else {**f, "value": copy.deepcopy(value)} for f in body["fields"]]
        return body, edited

    def _consistency_turn(self, slot: str, listing_id: str, changes: list) -> None:
        """The person's model agent brings the rest of their profile in line with a change they made (keep_change). It is
        told only what changed, from what to what, between two drafts it is given anyway (agent_llm.edit_message), and
        builds on the saved draft as with every message. Its save goes through build_draft and is held to one fixed rule
        (agent.keep_edit: build_draft refuses what breaks it, and whatever comes back is corrected by it here, for any
        agent, against the profile as it is shown when the turn ends), then kept like any agent draft: the person's
        choices of shown or hidden stand, those made meanwhile too. If anything had to be corrected, the chat says what actually changed (agent.updated_line), never what the
        agent claimed. The turn stays in its conversation without the wording (agent_llm.edit_note). If the agent is
        unavailable, the phone offers the change instead (agent.offers). A wish is never made a must here: keep_edit
        keeps every wish as the person kept it, required or not, and takes no new one, so the rule that a must comes
        from their own words (agent.ground_wishes) has nothing to do in this turn."""
        with self.lock:
            st = self.slots.get(slot) or self.handed_back.get(slot)
            mine = next((l for l in st.listings if l.id == listing_id), None) if st else None
            if mine is None:
                return  # the demo was reset, or the listing deleted
            generation, llm = self.generation, self.llm
            if llm is None:  # became unavailable meanwhile
                mine.agent_busy = False
                mine.offers = self._offers(mine, changes, self._live_offers(mine))
                return
            name, band, conversation = st.name, st.person.age_band, mine.agent_conversation
            saved = copy.deepcopy(mine.draft)
            current = {**self._without_guesses(mine), "choices": dict(mine.choices)}
            chosen_before = dict(mine.choices)
            in_use = self.details_in_use()
            text, start = edit_message(changes, agent.rereads(saved, changes), agent.reviews(saved, changes)), len(conversation)
        reply = draft = None
        try:
            if "edit" in inspect.signature(llm.respond).parameters:
                reply, draft = llm.respond(conversation, text, name, band, current, in_use=in_use, edit=changes)
            else:  # an agent of its own that doesn't know the rule: corrected by it below all the same
                reply, draft = llm.respond(conversation, text, name, band, current, in_use=in_use)
        except AgentUnavailable as e:
            with self.lock:
                if self.llm is llm:
                    self.llm, self.agent_status = None, f"Rule-based agent: {e}"
        except Exception:  # off the request: a turn that fails otherwise changes nothing and never leaves them waiting
            reply = draft = None
            del conversation[start:]  # nor anything half done in the conversation
        with self.lock:
            if generation != self.generation or mine not in st.listings:
                return
            mine.agent_busy = False
            if reply is None:
                mine.offers = self._offers(mine, changes, self._live_offers(mine))
                return
            latest = [d for d in [*(draft or {}).get("earlier", []), draft] if d and not d.get("separate")][-1:]
            updated, now = False, mine.draft
            if latest:
                d = {k: v for k, v in latest[0].items() if k != "earlier"}
                agent.carry_over(d, self._without_guesses(mine), mine.choices)
                # A choice of shown or hidden made while it was replying wins, as after a message, and counts in the
                # rule: its update is checked against the profile as it is shown now, so nothing it wrote from a part
                # hidden meanwhile is taken into a shown one.
                meanwhile = {k: v for k, v in mine.choices.items() if chosen_before.get(k) != v}
                as_shown = self._with_choices(saved, meanwhile)
                checked = [{**c, "private": c["private"] or agent._private(as_shown, c["key"])} for c in changes]
                result, fixes = agent.keep_edit(self._with_choices(d, meanwhile), as_shown, checked)
                # It worked from the draft as the turn began: what changed on it since (kept or saved live meanwhile)
                # stays as it is now, over its update.
                result, lost = agent.keep_meanwhile(result, saved, now)
                result = self._with_choices(result, meanwhile)
                updated = bool(agent.changes_made(now, result))
                if updated or result != now:
                    mine.draft = result
                if fixes or lost:
                    reply = agent.updated_line(now, result)
            elif not reply.endswith((EDIT_UNSAVED_NOTE, SINGLED_OUT_NOTE)):
                reply = agent.NOTHING_ELSE  # it saved nothing, whatever its words say (a failed save says so itself)
            mine.chat.append({"from": "agent", "text": reply, "draft_updated": updated})
            if len(conversation) > start:
                # Once it's over, the turn stays in the conversation without any wording, old or new: which parts they
                # changed, and which their agent updated (agent.updated_line, from what actually changed). Its saves,
                # refused ones too, and its own words go, so a value the person replaced is never sent again.
                kept = (agent.updated_line(now, mine.draft) if updated
                        else next((n for n in (EDIT_UNSAVED_NOTE, SINGLED_OUT_NOTE) if reply.endswith(n)), agent.NOTHING_ELSE))
                conversation[start:] = [{"role": "user", "content": edit_note(changes)}, {"role": "assistant", "content": kept}]

    def _offers(self, mine: _Listing, changes: list, offered=()) -> list:
        """What the phone offers after a kept change without a model agent (agent.offers), only where taking it would
        pass every check Save draft runs (_save_problems), so a tap never fails on its own. What it already offers
        (offered, still live) stays, and the new change is offered on top of it: a part already offered is offered
        with both changes made (agent.offers reads the draft as if those offers were taken), one offer per part."""
        before = {f["key"]: f for f in mine.draft.get("fields", [])}
        by_key = {o["key"]: o for o in offered}
        for offer in agent.offers(self._with_offers(mine.draft, list(offered)), changes):
            now = agent.part_value(mine.draft, offer["key"])
            by_key.pop(offer["key"], None)
            if not agent._same(offer["after"], now):
                by_key[offer["key"]] = {**offer, "before": now}
        out = []
        for key in agent.parts(mine.draft):
            offer = by_key.get(key)
            if offer is None:
                continue
            draft = self._with_offers(mine.draft, [offer])
            if not self._save_problems(mine, draft["fields"], draft.get("kind"), draft.get("intent_text", ""), before):
                out.append(offer)
        return out

    @staticmethod
    def _with_choices(draft: dict, choices: dict) -> dict:
        """A copy of the draft with these choices of shown or hidden made (a sensitive detail is only ever hidden by
        one: only the person shows it, on its own)."""
        draft = copy.deepcopy(draft)
        for f in draft.get("fields", []):
            vis = choices.get(f["key"])
            if vis is not None and (vis == HIDDEN or not f.get("sensitive")):
                f["visibility"] = vis
        return draft

    @staticmethod
    def _with_offers(draft: dict, offers: list) -> dict:
        """A copy of the draft with these offers taken."""
        draft = copy.deepcopy(draft)
        for offer in offers:
            if offer["key"] == agent.DESCRIPTION:
                draft["intent_text"] = offer["after"]
            for f in draft.get("fields", []):
                if f["key"] == offer["key"]:
                    f["value"] = copy.deepcopy(offer["after"])
        return draft

    @staticmethod
    def _live_offers(mine: _Listing) -> list:
        """The offers that still apply: each part still says what it did when it was offered."""
        return [o for o in mine.offers if agent.part_value(mine.draft, o["key"]) is not None
                and agent.part_value(mine.draft, o["key"]) == o["before"]]

    def answer_offers(self, slot: str, listing_id, keys) -> None:
        """The person's answer to what their phone offered to change too (keep_change): the parts they chose (keys)
        take the offered wording, checked as Save draft checks it (if that fails, nothing changes), and the others stay
        offered; an empty answer lets them all go. Only their draft changes (nothing reaches anyone else, the
        presenter's log or the Warden), and taking an offer isn't another change to offer more for."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        if mine.agent_busy:
            raise DemoError("Your agent is still replying. Try again once it has.")
        if not isinstance(keys, list) or not all(isinstance(k, str) for k in keys):
            raise DemoError("Nothing to change.")
        live = self._live_offers(mine)
        if not keys:
            mine.offers = []
            return
        chosen = [o for o in live if o["key"] in keys]
        mine.offers = live
        if not chosen:
            raise DemoError("That part has changed since, so nothing was changed.")
        draft = self._with_offers(mine.draft, chosen)
        problems = self._save_problems(mine, draft["fields"], draft.get("kind"), draft.get("intent_text", ""),
                                       {f["key"]: f for f in mine.draft.get("fields", [])})
        if problems:
            raise DemoError("; ".join(problems))
        mine.draft = draft
        mine.provisional -= {o["key"] for o in chosen}
        mine.offers = [o for o in live if o not in chosen]

    def _free_text_problems(self, mine: _Listing, fields: list, intent: str) -> list:
        """The Warden's specific-person rules for a draft's description and other free text, read with the fields the
        Warden reads at entry: the draft's city and country as it shows them, and the places on the Warden's last
        shared list with those this live listing was accepted with (never sent to the phone), so a description Save
        draft keeps is one entry would admit (agent_llm.description_save_problems, worded for the person)."""
        own_words = [(f["key"], f.get("value")) for f in fields if f["key"] in OWN_WORDS and f.get("enabled", True)]
        where = [(f["key"], f.get("value"), f.get("visibility")) for f in fields
                 if f["key"] in PLACE_KEYS and f.get("enabled", True)]
        places = [*self.warden.listed_places(), *(self.warden.kept_places(mine.id) if mine.id in self.board else [])]
        return description_save_problems(intent, own_words, where, places, for_person=True)

    @staticmethod
    def _list_problems(rows: list, before: dict, kind, in_use) -> list:
        """One name, one meaning for a save from the phone: only for an open detail whose definition (its name,
        how it compares, its steps) differs from the stored draft's, checked against today's shared list
        (in_use: called only then, so an ordinary save never builds it). Save draft keeps each definition as
        its agent wrote it (the phone changes only values), so today no row qualifies: the rule stands for any
        save that would change one. The problem is worded for the person, never quoting the list (phones never
        get it)."""
        how = lambda r: (r.get("compare"), tuple(canonical(s) for s in r.get("steps") or ()))
        changed = [r for r in rows if before.get(r["key"]) is None or how(r) != how(before[r["key"]])]
        if not changed:
            return []
        listed = in_use()
        return [f"Ask your agent to update the detail \u201c{r['name']}\u201d: other people's agents use that name "
                "another way." for r in changed if listed_name_problems([r], listed)]

    @staticmethod
    def _open_values(row: dict, value) -> list:
        """An open detail's values as the phone sent them, in the form they go live in: each in display form,
        and a step written in another case or spacing as that step. Whether they're allowed is the rule's to
        say (schema.detail_problems), never decided here."""
        given = value if isinstance(value, list) else [] if value in (None, "") else [value]
        values = [display_form(x)[:200] for x in given[:20] if isinstance(x, str) and x.strip()]
        steps = row.get("steps") or []
        return [next((s for s in steps if canonical(s) == canonical(v)), v) for v in values]

    @staticmethod
    def layout(kind) -> dict:
        """The profile's rows for a kind of match, for the editor when the person changes the kind."""
        return agent.kind_layout(kind)

    # -- 3. entering the room -----------------------------------------------------

    def submit(self, slot: str, body: dict, listing_id=None) -> None:
        """Start matching, or Save changes: the listing as the phone sends it goes to the Warden's entry check, and if
        it's accepted, live. Refused while the person's agent is replying or updating their profile, as Save draft is:
        its update would be made from the draft as it was. Changes made on an older copy of the draft (body["base"])
        are made on the draft as it now is first, as keep_change makes them (_rebased)."""
        st = self._verified(slot)
        mine = st.listing(listing_id)
        if mine.agent_busy:
            raise DemoError("Your agent is still replying. Save your changes once it has.")
        if isinstance(body, dict) and isinstance(body.get("base"), dict) and mine.draft:
            body, _ = self._rebased(mine, body, body["base"], live=True)  # made on the draft as it now is
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
                text = (f"Your listing is live: \u201c{self._live_title(listing)}\u201d. I'll tell you when Belong finds "
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
        """The listing as the phone sends it. An open detail goes live only as its agent wrote it on this
        listing's draft (its name, about, how it compares and its steps); the phone gives its values and shown
        or hidden. One not on the draft is left out, so a phone's old copy can't bring back a detail or a scale
        the agent has since changed. The Warden checks everything at entry."""
        drafted = {f["key"]: f for f in (mine.draft or {}).get("fields", []) if is_detail_key(f.get("key"))}
        fields, confirmed = [], []
        for f in body.get("fields", []) if isinstance(body.get("fields"), list) else []:
            if not isinstance(f, dict):
                continue
            key, value, vis = f.get("key"), f.get("value"), f.get("visibility")
            if is_detail_key(key):
                row = drafted.get(key)
                values = self._open_values(row, value) if row else []
                if values:
                    fields.append(agent.detail_field({**row, "value": values, "visibility": vis}))
                continue
            spec = FIELD_REGISTRY.get(key)
            if spec is None or value in (None, "", []):
                continue
            if spec.multi:
                value = tuple(str(x) for x in (value if isinstance(value, list) else [value]))
            elif not isinstance(value, str):
                value = str(value)
            if key == "wants_age":
                value = age_span_choice(value) or value  # "33 to 42" typed by hand; the Warden checks the form
            if spec.sensitive and vis == SHOWN and f.get("confirmed"):
                confirmed.append(key)
            fields.append(Field(key=key, value=value, visibility=vis, sensitive=spec.sensitive))
        # A detail only another kind has never goes live (agent.carries, the same rule as every draft): the kind
        # on the listing, or with none, the kind its description reads as.
        intent = str(body.get("intent_text", ""))
        kind = next((f.value for f in fields if f.key == "kind"), None) or parse_intent(intent).kind
        fields = [f for f in fields if agent.carries(kind, f.key)]
        version = self.board.listing(mine.id).version + 1 if mine.id in self.board else 1
        try:
            exactness = float(body.get("exactness", 0.6))
        except (TypeError, ValueError):
            exactness = -1.0  # the Warden's schema check will refuse it
        return Listing(person_id=mine.id, intent_text=intent,
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
        # Only the phones' matches, not everyone's; listings introduced a lot today rank a little lower.
        result = self.matcher.match(self.board, focus=demo_ids, load=self._introductions_today())
        mine = [p for p in result.proposals if {p.person_a, p.person_b} & demo_ids]
        self.last_proposals, self.last_proposals_at = mine, self.board.changes
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
                    f"{self._ai_note()}Nothing it read can be sent to anyone except through the Warden's checks.",
                    payload=[self._proposal_summary(p) for p in fresh])
        noted = len(self.events)

        for fu in result.followups:
            if fu.question_id != "broaden":
                self._ask(fu)

        for p in fresh:
            pair = frozenset((p.person_a, p.person_b))
            if any(self._within_hour(self.at_limit.get(x)) for x in (p.person_a, p.person_b)):
                continue  # at the daily limit: wait an hour of demo time before trying again
            start = len(self.warden.receipts)
            intro = self.warden.exit_introduction(p, self.board)
            owners = [o for o in (self._owner_of(p.person_a), self._owner_of(p.person_b)) if o[0]]
            slots = [s for s, _ in owners]
            receipts = self._receipts_since(start, None)
            # A close match is news only to its recipient: the other person's phone gets nothing of it, sent or held
            # back, not even the receipts of the Warden's checks (the public log still has them, with no content).
            for s, listing in owners:
                if not p.close or listing.id == p.close:
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
                # A one-way close match is only news to its recipient; the other person isn't told, so nothing on
                # their phone changes (not even "No matches yet. Want to include close matches?" going away).
                if not intro.one_way_to or intro.one_way_to == listing.id:
                    listing.followups = [q for q in listing.followups if q["question_id"] != "broaden"]  # no longer true
                    self._tell(listing, "match", self._waiting_intros(listing))
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
        # "Want to include close matches?" (or turning them on, by the person's settings) only once this run's
        # introductions are out, so the Warden's "no match yet" check sees them: a Matchmaker that proposes it with a
        # match in the same run is refused, whatever order it hands them over in.
        for fu in result.followups:
            if fu.question_id == "broaden":
                self._ask(fu)
        sent = (len(self.intros), len(self.outputs["followups"])) != sent_before
        if quiet and len(self.events) == noted:
            del self.events[events_before:]  # nothing happened: no trace in the log
            return sent
        waiting = self.matchmaker_state().get("decided", {}).get("waiting", 0)
        if not mine and waiting:
            self._event("inside", "Waiting for the Matchmaker AI",
                        f"It is still judging {waiting} pair{'' if waiting == 1 else 's'} for the demo phones. The "
                        "Matchmaker runs again when its answer comes, and nobody is asked about close matches meanwhile.")
        elif not mine and not any(l.followups for st in self.slots.values() for l in st.listings):
            self._event("inside", "No match yet for the demo phones",
                        "Nobody on the board meets their intent and is also looking for them.")
        return sent

    def _ai_note(self) -> str:
        """What the Matchmaker AI did in the last run, for the presenter's log ("" with the rules alone)."""
        state = self.matchmaker_state()
        if not state["ai"]:
            return ""
        how = state["decided"]
        judged, kept, waiting = how.get("judge", 0), how.get("kept", 0), how.get("waiting", 0)
        by_rules = how.get("cap", 0) + how.get("failed", 0) + how.get("unanswered", 0)
        note = f"The Matchmaker AI judged {judged + kept} pairs the rules allow ({kept} from its earlier answers)"
        if waiting:
            note += f", {waiting} wait for its answer"
        if by_rules:
            note += f"; the rules decided {by_rules} it didn't judge (over this run's cap, or no answer)"
        # What goes to the Matchmaker AI is the one thing that leaves without being sent to a person: say so.
        return note + (". It is sent only shown details and descriptions, and each request goes only once it passes "
                       "the Warden's check. ")

    def _ask(self, fu: FollowUp) -> None:
        """A question the Matchmaker proposed, for one of the phones: through the Warden's checks (_exit_followup), or,
        for "Want to include close matches?" when the person's settings say always, turned on instead
        (_close_by_setting). Not asked again once answered, while it waits on someone else's no, or when it was held
        back before and nothing has changed since."""
        slot, listing = self._owner_of(fu.person_id)
        if slot is None or any(q["question_id"] == fu.question_id for q in listing.followups):
            return
        if (fu.question_id, fu.about) in listing.asked:
            return  # answered before: not asked again
        # One person at a time: the second is asked only after the first said no. (A synthetic
        # person who'd be asked first declines, as they do in the simulation.)
        if fu.after and self._slot_of(fu.after) is not None and (fu.after, fu.person_id) not in self.declined:
            return
        key = (fu.person_id, fu.question_id, fu.about)
        if self.held_questions.get(key) == self.board.listing(fu.person_id).version:
            return  # held back before, and nothing has changed since
        if fu.question_id == "broaden" and self.slots[slot].settings["always_close"]:
            self._close_by_setting(fu, slot, listing)
            return
        self._exit_followup(fu, slot, listing)

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

    def _introductions_today(self) -> dict:
        """{listing id: introductions it got in the last 24 hours of demo time}, from the Warden's own
        send times, so the Matchmaker can rank busy listings a little lower (fair ranking)."""
        from datetime import datetime, timedelta
        now = datetime.fromisoformat(self.clock.now().rstrip("Z"))
        load = {}
        for intro in self.intros.values():
            if intro.sent_at and now - datetime.fromisoformat(intro.sent_at.rstrip("Z")) < timedelta(days=1):
                for pid in (intro.person_a, intro.person_b):
                    load[pid] = load.get(pid, 0) + 1
        return load

    @staticmethod
    def _tell(listing: _Listing, kind: str, waiting: int) -> None:
        """A notice from the person's own agent in that listing's chat (not sent to the model). waiting: how
        many of that kind (matches or questions) are waiting on the person now, so the number is always
        true. Notices of one kind since anything else was said are one line, moved to the end, so a busy
        listing isn't flooded with the same sentence."""
        for n in range(len(listing.chat) - 1, -1, -1):
            m = listing.chat[n]
            if not m.get("notice"):
                break
            if m["notice"] == kind:
                listing.chat.pop(n)
                break
        one, several = NOTICES[kind]
        waiting = max(1, waiting)  # told right after one arrives
        listing.chat.append({"from": "agent", "text": one if waiting == 1 else several.format(n=waiting),
                             "notice": kind, "count": waiting})

    def _waiting_intros(self, mine: _Listing) -> int:
        """Introductions on this listing waiting on the person's answer, as their phone shows them."""
        return sum(item["status"] == "pending" for item in self._listing_state(mine)["intros"])

    def _exit_followup(self, fu: FollowUp, slot: str, listing: _Listing) -> None:
        start = len(self.warden.receipts)
        out = self.warden.exit_followup(fu, self.board)
        if out is None:
            self._held_question(fu, slot, start)
            return
        d = out.to_dict()
        self.outputs["followups"].append(d)
        self._put_question(d, fu, slot, listing, self._receipts_since(start, slot))

    def _held_question(self, fu: FollowUp, slot: str, start: int) -> None:
        """The Warden refused a question (receipts from start on): it isn't tried again until the listing changes."""
        receipts = self._receipts_since(start, slot)
        failed = [r["check"] for r in receipts if r["result"] == "fail"]
        self.held_questions[(fu.person_id, fu.question_id, fu.about)] = self.board.listing(fu.person_id).version
        self._event("blocked", f"Question to {self.slots[slot].name} held back", f"Failed: {', '.join(failed)}",
                    receipts=receipts, slots=[slot])

    def _put_question(self, d: dict, fu: FollowUp, slot: str, listing: _Listing, receipts: list) -> None:
        """A question the Warden let out (d), on the listing for the person to answer, and their agent says so."""
        listing.followups.append({"question_id": d["question_id"], "question_text": d["question_text"],
                                  "offer_add": d["offer_add"], "seek_add": d["seek_add"], "pay": d["pay"], "about": fu.about})
        self._tell(listing, "question", len(listing.followups))
        self._event("out", f"Question to {self.slots[slot].name}", "A fixed-menu question, worded by the Warden:",
                    payload={"question": d["question_text"]}, receipts=receipts, slots=[slot])

    def _close_by_setting(self, fu: FollowUp, slot: str, listing: _Listing) -> None:
        """"Always include close matches" (the person's settings): where Belong would ask "Want to include close
        matches?", it turns that listing's close matches on by itself and says so in the listing's chat. Only where
        the question could be asked: the Warden lets it out first, by the question's own checks (the listing is active,
        has no match yet and its close matches are still off, and it's the first time ever), so a listing looks for
        exact matches first and is never switched on by its settings twice, even after the person turns them off. The
        listing's own choice then changes through set_prefs, re-checked by the Warden like any update (questions keep
        their own switch). If the Warden refuses that (the day's update limit, say), nothing changed, and the
        question is put to the person as it is without the setting."""
        start = len(self.warden.receipts)
        out = self.warden.exit_followup(fu, self.board)
        if out is None:
            self._held_question(fu, slot, start)
            return
        d = out.to_dict()
        self.outputs["followups"].append(d)  # audited as the question is (R5: once, and while close matches were off)
        try:
            self.set_prefs(slot, listing.id, {"notify_close": True, "notify_questions": listing.prefs["notify_questions"]},
                           why="No match yet, and their settings say always include close matches, so Belong turned "
                               "them on instead of asking. The Warden let it happen by the question's own checks, then "
                               "re-checked the listing:", start=start)
        except DemoError:
            self._put_question(d, fu, slot, listing, [])  # the receipts went with the refusal
            return
        listing.asked.add(("broaden", ""))  # as if they'd said yes: never asked, nor switched on again
        self._tell(listing, "close", 1)

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
        # An answer that lacks the value its question needs is refused, never filled in with a default.
        if question_id == "age_range":
            try:
                lo, hi = int(answer["min"]), int(answer["max"])
            except (KeyError, TypeError, ValueError):
                raise DemoError("Pick an age range from 18 up.")
            if not 18 <= lo <= hi <= 99:
                raise DemoError("Pick an age range from 18 up.")
            text += f" Aged {lo} or older." if hi >= 99 else f" Aged {lo} to {hi}."
            fields = list(agent.with_wants_age(tuple(fields), f"{lo}-{hi}"))  # recorded on the listing too
        question = next(q for q in mine.followups if q["question_id"] == question_id)
        if question_id == "broaden" and answer.get("yes"):
            # A yes is the opt-in to close matches only: questions keep their own switch. The question is used up only
            # once that's done: if the Warden refuses the update (the day's update limit, say), it stays on the card,
            # to answer again later, and nothing changed.
            self.set_prefs(slot, mine.id, {"notify_close": True, "notify_questions": mine.prefs["notify_questions"]})
        mine.followups = [q for q in mine.followups if q["question_id"] != question_id]
        mine.asked.add((question_id, question.get("about", "")))
        if question_id == "broaden":
            if answer.get("yes"):
                mine.chat.append({"from": "agent", "text": "You said yes, so you'll hear about close matches too. "
                                                           "They're one-way: the other person isn't told."})
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
            st = self.slots.get(slot) or self.handed_back.get(slot)  # handed back meanwhile: still their update
            if st is None:
                return  # the demo was reset
            generation, llm = self.generation, self.llm
            mine = st.listing(listing_id)
            name, band, conversation = st.name, st.person.age_band, mine.agent_conversation
            self._sync_draft(mine, updated)  # the agent sees the listing as it will be
            current = {**copy.deepcopy(mine.draft or {"fields": []}), "choices": dict(mine.choices)}
            headline_vis = mine.choices.get("headline", SHOWN)  # a headline they hid stays hidden
            in_use = self.details_in_use()
            said = self._said(mine)  # what they typed, never this turn: the app writes it
        ask = (f'Belong asked me: "{question_text}" and I said yes. Update my listing so its headline and '
               "description describe what I agreed to now. Keep everything else as it is, and keep "
               "separate_listing false.")
        draft = None
        try:
            if "said" in inspect.signature(llm.respond).parameters:
                _, draft = llm.respond(conversation, ask, name, band, current, in_use=in_use, said=said)
            else:
                _, draft = llm.respond(conversation, ask, name, band, current, in_use=in_use)
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
                    fields += (Field("headline", words["headline"], headline_vis, False),)
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
            if self._slot_of(other) is None and not self.threads.get(intro_id):
                self._persona_turn(intro_id, other)  # a simulated person you matched with says hello first

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
                        payload=[{"detail": agent.label_for(r["key"]), "value": ", ".join(r["values"])} for r in results],
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
        if self._slot_of(other) is None:
            self._persona_turn(intro_id, other)  # a simulated person answers, since someone wrote to them

    def _persona_turn(self, intro_id: str, me: str) -> None:
        """A simulated person writes (and maybe reacts): a hello when a real person matches with
        them, then only in answer to that person's messages. One reply at a time per conversation,
        at most PERSONA_REPLIES times. Two simulated people never chat."""
        if intro_id in self.persona_typing or self.persona_replies.get(intro_id, 0) >= PERSONA_REPLIES:
            return
        self.persona_typing.add(intro_id)
        generation = self.generation
        self._spawn(lambda: self._persona_reply(intro_id, me, generation))

    def _persona_reply(self, intro_id: str, me: str, generation: int) -> None:
        import time
        started = time.monotonic()
        with self.lock:
            if generation != self.generation:
                return
            intro = self.intros[intro_id]
            mine = self.board.listing(me)
            # Only what this person would know: their own shown details and request, and the
            # introduction they were shown. Never anyone's hidden details.
            ctx = persona_chat.context(self._name(me), mine.shown(), mine.intent_text,
                                       intro.a_view if intro.person_a == me else intro.b_view)
            thread = [dict(m) for m in self.threads.get(intro_id, [])]
            llm, rng = self.llm, random.Random(f"{intro_id}:{len(thread)}")
        try:
            text, react = persona_chat.reply(llm, ctx, thread, me, REACTIONS, rng)
        except Exception:  # the model is unavailable: answer simply rather than not at all
            text, react = persona_chat.fallback(thread, me, REACTIONS, rng)
        if not self.run_inline:
            time.sleep(max(0.0, 1.5 - (time.monotonic() - started)))  # people take a moment to type
        with self.lock:
            if generation != self.generation:
                return
            self.persona_typing.discard(intro_id)
            if self._slot_of(me) is not None:
                return  # taken over meanwhile: the presenter answers as them now
            live = self.threads.setdefault(intro_id, [])
            theirs = [n for n, m in enumerate(live[:len(thread)]) if m["from"] != me]
            if react and theirs:
                live[theirs[-1]]["reactions"][me] = react
            at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            live.append({"from": me, "text": text, "at": at, "reactions": {}})
            self.persona_replies[intro_id] = self.persona_replies.get(intro_id, 0) + 1
            other = intro.person_b if intro.person_a == me else intro.person_a
            self._event("out", f"{self._name(me)} replied to {self._name(other)}",
                        "A simulated person's reply, written by the AI model from what they were shown. "
                        "Its content isn't logged.", slots=[s for s in [self._slot_of(other)] if s])
            if any(m["from"] != me for m in live[len(thread):]):
                self._persona_turn(intro_id, me)  # they wrote again while this reply was being written

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
            (f"Use {a_name}'s hidden '{agent.label_for(target)}' as the reason for a match",
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

    # -- presenter: answering as a synthetic person ---------------------------------

    def take_over(self, persona_id) -> str:
        """Give a synthetic person a phone, so the presenter can answer as them (the inspector's
        "Take over"). Returns the phone's slot id: their person id, safe in URLs and HTML.

        Their phone holds their listing that is already live (same id, nothing re-entered), with a
        draft built from it, so their profile shows them their own shown and hidden details, and any
        introductions that already involve it. From then on the listing is matched like a phone's,
        what the Warden sends it reaches this phone, and nothing is decided for them any more
        (`_slot_of` finds them, so the simulated answers in run_matchmaker, decide and reveal stop).
        Only someone on the board who passed the Warden's entry check; never A or B. Taking over
        someone already on a phone returns that phone."""
        pid = persona_id if isinstance(persona_id, str) else ""
        if pid in self.slots and self.slots[pid].taken_over:
            return pid
        if pid in self.handed_back:  # their phone comes back as it was, with anything sent meanwhile
            st = self.slots[pid] = self.handed_back.pop(pid)
            self._attach_intros(st)
            self._event("inside", f"Presenter took over {st.name} again", "Their phone is back as it was.", slots=[pid])
            return pid
        persona = self.personas.get(pid)
        if persona is None or pid in SLOTS or not SLOT_ID.fullmatch(pid):
            raise DemoError("That person isn't one of the synthetic people in the live demo.")
        if pid not in self.board or self.board.listing(pid).owner != pid or not persona.person.verified:
            raise DemoError("That person didn't pass the Warden's entry check, so they have no live listing.")
        live = self.board.listing(pid)
        first = live.get("first_name")
        st = _Slot(pid, person_id=pid)
        st.person, st.name = persona.person, (first.value if first else persona.values.get("first_name") or pid)
        mine = _Listing(pid, 1, listing_id=pid)
        st.last_n = 1
        mine.prefs = {"notify_close": live.notify_close, "notify_questions": live.notify_questions}
        mine.entry = {"accepted": True, "problems": []}
        # Their profile, from what's live: every detail they have, hidden ones in their hidden well
        # (on their own phone only), and the sensitive ones they never turned on, still off.
        kind = live.get("kind").value if live.get("kind") else None  # as recorded on their listing
        mine.draft = {**agent.empty_draft(st.name, persona.person.age_band), "kind": kind, "by": "sync"}
        self._sync_draft(mine, live)
        # What they typed, as the simulation counts it (their description, and their words for each must), so a must
        # their agent writes later from it is theirs (agent.ground_wishes).
        mine.said_before = [persona.intent_text, *persona.said]
        mine.chat = [{"from": "agent", "text": f"Hi {st.name}, I'm your agent. Your listing is live: "
                                              f"“{self._live_title(live)}”. I'll tell you when Belong finds "
                                              "someone who is also looking for you."}]
        st.listings.append(mine)
        st.current = mine.id
        for intro_id in [i for i, x in self.intros.items() if pid in (x.person_a, x.person_b)]:
            self.persona_typing.discard(intro_id)
        for other in self.slots.values():
            for theirs in other.listings:
                asked_of_them = {q.get("about") for q in theirs.followups if q["question_id"] == "also_offer"}
                asked_of_them |= {about for q, about in theirs.asked if q == "also_offer"}
                if pid in asked_of_them:
                    mine.asked.add(("also_offer", theirs.id))  # they were simulated, so taken to have said no
        self.slots[pid] = st
        self._attach_intros(st)
        self._event("inside", f"Presenter took over {st.name}",
                    "A synthetic person, now on a phone of their own. Only what the Warden sends them reaches it, "
                    "and nothing is decided for them until they're handed back.", slots=[pid])
        return pid

    def hand_back(self, slot) -> None:
        """Stop answering as a synthetic person: their phone closes and they're simulated again. Their
        listings stay as they are; taking them over again brings the phone back as it was."""
        st = self._slot(slot)
        if not st.taken_over:
            raise DemoError("Only a synthetic person who was taken over can be handed back.")
        self.handed_back[slot] = self.slots.pop(slot)
        self._event("inside", f"Presenter handed {st.name} back", "Simulated again: they answer for themselves.")
        for mine in st.listings:  # anything left waiting on them is answered as a simulated person would
            for intro_id in list(mine.intro_ids):
                intro = self.intros[intro_id]
                other = intro.person_b if intro.person_a == mine.id else intro.person_a
                if (intro_id, mine.id) not in self.decisions and (
                        self.decisions.get((intro_id, other)) is True or intro.one_way_to == mine.id):
                    self._record(intro_id, mine.id, self.rng.random() < 0.8)
                    if self.warden.is_connected(intro_id):
                        self.connections.add(frozenset((intro.person_a, intro.person_b)))
                if self.warden.is_connected(intro_id) and (intro_id, other) in self.consents \
                        and (intro_id, mine.id) not in self.consents:
                    listing = self.board.listing(mine.id)
                    self.consents[(intro_id, mine.id)] = {f.key for f in listing.fields
                                                          if f.visibility == HIDDEN and self.rng.random() < 0.7}
                    self._settle_reveal(intro)

    def _attach_intros(self, st: _Slot) -> None:
        """Introductions sent to this phone's listings while they were simulated, with the details
        they chose to compare meanwhile and what the Warden's reveal (it goes to both people) showed."""
        for mine in st.listings:
            for intro_id, intro in self.intros.items():
                if mine.id not in (intro.person_a, intro.person_b):
                    continue
                if intro_id not in mine.intro_ids:
                    mine.intro_ids.append(intro_id)
                other = intro.person_b if intro.person_a == mine.id else intro.person_a
                if (intro_id, mine.id) in self.consents and intro_id not in mine.my_consents:
                    mine.my_consents[intro_id] = set(self.consents[(intro_id, mine.id)])
                if (intro_id, mine.id) in self.consents and (intro_id, other) in self.consents \
                        and intro_id not in mine.reveal_results:  # both chose, so the Warden settled it
                    pair = {intro.person_a, intro.person_b}
                    mine.reveal_results[intro_id] = [{"key": r["key"], "values": r["values"]}
                                                     for r in self.outputs["reveals"]
                                                     if {r["person_a"], r["person_b"]} == pair]

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
                item["simulated"] = self._slot_of(other) is None  # their replies are written by the AI model
                item["typing"] = intro_id in self.persona_typing
                item["their_assistant"] = self._slot_of(other) in self.assistant_slots
            intros.append(item)
        hidden_keys = []
        if listed:
            l = self.board.listing(mine.id)
            hidden_keys = [{"key": f.key, "label": agent.label_for(f.key)} for f in l.fields if f.visibility == HIDDEN]
        return {"id": mine.id, "n": mine.n,
                "title": self._live_title(self.board.listing(mine.id)) if listed else self._title(mine.draft),
                "unsaved": self._unsaved(mine),
                "active": self.board.listing(mine.id).active if listed else True,
                "kind": (mine.draft or {}).get("kind"), "listed": listed, "draft": mine.draft, "choices": dict(mine.choices), "chat": mine.chat,
                # Their own dating draft, as matching reads it: who and what ages it looks for.
                "wants": agent.dating_wants(mine.draft),
                "agent_busy": mine.agent_busy, "entry": mine.entry,
                "intros": intros, "hidden_keys": hidden_keys, "prefs": mine.prefs,
                "ready": self._ready(mine.draft), "has_profile": self._has_profile(mine),
                # What their phone offers to change too, after a change they made themselves (keep_change).
                "offers": [{"key": o["key"], "label": agent.part_label(o["key"]),
                            "after": ", ".join(o["after"]) if isinstance(o["after"], list) else o["after"]}
                           for o in self._live_offers(mine)],
                "followups": [{"question_id": q["question_id"], "question_text": q["question_text"],
                               "answer": self._answer_type(q["question_id"])} for q in mine.followups]}

    @staticmethod
    def _ready(draft) -> bool:
        """Can this draft go live (agent.ready: what it's for, and where unless it can meet online). Counted as the
        phone counts it (demo.html canSave), so every page offers Start matching by the same rule, and the phone
        first shows a new listing's profile by it too."""
        return agent.ready(draft)

    def _has_profile(self, mine: _Listing) -> bool:
        """Does the phone show this listing's profile: from the first time the listing is live or its draft is ready
        (_ready, what Start matching needs) on any page, for good, so a reload or another device shows it too, and
        the person can always get back to their draft and its shown or hidden choices, even once it stops being
        ready (a place hidden or cleared, a redraft). Until then the phone shows only the chat with their agent
        (demo.html profileShown), and the agent's replies don't point to a profile."""
        mine.has_profile = mine.has_profile or mine.id in self.board or self._ready(mine.draft)
        return mine.has_profile

    @staticmethod
    def _answer_type(question_id: str) -> dict:
        """What a question's answer is, so every page draws the right control (never a yes or no for a value):
        an age range, or yes or no."""
        if question_id == "age_range":
            return {"type": "age_range"}
        return {"type": "yes_no"}

    def phone_state(self, slot: str) -> dict:
        """Only what this person may know."""
        st = self._slot(slot)
        listings = [self._listing_state(l) for l in st.listings]
        current = next((l for l in listings if l["id"] == st.current), None) or \
            {"id": None, "listed": False, "draft": None, "chat": [], "agent_busy": False, "entry": None,
             "followups": [], "intros": [], "hidden_keys": []}
        return {"slot": slot, "taken_over": st.taken_over, "verified": st.person is not None, "name": st.name,
                "age_band": st.person.age_band if st.person else None,
                "settings": dict(st.settings),  # their own account settings, and nobody else's
                # What their own agent runs on, for Settings: a model, or rules (also once a model fell back). Never
                # which Matchmaker runs: matching internals are only for the presenter's panel.
                "agent_kind": "model" if self.llm is not None else "rules",
                "listings": listings, "current": st.current, "receipts": st.receipts[-60:],
                # The current listing, for clients that show one at a time.
                "chat": current["chat"], "draft": current["draft"], "entry": current["entry"],
                "listed": current["listed"], "followups": current["followups"], "agent_busy": current["agent_busy"],
                "hidden_keys": current["hidden_keys"],
                "intros": [i for l in listings for i in l["intros"]]}

    def room_state(self) -> dict:
        """Presenter view. Shows inside-the-room data for the demo listings only. A hidden detail's crowd is
        counted as matching counts it: different people with the same value (canonical), and it can raise a
        score only with K_CROWD other people besides the two being matched."""
        counts = self.board.value_counts()
        inside = []
        for st in self.slots.values():
            for mine in st.listings:
                if mine.id not in self.board:
                    continue
                l = self.board.listing(mine.id)
                rows = []
                for f in l.fields:
                    row = {"key": f.key, "label": agent.label_for(f.key), "value": ", ".join(f.strings()),
                           "visibility": f.visibility}
                    if f.visibility == HIDDEN:
                        best = max(counts[(f.key, canonical(s))] for s in f.strings())
                        row["crowd"] = best
                        row["usable"] = best >= K_CROWD + 2
                    rows.append(row)
                inside.append({"slot": st.slot, "name": st.name, "title": self._live_title(l),
                               "intent_text": l.intent_text, "fields": rows})
        # The audit runs again whenever anything it reads has changed, not on every page check-in.
        key = (self.generation, self.board.changes, len(self.intros), len(self.outputs["followups"]),
               len(self.outputs["reveals"]), len(self.outputs["details_in_use"]), len(self.warden.receipts),
               len(self.connections), len(self.consents), id(self.last_proposals), self._judge_sent_count())
        if getattr(self, "_audit", (None, None))[0] != key:
            self._audit = (key, self._leak_report())
        report = self._audit[1]
        return {"listings": len(self.board.listings()), "k": K_CROWD, "inside": inside,
                "events": self.events[-150:], "leak_report": {"ok": report["ok"], "findings": report["findings"],
                                                               "checked": report["checked"]},
                # The shared list as the Warden last built it (entries only: there are no counts on it).
                "details_in_use": self.in_use,
                "followup_menu": FOLLOWUP_MENU, "agent_status": self.agent_status,
                "matchmaker_status": self.matchmaker_status, "matchmaker": self.matchmaker_state(),
                "matchmaker_last_run": self.last_run,
                "taken_over": [{"slot": s, "name": st.name} for s, st in self.slots.items() if st.taken_over]}

    def _leak_report(self) -> dict:
        """The independent audit (leaks.leak_report) of everything sent, and of the last Matchmaker run's proposals
        against the board they were made on: once the board has changed since that run (someone hid a detail that
        was a reason), they are audited against the board as it was then (Marketboard.as_of), never against
        listings they were never made from, and never left out."""
        intros = [i.to_dict() for i in self.intros.values()]
        receipts = [r.to_dict() for r in self.warden.receipts[self._seed_receipts:]]
        current = self.board.changes == self.last_proposals_at
        judge_requests, wire, checked = self._judge_sent()
        report = leak_report(self.board, intros, self.outputs["followups"], self.outputs["reveals"], receipts,
                             self.last_proposals if current else [], self.connections, self.consents, K_CROWD,
                             APPROACH_LIMIT, details_lists=self.outputs["details_in_use"],
                             judge_requests=judge_requests, wire_requests=wire, wire_checked=checked)
        if current or not self.last_proposals:
            return report
        then = Marketboard()
        for listing in self.board.as_of(self.last_proposals_at):
            then.put(self.board.person(listing.person_id), listing)
        run = leak_report(then, [], [], [], [], self.last_proposals, set(), {}, K_CROWD, APPROACH_LIMIT)
        return {"ok": report["ok"] and run["ok"], "findings": report["findings"] + run["findings"],
                "checked": {**report["checked"], "proposals": run["checked"]["proposals"]}}

    def _judge_sent(self) -> tuple:
        """For the leak audit (A8), with the Matchmaker AI: (every request its judge was sent, from the Matchmaker's
        own record, each read against the board as it was when it was built; the requests the model's own record says
        went out since this Matchmaker started, each of which must be one of those; and those requests with the
        digests of every request the Warden's check let through since then, each of which must be one of them).
        (None, (), None) with the rules. A request the Matchmaker before a reset still had going out is left out of
        both comparisons: its board is gone (it was audited with that board). The first comparison stops once the
        Matchmaker has let old requests go (more than hybrid.PAYLOADS_KEPT), since a request the model holds may be
        one let go; the second never does (the Warden keeps every digest)."""
        if not isinstance(self.matcher, LLMMatcher):
            return None, (), None
        sent = self.matcher.sent_requests()
        model = getattr(self._judge, "judge", None)
        if not isinstance(model, OpenAIJudge):
            return sent, (), None
        wire = model.sent[max(0, self._wire_from - model.sent_dropped):]
        before = getattr(self, "_wire_before", set())
        wire = [r for r in wire if not before or json.dumps(r, ensure_ascii=False, sort_keys=True) not in before]
        return sent, ([] if self.matcher.payloads_dropped else wire), (wire, self.warden.judge_passed)

    def _judge_sent_count(self) -> tuple:
        """How many requests the Matchmaker AI's judge has been sent, by both records (the audit runs again on a new
        one)."""
        if not isinstance(self.matcher, LLMMatcher):
            return ()
        model = getattr(self._judge, "judge", None)
        wire = model.sent_dropped + len(model.sent) if isinstance(model, OpenAIJudge) else 0
        return len(self.matcher.payloads) + self.matcher.payloads_dropped, wire

    def state(self) -> dict:
        """Every phone (A, B and anyone taken over), each with only its own person's view, and the
        presenter's panel."""
        return {"phones": {s: self.phone_state(s) for s in self.slots}, "room": self.room_state()}
