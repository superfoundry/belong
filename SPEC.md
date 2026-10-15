# Belong — prototype spec (v0.1)

Belong connects two people (dating or functional needs) through AI agents and a
**locked room**. The product promise is trust: private details can improve a match
without ever being revealed.

This file is the handoff from the design canvas to the build. Put it in the repo
root (and/or copy the key rules into `CLAUDE.md`) so every coding session starts
from the same understanding.

---

## 1. Priorities (from Jad, in order)

1. **Perfect privacy**
2. **No spam, no fakes**
3. **Efficiency**

When two goals conflict, the higher one wins.

## 2. The system in one paragraph

Each person is KYC-verified by a third party (e.g. Stripe Identity) and has an AI
agent (their own if Belong-certified, or one Belong provides). The agent helps the
person write an **intent** in plain text and a **listing** of details, each marked
hidden or shown. The listing enters the locked room through the **Warden agent's**
entry check. Inside, the **Matchmaker agent** reads everything on the
**Marketboard**, narrows the field, and proposes pairs where *both* intents are met.
Nothing the Matchmaker agent sees ever leaves the room. Proposals go out only
through the Warden agent's exit check, which strips anything not marked shown.
Both people must say yes before they are introduced.

## 3. Roles

| Role | Where it runs | Can read | Can write / output |
|---|---|---|---|
| Person | App | Their own data, their matches | Intent, listing, yes/no |
| Personal agent (A, B…) | Outside the room | Its person's data | Drafts intent + listing (person approves); relays follow-up answers |
| KYC provider | Third party | ID documents | `verified: bool`, `age_band` only |
| Warden agent | Boundary (entry + exit) | Inputs at entry, proposals at exit | Accept/reject; sanitized outputs; tamper-proof check receipts (no content) |
| Matchmaker agent | Inside the room | Everything on the Marketboard | Proposals (internal only); fixed-menu follow-up questions |
| Marketboard | Inside the room | — | Stores listings; nothing on it acts |

**Separation rule:** the agent that sees everything (Matchmaker) has no path out;
the agent that controls the exit (Warden) never decides who matches.

## 4. Data model (v0.1)

```text
Person
  id, verified: bool, age_band, created_at

Listing                      # one per person, versioned
  person_id
  intent_text: str           # "A climbing partner in Osaka on weekends…"
  exactness: float 0..1      # 1 = exact, 0 = wide net
  fields: [Field]
  updated_at

Field
  key: str                   # "first_name", "income_band", "kinks", …
  value: any
  visibility: "shown" | "hidden"
  sensitive: bool            # sensitive categories start hidden + off

Proposal  (internal to the room, never leaves as-is)
  person_a, person_b
  a_intent_met: bool, b_intent_met: bool
  score: float
  reason_fields: [field key] # must all be "shown" for BOTH people
  used_hidden_fields: [field key]

Introduction  (the only thing that leaves, via Warden exit)
  person_a_view: { shown fields of B, reason text from shown fields only }
  person_b_view: { shown fields of A, reason text from shown fields only }
```

## 5. Matching rules

- **One kind of match.** No tiers. The person's intent text + exactness define it.
- **Both must be looking.** A pair is proposed only if A's intent is met by B *and*
  B's intent is met by A. Nobody is approached who didn't ask.
- **Never a specific person.** The Warden agent rejects intents narrow enough to
  identify one individual (names, handles, exact workplaces + physical details…).
- **AI suggests, the person decides.** Defaults for exactness and hidden/shown come
  from the intent; the person can override everything.

## 6. Privacy rules for hidden fields (always on, not user-configurable)

- **Never the only reason.** A proposal must be justified by shown fields alone; a
  hidden field may only *raise* a score.
- **Only counts in a crowd.** A hidden field value is used only if at least *k*
  people share it (start with k = 20).
- **Never in the reason.** Reason text is generated only from fields shown by both.
- **Mutual reveal.** A hidden detail both people share is revealed only if both opt in
  after connecting.

## 7. Warden agent checks (fixed rules, no AI judgment)

Entry:
- Person is KYC-verified.
- Listing matches the schema; sensitive fields default to hidden.
- Intent passes the specific-person check.
- Rate limits on listing updates.

Exit:
- Output matches the Introduction schema exactly.
- Every field in each view is marked shown by its owner.
- Reason text references only reason_fields; no hidden values appear (string check).
- Follow-up questions come from the fixed menu only.
- Approach limits per person per week.
- Write a receipt: `{check, result, timestamp, hash}`; never the content.

## 8. Prototype scope (Step 2)

Goal: prove that free-text intent matching produces good matches, and that nothing
hidden leaks. No secure hardware yet, but built the way the room will run.

1. Generate 300 synthetic people (mix of dating + functional intents, realistic
   hidden fields).
2. Narrow: embeddings + hard filters → shortlist per person.
3. Matchmaker (pluggable interface):
   - `OfflineMatcher`: embeddings + rules, no API key.
   - `LLMMatcher`: plugs in later (Claude API for fake data; self-hosted for real).
4. Warden exit check as plain code.
5. Mutual yes simulation.
6. Web view to inspect: people, proposals, introductions, and a **leak report**
   (any hidden value that reached an output = test failure).

Build constraints that mirror the room:
- Matchmaker module has no network access and no logging of raw data.
- Only the Warden module can produce output objects.
- Tests fail on any leak.

## 9. Later steps

- **Step 3 — Locked room:** move Matchmaker into confidential computing (AWS Nitro
  Enclaves / Google Confidential Space / Azure confidential VMs; confidential GPUs
  for a self-hosted model); add attestation; audit Warden rules.
- **Step 4 — Trust and safety + app:** KYC integration, approach limits, consent
  screens, closed pilot.

## 10. Open questions

- Which details go in the default hidden/shown sets per intent type.
- Value of *k* for the crowd threshold.
- Certification process for outside agents.
- Model choice for the real Matchmaker agent (must run inside the room).
