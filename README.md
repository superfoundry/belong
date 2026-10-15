# Belong prototype

Belong connects two people, for dating or practical needs, through AI agents: private
details can improve a match without ever being revealed.

This repository is the Step 2 prototype from [SPEC.md](SPEC.md). It uses 300
synthetic people and plain Python; secure hardware comes in Step 3.

- **Any kind of match.** Besides the named kinds (dating, climbing, running, language
  exchange, tutoring, cofounder, roommate), the agent can create an open-ended listing:
  a short "looking for" phrase and a "you are" phrase, written by the agent ("a fishing
  teacher" / "a fishing student", "a ska guitarist" / "a ska band"). Two people are
  introduced only when each one's "looking for" fits the other's "you are".
- **Anywhere in the world.** Location is a country plus an optional city; two people
  match on location when they share a city, or a country if one of them is open to
  anywhere in it.
- **[TRUST.md](TRUST.md)** lists every privacy promise with the code that enforces it,
  the test that proves it, and what isn't guaranteed yet. Start there.
- **[SPEC.md](SPEC.md)** is the product spec.

## Run it

Python 3.10+. Everything runs on the standard library, except the optional Claude-powered
personal agent in the demo.

**To use a real AI agent in the demo** (otherwise each phone falls back to a simple
rule-based agent, and the presenter panel says so): copy `.env.example` to `.env`, put your
OpenAI key in it (`OPENAI_API_KEY=...`), and restart `python3 -m belong serve`. `.env` is
git-ignored. To use Claude instead, `pip install -r requirements.txt`, set
`ANTHROPIC_API_KEY`, and leave `OPENAI_API_KEY` empty (or set `BELONG_AGENT=claude`).

## Layout

```text
belong/                                    [open] = published as open source
  schema.py        [open] data model, field registry, follow-up menu   SPEC §4
  warden.py        [open] entry + exit checks, sealed outputs,         §7
                          hash-chained receipts
  leaks.py         [open] independent leak audit                       §8.6
  attest.py        [open] measurements, simulated attestation, the
                          list of published files
  matching/        private matching: no network, files or logging
    __init__.py    [open] isolation rules for matching code
    contract.py    [open] the only shapes the Matchmaker may hand the Warden
    marketboard.py [open] stores listings
    matcher.py       OfflineMatcher (rules), LLMMatcher (planned)       §8.3
    intent.py        reads free-text intents into structure
    narrow.py        hard filters + text similarity -> shortlists       §8.2
    text.py          small TF-IDF embedding
  agent.py         personal agent: drafts listings, relays answers     §3
                   (rule-based fallback)
  agent_llm.py     personal agent powered by Claude, plus the instructions
                   and draft checks shared by every model-powered agent
  agent_openai.py  personal agent powered by OpenAI (standard library only)
  synth.py         300 synthetic people + ground truth                 §8.1
  evaluate.py      match quality vs ground truth
  sim.py           one simulated week, end to end                      §8.5
  demo.py          two-person demo session + boundary log
  probe.py         canary test + attack suite for outside agents
  server.py        local server (loopback only)
  web/             demo.html, trust.html, index.html (inspector)
tests/             privacy, entry, isolation, matching, demo, trust
```
