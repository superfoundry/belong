# Belong. prototype

Belong. connects two people, for dating or practical needs, through AI agents: private
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
- **Several listings per person.** One person can be looking for a tutor in one listing
  and a date in another. Each listing has its own profile, agent chat and matches, and
  a person is never matched with their own other listing. (SPEC.md assumes one listing
  per person; this is a deliberate change.)
- **Asked when it matters.** A new listing starts with exact matches only. If it has no match
  yet, Belong. asks once: "No matches yet. Want to include close matches?" A yes turns on close
  matches (one-way: the other person isn't told unless you say yes) and, with them, follow-up
  questions ("Someone is looking for an air conditioner repair technician. Would that work for
  you?", never saying who). People you've both said yes to appear under **Found.**, where you
  can message each other.
- **Connector (mockup).** `/connector` shows one person using Belong. inside a ChatGPT-style chat
  app through a connector, in dark mode: connecting, drafting, going live and
  matches, all in the conversation. Saying yes and going live always need the person's own tap.
  It runs on its own demo session, separate from the phones, and isn't affiliated with OpenAI.
  "Test with a phone" opens person B on a phone beside the chat, in the same session, so you
  can match yourself across the chat and a phone. Ask the same agent for something different
  ("I'm also looking for...") and it starts a separate listing; every live listing has its card.
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

## Run it online (Google Cloud Run)

The [Dockerfile](Dockerfile) runs the same server in a container. Google builds it from this
folder (`.gcloudignore` keeps `.env` and `.git` out), and the OpenAI key and a demo password come
from Secret Manager at run time. With `BELONG_PASSWORD` set, every page asks for it (any user
name). It stays one shared demo session, so use one instance (see TRUST.md §5, limit 17).

```bash
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com secretmanager.googleapis.com
# The key from your .env, and a password you choose (typed, not shown, not saved in history):
grep '^OPENAI_API_KEY=' .env | cut -d= -f2- | tr -d '\r\n "' | gcloud secrets create openai-api-key --data-file=-
read -s "PW?Demo password: " && printf %s "$PW" | gcloud secrets create belong-password --data-file=- && unset PW
# Let Cloud Run read them:
SA="$(gcloud projects describe "$(gcloud config get-value project)" --format='value(projectNumber)')-compute@developer.gserviceaccount.com"
for s in openai-api-key belong-password; do gcloud secrets add-iam-policy-binding $s --member="serviceAccount:$SA" --role=roles/secretmanager.secretAccessor; done
# Build and deploy (run again to update; each deploy starts the demo fresh):
gcloud run deploy belong --source . --region us-central1 --allow-unauthenticated \
  --min-instances 1 --max-instances 1 \
  --set-secrets OPENAI_API_KEY=openai-api-key:latest,BELONG_PASSWORD=belong-password:latest
```

`--allow-unauthenticated` lets the page load at all; the demo password is what keeps people out.
`--max-instances 1` keeps everyone on the same demo; `--min-instances 1` keeps it from being reset
when idle, at the cost of one instance kept warm. To take it down:
`gcloud run services delete belong --region us-central1`.

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
    matcher.py     [open] OfflineMatcher (rules), LLMMatcher (planned)  §8.3
    intent.py      [open] reads free-text intents into structure
    narrow.py      [open] hard filters + text similarity -> shortlists  §8.2
    text.py        [open] small TF-IDF embedding
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
  server.py        web server (loopback only unless hosted; optional password)
  web/             demo.html, trust.html, index.html (inspector), nav.js (menu)
    ds/            the Belong. design system: tokens.css (generated by
                   tools/make_tokens_css.py), belong.css (components), fonts
tests/             privacy, entry, isolation, matching, demo, trust
```
