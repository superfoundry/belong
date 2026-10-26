# Belong. prototype

Belong. connects two people, for dating or practical needs, through AI agents: private
details can improve a match without ever being revealed.

This repository is the Step 2 prototype from [SPEC.md](SPEC.md). It uses synthetic people
(5,000 in the demo, set with `BELONG_PEOPLE`; 300 in tests and the batch run), with names from
their region, and plain Python; secure hardware comes in Step 3.

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
- **Inspector.** `/inspector` shows the batch run (`python3 -m belong run`): every synthetic person,
  proposal, introduction and receipt. Its people are the live demo's first 300 (same ids, same seed).
  **Take over** on a person opens their phone in a small window, so you can answer as them in the live
  demo: their listing is matched like a phone's, what the Warden sends them arrives there, and nothing
  is decided for them until you **Hand back** (above their phone). The presenter panel lists who's taken over.
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

## Run it online

Either way the demo password (`BELONG_PASSWORD`) is asked for on a sign-in page before anything else, and the
hosted demo won't start without one. It stays one shared demo session (TRUST.md §5, limit 17).

### On a VM (Debian 12 or Ubuntu 24.04, e.g. Google Compute Engine)

The demo keeps listening only on the VM itself; [Caddy](deploy/Caddyfile) is the way in, over HTTPS
with an automatic certificate, and [systemd](deploy/belong.service) keeps it running.

1. Let web traffic reach the VM: on Google Cloud, edit the VM and tick *Allow HTTP traffic* and
   *Allow HTTPS traffic*. (Reserving its external IP as static keeps the address from changing.)
2. Copy the code over without `.env` or history: `git archive --format=tar.gz -o belong.tar.gz HEAD`,
   then upload it (the browser SSH window has *Upload file*) and `mkdir belong && tar -xzf belong.tar.gz -C belong && cd belong`.
3. `sudo sh deploy/setup.sh ADDRESS`, where ADDRESS is a domain pointing at the VM, or, without one,
   its external IP with dashes plus `.sslip.io` (e.g. `34-1-2-3.sslip.io`).
4. `sudo nano /opt/belong/.env`: put in `OPENAI_API_KEY=...` and `BELONG_PASSWORD=...`, then
   `sudo systemctl restart belong`. Open `https://ADDRESS`.

`sudo journalctl -u belong` shows the server's own output (startup and errors, never requests).

**Updates deploy themselves.** Once set up, every merge to `main` that passes the tests is sent to
the VM by GitHub ([ci.yml](.github/workflows/ci.yml)), and the demo restarts fresh. One-time setup:
a deploy key (its private half is a GitHub secret, `VM_SSH_KEY`), the VM's address and host key as
repo variables (`VM_HOST`, `VM_KNOWN_HOSTS`), and on the VM `sudo sh deploy/allow-deploys.sh "<public
key>"`. That key can only run [belong-update](deploy/belong-update), which replaces the app's code and
restarts it; it never runs anything from the upload as root and never touches `.env`. Without the
setup, upload a new archive and run step 3 again (`.env` is kept).

### Or on Google Cloud Run

The [Dockerfile](Dockerfile) runs the same server in a container. Google builds it from this
folder (`.gcloudignore` keeps `.env` and `.git` out), and the OpenAI key and a demo password come
from Secret Manager at run time. Everyone has to share the one demo, so it runs as one instance.

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
  synth.py         synthetic people (regional names) + ground truth §8.1
  evaluate.py      match quality vs ground truth
  sim.py           one simulated week, end to end                      §8.5
  demo.py          two-person demo session + boundary log, and phones for
                   synthetic people taken over from the inspector
  probe.py         canary test + attack suite for outside agents
  server.py        web server (loopback only unless hosted; optional password)
  web/             demo.html, trust.html, index.html (inspector), nav.js (menu)
    ds/            the Belong. design system: tokens.css (generated by
                   tools/make_tokens_css.py), belong.css (components), fonts
tests/             privacy, entry, isolation, matching, demo, trust
```
