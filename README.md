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
- **Anywhere in the world, in person or online.** Location is a country plus an optional
  city, and each listing says how it meets (in person, online, or either) and how far its
  person will travel to meet in person (same city, up to 30 km, up to 100 km). See "How
  matching works" below.
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
- **Chatting with a synthetic person.** If you match with one of the synthetic people, they say hello
  and reply to your messages, written by the AI model (OpenAI's `OPENAI_CHAT_MODEL`, gpt-5-mini by
  default) from only what they'd know, sometimes with a reaction. No key: short built-in replies.
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
  Each description is shown as it went live, as in the live demo; anything the batch run's follow-up
  answers added ("Aged 25 to 39.") is marked **Batch-run answer**, because the live demo doesn't have it.
- **[TRUST.md](TRUST.md)** lists every privacy promise with the code that enforces it,
  the test that proves it, and what isn't guaranteed yet. Start there.
- **[SPEC.md](SPEC.md)** is the product spec.

## How matching works

**First, what you want, in your agent's words and yours.** Your personal agent writes down what it
understood as details on your listing, where you can see and change them: the kind of match (what
you want to meet someone *for*: "I enjoy climbing" in a dating request is a hobby, not a search for
a climbing partner) and, for dating, who you'd like to meet and their ages ("Looking for: a woman,
30–49"). It works the ages out from what you said: "early 30s", "late 20s to mid 30s", "around my
age" (one age band either side of yours). When your words could mean two things (meeting a woman,
and climbing; or just "a partner"), it should ask you one short question before saving, and "both"
makes two listings, each with its own kind. The Claude or OpenAI agent is told to ask (an
instruction, not something the code can check); the rule-based agent asks when its own readers
see two readings, and never records a kind or an age range it only guessed. The Warden lets these details in only as fixed
choices (no listing can ask for anyone under 18), and matching uses them first. Your description is
read only for what they don't say, and it's never shown to anyone. A detail you hide never decides a
match: matching then reads your description as if the detail weren't there.

**Then the Matchmaker** ([matcher.py](belong/matching/matcher.py), [narrow.py](belong/matching/narrow.py))
looks at every listing in the same order, using only what people show:

1. **How you'd meet.** If both listings are happy to meet online, they can be matched wherever
   they are. If either one only meets in person, both must meet in person, and step 2 decides.
   A listing that doesn't say reads its description ("online is fine" means either; "not online"
   means in person). Only "online only" or "just online" makes a listing online only. Dating is
   always in person.
2. **Where, by travel distance.** Two people in the same city can meet. Two people in different
   cities can meet only if the distance between them is within what *both* said they'd travel:
   Osaka and Kobe (about 28 km) need "up to 30 km" from both; Osaka and Kyoto (about 43 km) need
   "up to 100 km". Someone who doesn't say stays in their own city. Distances come from a short
   list of approximate city centres, each known only in its own country, and are used only when
   both people show the same country: a Cambridge in the United States is never near London. Any
   other city, or a city shown without a country, only matches a city of the same name. Someone
   who gives a country but no city is open to anywhere in it.
3. **What, by meaning.** Named kinds (dating, language exchange, tutoring...) are compared with
   their own kind: the kind on the listing. Everything else is compared by its "looking for" and
   "you are" phrases, but only with listings that share a word (or the start of one, so "repair"
   meets "repairman") between what one looks for and what the other is, shows about themselves or
   lists as topics. This never drops a pair the rules below could accept;
   `python3 tools/compare_narrowing.py` checks it on 3,000 people.
4. **Both must be looking.** Two people are introduced only when each is what the other asked for,
   judged from shown details only. For dating: the gender each would like to meet, and an age in
   the other's range.
5. **Never the only reason.** A pair must also share something both show (the same city, an
   interest, when they're free) and score well enough on shown details alone. Hidden details can
   only raise the score of a pair that already qualifies, and only when at least 20 people share them.
6. **Fair ranking.** Among pairs that qualify, a listing that already had introductions today
   ranks a little lower, and one updated in the last day a little higher, so new and quiet
   listings get seen. This only changes the order; it never lets a pair skip a rule. When only the
   demo's phones are matched, their own introductions don't count against meeting each other.

**For dating, a few things worth knowing:**

- **What you didn't say isn't a strict preference.** No climbing level: any level will do. No ages:
  people within one age band of your own (25 to 39 if you're 30 to 34). If you've turned on
  questions, Belong. may ask what ages you'd like, to narrow it.
- **Your gender** is how people looking for a woman or a man find you, so the profile asks for it
  and says why. You can keep it hidden, but a hidden detail never decides a match, so then only
  people open to anyone can match you, and you may get fewer matches. Without knowing who you'd
  like to meet, Belong. can't match a dating listing at all.
- **When a listing records nothing,** for instance an older one, the description is read by simple
  rules: wanting to meet a man or a woman, a boyfriend or a girlfriend, or looking for love is
  dating; "a man with a van" or "a woman to clean my flat" isn't, and nor is selling, buying or
  giving something away. Ages work in your own words ("aged 25 to 34", "early 30s", "in their
  thirties", "around 30"), and what you say about your own age ("I'm 34") isn't read as the age you
  want. These rules are a fallback: what people type has no end, which is why your agent records
  what it understood instead.

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
    narrow.py      [open] how, where (travel), what (word index)       §8.2
                          -> shortlists
    text.py        [open] small TF-IDF embedding
  agent.py         personal agent: drafts listings, relays answers     §3
                   (rule-based fallback)
  agent_llm.py     personal agent powered by Claude, plus the instructions
                   and draft checks shared by every model-powered agent
  agent_openai.py  personal agent powered by OpenAI (standard library only)
  synth.py         synthetic people (regional names) + ground truth §8.1
  evaluate.py      match quality vs ground truth; narrowing's word index
                   vs the catch-all group it replaced
  sim.py           one simulated week, end to end                      §8.5
  demo.py          two-person demo session + boundary log, and phones for
                   synthetic people taken over from the inspector
  probe.py         canary test + attack suite for outside agents
  server.py        web server (loopback only unless hosted; optional password)
  web/             demo.html, trust.html, index.html (inspector), nav.js (menu)
    ds/            the Belong. design system: tokens.css (generated by
                   tools/make_tokens_css.py), belong.css (components), fonts
tests/             privacy, entry, isolation, matching (and its order), demo, trust
tools/             compare_narrowing.py, make_tokens_css.py
```
