# Belong prototype

Belong connects two people, for dating or practical needs, through AI agents: private
details can improve a match without ever being revealed.

This repository is the Step 2 prototype from [SPEC.md](SPEC.md). It uses synthetic people
(5,000 in the demo, set with `BELONG_PEOPLE`; 300 in tests and the batch run), with names from
their region, and plain Python; secure hardware comes in Step 3.

- **Any kind of match.** A listing is for dating, which has fixed rules of its own (who each
  person would like to meet, and their ages), or for anything else: a short "looking for" phrase
  and a "you are" phrase, written by the agent ("a fishing teacher" / "a fishing student", "a ska
  guitarist" / "a ska band"), plus whatever details matter to it, which the agent names on the
  spot ("My instrument is cello.", "I'd like their skill to be middle or high."). There is no
  built-in list of activities or details. Two people are introduced only when each one's
  "looking for" fits the other's "you are" and every wish each one shows is met by the other.
- **One vocabulary, from what people already show.** So that agents name the same thing the same
  way, every agent is given a shared list of the details, scales, phrases and places already in
  use, each one shown by at least 20 different people. The Warden builds it once a day from the
  listings alone, nothing hidden or sensitive ever goes on it, and phones never get it.
- **Anywhere in the world, in person or online.** Location is a country plus an optional
  city, and each listing says how it meets (in person, online, or either) and how far its
  person will travel to meet in person (same city, up to 30 km, up to 100 km). See "How
  matching works" below.
- **Several listings per person.** One person can be looking for a tutor in one listing
  and a date in another, up to five live listings. Each listing has its own profile, agent chat
  and matches, and a person is never matched with their own other listing.
- **Asked when it matters.** A new listing starts with exact matches only. If it has no match
  yet, Belong asks once: "No matches yet. Want to include close matches?" A yes turns on close
  matches (one-way: the other person isn't told unless you say yes). Follow-up questions ("Someone
  is looking for an air conditioner repair technician. Would that work for you?", never saying
  who) have a switch of their own. Each listing's gear has both switches, and Settings has two
  account-wide ones, both off until you turn one on: "Always allow questions" (new listings start
  with questions on) and "Always include close matches" (instead of asking, Belong turns them on
  when a listing has no match yet, and says so in its chat). People you've both said yes to appear
  under **Found**, where you can message each other.
- **Chatting with a synthetic person.** If you match with one of the synthetic people, they say hello
  and reply to your messages, written by the AI model (OpenAI's `OPENAI_CHAT_MODEL`, gpt-5-mini by
  default) from only what they'd know, sometimes with a reaction. No key: short built-in replies.
- **Connector (mockup).** `/connector` shows one person using Belong inside a ChatGPT-style chat
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
  `python3 -m belong run` then runs the same week again with a quarter of the people's agents writing
  one detail differently from most agents (another name, another scale, one left out or hidden), and
  prints both results: a detail written differently costs matches but should never make a wrong one.
  Then it runs both again with a quarter of the agents of listings that aren't dating adding a must the
  person never had, with and without the rule that a must comes from the person's own words, to show
  what that rule wins back, and with it while those agents quote the person's "no" as their answer, to
  show what a word rule can't win back; and last the second week with nobody typing their musts with
  their values (their descriptions still say most of them; it counts, in musts, how many nothing they
  typed says), with the person's yes to their agent's question and by values alone, to show what reading
  values alone would cost on those musts (TRUST.md #45).
- **[TRUST.md](TRUST.md)** lists every privacy promise with the code that enforces it,
  the test that proves it, and what isn't guaranteed yet. Start there.
- **[SPEC.md](SPEC.md)** is the product spec.

## How matching works

**First, what you want, in your agent's words and yours.** Your personal agent writes down what it
understood as details on your listing, where you can see and change them: the kind of match (dating,
or something else: what you want to meet someone *for*: "I enjoy sailing" in a dating request is a
hobby, not a search for a sailing partner); for dating, who you'd like to meet and their ages
("Looking for: a woman, 30–49"); otherwise your "looking for" and "you are" phrases; and any details
that matter, named on the spot: about you ("My instrument is cello."), what you'd like in the other
person ("I'd like their skill to be middle or high.", required unless you said "ideally"), or what
the request needs from both of you ("The tempo is brisk."). It works the ages out from what you said:
"early 30s", "late 20s to mid 30s", "around my age" (one age band either side of yours). When your
words could mean two things (meeting a woman, and a sport; or just "a partner"), it should ask you
one short question before saving, and "both" makes two listings, each with its own kind. The Claude
or OpenAI agent is told to ask, and to reuse the names and scales on the shared list when it means
the same thing (instructions, not something the code can check; a name on the list can't be reused
with another meaning). The rule-based agent asks when its own readers see two readings, and never
records a kind or an age range it only guessed. It also writes only words it recognises into
anything shown (words on the shared list, a known place, a fixed choice), and a value about you only
when you say it about yourself ("I'm an intermediate player", "I like hiking", "my level is…"),
never from a wish, a description of someone else (including a value of someone's: "a seasoned crew's
coach"), a value after a comma that starts another clause ("I'm a thrower, a glazer would suit me" writes
only "thrower"), the object of something you do ("I'm leading a weekend group") or a denial ("not", "hardly",
"far from", "neither … nor", "doubt", also over a clause that depends on it: "I don't think I'm…"):
the rest of what you type stays in
your description, and it asks when it doesn't recognise what you want ("What would you like a
partner for?"). The Warden lets what the agent understood in only as fixed choices (no listing can
ask for anyone under 18), and the details it names on the spot only in a fixed form: short plain
words, no contact details, no claim to be "verified", nothing that singles out one person, and
nothing sensitive (religion, health, politics, income and intimacy go only in their own hidden
categories). What you'd like in the other person, and what you both need to share, is always shown.
Nobody can use it to find one particular person, in three layers where the AI only ever adds a refusal.
The Warden's fixed floor refuses handles, contact details, an ex, clues that together narrow to one
person, and two capitalised words in a row in the middle of a sentence, which look like a full name (also with
"van", "de la" or another lower-case name particle between them, or after an initial), unless one of them
is part of the city or country your listing shows or of a place on the shared list, written out whole, the
pronoun I, a word in capitals, a month or a weekday (no list of places, languages or titles). A live listing
keeps the listed places it was accepted with, so a later list never refuses its next update. The Claude or OpenAI agent
also judges from your words and the context (a first name alone, a nickname, a name in lower case): it
saves nothing it judges singles someone out and tells you why, and when the floor refuses two capitalised
words that are really a place or a title, it rewords them and says so. The rule-based agent has no model
and relies on the floor; so does Save draft on the phone, which runs the same rules. The Matchmaker AI
(below) may also decline to match a listing it judges is hunting one person, which can only leave pairs
out. Clues are counted only about the person you're looking for, never your own job or looks (what you
say about yourself, in what you offer or a detail about you, is read as yours, so only a clue about someone
else written there counts): one or two common details of how they look are fine, but the Warden refuses a
place or time where they can be found (where they work or live, a workplace you share with them, someone you
saw or met, "the" and a clock time) together with how they look, three or more details of how they look, or a
scar, birthmark, tattoo or piercing described with where it is or what it shows. A character that shows
nothing never hides any of these rules. A looks word counts only where it describes a person, never a thing. The rule reads
English sentences with a few words for looks, marks, workplaces, homes, meeting someone and people;
anything else is left to the model agent's judgement (its instructions, untested against a live model),
and the rule-based agent has only the rule (TRUST.md limit 26).
Matching uses these details first. Your description is never shown to anyone and is never part of a
reason, but matching does read it: to tell dating from the rest, for who and what ages a dating
listing wants and how a listing meets when those details aren't shown, and as text similarity, which
counts toward a pair's score and orders shortlists (TRUST.md §5, limit 4). A detail you hide never decides a match:
matching then reads your description as if the detail weren't there.

**Then the Matchmaker** ([matcher.py](belong/matching/matcher.py), [narrow.py](belong/matching/narrow.py))
looks at every listing in the same order, using only what people show (and their descriptions, as
above):

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
3. **What, by meaning.** Dating is compared with dating. Everything else is compared by its
   "looking for" and "you are" phrases, but only with listings that share a word (or the start of
   one, so "repair" meets "repairman") between what one looks for and what the other is, shows
   about themselves or lists as topics; a role word alone ("partner", "teacher") doesn't count.
   This never drops a pair the rules below could accept; `python3 tools/compare_narrowing.py`
   checks it on 3,000 people.
4. **Both must be looking.** Two people are introduced only when each is what the other asked for,
   judged from shown details only. For dating: the gender each would like to meet, and an age in
   the other's range. For anything else: what one looks for is what the other is, and what you look
   for but aren't yourself, they must be (two first violinists both looking for a second violinist
   aren't each other's answer). Two phrases with the same main noun are compared on their other
   words, in any language: "a hockey goalkeeper" isn't "a handball goalkeeper", and at the usual
   exactness two phrases that each say something the other doesn't never meet, whatever they share
   and in any word order: "an evening sketching partner" isn't "an evening baking partner", nor
   "amico di pesca serale" "amico di corsa serale" (a wide net, under 0.5, accepts half of those
   words). And always: every wish each one shows is met by the other's own
   shown detail of that name (the same value, or one of the wished steps; a wide net also takes the
   step next to one). A wish for "any" (or for every step of its scale) says nothing, so it is always
   met and never asks for the detail. An own "any" meets a wish only for a listing that isn't strict
   (exactness under 0.8), and it is never a reason: it says nothing two people have in common. A
   detail the other doesn't show, or names or scales differently, fails a required wish: that costs a
   match but never makes a wrong one. A wish stays required only when you said it: what you typed
   says one of its values; or a save kept it as "ideally", so your agent was told to ask whether it
   must be met, and it quotes your answer from the message you typed next; or it was already a must on
   your draft. One your agent saves as a must that you never said is kept as "ideally", so it no longer
   shuts out people who don't show that detail, though it still fails anyone who shows a value that
   doesn't meet it. At exactness 0.8 or more every wish shuts out who doesn't show it, so a listing that
   exact with a wish you never said is made less exact (0.7) until you answer. It is a word rule, not a
   reading of meaning: your answer counts whatever it says, any one value counts, and a must you never
   put into words is only "ideally" until you say it (TRUST.md #45, limit 28).
5. **Never the only reason.** A pair must also share something both show (where they can meet,
   their phrases, or a detail they have in common) and score well enough without any hidden detail
   (on shown details, and how alike the two descriptions read).
   Hidden details can only raise the score of a pair that already qualifies, and only when at least
   20 other people share them.
6. **Fair ranking.** Among pairs that qualify, a listing that already had introductions today
   ranks a little lower, and one updated in the last day a little higher, so new and quiet
   listings get seen. This only changes the order; it never lets a pair skip a rule. When only the
   demo's phones are matched, their own introductions don't count against meeting each other.

**For dating, a few things worth knowing:**

- **What you didn't say isn't a strict preference.** A wish you didn't make is no wish: any value
  will do. No ages: people within one age band of your own (25 to 39 if you're 30 to 34). If you've
  turned on questions, Belong may ask what ages you'd like, to narrow it.
- **Your gender** is how people looking for a woman or a man find you, so the profile asks for it
  and says why. You can keep it hidden, but a hidden detail never decides a match, so then only
  people open to anyone can match you, and you may get fewer matches. Without knowing who you'd
  like to meet, Belong can't match a dating listing at all.
- **When a listing records nothing,** for instance an older one, the description is read by simple
  rules: wanting to meet a man or a woman, a boyfriend or a girlfriend, or looking for love is
  dating; "a man with a van to help me move" or "a woman to clean my flat" isn't, and nor is
  selling, buying or giving something away. They read how a sentence is built and a few general
  words (a date, a drink, love), never a list of trades or hobbies: "dating" before a noun leaves the
  noun to say what it is ("a dating romance" is dating, "a dating podcast" and "dating coaches"
  aren't), "dating" before a person asks for them ("a dating partner", "dating women", also with describing
  words first, "dating older men", or a word made from a verb, "dating widowed men"), and in "a man
  with a nose for a bargain" the "for" describes him. Ages work in your own words ("aged 25 to 34",
  "early 30s", "in their thirties", "around 30", "a woman late 40s", "a man, ideally 40s", "a man in
  late 30s", "a 40-something woman", "a woman 40s+", "of thirties or forties"), but not decades that
  describe something else ("a fan of 70s cinema", "a fan of 80s and 90s films"), and what you say
  about your own age ("I'm 34", or "Me: woman mid-40s" in a sentence of its own) isn't read as the age
  you want. These rules are a fallback: what people type has no end, which is why your agent records
  what it understood instead.

**The Matchmaker AI** ([hybrid.py](belong/matching/hybrid.py), [matchmaker_openai.py](belong/matchmaker_openai.py))
is a mock of the model that will run enclosed: OpenAI's API stands in for it, for synthetic and test profiles
only. The rules above still narrow and decide everything hard and exact; the AI judges only the fuzzy part of
each pair they allow (does each give what the other wants, also where two agents named or scaled a detail
differently), from what both listings show and their descriptions, with every first name, id and hidden text
taken out. Its answer is read as verdicts and a number only, the proposal is the rules' own contract, and the
Warden checks it as before. Every request goes only once the Warden's check passes it, against the board as it
is just before it goes (`warden.exit_judge_request`), and is kept and audited again by the leak report
(`leaks.py`, A8); every security check runs as if it were enclosed (TRUST.md #39 to #44, limit 27).

## Run it

Python 3.10+. Everything runs on the standard library, except the optional Claude-powered
personal agent in the demo.

**To use a real AI agent in the demo** (otherwise each phone falls back to a simple
rule-based agent, and the presenter panel says so): copy `.env.example` to `.env`, put your
OpenAI key in it (`OPENAI_API_KEY=...`), and restart `python3 -m belong serve`. `.env` is
git-ignored. To use Claude instead, `pip install -r requirements.txt`, set
`ANTHROPIC_API_KEY`, and leave `OPENAI_API_KEY` empty (or set `BELONG_AGENT=claude`).

**The Matchmaker AI in the demo** is on when `OPENAI_API_KEY` is set (the presenter panel says
"Matchmaker AI: OpenAI ... (a mock of the enclosed model; synthetic and test profiles only)", or
"Matchmaker: rules"). `BELONG_MATCHMAKER=rules` turns it off, `BELONG_MATCHMAKER=ai` on. Its settings:
`OPENAI_MATCHMAKER_MODEL` (else `OPENAI_MODEL`, else gpt-5); `BELONG_MATCHMAKER_URL` to use another
server that speaks the Chat Completions format (a self-hosted open-weights model behind vLLM, llama.cpp
or Ollama; https, or http on this machine only, and no redirect is followed) with `BELONG_MATCHMAKER_KEY`
if it needs one (the OpenAI key only ever goes to OpenAI); `BELONG_MATCHMAKER_TIMEOUT` (seconds, default
120). Only the phones'
listings are matched in the demo, so only their pairs reach it. **Cost:** one request per phone listing
with up to 40 candidates, plus one small request asking about each phone listing alone (once for each version
of it), at most 80 pairs a run (the rest decided by the rules), each verdict kept until a listing of its request
changes; the presenter panel shows the tokens used. If it fails, or the Warden's check stops a request, the rules
decide.

**The batch run stays on the rules**, network-free, as in CI. `python3 -m belong run --matcher ai
[--ai-pairs 80]` runs the week with the Matchmaker AI (live: it costs money); `--matcher hostile` runs it
with a stand-in that says yes to every pair and answers out of shape (no model, no network, writes
nothing), to show the leak reports still pass. `python3 tools/compare_judges.py [--n 300] [--seed 7]`
compares the rules with the Matchmaker AI on the same shortlists and on pairs outside them, by
disagreement bucket, with the simulation's ground truth saying who is right, and the tokens used; it
sends nothing without `--yes` (live, it costs money).

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

## The website

`site/` is the public website, two pages by choice: the landing page (`index.html`, short and visual, for anyone)
and the Use page (`use.html`): the ways to use Belong first, then everything an AI agent or developer needs to
connect. That technical part is Markdown, `site/use.md`, made into the page through `site/_page.html`; the Markdown
itself is published beside it, with `site/llms.txt` pointing to it, so agents can read it as plain text. It is plain
HTML on the Belong design system, with its logo files, and runs nothing on the visitor's side but the hero scene's
pause button. `python3 tools/build_site.py` builds it into `out/site` with the design system's files, fonts and logo
copied in, so it loads nothing from any other host. It never links to the demo, which is internal, or to the source
repository. Locally, `python3 -m http.server 8100 -d out/site` (or the `site` preview in `.claude/launch.json`).

**It deploys itself.** Every merge to `main` that passes the tests is built and published with GitHub Pages
([ci.yml](.github/workflows/ci.yml), the `site` job). One-time setup: Settings > Pages > Source: *GitHub
Actions*.

`tests/test_site.py` holds the built pages and the plain text to the same rules as the app's pages: the name without
its stop and the design system's logo, "shown" and "hidden" only, colours from the tokens alone, motion only without
reduced motion and a pause button on anything that loops, saying what is simulated, none of the demo's examples,
no link to the demo or the repository, every local link working, and nothing loaded from elsewhere. It checks there
are exactly the two pages.

## Layout

```text
belong/                                    [open] = published as open source
  schema.py        [open] data model: built-in fields, open-detail     SPEC §4
                          rules, the one normaliser, comparisons,
                          sensitive words, follow-up menu
  warden.py        [open] entry + exit checks, sealed outputs, the     §7
                          shared list, hash-chained receipts
  leaks.py         [open] independent leak audit (and of what the      §8.6
                          Matchmaker AI is sent)
  attest.py        [open] measurements, simulated attestation, the
                          list of published files
  matching/        private matching: no network, files or logging
    __init__.py    [open] isolation rules for matching code
    contract.py    [open] the only shapes the Matchmaker may hand the Warden
    marketboard.py [open] stores listings
    matcher.py     [open] OfflineMatcher (rules)                        §5, §8.3
    hybrid.py      [open] LLMMatcher: the rules narrow, an injected     §3, §8.3
                          judge judges the fuzzy part (no network here)
    intent.py      [open] reads free-text intents into structure
    narrow.py      [open] how, where (travel), what (word index)       §8.2
                          -> shortlists
    text.py        [open] small TF-IDF embedding
  agent.py         personal agent: drafts listings, relays answers     §3
                   (rule-based fallback)
  agent_llm.py     personal agent powered by Claude, plus the instructions
                   and draft checks shared by every model-powered agent
  agent_openai.py  personal agent powered by OpenAI (standard library only)
  matchmaker_openai.py [open] the Matchmaker AI's judge over a Chat     §3, §8.3
                   Completions API (OpenAI's, as a mock; standard library only)
  synth.py         synthetic people (regional names) + ground truth §8.1
  evaluate.py      match quality vs ground truth; narrowing's word index
                   vs the catch-all group it replaced
  sim.py           one simulated week, end to end                      §8.5
  demo.py          two-person demo session + boundary log, and phones for
                   synthetic people taken over from the inspector
  probe.py         canary test + attack suite for outside agents, and stand-in
                   Matchmaker AIs that answer as badly as they can
  server.py        web server (loopback only unless hosted; optional password)
  web/             demo.html, trust.html, index.html (inspector), nav.js (menu)
    ds/            the Belong design system: tokens.css (generated by
                   tools/make_tokens_css.py), belong.css (components), fonts
tests/             privacy, entry, isolation, matching (and its order), demo, trust
tools/             compare_narrowing.py, compare_judges.py (rules vs the Matchmaker AI;
                   live, opt-in), make_tokens_css.py
```
