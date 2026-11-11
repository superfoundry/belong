# Belong: working rules for every coding session

Read SPEC.md (the product spec) and TRUST.md (how each privacy promise is enforced
and tested) before changing code.

## Priorities (in order; the higher one wins a conflict)

1. Perfect privacy
2. No spam, no fakes
3. Efficiency

## How we fix things

- **Fix the general mechanism, never the example.** A reported case ("a beginner ice hockey coach
  didn't meet a figure skater") is one instance of a wider problem. Name the general rule it breaks
  and fix that. Never add example-specific words, activity lists, synonyms or prompt examples that
  mirror the case; prefer letting the person's agent supply meaning (fields like `topics`) over lists
  in matching code. Then check the fix on new, unrelated examples, and keep prompt examples
  different from the reported ones.
- **Demo examples never define the product.** The early examples (climbing, running, the demo's cities,
  the synthetic vocabularies in `synth.py`) are never a detail's label, its allowed values, an allow-list
  or a matching rule. Details beyond the built-ins are open details the person's agent names on the spot,
  held to one fixed rule (`schema.detail_problems`), and agents keep names consistent through the
  Warden's shared list (`Warden.details_in_use`), never through a list in code. A label, a Warden sentence
  or a question's answer must be true for any listing that can carry it. Where a detail may go is a fixed
  rule, not a list: open details go on either kind, intimacy only on dating (`schema.detail_problems`),
  who and what ages only on dating (`agent.carries`), and the only fixed choices are the built-ins' own
  (`agent.field_options`). `tests/test_neutral.py` guards this: only the simulation, its grading, the
  command line, the demo's board and the probe's sandbox import `synth.py` (each only the names it is
  allowed), and no word of its stories, hobbies, genres or subjects is in product code or a page's script.
  It reads every string and name from the syntax tree (f-strings included, on every Python version), and
  no product module or page may write out two or more words of one demo vocabulary together (its
  languages, levels, paces or band roles), in a list, a set, a dict's keys or one string.

## Rules that must never break

- **The room has no way out.** Code in `belong/matching/` imports only `re`, `math`,
  `collections`, `dataclasses`, `typing`, `unicodedata` and `belong.schema`. No network, files,
  logging, print, subprocess. `tests/test_isolation.py` enforces this. The Matchmaker AI's judge is
  injected from outside it (`hybrid.LLMMatcher(judge)`, plain data in, plain data out): what it is sent
  is a fixed rule (shown fields and descriptions, every first name, id and hidden text taken out, labels of
  one request), each request goes only once the Warden's check passes it against the board as it is then
  (`Warden.exit_judge_request`: whoever hands a judge to the hybrid puts `checked_judge` or the demo's
  `_JudgeOffLock` screen in front of it), every request is kept, and `leaks.py` audits each one (A8).
  Whether a listing hunts one person is asked about it alone; a verdict counts only while every listing of
  its request is unchanged.
- **The Matchmaker AI is a mock while it runs on OpenAI.** Synthetic and test profiles only, and every
  security check runs as if it were enclosed; TRUST.md and the trust page must keep saying so. It only
  judges inside the rules (the hard rules, the Proposal contract, the Warden's checks) and its answer is
  read strictly (`hybrid.read_answer`). Any new way it could misbehave gets a stub-judge attack in
  `belong/probe.py` with a remove-the-rule case in `tests/test_trust.py`.
- **Only the Warden creates outputs.** Introduction / FollowUpOut / Reveal / DetailsInUse need the
  private seal in `belong/warden.py`. Never reference `_SEAL` elsewhere.
- **The Warden checks; it never decides who matches.** Its checks are fixed rules,
  no AI judgment.
- **Hidden fields:** never the only reason (qualifying logic gets a `ShownView`);
  only counts in a crowd (k = 20 other people, counted in people, not listings); never in the reason (the
  Warden writes reasons from templates using fields both people show, and accepts open-detail reasons only
  if they are exactly `schema.detail_reasons`; its string checks read only what fills the templates, and
  `leaks.py` keeps its own copy of them); mutual reveal only with both opt-ins, never of a wish. A reissue
  never leaves something since hidden on screen: one that fails its checks is reduced (`reissue_reduced`).
- **The shared list of details in use** is built only by the Warden, from the board alone, once a day:
  nothing on it until K_CROWD different people show it on listings unchanged for a week, never anything
  hidden or sensitive, no counts, each text shaped like what it stands for (`schema.LIST_PHRASE_WORDS`,
  `LIST_VALUE_WORDS`, no sentence punctuation). `leaks.py` rebuilds it independently (A7) and `probe.py`
  attacks it. Agents get it as data after `schema.LIST_PREFIX`, never in their instructions, and recheck
  its shape with their own code; phones never get it.
- **Tests fail on any leak.** Never weaken a test, an audit in `belong/leaks.py`, or
  a Warden check to make something pass. If a change needs a new output type, add a
  Warden exit check *and* a leak audit for it.
- **leaks.py stays independent of warden.py.** Don't import Warden helpers into it.
- **Both agents are open source; the Matchmaker is still treated as hostile.** Everything in
  `OPEN_SOURCE` (belong/attest.py: `WARDEN_FILES` + `MATCHMAKER_FILES`) is published and
  measured per agent. Published code may only import published code, and Warden files must
  never import Matchmaker files (tests/test_trust.py enforces both). The privacy guarantee
  must never depend on the Matchmaker behaving well: add an attack to belong/probe.py for
  any new way it could misbehave.
- **Notifications are opt-in, per listing, and enforced by the Warden.** Questions only to
  listings with `notify_questions`; close matches only one-way to listings with `notify_close`;
  an "also offer" question needs both and never names who asked. Never default either to on.
  The person's account settings ("Always include close matches", "Always allow questions", both off
  until they turn one on) only set a listing's own choices, through the same Warden-checked update,
  never bypass them (`DemoSession.set_settings`, `_close_by_setting`).
- **Messages only between connected people** (`DemoSession.send_message`), never on the
  Marketboard, never in the presenter log.
- **The receipt log is append-only and hash-chained.** Never rewrite or drop receipts.
- **Phones see only what their person may know** (`DemoSession.phone_state`). Matching
  internals go only to the presenter panel.
- **Don't overclaim.** "Not even we can access it" needs the Step 3 enclave work
  (see TRUST.md §5). Pages and docs must keep saying what is simulated.

## How we talk about it (user-facing copy)

- In text the name is **Belong**, with no full stop: page copy, shown attributes, page titles, agent
  messages, docs and comments. "Belong." with its full stop is the logo only: the wordmark, and the
  phone's tab labels "Belong." and "Found.", which keep the logo style (both the user's choice). A
  sentence that ends on the name takes its normal stop ("the people who run Belong."). Text that
  mentions a tab drops the stop too ("Open chat in Found"). `tests/test_copy.py` fails on the logo's
  stop inside a sentence (followed by a lowercase word, or straight away by other punctuation or a
  letter) in every page's shown strings (not its comments), every string in the Python under `belong/`,
  and TRUST.md, SPEC.md, README.md and CLAUDE.md; checks that the wordmark and the tab labels still say
  "Belong." and "Found."; and reads TRUST.md and the published files for "room".
- The phone has two tabs, **Belong.** (your listings, each with its matches) and **Found.**
  (people you've both said yes to, and your chat with each). Profile, settings, receipts and
  log out live in the top menu. By the user's choice there is no privacy step before a new listing:
  when a listing has no match yet, Belong asks once whether to include close matches, and a yes
  turns on close matches only (questions have their own switch). With "Always include close matches"
  on in Settings, it turns them on by itself at that moment instead and says so in the listing's chat;
  with "Always allow questions" on, a new listing starts with questions on. Both settings start off,
  and each listing's gear has its own "Close matches" and "Questions" switches.

- "The locked room" is an internal metaphor only. Never show it to users: no "room" in
  page copy, agent messages, API responses, published (open-source) files or TRUST.md.
  Talk about trust and security instead: hidden details, the Warden's checks, what is
  shown and what is never shown.
- The Matchmaker is open source; say so, and say the Warden still treats it as hostile.
- Internal code comments may still say "the room" as shorthand.

## Design

All pages use the Belong design system (artifact https://claude.ai/artifact/KDPcQmxczT7iQ7Kcgt1et6):
`belong/web/ds/tokens.css` (generated by tools/make_tokens_css.py; never edit by hand) and
`belong/web/ds/belong.css` (its components as CSS classes). Style only with its tokens. The ground is white; cream (`surface-raised`) is an accent surface for
bands and highlight panels, not a tint on everything. Coral (`accent-default`) is the one action colour,
one primary button per screen, with a dark ink label (`text-on-accent`); a thin coral line (tab underline,
focus ring, selected border) uses the deeper `border-accent`. Periwinkle (`accent-ai`) marks agent output
and every hidden field; mint (`accent-olive`) is live status, at most once per screen. The phone's top bar carries no status pill (the user's choice). On a listing's profile there is no back row taking up height: a home icon left of the menu button takes the person back to their listings, and the profile's lower part is the phone's own off-white, not cream (both the user's choice). Hidden fields use the fixed look: sunken well, periwinkle hairline, "Hidden" badge, "Used for
matching, never shown". Sentence case, no emoji, no lock/shield/eye imagery; say "shown" and
"hidden", never synonyms. Exception, by the user's choice: after a match say "See what else you
have in common" / "Compare", not the design system's "Reveal to each other". Also by the user's choice: no "A SuperFoundry company" family mark in the footer. And notification dots on the phone are pure red (`--notify: #ff0000`), outside the token palette. And message reactions use a fixed few emoji (`REACTIONS` in demo.py), the one place emoji appear. And chat bubbles are rounder (18px) than the system's radii. And the title screen's "Sign in / Register" is orange text alone (`border-accent`, 19px, regular weight, though lighter than the contrast the system asks of small text), not a button. And a new listing starts as the agent's
chat, full screen: no profile behind it and no way to minimise it, its header saying the profile appears once the
agent knows what they're looking for and where. The first time an agent reply leaves the draft ready (what Start
matching needs: `canSave`, `agent.ready`, the same rule), the phone's own status line "Putting your profile together"
(role="status", in the agent's colour, a pulse on the motion tokens) shows for at least 1.5 s and until the phone's
saves finish, then, if the draft is still ready, the chat slides down to its header and the profile is there
(`profileShown` in demo.html); that hold is only for a reply that arrives while the chat is on screen. From then on
the listing has its profile for good (`DemoSession._has_profile`, sent as `has_profile`): a listing that is live, has
had its profile or is ready opens with it and the chat minimised, on any page and after a reload, even once its draft
stops being ready, and never a profile for a listing that has none. With reduced motion, no pulse or slide. Agents
never point at a profile that isn't on screen: the offline agent's reply depends on it, and model agents are told by
each save whether the profile is on screen (`agent_llm._summary`) and to reply briefly, without describing it, when
it appears. Once the profile is there, a change the person makes on it themselves (a detail's sheet closed, the editor
done, Save draft) is kept on the draft and told to their agent, which brings the parts that still say the old thing in
line while the chat's peek says "Updating your profile"; the chat shows a short line in the agent's colour ("You
changed ... on your profile"). Without a model agent the phone offers that change in those parts instead, in a cream
panel ("Change these too?"), one tap each or all, never coral. A change made meanwhile waits on the phone and goes next
(`keepChange` in demo.html, `DemoSession.keep_change`, TRUST.md #34). And on the phone's profile, hidden
details sit in a red band of their own (`--accent-error-muted` over white), between the shown details and
what's left to do, there only when something is hidden; it keeps the Hidden badge and "Used for matching,
never shown".

## Documentation

When you add or change a privacy promise, update TRUST.md in the same change:
the promise in plain English, the code that enforces it, and the test that proves it.
Module docstrings say which SPEC section they implement.

## Commands

```bash
python3 -m unittest -v      # all tests
python3 -m belong run       # 300-person simulation, then again with agent dialects; leak reports (exit 1 on a leak)
python3 -m belong run --matcher hostile   # the same with a stand-in Matchmaker AI that answers as badly as it can (no network)
python3 tools/compare_judges.py           # rules vs the Matchmaker AI (live, costs money: sends nothing without --yes; never in tests)
python3 -m belong serve     # demo / (and /demo), connector mockup /connector, landing /welcome, trust /trust, inspector /inspector
python3 tools/build_site.py # the public website (site/) into out/site; it never links to the internal demo
```

Pure standard library: don't add dependencies without asking. The one exception is the
optional `anthropic` package (requirements.txt), used only by belong/agent_llm.py. The OpenAI
agent (belong/agent_openai.py) uses the standard library. Agent choice: OPENAI_API_KEY ->
OpenAI, else Claude if configured, else rules (BELONG_AGENT overrides). Matchmaker choice (the demo):
BELONG_MATCHMAKER=ai|rules, otherwise the Matchmaker AI (belong/matchmaker_openai.py, standard library)
when OPENAI_API_KEY is set; `belong run` and CI stay on the rules unless `--matcher ai`. Its requests go only to
`BELONG_MATCHMAKER_URL` (https, or http on this machine; no redirect followed), and the OpenAI key only to
OpenAI's own endpoint. Keys live in `.env`
(git-ignored, loaded by `python3 -m belong serve`); never write a key into the repo. Every
model-powered agent must route drafts through `build_draft`. Tests never call the network: construct
`DemoSession(..., agent=None)` or pass a scripted stub client (see tests/test_agent_llm.py), and for the
Matchmaker AI a stub judge (`LLMMatcher(judge=stub)`, `DemoSession(..., matchmaker=stub)`) or a stub HTTP
layer (`OpenAIJudge(post=stub)`); tests/__init__.py forces BELONG_MATCHMAKER=rules.

## Working together

The shared code is the private GitHub repo `superfoundry/belong`. Work on a branch and open a pull
request; GitHub runs every test and the leak audit (`.github/workflows/ci.yml`), and only a passing
change can be merged. A merge to `main` deploys to the demo VM by itself (README, "Run it online").
Never push straight to `main`, never put a key or the demo password in the repo or a workflow, and
each person keeps their own `.env`.
