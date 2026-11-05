"""Measurements and a (simulated) attestation document. OPEN SOURCE.

What attestation is for: showing that the code running is the published open-source code, so anyone can
check that no one learns about someone else without a two-way match. A signed document, compared with builds
anyone makes from the open-source code, is still to come.

Both agents are open source. What this prototype can do: compute the same
measurements (SHA-256 of the files), one for the Warden agent and one for the
Matchmaker agent, and publish them with the code, so a verifier can see that the code
it downloaded is the code that was measured. It CANNOT prove that this code is what is actually running; only
a signature over the measurements would. The document says so in `signed_by`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .schema import (K_CROWD, LIST_CAPS, LIST_LONG_RUN, LIST_LONG_RUN_CHARS, LIST_MIN_AGE_DAYS, LIST_PHRASE_WORDS,
                     LIST_UNSPACED_CHARS, LIST_VALUE_WORDS, MAX_LISTINGS_PER_OWNER)

PKG = Path(__file__).resolve().parent

# The Warden agent: the boundary every listing and introduction passes through.
WARDEN_FILES = [
    "schema.py",          # built-in fields, open-detail rules, sensitive words, follow-up menu
    "warden.py",          # entry + exit checks, sealed outputs, receipt chain
    "leaks.py",           # independent leak audit
    "attest.py",          # this file
    "matching/__init__.py",   # isolation rules for matching code
    "matching/contract.py",   # the only shapes the Matchmaker may hand the Warden
    "matching/marketboard.py",
]
# The Matchmaker agent: also open source. The Warden agent still treats it as hostile,
# so the privacy guarantee holds even if someone runs a modified Matchmaker.
MATCHMAKER_FILES = [
    "matching/matcher.py",
    "matching/hybrid.py",  # the hybrid: the rules narrow, an injected judge judges the fuzzy part
    "matchmaker_openai.py",  # that judge: a model over a Chat Completions API (OpenAI's by default), outside matching/
    "matching/intent.py",
    "matching/narrow.py",
    "matching/text.py",
]
# Everything published for anyone to read and scrutinise.
OPEN_SOURCE = WARDEN_FILES + MATCHMAKER_FILES

POLICY = {
    "matching_output": "Warden exit checks only",
    "matching_allowed_imports": ["__future__", "collections", "dataclasses", "math", "re", "typing", "unicodedata",
                                 "belong.schema"],
    # A hidden value counts only if this many people besides the two being matched share it (counted
    # as different people, not listings).
    "k_crowd_threshold": K_CROWD,
    "approach_limit_per_day": 15,
    "listing_updates_per_day": 5,
    "listings_per_person": MAX_LISTINGS_PER_OWNER,
    "output_types": ["Introduction", "FollowUpOut", "Reveal", "DetailsInUse", "Receipt"],
    "details_in_use": {
        "built_by": "the Warden alone, from the board (no Matchmaker input), at most once a day",
        "min_people": K_CROWD,
        "counted_by": "different people",
        "from": "shown, non-sensitive fields of active listings unchanged for at least "
                f"{LIST_MIN_AGE_DAYS} days",
        "text": f"short plain text: a phrase at most {LIST_PHRASE_WORDS} words, anything else at most "
                f"{LIST_VALUE_WORDS} (a word is a run of letters and digits, whatever joins them, and a capital "
                f"after a lower-case letter starts one; in a script written without spaces, one word per "
                f"{LIST_UNSPACED_CHARS} letters; in any other, a run longer than {LIST_LONG_RUN} letters is one word "
                f"per {LIST_LONG_RUN_CHARS} letters), no sentence punctuation, "
                "no contact details, no sensitive words",
        "max_items": LIST_CAPS,
        "order": "alphabetical, no counts",
    },
    # The Matchmaker AI (matching/hybrid.py, matchmaker_openai.py): whether it is on right now is at
    # /api/trust/matchmaker; what it may receive and answer is fixed.
    "matchmaker_ai": {
        "status": "OPENAI: when it is on, a model over OpenAI's API (or a self-hosted model at the endpoint set), "
                  "treated as hostile; this demo runs on synthetic and test profiles only",
        "receives": "for each pair the hard rules allow, what both listings show and their descriptions, with every "
                    "first name, id and hidden text taken out, under labels of one request; each request checked by "
                    "the Warden against the board as it is just before it goes (warden.exit_judge_request), then "
                    "audited again (leaks.py, A8: the last 2000 requests kept; the model client's record compared "
                    "with every request the Warden passed)",
        "answers": "verdicts only, read strictly: match, close or no each way, a fit from 0 to 1, and, asked about "
                   "one listing alone, whether it seems to hunt one person (which can only drop its pairs); it never "
                   "adds a pair the hard rules refuse",
    },
}


def file_sha256(rel: str) -> str:
    return hashlib.sha256((PKG / rel).read_bytes()).hexdigest()


def measurements() -> dict:
    files = {rel: file_sha256(rel) for rel in OPEN_SOURCE}
    return {
        "warden": _combined({rel: files[rel] for rel in WARDEN_FILES}),
        "matchmaker": _combined({rel: files[rel] for rel in MATCHMAKER_FILES}),
        "policy": hashlib.sha256(json.dumps(POLICY, sort_keys=True).encode()).hexdigest(),
        "open_source_files": files,
        "groups": {"warden": WARDEN_FILES, "matchmaker": MATCHMAKER_FILES},
    }


def _combined(files: dict) -> str:
    """One measurement for a set of files: SHA-256 over 'path:sha256' lines, sorted."""
    lines = "\n".join(f"{p}:{h}" for p, h in sorted(files.items()))
    return hashlib.sha256(lines.encode()).hexdigest()


def attestation(nonce: str) -> dict:
    """The document a verifier asks for. `nonce` is chosen by the verifier, so an old
    document can't be replayed (in production it is inside the signed payload)."""
    return {
        "nonce": str(nonce)[:128],
        "issued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measurements": measurements(),
        "policy": POLICY,
        "signed_by": None,
        "status": ("SIMULATED: prototype. Measurements are real file hashes, but nothing proves this "
                   "code is what is running: this document is unsigned, and a signed one is still to come."),
    }


def open_source_text(rel: str):
    """Source of an open-source file, or None if it isn't one."""
    if rel not in OPEN_SOURCE:
        return None
    return (PKG / rel).read_text()
