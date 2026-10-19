"""Measurements and a (simulated) attestation document. OPEN SOURCE.

What "attestation" means in production (Step 3): matching runs in a hardware enclave
(e.g. AWS Nitro). The hardware measures the exact code it booted and signs those
measurements. Anyone can fetch the signed document and compare the measurements with
builds they made themselves from the open-source code.

Both agents are open source. What this prototype can do: compute the same
measurements (SHA-256 of the files), one for the Warden agent and one for the
Matchmaker agent, and publish them with the code, so a verifier can see that the code
it downloaded is the code that was measured. It CANNOT prove that this code is what is actually running; only
the hardware signature does that. The document says so in `signed_by`.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

PKG = Path(__file__).resolve().parent

# The Warden agent: the boundary every listing and introduction passes through.
WARDEN_FILES = [
    "schema.py",          # data model, field registry, follow-up menu
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
    "k_crowd_threshold": 20,
    "approach_limit_per_week": 3,
    "listing_updates_per_day": 5,
    "output_types": ["Introduction", "FollowUpOut", "Reveal", "Receipt"],
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
        "status": ("SIMULATED: prototype running outside an enclave. Measurements are real file hashes, "
                   "but nothing proves this code is what is running. In production this document is "
                   "signed by the enclave hardware (e.g. the AWS Nitro attestation PKI)."),
    }


def open_source_text(rel: str):
    """Source of an open-source file, or None if it isn't one."""
    if rel not in OPEN_SOURCE:
        return None
    return (PKG / rel).read_text()
