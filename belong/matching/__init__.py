"""Private matching (SPEC §2, §3).

Everything in this package may read every listing, including hidden fields, so it
must have no way to send anything anywhere. Rules for code in this package, enforced by
tests/test_isolation.py:

  - import only the standard library's pure modules and `belong.schema`;
  - no network, subprocess, file or logging modules; no print();
  - never import `belong.warden`, `belong.synth` or anything outside this package.

The only things this package hands back are Proposal and FollowUp objects, and those go
straight to the Warden's exit check. They are never shown to anyone as-is.
"""
