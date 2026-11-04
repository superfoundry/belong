"""Private matching (SPEC §2, §3).

Everything in this package may read every listing, including hidden fields, so it
must have no way of its own to send anything anywhere. Rules for code in this package, enforced by
tests/test_isolation.py:

  - import only the standard library's pure modules and `belong.schema`;
  - no network, subprocess, file or logging modules; no print();
  - never import `belong.warden`, `belong.synth` or anything outside this package.

The only things this package hands back are Proposal and FollowUp objects, and those go
straight to the Warden's exit check. They are never shown to anyone as-is.

One thing goes the other way, by design: the hybrid Matchmaker (hybrid.LLMMatcher) calls a judge the
caller passes in (code outside this package, which may reach a model). What it hands the judge is built
from shown fields and descriptions only, with every first name, id and hidden text taken out; the caller puts
the Warden's check in front of the judge (warden.exit_judge_request), so a request goes only if it passes, and
each request is kept so an audit can check it again. This package itself still has no network, files or logging.
"""
