"""Compare narrowing with and without its word index (SPEC §8.2; step 3 of the matching order).

    python3 tools/compare_narrowing.py [people] [seed]      # default: 3000 people, seed 7

Builds a board of synthetic people and narrows every listing twice with no size limit: with one
catch-all group for everything compared by phrase (as before), and with the word index. The index
must only drop candidates, and never one a decision could accept. Then it runs the Matchmaker both
ways at its usual shortlist size and compares the proposals with the ground truth.
Exits 1 if the index drops anything a decision could accept.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from belong.evaluate import compare_narrowing  # noqa: E402
from belong.synth import generate  # noqa: E402

n = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 7
start = time.perf_counter()
r = compare_narrowing(generate(n, seed))
print(f"{r['listings']} listings (generate({n}, {seed})), {time.perf_counter() - start:.1f}s")
print(f"Candidates, no size limit: {r['candidates_before']} with the catch-all group, "
      f"{r['candidates_after']} with the word index; only dropped: {r['subset']}")
print(f"  of which for the {r['phrase_listings']} listings compared by phrase: {r['phrase_candidates_before']} -> "
      f"{r['phrase_candidates_after']} (dating keeps its own group)")
print(f"Pairs a decision could accept: {r['acceptable_pairs_kept']} kept, {len(r['lost'])} lost {r['lost'][:5]}")
print(f"Proposals at the usual shortlist size: before {r['proposals_before']} ({r['correct_before']} correct), "
      f"after {r['proposals_after']} ({r['correct_after']} correct); correct pairs found only before: "
      f"{r['correct_found_only_before']}, only after: {r['correct_found_only_after']}")
sys.exit(0 if r["subset"] and not r["lost"] else 1)
