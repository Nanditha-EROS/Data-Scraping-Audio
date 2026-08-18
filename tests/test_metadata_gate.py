"""Standalone test/CLI for the Metadata Relevance Gate (pipeline gate #1).

Runs entirely offline on title/description strings -- no download, no models.
This is the "small real test" for gate 1 before we move to the next gate.

Usage:
    # run the built-in cases:
    python tests/test_metadata_gate.py

    # test a single title/description against a category slug:
    python tests/test_metadata_gate.py --slug kummi \
        --title "Kummi Paatu Tamil Folk Song" --desc "traditional village song"
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tnfolk.config import load_pipeline_config, load_taxonomy  # noqa: E402
from tnfolk.logging_utils import setup_logging  # noqa: E402
from tnfolk.models import Candidate, Decision, RejectionCode  # noqa: E402
from tnfolk.gates import metadata_relevance  # noqa: E402


def _run(slug: str, title: str, desc: str, cfg, taxonomy):
    category = taxonomy.get(slug)
    cand = Candidate(recording_id="test", category_slug=slug, title=title, description=desc)
    result = metadata_relevance.evaluate(cand, category, cfg)
    return result


# (slug, title, description, expected_decision, expected_code_or_None)
CASES = [
    ("kummi", "Kummi Paatu | Tamil Folk Song | Village Special", "traditional kummi folk dance song",
     Decision.PENDING, None),
    ("thalattu-lullaby", "தாலாட்டுப் பாடல் | அழகான தமிழ் நாட்டுப்புற பாடல்", "",
     Decision.PENDING, None),
    ("villupattu", "Villupattu speech and narration by artist", "villu paatu performance",
     Decision.PENDING, None),  # narration allowed for villupattu -> not auto-rejected on 'speech'
    ("kummi", "Political Leader Interview about elections", "news press meet interview",
     Decision.REJECT, RejectionCode.INTERVIEW),
    ("kummi", "Morning News Headlines Tamil", "today news update",
     Decision.REJECT, RejectionCode.NEWS),
    ("kummi", "How to invest in stocks - full lecture", "finance class tutorial",
     Decision.REJECT, RejectionCode.LECTURE),
    ("kummi", "Random cooking recipe video", "how to make biryani at home",
     Decision.REJECT, RejectionCode.NON_FOLK),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Metadata Relevance Gate tester")
    ap.add_argument("--slug")
    ap.add_argument("--title", default="")
    ap.add_argument("--desc", default="")
    args = ap.parse_args()

    setup_logging(level="INFO")
    cfg = load_pipeline_config()
    taxonomy = load_taxonomy()
    print(f"Loaded taxonomy: {len(taxonomy)} categories; state={taxonomy.state_name}\n")

    if args.slug:
        r = _run(args.slug, args.title, args.desc, cfg, taxonomy)
        print(f"decision={r.decision.value} code={r.rejection_code.value if r.rejection_code else '-'} "
              f"score={r.score} review={r.review}")
        print(f"message: {r.message}")
        print(f"details: {r.details}")
        return 0

    passed = 0
    for slug, title, desc, exp_dec, exp_code in CASES:
        r = _run(slug, title, desc, cfg, taxonomy)
        got_dec = r.decision
        got_code = r.rejection_code
        ok = (got_dec == exp_dec) and (got_code == exp_code)
        passed += ok
        status = "PASS" if ok else "FAIL"
        code_str = got_code.value if got_code else "-"
        print(f"[{status}] {slug:<16} dec={got_dec.value:<8} code={code_str:<10} "
              f"score={r.score} :: {title[:45]}")
        if not ok:
            print(f"        expected dec={exp_dec.value} code={exp_code.value if exp_code else '-'}"
                  f" | msg: {r.message}")
    print(f"\n{passed}/{len(CASES)} cases passed")
    return 0 if passed == len(CASES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
