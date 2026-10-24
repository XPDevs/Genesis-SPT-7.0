"""Validate model/Genesis-SPT-7.0.json against the self-validating contract."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MODEL_JSON

RESERVED = {"ver", "NOM", "SOS", "EOS"}
RESERVED_ORDER = ["ver", "SOS", "EOS", "NOM"]


def _norm_key(s: str) -> str:
    """Keep this identical to build_model.py and the JS `_norm`."""
    s = str(s).strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def main() -> None:
    data = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
    assert "ver" in data and data["ver"] == "Genesis-SPT-7.0"
    assert "NOM" in data and data["NOM"]

    keys = list(data.keys())
    # FIX(issue-8): the old `keys == sorted(keys) or keys[:4] == ...` was dead:
    # reserved keys guarantee the left side is false. Check both contracts.
    assert keys[:4] == RESERVED_ORDER, "reserved keys must come first"
    content = keys[4:]
    assert content == sorted(content), "content keys must be sorted"

    # FIX(issue-8): explicit duplicate/empty/normalization checks.
    assert len(keys) == len(set(keys)), "duplicate keys found"
    content_keys = [k for k in keys if k not in RESERVED]
    assert all(_norm_key(k) for k in content_keys), "empty key after normalization"
    assert all(k == _norm_key(k) for k in content_keys), (
        "content keys must be lowercase, punctuation-free, single-spaced"
    )

    assert all(isinstance(v, str) and v.strip() for v in data.values())

    # think balance
    for k, v in data.items():
        if k in RESERVED:
            continue
        assert v.count("<|think|>") == v.count("</|think|>")

    # FIX(issue-8): warn, don't fail, on suspicious keys.
    long_keys = [k for k in content_keys if len(k) > 60]
    many_token_keys = [k for k in content_keys if len(k.split()) > 8]
    if long_keys:
        print(f"warning: {len(long_keys)} content key(s) longer than 60 chars", file=sys.stderr)
        for k in long_keys[:5]:
            print(f"  {k!r}", file=sys.stderr)
    if many_token_keys:
        print(f"warning: {len(many_token_keys)} content key(s) with > 8 tokens", file=sys.stderr)
        for k in many_token_keys[:5]:
            print(f"  {k!r}", file=sys.stderr)

    print(f"OK: {len(data)} keys, {len(content_keys)} content")


if __name__ == "__main__":
    main()
