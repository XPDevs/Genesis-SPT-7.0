"""Count training-data entries and estimate model parameters.

Usage:
    python3 tools/count_training_data.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TD = ROOT / "training data"

MAX_EXAMPLE_PART = 120


def _norm_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def _short(s: str, limit: int = MAX_EXAMPLE_PART) -> str:
    return s if len(s) <= limit else s[:limit] + "..."


def _format_example(sep: str, line: str) -> str:
    key_part, value_part = line.split(sep, 1)
    return (
        f"sep={sep!r} key={_short(key_part.strip())!r} "
        f"value={_short(value_part.strip())!r}"
    )


def main() -> int:
    files = sorted(TD.glob("*.txt"))
    total_lines = 0
    valid = 0
    seen: set[str] = set()
    for f in files:
        size = f.stat().st_size
        read_bytes = 0
        n = 0
        example: str | None = None
        with f.open(encoding="utf-8", errors="ignore") as fh:
            for raw in fh:
                read_bytes += len(raw.encode("utf-8", errors="ignore"))
                line = raw.strip()
                pct = (read_bytes / size * 100) if size else 100.0
                if not line or line.startswith("#") or line.startswith("//"):
                    if n % 100000 == 0:
                        print(f"{f.name}: {n} lines, {pct:.1f}%", flush=True)
                    n += 1
                    continue
                if "|||" not in line and "=>" not in line:
                    if n % 100000 == 0:
                        print(f"{f.name}: {n} lines, {pct:.1f}%", flush=True)
                    n += 1
                    continue
                total_lines += 1
                sep = "|||" if "|||" in line else "=>"
                if example is None:
                    example = _format_example(sep, line)
                key, _ = line.split(sep, 1)
                nk = _norm_key(key)
                if nk:
                    seen.add(nk)
                    valid += 1
                n += 1
                if n % 100000 == 0:
                    print(f"{f.name}: {n} lines, {pct:.1f}%", flush=True)
        print(f"{f.name}: {n} lines, 100.0%", flush=True)
        if example is None:
            print("  example: (no lines with ||| or => found)", flush=True)
        else:
            print(f"  example: {example}", flush=True)
    print(f"total valid lines: {valid}")
    print(f"unique normalized keys: {len(seen)}")
    print(f"estimated parameters (entries/59): {len(seen) / 59:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
