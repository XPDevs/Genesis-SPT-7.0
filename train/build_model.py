"""Build Genesis-SPT-7.0.json from training data (+ optional base model)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import TRAINING_DATA, MODEL_JSON, MODEL

RESERVED = {"ver", "NOM", "SOS", "EOS"}
RESERVED_ORDER = ("ver", "SOS", "EOS", "NOM")
LARGE_FILE_WARNING = 10_000
PROGRESS_EVERY = 250_000


def _norm_key(s: str) -> str:
    """Normalize a key exactly the way the shipped JS `_norm` does.

    FIX(issue-6): build_model.py used to only lowercase/strip keys while the
    JavaScript runtime also replaced punctuation with spaces. That made keys
    such as "what's up" unreachable. Keeping this in sync is essential.
    """
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def parse_training_file(path: Path) -> tuple[list[tuple[str, str]], int]:
    """Return (pairs, skipped_empty_key_count) for one training file.

    FIX(issue-7a): stream line-by-line instead of path.read_text().splitlines().
    The old version (a) raised UnicodeDecodeError on any non-UTF-8 byte in a
    corpus file and (b) materialised the entire file as one str plus a list of
    ~1 line objects, which triggers MemoryError on multi-hundred-MB corpora
    like hf_wildchat.txt. errors="ignore" matches count_training_data.py.
    """
    pairs: list[tuple[str, str]] = []
    skipped = 0
    lineno = 0
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            lineno += 1
            if lineno % PROGRESS_EVERY == 0:
                print(
                    f"  {path.name}: {lineno:,} lines, {len(pairs):,} pairs ...",
                    flush=True,
                )
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue
            if "|||" in line:
                key, val = line.split("|||", 1)
            elif "=>" in line:
                key, val = line.split("=>", 1)
            else:
                continue
            key = key.strip()
            val = val.strip()

            # FIX(issue-6): reserved-key check happens before normalization, as
            # before, so an explicit NOM/SOS/EOS/ver line cannot become content.
            if key in RESERVED:
                continue

            key = _norm_key(key)
            if not key:
                # FIX(issue-6): punctuation-only keys (e.g. "!!!") normalize to an
                # empty string and must not create an unreachable model entry.
                if val:
                    skipped += 1
                continue
            if val:
                pairs.append((key, val))
    return pairs, skipped


def _normalize_base(raw: dict) -> tuple[dict, list[tuple[str, str]], int, int]:
    """Normalize content keys loaded from a legacy model JSON.

    FIX(issue-6): re-normalizing the base model keeps an incremental build from
    preserving old punctuation/uppercase keys that the runtime cannot reach.
    """
    out: dict = {}
    changed: list[tuple[str, str]] = []
    skipped = 0
    collisions = 0
    for k, v in raw.items():
        if k in RESERVED:
            out[k] = v
            continue
        nk = _norm_key(k)
        if not nk:
            skipped += 1
            continue
        if nk != k:
            changed.append((k, nk))
        if nk in out:
            collisions += 1
        out[nk] = v
    return out, changed, skipped, collisions


def main() -> None:
    base: dict = {}
    if MODEL_JSON.exists():
        try:
            loaded = json.loads(MODEL_JSON.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("top-level JSON value is not an object")
            base, changed, skipped, collisions = _normalize_base(loaded)
            print(f"loaded base model: {len(base)} keys")
            if changed:
                # FIX(issue-6): warn (with a sample) if a legacy model needed
                # normalization; the full list can be huge on old builds.
                sample = ", ".join(f"{old!r}->{new!r}" for old, new in changed[:10])
                print(
                    f"warning: normalized {len(changed)} legacy base key(s); "
                    f"sample: {sample}",
                    file=sys.stderr,
                )
            if collisions:
                print(
                    f"warning: {collisions} legacy base key collision(s) after normalization",
                    file=sys.stderr,
                )
            if skipped:
                print(
                    f"warning: skipped {skipped} base key(s) empty after normalization",
                    file=sys.stderr,
                )
        except Exception as e:
            print("could not load base model:", e)
            base = {}

    data = dict(base)
    # ensure reserved
    data.setdefault("ver", "Genesis-SPT-7.0")
    data.setdefault("SOS", "<|SOS|>")
    data.setdefault("EOS", "<|EOS|>")
    data.setdefault("NOM", "I'm not sure how to respond to that. Could you rephrase?")

    files = sorted(
        (f for f in TRAINING_DATA.iterdir() if f.is_file()),
        key=lambda p: p.name.lower(),
    )
    # FIX(issue-1): generated data is processed first and hand-written files
    # last. Because later assignments overwrite earlier ones, a curated
    # examples.txt answer always wins over a colliding generated.txt entry.
    generated_files = [f for f in files if f.name.lower().startswith("generated")]
    handwritten_files = [f for f in files if not f.name.lower().startswith("generated")]

    added = 0
    for f in generated_files + handwritten_files:
        pairs, skipped = parse_training_file(f)
        if skipped:
            print(
                f"warning: {f.name}: skipped {skipped} key(s) empty after normalization",
                file=sys.stderr,
            )
        # FIX(issue-1): surface suspiciously large files instead of silently
        # producing an enormous (and slow) model.
        if len(pairs) > LARGE_FILE_WARNING:
            print(
                f"warning: {f.name} contributes {len(pairs)} keys "
                f"(> {LARGE_FILE_WARNING}); this will make the model large",
                file=sys.stderr,
            )
        for k, v in pairs:
            data[k] = v
            added += 1
        print(f"{f.name}: {len(pairs)} pairs")

    # rebuild ordered
    ordered = {}
    for r in RESERVED_ORDER:
        if r in data:
            ordered[r] = data[r]
    for k in sorted(x for x in data if x not in RESERVED):
        ordered[k] = data[k]

    MODEL.mkdir(parents=True, exist_ok=True)
    # FIX(issue-7b): json.dump to a file handle instead of json.dumps + write_text.
    # json.dumps materialises the whole JSON string in memory on top of the dict
    # itself, which MemoryErrors on ~1.25M-key models. json.dump streams via
    # iterencode and avoids the extra full-size string.
    with MODEL_JSON.open("w", encoding="utf-8") as fh:
        json.dump(ordered, fh, ensure_ascii=False)
    print(f"added/updated {added} keys; total {len(ordered)} -> {MODEL_JSON}")


if __name__ == "__main__":
    main()
