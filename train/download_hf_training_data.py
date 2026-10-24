"""Download a HuggingFace conversational dataset and convert it to Genesis training data.

Usage:
    python3 train/download_hf_training_data.py
    python3 train/download_hf_training_data.py --repo allenai/WildChat-1M --out "training data/wildchat_1m.txt"
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRAINING_DATA = ROOT / "training data"

RESERVED = {"ver", "NOM", "SOS", "EOS"}


def _norm_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return " ".join(s.split())


def _clean_value(v) -> str:
    return " ".join(str(v).replace("\r", " ").replace("\n", " ").split())


def convert_parquet(path: Path, fout, seen: set[str], seen_conv: set[str], batch_size: int = 1000) -> int:
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(str(path))
    cols = pf.schema_arrow.names

    conv_col = None
    for c in ("conversation", "messages", "chat"):
        if c in cols:
            conv_col = c
            break
    if conv_col is None:
        for c in cols:
            if "conversation" in c.lower() or "message" in c.lower():
                conv_col = c
                break
    if conv_col is None:
        return 0

    id_col = None
    for c in ("conversation_hash", "conversation_id", "id", "uuid"):
        if c in cols:
            id_col = c
            break

    added = 0
    for batch in pf.iter_batches(batch_size=batch_size):
        for row in batch.to_pylist():
            cid = row.get(id_col) if id_col else None
            if cid is not None and cid in seen_conv:
                continue
            msgs = row.get(conv_col)
            if not isinstance(msgs, list):
                continue
            if cid is not None:
                seen_conv.add(cid)

            for i in range(len(msgs) - 1):
                a = msgs[i]
                b = msgs[i + 1]
                if not isinstance(a, dict) or not isinstance(b, dict):
                    continue
                role_a = str(a.get("role", "")).lower()
                role_b = str(b.get("role", "")).lower()
                if role_a not in ("user", "human", "prompter"):
                    continue
                if role_b not in ("assistant", "gpt", "chatbot", "bot"):
                    continue
                user_text = _clean_value(a.get("content", ""))
                asst_text = _clean_value(b.get("content", ""))
                if not user_text or not asst_text:
                    continue
                key = _norm_key(user_text)
                if not key or key in RESERVED:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                fout.write(f"{key} ||| {asst_text}\n")
                added += 1
        # flush periodically
        fout.flush()
    return added


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="allenai/WildChat,allenai/WildChat-1M", help="HuggingFace dataset repo id (comma-separated for multiple)")
    ap.add_argument("--out", default="training data/hf_wildchat.txt", help="output training-data file (relative to project root)")
    ap.add_argument("--keep-data", action="store_true", help="keep downloaded dataset files after conversion")
    ap.add_argument("--batch-size", type=int, default=1000, help="parquet batch size (lower = less RAM)")
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except Exception as e:
        print("huggingface_hub not installed:", e, file=sys.stderr)
        return 1

    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    repos = [r.strip() for r in args.repo.split(",") if r.strip()]

    seen: set[str] = set()
    seen_conv: set[str] = set()
    total = 0
    with out_path.open("w", encoding="utf-8") as fout:
        for repo in repos:
            data_dir = ROOT / "data" / "hf_download"
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"downloading {repo} ...", flush=True)
            snapshot_download(repo_id=repo, repo_type="dataset", local_dir=str(data_dir))

            parquets = sorted(data_dir.rglob("*.parquet"))
            if not parquets:
                print(f"no parquet files found for {repo}", file=sys.stderr)
                if not args.keep_data:
                    import shutil
                    shutil.rmtree(data_dir, ignore_errors=True)
                continue

            for pq in parquets:
                n = convert_parquet(pq, fout, seen, seen_conv, batch_size=args.batch_size)
                total += n
                print(f"{repo} {pq.name}: +{n} (total {total})", flush=True)

            if not args.keep_data:
                import shutil
                shutil.rmtree(data_dir, ignore_errors=True)
                print(f"removed downloaded files for {repo}", flush=True)

    print(f"wrote {total} pairs to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())