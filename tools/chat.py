#!/usr/bin/env python3
"""Chat with Genesis-SPT-7.0 without loading the whole model into RAM at once.

Usage:
    python3 chat.py [path/to/Genesis-SPT-7.0.json|.bin]
Type 'quit' to exit, '/think' to toggle reasoning display.

Memory notes
------------
* The .bin and .json loaders both STREAM: the file never sits in RAM as a
  single string. The only large structure is the content dict itself.
* Only `index_first` is built; prefix/bigram indexes are skipped because they
  add ~2 GB on top of the dict on a 1.25M-key model.
* Fuzzy fallback is capped at MAX_FUZZY_SECONDS so a miss never freezes.
"""
import json
import os
import re
import sys
import time
import functools
from collections import deque
from datetime import datetime

_dir = os.path.dirname(os.path.abspath(__file__))
_default_bin = os.path.join(_dir, "..", "model", "Genesis-SPT-7.0.bin")
_default_json = os.path.join(_dir, "..", "model", "Genesis-SPT-7.0.json")
MODEL_PATH = sys.argv[1] if len(sys.argv) > 1 else (
    _default_bin if os.path.exists(_default_bin) else _default_json
)

RESERVED = {"ver", "NOM", "SOS", "EOS"}
MAX_CANDIDATES = 5000
MAX_FUZZY_SECONDS = 0.5


# --- streaming JSON loader ---------------------------------------------

def _seek_to_object(fh, max_skip=4096):
    """Advance fh past any prefix (e.g. 'const model = ') up to the first '{'."""
    for _ in range(max_skip):
        ch = fh.read(1)
        if not ch:
            raise ValueError("no '{' found in file")
        if ch == '{':
            return
    raise ValueError("no '{' found in first 4 KB")


def _iter_json_object_entries(fh, chunk_size=1 << 16):
    """Stream-parse a JSON object; fh must be positioned just after '{'.

    Yields (key, value) one pair at a time. Uses a sliding buffer so peak
    RAM during parse is O(chunk_size), not O(file).
    """
    dec = json.JSONDecoder()
    buf = ""
    pos = 0
    eof = False

    def refill():
        nonlocal buf, eof
        if eof:
            return False
        chunk = fh.read(chunk_size)
        if not chunk:
            eof = True
            return False
        buf += chunk
        return True

    def skip_ws():
        nonlocal pos
        while True:
            while pos < len(buf) and buf[pos].isspace():
                pos += 1
            if pos < len(buf):
                return True
            if not refill():
                return False

    def compact():
        nonlocal buf, pos
        if pos > chunk_size:
            buf = buf[pos:]
            pos = 0

    def decode_at():
        while True:
            try:
                value, end = dec.raw_decode(buf, pos)
                return value, end
            except json.JSONDecodeError:
                if not refill():
                    raise ValueError(f"JSON parse failed at offset {pos}")

    while True:
        compact()
        if not skip_ws():
            raise ValueError("unexpected EOF inside object")
        if buf[pos] == "}":
            return
        if buf[pos] == ",":
            pos += 1
            continue
        key, end = decode_at()
        pos = end
        if not isinstance(key, str):
            raise ValueError(f"non-string key: {type(key).__name__}")
        if not skip_ws() or buf[pos] != ":":
            raise ValueError(f"expected ':', got {buf[pos]!r}")
        pos += 1
        if not skip_ws():
            raise ValueError("unexpected EOF before value")
        value, end = decode_at()
        pos = end
        yield key, value


def _load_model(path):
    """Stream a model file into (content_dict, reserved_dict)."""
    content: dict[str, object] = {}
    reserved: dict[str, object] = {}
    t0 = time.time()
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        _seek_to_object(fh)
        n = 0
        for k, v in _iter_json_object_entries(fh):
            if k in RESERVED:
                reserved[k] = v
            else:
                content[k] = v
            n += 1
            if n % 250_000 == 0:
                print(f"  loaded {n:,} entries ...", file=sys.stderr, flush=True)
    print(
        f"loaded {len(content):,} content + {len(reserved)} reserved "
        f"in {time.time() - t0:.1f}s",
        file=sys.stderr,
    )
    return content, reserved


print(f"Loading model from {MODEL_PATH} ...", file=sys.stderr)
content, _meta = _load_model(MODEL_PATH)
NOM = _meta.get("NOM", "I'm not sure how to respond to that.")
VERSION = _meta.get("ver")

# --- index: first token -> [keys] -------------------------------------

t0 = time.time()
index_first: dict[str, list[str]] = {}
for k in content:
    parts = k.split(None, 1)
    if not parts:
        continue
    bucket = index_first.get(parts[0])
    if bucket is None:
        index_first[parts[0]] = [k]
    else:
        bucket.append(k)
print(
    f"built index_first ({len(index_first):,} tokens) "
    f"in {time.time() - t0:.1f}s",
    file=sys.stderr,
)


# --- helpers -----------------------------------------------------------

def apply_placeholders(s):
    now = datetime.now()
    return (
        s.replace("%DATE%", now.strftime("%B %d, %Y"))
         .replace("%TIME%", now.strftime("%I:%M %p"))
    )


def strip_think(s):
    return re.sub(r"<\|think\|>.*?</\|think\|>", "", s, flags=re.S).strip()


SYNONYMS = {
    "hi": ["hello", "hey", "howdy", "greetings", "yo"],
    "hello": ["hi", "hey", "howdy", "greetings"],
    "hey": ["hi", "hello", "yo"],
    "sad": ["unhappy", "down", "depressed", "blue", "upset"],
    "happy": ["glad", "joyful", "cheerful", "content", "excited"],
    "good": ["great", "fine", "well", "okay", "nice"],
    "bad": ["terrible", "awful", "poor", "horrible"],
    "what": ["tell", "explain", "describe", "define", "show"],
    "how": ["way", "method", "steps", "guide"],
    "why": ["reason", "cause", "explain"],
    "make": ["create", "build", "generate", "produce"],
    "love": ["like", "enjoy", "adore", "appreciate"],
    "hate": ["dislike", "loathe", "detest"],
    "big": ["large", "huge", "massive"],
    "small": ["little", "tiny", "mini"],
    "fast": ["quick", "rapid", "speedy"],
    "slow": ["sluggish", "gradual"],
}


@functools.lru_cache(maxsize=200_000)
def _lev(a, b):
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(a) + 1))
    for i in range(1, len(b) + 1):
        cur = [i] + [0] * len(a)
        for j in range(1, len(a) + 1):
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + (a[j - 1] != b[i - 1]),
            )
        prev = cur
    return prev[len(a)]


# --- retrieval ---------------------------------------------------------

def respond(user_input):
    q = user_input.lower().strip()
    if not q:
        return apply_placeholders(NOM)
    if q in content:
        return apply_placeholders(content[q])

    q_tokens = q.split()
    if not q_tokens:
        return apply_placeholders(NOM)

    # 1. First-token lookup, with synonyms.
    first = q_tokens[0]
    cand: set[str] = set()
    for t in [first] + SYNONYMS.get(first, []):
        for k in index_first.get(t, ()):
            cand.add(k)
            if len(cand) >= MAX_CANDIDATES:
                break
        if len(cand) >= MAX_CANDIDATES:
            break

    # 2. Fuzzy fallback on the first token (bounded scan of index_first).
    if not cand and len(first) > 3:
        for token, keys in index_first.items():
            if abs(len(token) - len(first)) > 1:
                continue
            if _lev(first, token) <= 1:
                for k in keys:
                    cand.add(k)
                    if len(cand) >= 500:
                        break
                if len(cand) >= 500:
                    break

    if not cand:
        return apply_placeholders(NOM)

    # 3. Score by token overlap + containment + length fit.
    qset = set(q_tokens)
    best_key = None
    best_score = 0.0
    deadline = time.time() + MAX_FUZZY_SECONDS
    for k in cand:
        kt = k.split()
        if not kt:
            continue
        score = len(qset & set(kt)) * 1.2
        if q in k:
            score += 6
        elif k in q:
            score += 4
        score -= 0.05 * abs(len(kt) - len(q_tokens))
        if score > best_score:
            best_score = score
            best_key = k
        if time.time() > deadline:
            break

    if best_key and best_score > 0.5:
        return apply_placeholders(content[best_key])
    return apply_placeholders(NOM)


def split_think(raw):
    m = re.search(r"<\|think\|>(.*?)</\|think\|>", raw, re.S)
    think = (m.group(1).strip() if m else "")
    answer = strip_think(raw)
    return apply_placeholders(think), apply_placeholders(answer)


def stream_text(text, delay=0.004):
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    sys.stdout.write("\n")
    sys.stdout.flush()


def main():
    show_think = True
    print(f"Genesis-SPT-7.0 ready. ver={VERSION} keys={len(content):,}")
    print("Commands: quit | /think (toggle reasoning)\n")
    history = deque(maxlen=50)
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        if not user:
            continue
        if user.lower() in {"quit", "exit", "bye"}:
            print("Genesis: Take care!")
            break
        if user.lower() == "/think":
            show_think = not show_think
            print(f"Reasoning display: {'ON' if show_think else 'OFF'}")
            continue
        t0 = time.time()
        raw = respond(user)
        dt = time.time() - t0
        think, answer = split_think(raw)
        if show_think and think:
            print("Thinking: ", end="", flush=True)
            stream_text(think, delay=0.003)
        print("Genesis: ", end="", flush=True)
        stream_text(answer, delay=0.004)
        print(f"  ({dt*1000:.0f} ms)\n")
        history.append((user, raw))


if __name__ == "__main__":
    main()
