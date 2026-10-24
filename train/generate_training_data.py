"""Generate a training-data file into 'training data/generated.txt'."""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "training data" / "generated.txt"

# FIX(issue-1.1): safe defaults. The old script unconditionally wrote 500,000
# template keys, which drowned the curated examples and made every query match.
DEFAULT_COUNT = 1_000
MAX_COUNT = 10_000

EMOJIS = ["😊", "😄", "🎉", "🤓", "🤔", "💙", "📚", "✨", "🙌", "💡", "🌟"]

# FIX(issue-1.2): small embedded fallback so Windows/macOS (or any machine
# without /usr/share/dict/words) can run this script without crashing.
FALLBACK_WORDS = """
about above across action active actual advice afford after again against age
agree ahead alive allow almost alone along already also always among amount
animal answer anyone appear apple apply area around arrive art ask attempt
attention audio author available avoid away baby back balance ball bank base
basic beach bear beat beautiful because become before begin behavior behind
believe below benefit best better between beyond big bill bird black blue board
body book born both box boy break bring brother brown build business busy but
buy call calm camera can care carry case catch cause center certain chance
change check child choice choose city class clean clear close cloud coffee cold
collect color come common community company compare complete concern connect
consider contain continue control cook cool copy corner cost could country
course cover create cross current cut dark data daughter day deal dear debate
decide deep detail develop die difference different difficult dinner direction
discover discuss do doctor dog door down draw dream drive drop during each early
earth east easy eat edge education effect effort either else end energy enjoy
enough enter entire environment even evening event ever every example expect
experience explain eye face fact fall family far fast father fear feel few field
fight figure fill film final find fine finger finish fire first fish five floor
fly focus follow food foot force foreign forget form forward free friend from
front full future game garden general get girl give glass go goal good great
green ground group grow guess guide gun guy hair half hand happen happy hard
have head health hear heart help here high history hold home hope horse hospital
hot hour house how however human hundred idea identify if image imagine impact
important improve include increase indeed information inside instead interest
into involve issue item its job join just keep key kid kind kitchen know land
language large last late later laugh law lead learn leave left leg less let
letter level lie life light like line list listen little live local long look
lose lot love low machine main maintain major make man manage many market matter
may maybe mean measure media meet member mention message method middle might
military million mind minute miss mission model modern moment money month more
morning most mother move movie much music must name nation natural nature near
need network never new news next nice night no north note nothing notice now
number occur off offer office often oil old on once one only open operation
opportunity option or order organization other our out outside over own owner
page pain painting paper parent part partner party pass past patient pattern pay
peace people per perform perhaps period person phone pick picture piece place
plan plant play player point police policy political poor popular position
possible power practice prepare present president pressure pretty prevent price
private probably problem process produce product program project property
protect prove provide public pull purpose push put quality question quickly
quiet quite race radio raise range rate rather reach read ready real reality
realize really reason receive recent recognize record red reduce reflect region
relate relationship religious remain remember remove report represent require
research resource respond response rest result return reveal rich right rise
risk river road rock role room rule run safe same save say scene school science
score sea season seat second section security see seek seem sell send senior
sense series serve service set seven several shake share she shoot short should
show side sign significant similar simple since sing single sister sit site
situation six size skill skin small smile so social society soldier some
someone something sometimes son song soon sort sound source south space speak
special spend sport spring staff stage stand standard star start state stay step
still stock stop store story strategy street strong structure student study
stuff style subject success such suddenly suffer suggest summer support sure
surface system table take talk task tax teach teacher team technology tell ten
tend term test than thank that the their them then theory there these they thing
think third this those though thought thousand threat three through throw thus
time to today together tonight too top total tough toward town trade traditional
travel treat tree trial trip trouble true truth try turn two type under
understand unit until up upon us use usually value various very victim view
violence visit voice vote wait walk wall want war watch water way we weapon wear
week weight well west what whatever when where whether which while white who
whole whom whose why wide wife will win wind window wish with within without
woman wonder word work world worry would write wrong yard yeah year yes yet you
young your yourself
""".split()


def _load_words(path: Path | None) -> tuple[list[str], str]:
    """Load a word list from --words or the system dictionary, else fallback."""
    if path is not None:
        candidates = [path]
        if not path.exists():
            raise FileNotFoundError(f"word list not found: {path}")
    else:
        # Deliberately optional: these are conventional paths, not requirements.
        candidates = [
            Path("/usr/share/dict/words"),
            Path("/usr/share/dict/words.txt"),
            Path("/usr/dict/words"),
        ]

    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            raw = candidate.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            continue
        # FIX(issue-1.2/6): filter to plain ASCII lowercase words so generated
        # keys already satisfy the runtime/build normalization rule.
        words = sorted(
            {
                w.strip().lower()
                for w in raw
                if re.fullmatch(r"[a-z]{3,12}", w.strip().lower())
            }
        )
        if words:
            return words, str(candidate)

    # FIX(issue-1.2): use the embedded list when no system dictionary exists.
    words = sorted({w for w in FALLBACK_WORDS if re.fullmatch(r"[a-z]{3,12}", w)})
    return words, "embedded fallback"


def _count_arg(value: str) -> int:
    try:
        count = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("count must be an integer") from exc
    if count < 1:
        raise argparse.ArgumentTypeError("count must be >= 1")
    if count > MAX_COUNT:
        raise argparse.ArgumentTypeError(f"count must be <= {MAX_COUNT}")
    return count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=_count_arg,
        default=DEFAULT_COUNT,
        help=f"number of pairs to generate (default {DEFAULT_COUNT}, max {MAX_COUNT})",
    )
    parser.add_argument("--words", type=Path, default=None, help="optional word-list file")
    parser.add_argument("--out", type=Path, default=OUT, help="output path")
    parser.add_argument("--seed", type=int, default=123, help="deterministic RNG seed")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing output file",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out = Path(args.out)

    # FIX(issue-1.3): never clobber an existing dataset by accident.
    if out.exists() and not args.force:
        print(
            f"refusing to overwrite {out}; pass --force to replace it",
            file=sys.stderr,
        )
        return 1

    try:
        words, source = _load_words(args.words)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if not words:
        print("error: no usable words (need lowercase ASCII words of length 3-12)", file=sys.stderr)
        return 2

    # FIX(issue-9.7): local RNG so importing this module does not mutate the
    # global random state.
    rng = random.Random(args.seed)

    keys: set[str] = set()
    lines: list[str] = []
    attempts = 0
    max_attempts = max(args.count * 100, 10_000)
    while len(keys) < args.count and attempts < max_attempts:
        attempts += 1
        n = rng.randint(2, 5)
        key = " ".join(rng.choice(words) for _ in range(n))
        if key in keys:
            continue
        keys.add(key)
        e = rng.choice(EMOJIS)
        resp = (
            f"<|think|>The user asked about {key}. I should respond helpfully.</|think|> "
            f"Here's what I can share about {key}: it's a great topic to explore! {e}"
        )
        lines.append(f"{key} ||| {resp}")

    if len(lines) < args.count:
        print(
            f"error: only generated {len(lines)} unique pairs from {len(words)} words "
            f"(wanted {args.count}); supply a larger --words list",
            file=sys.stderr,
        )
        return 2

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {len(lines)} pairs to {out} (words: {source})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
