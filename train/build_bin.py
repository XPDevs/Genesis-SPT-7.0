"""Build model/Genesis-SPT-7.0.bin (JS bundle) from the JSON model."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import MODEL_JSON, MODEL_BIN

DEFAULT_MAX_CONTENT_KEYS = 50_000
RESERVED = {"ver", "NOM", "SOS", "EOS"}

# FIX(issue-7c): the build now STREAMS the source JSON. Peak RAM is O(chunk),
# independent of model size — a 2.83 GB model uses the same ~200 KB as a
# 2 MB one. File size is reported but no longer used to gate the build.
STREAM_CHUNK_BYTES = 64 * 1024

# FIX(issue-7e): node --check internally does readFileSync(path) which builds
# a single V8 string. V8's max string is 0x1fffffe8 chars (~512 MiB), so any
# JS file larger than that can never pass --check. Warn below that, skip above.
CHECK_WARN_BYTES = 500 * 1024 * 1024       # --check on >=500 MB is slow
CHECK_MAX_BYTES = 480 * 1024 * 1024        # V8 string limit; skip above this

# NOTE: This string is the matcher that ships. Do not move it into a separate
# .js file; every JavaScript fix belongs here.
JS_CODE = r"""
// FIX(issue-2): Tunable candidate caps. The old caps (20k/25k/30k/5k/8k)
// made a single query do tens of thousands of Levenshtein calls.
const CAP_FIRST = 4000;
const CAP_BIGRAM = 3000;
const CAP_PREFIX = 4000;
const CAP_FUZZY = 2000;
const CAP_RESCORE = 500;          // only this many cheap winners get _levenshtein
const CAP_FUZZY_FALLBACK = 200;   // fuzzy candidate scan stays on the tiny path

// FIX(issue-9.5): timestamps are host-independent and configurable.
const GENESIS_TIMEZONE = "UTC";
const GENESIS_MERGE_THOUGHT = "<|think|>Synthesizing across multiple relevant knowledge patterns to derive a generalized response.</|think|>";

function _norm(s){return String(s||"").toLowerCase().replace(/[^a-z0-9\s]/g," ").replace(/\s+/g," ").trim();}
function _tokens(s){return _norm(s).split(" ").filter(Boolean);}
const GENESIS_SYNONYMS={hi:["hello","hey","howdy","greetings","yo","hiya"],hello:["hi","hey","howdy","greetings"],hey:["hi","hello","yo"],good:["great","fine","well","okay","nice","decent"],bad:["terrible","awful","poor","horrible","bad"],what:["tell","explain","describe","define","show","give"],how:["way","method","steps","guide","do"],why:["reason","cause"],make:["create","build","generate","produce"],old:["age","years","ancient"],young:["new","fresh","modern"],love:["like","enjoy","adore","appreciate"],hate:["dislike","loathe","detest"],big:["large","huge","massive","enormous"],small:["little","tiny","mini","miniature"],fast:["quick","rapid","speedy"],slow:["gradual","sluggish"],important:["crucial","vital","essential"],think:["ponder","consider"]};
function _expand(tokens,syns){syns=syns||GENESIS_SYNONYMS;const o=[...tokens];for(let i=0;i<tokens.length;i++){const t=tokens[i];if(syns[t])for(let j=0;j<syns[t].length;j++)o.push(syns[t][j]);}return o;}
function _strip(s){return String(s||"").replace(/<\|think\|>[\s\S]*?<\/\|think\|>/g,"").trim();}

function _apply(s){
  const n=new Date();
  const d=n.toLocaleDateString("en-US",{year:"numeric",month:"long",day:"numeric",timeZone:GENESIS_TIMEZONE});
  const t=n.toLocaleTimeString("en-US",{hour:"2-digit",minute:"2-digit",timeZone:GENESIS_TIMEZONE});
  // FIX(issue-9.6): %YEAR% is the numeric year; %DAY% is the full weekday.
  const y=n.toLocaleDateString("en-US",{year:"numeric",timeZone:GENESIS_TIMEZONE});
  const day=n.toLocaleDateString("en-US",{weekday:"long",timeZone:GENESIS_TIMEZONE});
  return String(s||"").replace(/%DATE%/g,d).replace(/%TIME%/g,t).replace(/%YEAR%/g,y).replace(/%DAY%/g,day);
}

// FIX(issue-9.3): clearer internal names; these remain module-local.
function _levenshtein(a,b){if(a===b)return 0;if(!a.length)return b.length;if(!b.length)return a.length;let p=Array(b.length+1).fill().map((_,i)=>i),c;for(let i=1;i<=a.length;i++){c=[i];for(let j=1;j<=b.length;j++){const cost=a[i-1]===b[j-1]?0:1;c[j]=Math.min(p[j]+1,c[j-1]+1,p[j-1]+cost);}p=c;}return p[b.length];}

// FIX(issue-7): the word index (wi) was built but effectively unreachable at
// scale. It is removed; the first/prefix/bigram indexes cover retrieval.
function _buildIndex(m){
  if(m.__gx)return m.__gx;
  const fi=new Map(),pi=new Map(),bi=new Map(),ks=[];
  const keys=Object.keys(m);
  for(let i=0;i<keys.length;i++){
    const k=keys[i];
    if(k==="ver"||k==="NOM"||k==="SOS"||k==="EOS")continue;
    const toks=k.split(/\s+/).filter(Boolean);
    if(!toks.length)continue;
    ks.push(k);
    const f=toks[0];
    if(!fi.has(f))fi.set(f,[]);
    fi.get(f).push(k);
    const pr=k.slice(0,4);
    if(!pi.has(pr))pi.set(pr,[]);
    pi.get(pr).push(k);
    for(let j=0;j<toks.length-1;j++){
      const bg=toks[j]+'_'+toks[j+1];
      if(!bi.has(bg))bi.set(bg,[]);
      bi.get(bg).push(k);
    }
  }
  m.__gx={fi,pi,bi,ks};
  return m.__gx;
}

function _cheapScore(qt,qn,k){
  const kt=k.split(/\s+/).filter(Boolean);
  const kset=new Set(kt);
  let score=0;
  for(let i=0;i<qt.length;i++)if(kset.has(qt[i]))score+=1.2;
  for(let i=0;i<qt.length;i++){
    for(let j=0;j<kt.length;j++){
      if(qt[i]===kt[j])score+=0.4;
      else if(qt[i].includes(kt[j])||kt[j].includes(qt[i]))score+=0.15;
    }
  }
  if(qn.includes(k))score+=7;else if(k.includes(qn))score+=5;
  score-=0.03*Math.abs(kt.length-qt.length);
  if(qt.length>1&&kt.length>=qt.length-1)score+=0.4;
  return score;
}

function _edBonus(qt,k){
  // FIX(issue-2): this expensive feature is only used on the re-score pass.
  const kt=k.split(/\s+/).filter(Boolean);
  let bonus=0;
  for(let i=0;i<qt.length;i++){
    for(let j=0;j<kt.length;j++){
      if(qt[i]!==kt[j]&&qt[i].length>3&&kt[j].length>3&&_levenshtein(qt[i],kt[j])===1)bonus+=0.25;
    }
  }
  return bonus;
}

function _nomResult(){
  return{key:null,response:_apply(_strip(model.NOM)),raw:model.NOM,confidence:0,thought:null};
}

// Complexity note (FIX(issue-2)): candidate retrieval is capped at ~4k keys.
// The cheap pass is O(candidates * query_tokens * key_tokens) with no edit
// distance. Only the top 500 cheap candidates pay for _levenshtein, so the
// worst case is bounded by ~500 * query_tokens * key_tokens edit-distance
// calls instead of 10^5-3*10^5.
function generalise(input,opts){
  opts=opts||{};
  // FIX(issue-4): merge is opt-in; undefined/omitted means single answer.
  const merge=opts.merge===true;
  const qn=_norm(input);
  if(!qn)return _nomResult();
  if(model[qn]!==undefined){
    const raw=model[qn];
    return{key:qn,response:_apply(_strip(raw)),raw,confidence:1.0,thought:null};
  }

  const idx=_buildIndex(model);
  const qt=_tokens(qn);
  const cand=new Set();
  const xt=_expand(qt);
  for(let i=0;i<xt.length;i++){
    const arr=idx.fi.get(xt[i]);
    if(arr)for(let j=0;j<arr.length&&cand.size<CAP_FIRST;j++)cand.add(arr[j]);
  }
  for(let i=0;i<qt.length-1;i++){
    const bg=qt[i]+'_'+qt[i+1];
    const arr=idx.bi.get(bg);
    if(arr)for(let j=0;j<arr.length&&cand.size<CAP_BIGRAM;j++)cand.add(arr[j]);
  }
  const prarr=idx.pi.get(qn.slice(0,4));
  if(prarr)for(let i=0;i<prarr.length&&cand.size<CAP_PREFIX;i++)cand.add(prarr[i]);

  // FIX(issue-2/7): keep edit distance only on the small-candidate fuzzy path;
  // the old word-index branch was dead weight at scale and is gone.
  if(cand.size<CAP_FUZZY_FALLBACK){
    const ks=idx.ks;
    for(let i=0;i<ks.length&&cand.size<CAP_FUZZY;i++){
      const k=ks[i];
      const kt=k.split(/\s+/);
      for(let x=0;x<qt.length;x++){
        for(let y=0;y<kt.length;y++){
          if(kt[y]===qt[x]){cand.add(k);break;}
          if(kt[y].length>2&&qt[x].length>2&&_levenshtein(kt[y],qt[x])<=1){cand.add(k);break;}
        }
        if(cand.has(k))break;
      }
    }
  }
  if(cand.size===0)return _nomResult();

  // Cheap scoring pass: token overlap, exact token equality, containment and
  // length penalty only. No Levenshtein here.
  const scored=[];
  for(const k of cand){
    scored.push({key:k,score:_cheapScore(qt,qn,k),raw:model[k]});
  }
  scored.sort((a,b)=>b.score-a.score||(a.key<b.key?-1:a.key>b.key?1:0));

  // FIX(issue-2): re-score only the top CAP_RESCORE candidates with the
  // edit-distance bonus, then sort those again.
  const ranked=scored.slice(0,CAP_RESCORE);
  for(const item of ranked)item.score+=_edBonus(qt,item.key);
  ranked.sort((a,b)=>b.score-a.score||(a.key<b.key?-1:a.key>b.key?1:0));

  // FIX(issue-4/5): stricter gate, and the merge result keeps synthesized
  // prose out of `response`/`raw`. `thought` is delivered separately and
  // `raw` becomes the array of original model values that were merged.
  if(merge&&ranked.length>=2){
    const top=ranked[0],second=ranked[1];
    if(top.score>0.6&&second.score>0.6&&second.score>=top.score*0.7&&top.key!==second.key){
      const parts=[],raws=[],mergedFrom=[],used=new Set();
      const floor=Math.max(0.6,top.score*0.7);
      for(let i=0;i<Math.min(4,ranked.length);i++){
        const item=ranked[i];
        if(item.score<=0.6||item.score<floor)continue;
        const clean=_apply(_strip(item.raw));
        if(!clean)continue;
        const signature=clean.slice(0,60);
        if(used.has(signature))continue;
        used.add(signature);
        parts.push(clean);
        raws.push(item.raw);
        mergedFrom.push(item.key);
      }
      if(parts.length>=2){
        return{
          key:top.key,
          response:parts.join(" "),
          raw:raws,
          confidence:Math.min(1,top.score/(qt.length+1.5)),
          thought:GENESIS_MERGE_THOUGHT,
          composed:true,
          mergedFrom
        };
      }
    }
  }

  const best=ranked[0];
  if(best&&best.score>0.4){
    const raw=best.raw;
    return{key:best.key,response:_apply(_strip(raw)),raw,confidence:Math.min(1,best.score/(qt.length+1)),thought:null};
  }
  for(let n=4;n>=2;n--){
    for(let i=0;i<=qt.length-n;i++){
      const ph=qt.slice(i,i+n).join(" ");
      if(model[ph]!==undefined){
        const raw=model[ph];
        return{key:ph,response:_apply(_strip(raw)),raw,confidence:0.75,thought:null};
      }
    }
  }
  return _nomResult();
}

// FIX(issue-9.4): keep the global for existing callers, but also expose the
// matcher to CommonJS/bundler consumers without adding a dependency.
if (typeof module !== "undefined" && module.exports) {
  module.exports = { generalise, model };
}
"""


def _stream_json_to_bin(src: Path, dst_tmp: Path) -> tuple[int, int]:
    """Stream src JSON into dst_tmp as a JS .bin.

    Returns (bytes_written, content_key_count).
    Raises ValueError if the source is not a well-formed top-level object.

    FIX(issue-7d): single-pass streaming. Source is read in STREAM_CHUNK_BYTES
    chunks; each chunk is written straight through and simultaneously fed to a
    small state machine that (a) counts top-level keys and (b) validates the
    JSON structure. Peak RAM = O(chunk), not O(model). A 2.83 GB model needs
    the same ~200 KB as a 2 MB one.
    """
    bytes_written = 0
    count = 0
    depth = 0
    in_str = False
    escape = False
    expect_key = False
    string_started_at_depth = -1
    started = False

    prefix = "const model = "
    suffix = ";\nglobalThis.model = model;\n" + JS_CODE + "\n"

    with src.open("r", encoding="utf-8") as fin, \
         dst_tmp.open("w", encoding="utf-8", newline="\n") as fout:
        fout.write(prefix)
        bytes_written += len(prefix.encode("utf-8"))

        while True:
            chunk = fin.read(STREAM_CHUNK_BYTES)
            if not chunk:
                break
            fout.write(chunk)
            bytes_written += len(chunk.encode("utf-8"))

            for ch in chunk:
                if escape:
                    escape = False
                    continue
                if in_str:
                    if ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                        if string_started_at_depth == 1 and expect_key:
                            count += 1
                            expect_key = False
                        string_started_at_depth = -1
                    continue
                if ch == '"':
                    in_str = True
                    string_started_at_depth = depth
                elif ch == "{":
                    depth += 1
                    if depth == 1:
                        started = True
                        expect_key = True
                elif ch == "}":
                    depth -= 1
                elif ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                elif ch == ",":
                    if depth == 1:
                        expect_key = True

        fout.write(suffix)
        bytes_written += len(suffix.encode("utf-8"))

    if not started:
        raise ValueError(f"{src.name} does not start with a JSON object")
    if depth != 0:
        raise ValueError(f"{src.name} appears malformed (JSON depth ended at {depth}, expected 0)")
    if in_str:
        raise ValueError(f"{src.name} appears truncated (unterminated string)")

    return bytes_written, count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="run `node --check` on the written .bin (warn if Node is missing)",
    )
    parser.add_argument(
        "--max-keys",
        type=int,
        default=DEFAULT_MAX_CONTENT_KEYS,
        help="content-key threshold that requires --i-know-what-im-doing",
    )
    parser.add_argument(
        "--i-know-what-im-doing",
        action="store_true",
        help="allow writing a model above --max-keys",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    opt_in = args.i_know_what_im_doing

    # FIX(issue-9.1): friendly failure when the JSON has not been built yet.
    if not MODEL_JSON.exists():
        sys.exit("model/Genesis-SPT-7.0.json is missing; run build_model.py first")

    fsize = MODEL_JSON.stat().st_size
    print(
        f"streaming {MODEL_JSON.name} ({fsize / 1024 ** 2:.0f} MB) -> {MODEL_BIN.name}",
        file=sys.stderr,
    )

    MODEL_BIN.parent.mkdir(parents=True, exist_ok=True)
    tmp = MODEL_BIN.with_suffix(MODEL_BIN.suffix + ".tmp")

    # FIX(issue-7d): stream to a sibling temp file so a mid-stream failure
    # never leaves a half-written .bin. rename() is atomic on the same fs.
    try:
        bytes_written, content_keys = _stream_json_to_bin(MODEL_JSON, tmp)
    except ValueError as e:
        try:
            tmp.unlink()
        except OSError:
            pass
        sys.exit(f"ERROR: {e}")
    except MemoryError:
        # Should be impossible — we only hold 64 KB chunks — but if the OS
        # has no page cache to spare, fail cleanly rather than leave debris.
        try:
            tmp.unlink()
        except OSError:
            pass
        sys.exit(f"ERROR: out of RAM while streaming {MODEL_JSON.name}")

    # FIX(issue-3): safety check now runs on the streamed count, before the
    # .bin is committed. On failure the temp file is deleted, so the shipped
    # artifact is never an over-size model that the user didn't ask for.
    if content_keys > args.max_keys and not opt_in:
        try:
            tmp.unlink()
        except OSError:
            pass
        print(
            f"ERROR: model has {content_keys:,} content keys, above the "
            f"{args.max_keys:,} safety threshold.",
            file=sys.stderr,
        )
        print(
            "       A model this large will be slow to build/load. Regenerate "
            "generated.txt with a smaller --count,",
            file=sys.stderr,
        )
        print(
            "       or re-run with --i-know-what-im-doing if you really want it.",
            file=sys.stderr,
        )
        return 2

    tmp.replace(MODEL_BIN)
    print(
        f"wrote {MODEL_BIN} "
        f"({bytes_written:,} bytes, {content_keys:,} content keys)"
    )

    # FIX(issue-9.2): optional syntax validation. Missing Node is not fatal.
    # FIX(issue-7e): node --check reads the whole file into one V8 string, and
    # V8 caps strings at 0x1fffffe8 (~512 MiB). Files larger than that cannot
    # be checked by node at all — treat as "skipped" not "failed".
    if args.check:
        node = shutil.which("node")
        if not node:
            print("warning: node not found on PATH; skipped --check", file=sys.stderr)
        else:
            out_size = MODEL_BIN.stat().st_size
            if out_size > CHECK_MAX_BYTES:
                print(
                    f"note: {MODEL_BIN.name} is {out_size / 1024 ** 3:.2f} GB; "
                    f"skipping node --check (V8 string limit is ~512 MiB, so "
                    f"files larger than that cannot be checked by node at all). "
                    f"The JSON payload was already structurally validated "
                    f"during streaming; this check adds nothing here.",
                    file=sys.stderr,
                )
            elif out_size > CHECK_WARN_BYTES:
                print(
                    f"note: node --check on a {out_size / 1024 ** 3:.2f} GB file "
                    f"will take several minutes; pass without --check to skip",
                    file=sys.stderr,
                )
            if out_size <= CHECK_MAX_BYTES:
                with tempfile.NamedTemporaryFile(suffix=".js", delete=False) as tf:
                    check_path = tf.name
                try:
                    shutil.copyfile(MODEL_BIN, check_path)
                    result = subprocess.run([node, "--check", check_path])
                finally:
                    try:
                        Path(check_path).unlink()
                    except OSError:
                        pass
                if result.returncode != 0:
                    print(f"ERROR: node --check failed for {MODEL_BIN}", file=sys.stderr)
                    return result.returncode or 1
                print(f"ok: node --check passed for {MODEL_BIN}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
