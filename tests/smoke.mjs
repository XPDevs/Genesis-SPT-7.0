// Standalone acceptance smoke test for Genesis-SPT-7.0.
// Run with: node tests/smoke.mjs
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  mkdtempSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const HERE = dirname(fileURLToPath(import.meta.url));
const ROOT = resolve(HERE, "..");
const PYTHON = process.env.PYTHON || "python3";
const BUNDLE = join(ROOT, "model", "Genesis-SPT-7.0.bin");
const SCRIPTS = [
  "paths.py",
  "build_model.py",
  "build_bin.py",
  "validate.py",
  "generate_training_data.py",
];

const RESERVED = {
  ver: "Genesis-SPT-7.0",
  SOS: "<|SOS|>",
  EOS: "<|EOS|>",
  NOM: "<|think|>No match.</|think|> I'm not sure how to respond to that.",
};

let passed = 0;
let failed = 0;

async function check(name, fn) {
  try {
    await fn();
    passed += 1;
    console.log(`ok ${passed} - ${name}`);
  } catch (err) {
    failed += 1;
    console.error(`not ok - ${name}`);
    console.error(err && err.stack ? err.stack : err);
  }
}

function makeModel(extra = {}) {
  return { ...RESERVED, ...extra };
}

function tailOfBundle(source) {
  const marker = "globalThis.model = model;";
  const at = source.indexOf(marker);
  if (at < 0) throw new Error("bundle does not contain globalThis.model = model;");
  return source.slice(at + marker.length);
}

function loadSource(source, filename = "genesis-smoke-bundle.js") {
  const module = { exports: {} };
  const sandbox = { console, module };
  const context = vm.createContext(sandbox);
  vm.runInContext(source, context, { filename });
  if (typeof context.generalise !== "function") {
    throw new Error("bundle did not define generalise()");
  }
  return { ...context, exports: module.exports, model: context.model };
}

function loadModel(modelObject) {
  const tail = tailOfBundle(readFileSync(BUNDLE, "utf8"));
  const source =
    "const model = " + JSON.stringify(modelObject) + ";\n" +
    "globalThis.model = model;\n" +
    tail;
  return loadSource(source, "genesis-smoke-model.js");
}

function loadShipped() {
  return loadSource(readFileSync(BUNDLE, "utf8"), "genesis-smoke-shipped.bin");
}

function makeProject(label) {
  const dir = mkdtempSync(join(tmpdir(), `genesis-${label}-`));
  mkdirSync(join(dir, "train"), { recursive: true });
  mkdirSync(join(dir, "training data"), { recursive: true });
  mkdirSync(join(dir, "model"), { recursive: true });
  for (const name of SCRIPTS) {
    copyFileSync(join(ROOT, "train", name), join(dir, "train", name));
  }
  return dir;
}

function runPython(dir, script, args = []) {
  return spawnSync(PYTHON, [join(dir, "train", script), ...args], {
    cwd: dir,
    encoding: "utf8",
    env: { ...process.env },
  });
}

function assertRunOk(result, label) {
  assert.equal(
    result.status,
    0,
    `${label} failed (status ${result.status})\nSTDOUT:\n${result.stdout}\nSTDERR:\n${result.stderr}`,
  );
}

const shipped = loadShipped();

await check("1 exact match works and returns confidence 1", () => {
  const r = shipped.generalise("hello");
  assert.equal(r.key, "hello");
  assert.equal(r.confidence, 1.0);
  assert.match(r.response, /Hey there! Genesis here/);
  assert.equal(typeof r.raw, "string");
  assert.equal(r.thought, null);
});

await check("2 punctuation-safe normalization hits the apostrophe key", () => {
  const js = loadModel(makeModel({ "what s up": "<|think|>x</|think|> Not much!" }));
  for (const q of ["what's up", "whats up"]) {
    const r = js.generalise(q);
    assert.equal(r.key, "what s up", `${q} should hit the normalized key`);
    assert.ok(r.confidence >= 0.6, `${q} confidence ${r.confidence} should be >= 0.6`);
  }
});

await check("3 fallback returns NOM with confidence 0", () => {
  const r = shipped.generalise("qwerty zxcvb asdf");
  assert.equal(r.key, null);
  assert.equal(r.confidence, 0);
  assert.equal(r.raw, shipped.model.NOM);
  assert.ok(!r.response.includes("<|think|>"));
});

await check("4 merge is opt-in and returns thought/raw-array schema", () => {
  const js = loadModel(
    makeModel({
      "hello there": "<|think|>t</|think|> There",
      "hello friend": "<|think|>f</|think|> Friend",
      "hello world": "<|think|>w</|think|> World",
    }),
  );

  const single = js.generalise("hello");
  assert.equal(single.thought, null);
  assert.equal(typeof single.raw, "string");
  assert.ok(!single.composed);

  const merged = js.generalise("hello", { merge: true });
  assert.match(merged.thought, /^<\|think\|>[\s\S]*<\/\|think\|>$/);
  assert.ok(!merged.response.includes("<|think|>"), "think block leaked into response");
  assert.ok(!merged.response.includes("</|think|>"), "closing think tag leaked into response");
  assert.ok(Array.isArray(merged.raw), "merge raw must be an array");
  assert.ok(merged.raw.length >= 2, "merge should include at least two raw values");
  assert.ok(merged.raw.every((v) => typeof v === "string"));
  assert.equal(merged.composed, true);
  assert.ok(Array.isArray(merged.mergedFrom) && merged.mergedFrom.length >= 2);
});

await check("5 no think block leaks under 100 random inputs, both merge modes", () => {
  const words = [
    "hello", "there", "friend", "what", "is", "your", "name", "today", "time",
    "alpha", "beta", "gamma", "delta", "quantum", "computing", "genesis", "model",
    "random", "phrase", "test", "the", "a", "of", "and", "how", "old", "are", "you",
  ];
  let state = 0x12345678;
  const rand = () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
  const pick = () => words[Math.floor(rand() * words.length)];

  for (let i = 0; i < 100; i += 1) {
    const n = 1 + Math.floor(rand() * 5);
    const q = Array.from({ length: n }, pick).join(" ") + (i % 3 === 0 ? "?!" : "");
    for (const merge of [false, true]) {
      const r = shipped.generalise(q, { merge });
      assert.ok(!r.response.includes("<|think|>"), `leak for ${JSON.stringify(q)} merge=${merge}`);
      assert.ok(!r.response.includes("</|think|>"), `close-tag leak for ${JSON.stringify(q)} merge=${merge}`);
    }
  }
});

await check("6 placeholder substitution uses UTC date/time plus %YEAR%/%DAY%", () => {
  const js = loadModel(makeModel({ today: "Today is %DATE% at %TIME% %YEAR% %DAY%" }));
  const r = js.generalise("today");
  assert.ok(!r.response.includes("%DATE%"));
  assert.ok(!r.response.includes("%TIME%"));
  assert.ok(!r.response.includes("%YEAR%"));
  assert.ok(!r.response.includes("%DAY%"));
  const now = new Date();
  const year = now.toLocaleDateString("en-US", { year: "numeric", timeZone: "UTC" });
  const day = now.toLocaleDateString("en-US", { weekday: "long", timeZone: "UTC" });
  assert.ok(r.response.includes(year), `missing UTC year ${year}: ${r.response}`);
  assert.ok(r.response.includes(day), `missing UTC weekday ${day}: ${r.response}`);
  assert.match(r.response, /[A-Z][a-z]+ \d{1,2}, \d{4}/);
  assert.match(r.response, /\d{2}:\d{2}\s?(AM|PM)/);
});

await check("7 build_model normalizes apostrophe/em-dash/uppercase keys", () => {
  const dir = makeProject("norm");
  try {
    writeFileSync(
      join(dir, "model", "Genesis-SPT-7.0.json"),
      JSON.stringify({
        ...RESERVED,
        "WHAT'S UP — FRIEND": "<|think|>x</|think|> Punctuation survives.",
        "!!!": "empty after normalization",
      }),
    );
    const res = runPython(dir, "build_model.py");
    assertRunOk(res, "build_model.py");

    const data = JSON.parse(readFileSync(join(dir, "model", "Genesis-SPT-7.0.json"), "utf8"));
    assert.ok(!("WHAT'S UP — FRIEND" in data));
    assert.ok("what s up friend" in data);
    for (const key of Object.keys(data)) {
      if (key in RESERVED) continue;
      assert.match(key, /^[a-z0-9 ]+$/, `invalid normalized key ${JSON.stringify(key)}`);
      assert.equal(key, key.trim().replace(/\s+/g, " "));
    }

    const js = loadModel(data);
    const r = js.generalise("WHAT'S UP — FRIEND");
    assert.equal(r.key, "what s up friend");
    assert.equal(r.confidence, 1.0);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

await check("8 50k-key model answers a typical query under 200ms median", () => {
  const big = makeModel();
  const vocab = [
    "common", "query", "words", "topic", "term", "subject", "answer", "question",
    "simple", "model", "lookup", "phrase", "search", "match", "response", "input",
    "output", "data", "index", "score",
  ];
  for (let i = 0; i < 50_000; i += 1) {
    const key = `alpha ${vocab[i % vocab.length]} topic${i} word${i % 97} term${i % 53}`;
    big[key] = `response ${i}`;
  }
  const js = loadModel(big);
  for (let i = 0; i < 3; i += 1) js.generalise("alpha common query words");
  const times = [];
  for (let i = 0; i < 20; i += 1) {
    const t0 = performance.now();
    const r = js.generalise("alpha common query words");
    times.push(performance.now() - t0);
    assert.equal(typeof r.response, "string");
  }
  times.sort((a, b) => a - b);
  const median = times[Math.floor(times.length / 2)];
  assert.ok(median < 200, `median latency ${median.toFixed(1)}ms is >= 200ms`);
  console.log(`   median latency: ${median.toFixed(1)}ms`);
});

let builtProject = null;
await check("9 build_bin writes a .bin that passes node --check", () => {
  const dir = makeProject("build");
  builtProject = dir;
  writeFileSync(
    join(dir, "training data", "examples.txt"),
    "hello ||| <|think|>Greeting.</|think|> Hey there!\n",
  );
  assertRunOk(runPython(dir, "build_model.py"), "build_model.py");
  const res = runPython(dir, "build_bin.py", ["--check"]);
  assertRunOk(res, "build_bin.py --check");
  assert.ok(existsSync(join(dir, "model", "Genesis-SPT-7.0.bin")));
  assert.match(res.stdout + res.stderr, /node --check passed/);
});

await check("10 validate.py passes the new ordering/normalization assertions", () => {
  assert.ok(builtProject, "build project from test 9 is missing");
  const res = runPython(builtProject, "validate.py");
  assertRunOk(res, "validate.py");
  assert.match(res.stdout, /^OK: /m);
  rmSync(builtProject, { recursive: true, force: true });
  builtProject = null;
});

await check("11 generate_training_data defaults are safe and refuse overwrite", () => {
  const dir = makeProject("gen");
  try {
    const out = join(dir, "training data", "generated.txt");
    const first = runPython(dir, "generate_training_data.py");
    assertRunOk(first, "generate_training_data.py (no args)");
    const lines = readFileSync(out, "utf8").trimEnd().split(/\r?\n/);
    assert.ok(lines.length > 0 && lines.length <= 1000, `wrote ${lines.length} lines`);

    writeFileSync(out, "SENTINEL\n");
    const second = runPython(dir, "generate_training_data.py");
    assert.notEqual(second.status, 0, "no-args run must refuse to overwrite");
    assert.equal(readFileSync(out, "utf8"), "SENTINEL\n");
    assert.match(second.stderr, /--force/);

    const capped = runPython(dir, "generate_training_data.py", ["--count", "10001"]);
    assert.notEqual(capped.status, 0, "count above hard cap must fail");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

await check("12 hand-written examples win over generated collisions", () => {
  const dir = makeProject("order");
  try {
    writeFileSync(join(dir, "training data", "generated.txt"), "hello ||| GENERATED\n");
    writeFileSync(join(dir, "training data", "examples.txt"), "hello ||| REAL\n");
    assertRunOk(runPython(dir, "build_model.py"), "build_model.py");
    const data = JSON.parse(readFileSync(join(dir, "model", "Genesis-SPT-7.0.json"), "utf8"));
    assert.equal(data.hello, "REAL");
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

if (builtProject) {
  rmSync(builtProject, { recursive: true, force: true });
}

if (failed > 0) {
  console.error(`\n${failed} smoke test(s) failed, ${passed} passed.`);
  process.exit(1);
}
console.log(`\nAll ${passed} smoke tests passed.`);
