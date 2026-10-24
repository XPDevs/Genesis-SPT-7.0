function _norm(s) { return String(s || "").toLowerCase().trim().replace(/\s+/g, " "); }
function _tokens(s) { return _norm(s).split(" ").filter(Boolean); }
const GENESIS_SYNONYMS = {
  hi: ["hello","hey","howdy","greetings"], hello: ["hi","hey","howdy"],
  sad: ["unhappy","down","depressed","blue"], happy: ["glad","joyful","cheerful","content"],
  good: ["great","fine","well","okay"], bad: ["terrible","awful","poor"],
  what: ["tell","explain","describe","define"], how: ["way","method","steps"],
  why: ["reason","cause"], make: ["create","build","generate"],
  old: ["age","years"], young: ["new","fresh"], love: ["like","enjoy","adore"]
};
function _expandTokens(tokens, syns) {
  syns = syns || GENESIS_SYNONYMS;
  const out = [...tokens];
  for (const t of tokens) if (syns[t]) out.push(...syns[t]);
  return out;
}
function _stripThink(s) { return String(s || "").replace(/<\|think\|>.*?<\/\|think\|>/gs, "").trim(); }
function _applyPlaceholders(s) {
  const now = new Date();
  const date = now.toLocaleDateString("en-US", { year: "numeric", month: "long", day: "numeric" });
  const time = now.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" });
  return String(s || "").replace(/%DATE%/g, date).replace(/%TIME%/g, time);
}
function _ensureIndex(model) {
  if (model.__idx) return model.__idx;
  const firstIndex = new Map();
  const prefixIndex = new Map();
  const keys = Object.keys(model);
  for (let i = 0; i < keys.length; i++) {
    const k = keys[i];
    if (k === "ver" || k === "NOM" || k === "SOS" || k === "EOS") continue;
    const toks = k.split(/\s+/).filter(Boolean);
    const first = toks[0] || k;
    if (!firstIndex.has(first)) firstIndex.set(first, []);
    firstIndex.get(first).push(k);
    const pref = k.slice(0, 4);
    if (!prefixIndex.has(pref)) prefixIndex.set(pref, []);
    prefixIndex.get(pref).push(k);
  }
  model.__idx = { firstIndex, prefixIndex };
  return model.__idx;
}
function _editDistance(a, b) {
  if (a === b) return 0;
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  let prev = Array.from({length: b.length + 1}, (_, i) => i);
  for (let i = 1; i <= a.length; i++) {
    let cur = [i];
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i-1] === b[j-1] ? 0 : 1;
      cur[j] = Math.min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost);
    }
    prev = cur;
  }
  return prev[b.length];
}
function generalise(input, options) {
  options = options || {};
  const syns = options.synonyms || GENESIS_SYNONYMS;
  const q = _norm(input);
  if (!q) return { key: null, response: _applyPlaceholders(_stripThink(model.NOM)), raw: model.NOM, confidence: 0 };
  if (model[q] !== undefined) {
    const raw = model[q];
    return { key: q, response: _applyPlaceholders(_stripThink(raw)), raw, confidence: 1.0 };
  }
  const idx = _ensureIndex(model);
  const qTokens = _tokens(q);
  const candSet = new Set();
  for (const t of _expandTokens(qTokens, syns)) {
    const arr = idx.firstIndex.get(t);
    if (arr) for (const k of arr) { candSet.add(k); if (candSet.size > 6000) break; }
    if (candSet.size > 6000) break;
  }
  if (candSet.size === 0) {
    const arr = idx.prefixIndex.get(q.slice(0, 4));
    if (arr) for (const k of arr) { candSet.add(k); if (candSet.size > 6000) break; }
  }
  if (candSet.size < 50) {
    const allKeys = Object.keys(model);
    for (let i = 0; i < allKeys.length && candSet.size < 2000; i++) {
      const k = allKeys[i];
      if (k === "ver" || k === "NOM" || k === "SOS" || k === "EOS") continue;
      const kt = k.split(/\s+/);
      for (const t of qTokens) {
        for (const w of kt) {
          if (w.length >= 3 && _editDistance(t, w) <= 1) { candSet.add(k); break; }
        }
        if (candSet.size > 2000) break;
      }
    }
  }
  if (candSet.size === 0) return { key: null, response: _applyPlaceholders(_stripThink(model.NOM)), raw: model.NOM, confidence: 0 };

  const qset = new Set(qTokens);
  const scored = [];
  for (const k of candSet) {
    const kt = new Set(k.split(/\s+/).filter(Boolean));
    let score = 0;
    for (const t of qset) if (kt.has(t)) score += 1;
    if (q.includes(k) || k.includes(q)) score += 5;
    score -= 0.1 * Math.abs(kt.size - qset.size);
    scored.push({ key: k, score });
  }
  scored.sort((a, b) => b.score - a.score);
  const best = scored[0];
  if (best && best.score > 0) {
    const raw = model[best.key];
    let response = _applyPlaceholders(_stripThink(raw));
    if (options && options.merge && scored.length > 1 && scored[1].score >= best.score * 0.6) {
      const raw2 = model[scored[1].key];
      const extra = _applyPlaceholders(_stripThink(raw2));
      if (extra && extra !== response) response = response + " " + extra;
    }
    return { key: best.key, response, raw, confidence: Math.min(1, best.score / (qTokens.length + 1)) };
  }
  for (let n = 3; n >= 2; n--) {
    for (let i = 0; i <= qTokens.length - n; i++) {
      const phrase = qTokens.slice(i, i + n).join(" ");
      if (model[phrase] !== undefined) {
        const raw = model[phrase];
        return { key: phrase, response: _applyPlaceholders(_stripThink(raw)), raw, confidence: 0.7 };
      }
    }
  }
  return { key: null, response: _applyPlaceholders(_stripThink(model.NOM)), raw: model.NOM, confidence: 0 };
}
