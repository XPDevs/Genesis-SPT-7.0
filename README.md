# Genesis-SPT-7.0

A from-scratch, non-neural "Genesis" model. It is a large JSON lookup table
(input phrase -> response) plus a small generalisation program. It is **not**
a trained transformer and does not use Hugging Face weights.

Current dataset: ~2,000,000 keys.

## Layout

```
Genesis-SPT-7.0/
  training data/        input pairs (input ||| response)
    examples.txt          curated examples
    generated.txt         generated expansion data
  train/
    paths.py              shared paths
    generate_training_data.py  expand training data
    build_model.py        training data + base -> model/Genesis-SPT-7.0.json
    build_bin.py          JSON -> model/Genesis-SPT-7.0.bin (JSON + JS bundle)
    validate.py           validates the model contract
  tools/
    chat.py               streaming terminal chat (thinking + answer)
  runtime/
    genesis-spt-7.0.js    JS runtime note/helper
  model/
    Genesis-SPT-7.0.json  source model
    Genesis-SPT-7.0.bin   JS bundle: const model = {...}; function generalise(...)
  data/                   logs / processed data
  checkpoints/            local checkpoints
  run.sh / train.sh / chat.sh / clean.sh
```

## Build pipeline

```bash
./train.sh          # training data -> JSON -> .bin
./run.sh            # build + validate
./chat.sh           # chat
```

Add your own data to `training data/*.txt` using one pair per line:

```
input phrase ||| response text
```

Then run `./train.sh`.

## Model format

Flat JSON object:

```json
{
  "ver": "Genesis-SPT-7.0",
  "SOS": "<|SOS|>",
  "EOS": "<|EOS|>",
  "NOM": "<|think|>...</|think|> fallback",
  "hello": "<|think|>...</|think|> Hey there! ..."
}
```

Responses may contain `<|think|>...</|think|>` reasoning. `%DATE%` and `%TIME%`
are substituted at runtime.

## Chat: thinking + streaming

`tools/chat.py` prints the model's thinking, then streams the answer
character-by-character like a real AI. Use `/think` to toggle reasoning display.

```bash
./chat.sh
# or: python3 tools/chat.py model/Genesis-SPT-7.0.bin
```

## Generalisation

The generalisation program is data-driven — it uses the model's own keys and
responses, not hardcoded answers:

1. Normalise input.
2. Exact lookup.
3. Synonym expansion (extendable via `options.synonyms`).
4. Candidate gathering through first-token and prefix indexes (cached).
5. Scoring by token overlap, containment and length similarity.
6. Edit-distance rescue for typos.
7. N-gram fallback for partial phrases.
8. Optional response merging (`{ merge: true }`).

## `.bin` bundle

`model/Genesis-SPT-7.0.bin` is a JavaScript bundle:

```js
const model = { ...model JSON... };
globalThis.model = model;

function generalise(input, options) { ... }
```

`tools/chat.py` extracts the JSON via `json.JSONDecoder().raw_decode`.
JavaScript can evaluate the bundle directly:

```js
const src = await (await fetch('model/Genesis-SPT-7.0.bin')).text();
(0, eval)(src);
const r = generalise("tell me about quantum computing", { merge: true });
console.log(r.response);
```

## Genesis-AI website compatibility

The JSON is a standard Genesis model (`ver`, `NOM`, `SOS`, `EOS`, lowercase
keys, string values) and can be loaded by Genesis-AI as a custom model without
modifying the website.
