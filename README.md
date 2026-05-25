# Crossmodal Search

Natural-language search over local Waymo End-to-End Camera TFRecord shards.

This repo is an MVP for indexing a local shard, captioning synchronized
8-camera frames, embedding the grounded text, and serving a search UI with a
cockpit detail page. It is not a canned demo: search results come from the
local generated index under `crossmodal_search/data/`, which is intentionally
gitignored.

## Current State

As of 2026-05-25, the implementation supports the full local path:

- scan one local E2E TFRecord shard into `frames.parquet`
- hydrate frame images from byte offsets
- build 4x2 camera mosaics
- call a managed vision captioning agent
- embed captions/tags into a local vector index
- serve a Flask search UI and frame detail page

The active local index in this workspace is still small: 20 sampled rows, with
1 real VLM-captioned frame and 19 provider failures. The real visual frame is:

```text
e0ae38e2d9675d9d889eaffcf10fe392-142
intent: GO_RIGHT
caption: A vehicle navigating an urban environment with signs of road
construction and multiple intersections, transitioning towards a curved road
section marked by retaining walls and chevron signs, under clear daytime
conditions.
```

That means visual search currently works only for concepts present in that
caption/tags. For example, `intersection` can return the real indexed frame.
`crash` returns no result because no indexed visual caption contains crash
evidence. The UI reports how many visual frames are indexed and does not show
placeholder rows as visual search hits.

## Managed Agents

The project uses managed agents when provider keys are available in `.env` or
the process environment.

- `QueryRouterAgent` routes user queries into supported filters and visual
  requirements. It prevents unsupported visual terms from being treated as
  indexed facts.
- `VisionCaptionAgent` captions a 4x2 mosaic of the 8 synchronized cameras and
  emits a caption plus structured tags.

Provider selection is automatic by default:

- Gemini: `GEMINI_API_KEY`, default model `gemini-2.5-flash`
- OpenAI: `OPENAI_API_KEY`, default model `gpt-5-mini`
- Override provider: `CROSSMODAL_AGENT_PROVIDER=gemini|openai|auto`
- Disable hosted agents: `CROSSMODAL_MANAGED_AGENTS=0`

Lowercase aliases such as `gemini_api_key` and `openai_api_key` are also
accepted. `.env` can live at the repo root or inside `crossmodal_search/`.
Never commit `.env`.

Known provider blockers:

- Gemini captioning has produced HTTP/API failures during batch captioning,
  including an invalid-key response in one retry.
- OpenAI routing/captioning has hit account rate limits in this environment.

Until provider reliability is fixed, the product should be presented as a
working local MVP with a partial visual index, not as a complete dataset-wide
visual search engine.

## Search Behavior

Search is intentionally strict:

- visual queries search only rows with `caption_status == "vlm"`
- visual terms must appear in the grounded caption, tags, or structured fields
- intent and motion terms are hard filters, not score boosts
- results are deduplicated by segment
- a query such as `left turn` cannot return a `GO_RIGHT` frame
- a query such as `crash` returns no results unless a real VLM-captioned row
  contains crash evidence

The fallback structural rows still help with development and ego-motion
commands, but they are not surfaced as visual evidence.

## Setup

Use Python 3.11 or newer.

```bash
cd /Users/dalrae/Downloads/Developed/crossmodal_search
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Add provider keys if you want managed routing or managed captioning:

```bash
cat > .env <<'EOF'
GEMINI_API_KEY=...
# OPENAI_API_KEY=...
CROSSMODAL_AGENT_PROVIDER=auto
EOF
```

The default shard path is the repo root:

```text
test_202504211836-202504220845.tfrecord-00004-of-00266
```

You can also pass any local shard with `--shard /path/to/shard`.

## Build A Local Index

Scan the local shard:

```bash
python3 -m crossmodal_search.indexer.scan \
  --shard /path/to/test_202504211836-202504220845.tfrecord-00004-of-00266 \
  --progress-every 100
```

Caption a small batch with the managed vision agent:

```bash
python3 -m crossmodal_search.indexer.caption \
  --model managed \
  --limit 20 \
  --progress-every 1 \
  --shard /path/to/test_202504211836-202504220845.tfrecord-00004-of-00266
```

Resume and retry failed VLM rows:

```bash
python3 -m crossmodal_search.indexer.caption \
  --model managed \
  --resume-from crossmodal_search/data/frames_tagged.parquet \
  --retry-failed \
  --limit 20 \
  --progress-every 1 \
  --shard /path/to/test_202504211836-202504220845.tfrecord-00004-of-00266
```

Build the local vector index:

```bash
python3 -m crossmodal_search.indexer.embed
```

Run the web UI:

```bash
python3 -m crossmodal_search.ui.app
```

Open:

```text
http://127.0.0.1:5003
```

## Useful Checks

Inspect local index status:

```bash
python3 -c "import pandas as pd; df=pd.read_parquet('crossmodal_search/data/frames_embedded.parquet'); print(len(df)); print(df['caption_status'].value_counts(dropna=False))"
```

Try the CLI search path:

```bash
python3 -m crossmodal_search.search.query "intersection"
python3 -m crossmodal_search.search.query "crash"
python3 -m crossmodal_search.search.query "left turn"
```

Expected behavior with the current partial local index:

- `intersection` can return the one real VLM-captioned frame
- `crash` returns no result
- `left turn` returns no result if there is no `GO_LEFT` VLM row

## Data Products

Generated files live under `crossmodal_search/data/` and are ignored by git.

- `frames.parquet`: scanned shard metadata, frame names, byte offsets, ego
  intent, and past state summaries
- `frames_tagged.parquet`: captions, tags, structured fields, caption status,
  and optional composite paths
- `frames_embedded.parquet`: tagged rows plus `search_text`
- `text_vectors.npy`: local HashingVectorizer vectors
- `search_index.json`: manifest consumed by the CLI and Flask app
- `composites/`: optional 4x2 camera mosaic JPEG cache

## UI

The UI has two screens:

- `/`: search bar plus result cards, each using a 4x2 mosaic in the Waymo
  camera order
- `/frame/<frame_name>`: cockpit detail page with camera sidebar, frame
  metadata, and segment scrubber

Image payloads are hydrated from the local shard using `shard_path`,
`byte_offset`, and `byte_length`.

## Roadmap

Immediate work:

- fix provider/key reliability so the managed captioner can complete a full
  shard batch
- index all frames in one shard with real VLM captions
- add an eval set for "must return" and "must not return" queries
- replace the current hashing-vectorizer backend when the caption corpus is
  large enough to justify a stronger embedding store
- create a demo video only from the real active index after visual coverage is
  broad enough

Deferred work:

- GCS range reads instead of local-only shard access
- all 266 test shards
- Motion/E2E joins
- image embeddings or CLIP-style retrieval
