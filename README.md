# Crossmodal Search

Natural-language search over the Waymo End-to-End Camera dataset.

Type `"left turn at an intersection with a cyclist, dusk"` → get back ranked
moments, each showing all 8 synchronized camera views + ego intent + past
trajectory.

Status: MVP vertical slice built for one local shard. The local data products
are generated under `crossmodal_search/data/` and are gitignored.

## Why this exists

Waymo publishes thousands of TFRecord shards. To answer the question *"show me
a moment where the car turns left and there is a cyclist"* you currently have
to download terabytes and scan TFRecords by hand. There is no index, no search,
no metadata table — just blobs in a bucket. Every researcher who touches this
data ends up writing their own "scan 2000 shards" loop. We are writing the
thing they all wished existed.

## Scope (deliberate narrowing)

Waymo ships three datasets in three buckets with three ID schemes that do not
cleanly join:

- **Motion** (`Scenario` proto) — tracks, agents, map features
- **End-to-End Camera** (`E2EDFrame`) — 8 cameras + ego state + intent
- **Perception** — LiDAR + 3D boxes

We are doing **E2E only**. The hard "join Motion ↔ E2E" problem (different
IDs, different time bases, ambiguous matching) is deferred. The E2E dataset
is already self-contained and rich enough to support magical NL queries.

## Unit of data

One **frame** = one TFRecord record in an E2E shard:

```text
frame = {
  frame_name:    "<segment_id>-<frame_index>"   # e.g. e0ae38e2…-142
  segment_id:    string                          # groups frames into a drive
  frame_index:   int                             # ordering inside segment
  intent:        GO_STRAIGHT | GO_LEFT | GO_RIGHT | UNKNOWN
  past_states:   16 steps × {pos_xyz, vel_xy, accel_xy}
  images:        { FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT,
                   REAR, REAR_LEFT, REAR_RIGHT }   # 8 JPEGs
  shard_object:  string   # GCS object name
  byte_offset:   int      # where the record starts in the shard
  byte_length:   int      # record payload length
}
```

A **segment** is a contiguous drive — frames from one segment are written
sequentially and almost certainly live in a single shard (verify on a few
segments before relying on this).

For the E2E test split alone: 266 shards × ~775 records ≈ **~206K frames**.

## Architecture

```
                ┌──────────────────────────────────────────┐
                │             GCS Bucket                    │
                │  waymo_open_dataset_end_to_end_camera…    │
                │  test_*.tfrecord-*-of-00266  (×266)       │
                └──────────────┬───────────────────────────┘
                               │ HTTP range reads
                ┌──────────────▼───────────────────────────┐
                │  Stage 1: Structural Scan                │
                │  - Walk every record header              │
                │  - Extract frame_name, intent, past      │
                │  - Record shard_object + byte_offset     │
                │  → frames.parquet  (~206K rows, ~50 MB)  │
                └──────────────┬───────────────────────────┘
                               │ for each frame, fetch 8 JPEGs
                ┌──────────────▼───────────────────────────┐
                │  Stage 2: Visual Tagging                 │
                │  - Composite 8 views into one image      │
                │  - VLM pass: caption + structured tags   │
                │    (weather, time-of-day, road type,     │
                │     agents present, traffic-light)       │
                │  → frames_tagged.parquet                 │
                └──────────────┬───────────────────────────┘
                               │
                ┌──────────────▼───────────────────────────┐
                │  Stage 3: Embedding                      │
                │  - Embed caption + structured-tag bag    │
                │  - Optional: CLIP image embedding        │
                │  → vector index (LanceDB / FAISS)         │
                └──────────────┬───────────────────────────┘
                               │
                ┌──────────────▼───────────────────────────┐
                │  Query API                                │
                │  NL query → embedding                     │
                │  + structured WHERE (intent='GO_LEFT')    │
                │  → top-K frame_names                       │
                └──────────────┬───────────────────────────┘
                               │
                ┌──────────────▼───────────────────────────┐
                │  Viewer                                   │
                │  - Result grid: 8 camera views per hit    │
                │  - Ego intent + trajectory overlay        │
                │  - Click → full segment scrubber          │
                └───────────────────────────────────────────┘
```

## Index schema (`frames.parquet`)

| column | type | source |
|---|---|---|
| `frame_name` | string | E2EDFrame.frame.context.name (field 1.1.1) |
| `segment_id` | string | split `frame_name` on last `-` |
| `frame_index` | int | suffix of `frame_name` |
| `intent` | enum | E2EDFrame.intent (field 7) |
| `past_pos_x` | float[16] | E2EDFrame.past_states (field 6.1) |
| `past_pos_y` | float[16] | field 6.2 |
| `past_vel_x` | float[16] | field 6.4 |
| `past_vel_y` | float[16] | field 6.5 |
| `past_speed_mean` | float | derived: mean(‖vel‖) |
| `past_yaw_change` | float | derived: heading delta over window |
| `shard_object` | string | GCS object name |
| `byte_offset` | int | record start in shard |
| `byte_length` | int | record payload length |
| `caption` | string | Stage 2 output |
| `tags` | string[] | Stage 2 output (structured) |
| `text_embedding` | float[768] | Stage 3 |
| `image_embedding` | float[512] | Stage 3, optional |

## Stage 2: tagging prompts

One pass per frame, given a 4×2 grid composite of the 8 views:

```
Describe this driving scene. Return JSON with:
- one_line_caption: a single sentence
- time_of_day: day | dusk | dawn | night | unknown
- weather: clear | rain | snow | fog | wet_road | unknown
- road_type: highway | urban_street | intersection | residential | parking_lot | other
- traffic_lights_visible: bool
- agents: list of {type: vehicle|pedestrian|cyclist|truck|bus, count: int}
- notable: list of free-text tokens for unusual things
```

Structured tags become indexable columns (`time_of_day='dusk' AND
agents.cyclist > 0`). Captions go to the text embedding.

## Cost model (one-time, 206K frames)

| stage | per-frame | total |
|---|---|---|
| Stage 1 scan (HTTP range, no captioning) | ~1 s, parallelized | ~10 min @ 32 workers |
| Stage 2 captioning with Claude Haiku | ~$0.003 | **~$600** |
| Stage 2 captioning with local BLIP-2 / LLaVA | GPU time only | $50–100 |
| Stage 3 text embedding | ~$0.0001 | ~$20 |
| Storage | — | <1 GB |

Captioning dominates. Open-model captioning is the right MVP path; upgrade to
Claude Haiku later if recall isn't good enough.

## Query path

```
user query
    │
    ▼
[router] ──► structured? (regex / LLM classifier)
    │           │
    │           ├─► "left turns at intersections with cyclists"
    │           │      → WHERE intent='GO_LEFT'
    │           │        AND tags @> '{intersection}'
    │           │        AND agents.cyclist > 0
    │           │
    │           └─► semantic-only? skip filters
    │
    ▼
[embed]  query → text_embedding
    │
    ▼
[search] vector_search(text_embedding, k=50) WHERE <filters>
    │
    ▼
[rerank]  optional: cross-encoder rerank top-50 → top-10
    │
    ▼
[hydrate] for each hit: fetch the 8 JPEGs via HTTP range
          using (shard_object, byte_offset, byte_length)
    │
    ▼
[viewer]  show 8-camera grid + ego trajectory + segment context
```

## UI

Two screens. Single-page app.

### Screen 1 — search + results grid

Top: one big search bar. Below: a grid of result cards. Each card is a
**4×2 mosaic** of the 8 cameras (FRONT_LEFT, FRONT, FRONT_RIGHT, SIDE_LEFT
on row 1; SIDE_RIGHT, REAR_RIGHT, REAR, REAR_LEFT on row 2 — same order
as `e2e_viewer/`), with a caption and chips underneath.

```text
┌────────────────────────────────────────────────────────────────────┐
│  🔍  left turn at intersection with cyclist                        │
└────────────────────────────────────────────────────────────────────┘

  ┌────────────────────────┐   ┌────────────────────────┐
  │ ┌──┬──┬──┬──┐          │   │ ┌──┬──┬──┬──┐          │
  │ │FL│FR│FR│SL│          │   │ │FL│FR│FR│SL│          │
  │ ├──┼──┼──┼──┤          │   │ ├──┼──┼──┼──┤          │
  │ │SR│RR│R │RL│          │   │ │SR│RR│R │RL│          │
  │ └──┴──┴──┴──┘          │   │ └──┴──┴──┴──┘          │
  ├────────────────────────┤   ├────────────────────────┤
  │ Left turn at busy      │   │ Left turn, dusk,       │
  │ intersection, dusk.    │   │ cyclist at crosswalk.  │
  │ ▸ GO_LEFT ▸ dusk       │   │ ▸ GO_LEFT ▸ cyclist    │
  │ segment e0ae38e2…      │   │ segment 1b0a4771…      │
  │ frame 142 / 287        │   │ frame  88 / 312        │
  └────────────────────────┘   └────────────────────────┘
        click anywhere → detail view
```

**Result grouping**: one card per *segment*, not per *frame*. When 20
consecutive frames from one drive all match the query, we pick the
best-matching frame as the thumbnail and show "frame 142 of 287" so the
user knows there's more to scrub. Reason: 20 near-identical cards is
noise; one card with "20 frames matched here" is signal.

### Screen 2 — frame detail (cockpit layout)

Clicking a card opens the frame in a **cockpit-layout** detail view —
the 8 cameras arranged in their spatial position around an imaginary
car, with a metadata sidebar on the right and a segment scrubber along
the bottom.

```text
┌──── ← back to results ──────────────────────────────────────────────┐
│                                                                      │
│  ┌────────┐ ┌──────────────┐ ┌────────┐  │ INTENT                   │
│  │ FRONT  │ │    FRONT     │ │ FRONT  │  │ GO_LEFT                  │
│  │ LEFT   │ │              │ │ RIGHT  │  │                          │
│  └────────┘ └──────────────┘ └────────┘  │ CAPTION                  │
│                                          │ Left turn at a busy      │
│  ┌────────┐                  ┌────────┐  │ intersection at dusk.    │
│  │ SIDE   │                  │ SIDE   │  │ A cyclist crosses        │
│  │ LEFT   │                  │ RIGHT  │  │ the crosswalk.           │
│  └────────┘                  └────────┘  │                          │
│                                          │ TAGS                     │
│  ┌────────┐ ┌──────────────┐ ┌────────┐  │ dusk · urban · cyclist   │
│  │ REAR   │ │    REAR      │ │ REAR   │  │ · intersection           │
│  │ LEFT   │ │              │ │ RIGHT  │  │                          │
│  └────────┘ └──────────────┘ └────────┘  │ EGO STATE                │
│                                          │ vel: 3.2 m/s             │
│                                          │ speed ▁▂▃▅▆▇▆▅▃▂         │
├──────────────────────────────────────────┴──────────────────────────┤
│  ◀  ━━━━━━━●━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━  ▶  frame 142/287 │
│            segment scrubber                                           │
└──────────────────────────────────────────────────────────────────────┘
```

The scrubber loads adjacent frames in the same segment so you can see
what happens immediately before and after the moment that matched.

Click on any individual camera in the cockpit grid → fullscreen that
camera (keyboard `esc` or click again to dismiss).

## MVP scope (the smallest thing worth shipping)

1. **Index one shard.** ~775 frames from
   `test_202504211836-202504220845.tfrecord-00004-of-00266` (already
   local). Validate the scan + parse + parquet pipeline end-to-end.
2. **Caption with a local model.** BLIP-2 or LLaVA-Next. ~775 calls.
   Tag every frame with structured fields too.
3. **Text-only embedding + cosine search.** No image embeddings yet.
4. **Minimal Flask UI** — exactly the two screens above:
   - search bar + 4×2-mosaic results grid (segment-deduped)
   - cockpit detail view with sidebar + scrubber
5. **Read JPEGs from the local shard via byte offsets** (no GCS in MVP).

Ship that against 775 frames. Then scale: index all 266 test shards,
swap local-shard reads for GCS range reads, add image embeddings, add
structured filter UI (intent dropdown, time-of-day toggles).

## Open questions

- **Is one segment always in one shard?** Need to verify on 3–5 known
  segments. If true, segment-level retrieval is free; if not, we need
  segment-spanning logic.
- **Composite vs per-camera captioning?** A 4×2 grid loses resolution but is
  4× cheaper. Front camera alone may capture 80% of the query-relevant signal.
- **Open-model caption quality at night / rain?** Likely worst case. May need
  Haiku fallback for low-confidence Stage 2 outputs.
- **Segment grouping on retrieval.** When 20 frames from the same segment
  all match, do we show 20 results or one segment with 20 hits? Probably the
  latter — dedupe to segment-level then expand on click.
- **Should we run Stage 1 in Google Cloud (same region as bucket)?** Egress
  savings + much faster range reads. Probably yes for the full 206K pass.

## Layout (proposed)

```
crossmodal_search/
  README.md            ← this file
  indexer/
    scan.py            ← Stage 1: walk shards, write frames.parquet
    caption.py         ← Stage 2: VLM tagging
    embed.py           ← Stage 3: text + image embeddings
  search/
    query.py           ← NL query → top-K
    hydrate.py         ← fetch JPEGs by (shard, offset, length)
  ui/
    app.py             ← Flask: search box + result grid
    templates/
  data/
    frames.parquet     ← committed? no — too big. write to .gitignore.
    vectors.lance/     ← LanceDB on-disk index
```

## Next step

Run or refresh the one-shard MVP:

```bash
python3 -m crossmodal_search.indexer.scan
python3 -m crossmodal_search.indexer.caption --model heuristic
python3 -m crossmodal_search.indexer.embed
python3 -m crossmodal_search.ui.app
```

Open `http://127.0.0.1:5003`.

The captioner has an optional BLIP path (`--model auto` or `--model blip`),
but this workspace currently falls back to metadata-only placeholders unless
the local `transformers` install is repaired and the model is already cached.
In metadata-only mode, only ego metadata queries such as `left turn`,
`right turn`, `straight`, `stopped`, `slow`, or `moving` are accepted. Visual
queries such as `cyclist`, `intersection`, `night`, `rain`, `traffic light`,
or arbitrary scene text intentionally return no results instead of fabricated
matches.
