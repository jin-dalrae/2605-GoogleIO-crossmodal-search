# Product Requirements Document: Crossmodal Search

Date: 2026-05-25

## Summary

Crossmodal Search is a local-first search product for Waymo End-to-End Camera
TFRecord shards. A user types a natural-language driving-scene query and gets
grounded frame results with all 8 synchronized camera views, ego intent, frame
position, and a cockpit detail page.

The product goal is not to stage a demo. The product goal is to build a real
visual index where search results are traceable to actual frame captions and
local shard bytes.

## Problem

Waymo E2E camera data is stored as TFRecord shards. Researchers and builders
who want examples like "intersection with construction" or "left turn with a
cyclist" must manually scan records, decode images, and inspect frames. There
is no small local index that maps natural-language scene descriptions back to
specific camera frames.

## Target Users

- autonomy researchers inspecting driving-scene examples
- ML engineers building retrieval/evaluation sets
- demo builders who need a product that only shows real indexed evidence
- internal developers validating captioning and search behavior on local shards

## Goals

- Index one local Waymo E2E camera shard end to end.
- Caption each frame using the 8-camera mosaic, not just structural metadata.
- Store captions, structured tags, ego intent, frame offsets, and segment
  position in a searchable local index.
- Return visual search results only when real VLM evidence exists.
- Provide a search results screen and cockpit detail screen.
- Make provider failures visible and resumable.

## Non-Goals

- No fabricated visual labels.
- No staged or hand-authored search results.
- No demo video until the active index has enough real visual coverage.
- No GCS round trips in the MVP.
- No Motion/E2E dataset join in the MVP.
- No claim of dataset-wide visual search until all required shards are indexed.

## Current Implementation Status

The MVP code path exists:

- `crossmodal_search/indexer/scan.py` scans one local shard.
- `crossmodal_search/indexer/caption.py` hydrates images, builds mosaics, and
  calls a managed/local captioner.
- `crossmodal_search/indexer/embed.py` builds a local text-vector index.
- `crossmodal_search/search/query.py` routes and searches queries.
- `crossmodal_search/ui/app.py` serves the Flask UI.

Current local index status in this workspace:

- 20 sampled rows are embedded.
- 1 row has `caption_status == "vlm"`.
- 19 rows have provider captioning failures.
- The one VLM row is a real `GO_RIGHT` frame with intersection/construction
  evidence.

This is enough to prove the full vertical path, but not enough to market broad
visual search yet.

## User Stories

1. As a researcher, I can search for an indexed visual concept such as
   `intersection` and inspect the exact 8-camera frame that matched.
2. As a researcher, I can search for `crash` and get no result when the index
   has no real crash evidence.
3. As an evaluator, I can search for `left turn` and never receive a `GO_RIGHT`
   frame.
4. As a developer, I can rerun captioning for failed rows without losing
   already successful VLM captions.
5. As a demo builder, I can see the number of visual frames indexed before
   deciding whether the product is ready to record.

## Functional Requirements

### Local Shard Scan

- Read a local Waymo E2E TFRecord shard.
- Validate TFRecord lengths and CRCs.
- Emit `frames.parquet`.
- Required fields:
  - `frame_name`
  - `segment_id`
  - `frame_index`
  - `record_index`
  - `intent_name`
  - `past_speed_now`
  - `past_speed_mean`
  - `past_yaw_change`
  - `segment_frame_position`
  - `segment_frame_count`
  - `shard_path`
  - `shard_object`
  - `byte_offset`
  - `byte_length`

### Visual Captioning

- Hydrate the 8 JPEG camera views from local shard offsets.
- Compose a 4x2 mosaic in the same camera order used by the UI.
- Call `VisionCaptionAgent` when `--model managed` is selected.
- Emit:
  - `caption`
  - `tags`
  - `time_of_day`
  - `weather`
  - `road_type`
  - `traffic_lights_visible`
  - `agents_json`
  - `notable`
  - `caption_model`
  - `caption_provider`
  - `caption_status`
- Preserve successful `vlm` rows when resuming.
- Retry failed rows only when `--retry-failed` is passed.

### Managed Agents

- Load `.env` from the repo root or `crossmodal_search/.env`.
- Accept canonical and lowercase key names.
- Support Gemini and OpenAI providers.
- Use `QueryRouterAgent` to separate intent/motion filters from visual
  requirements.
- Use `VisionCaptionAgent` to produce grounded visual captions.
- Fall back to deterministic routing when a query agent is unavailable.
- Do not silently turn provider failures into visual facts.

### Embedding And Index

- Build `search_text` from captions, tags, structured fields, intent, and IDs.
- Write `frames_embedded.parquet`, `text_vectors.npy`, and `search_index.json`.
- Use deterministic local vectors for the current MVP.
- Keep generated data out of git.

### Search

- For visual queries, search only rows with `caption_status == "vlm"`.
- Require visual query terms to appear in the caption/tags/structured haystack.
- Apply intent and motion as hard filters.
- Deduplicate result cards by segment.
- Return an empty result set instead of an unrelated result.

### UI

- `/` shows a search bar, status line, and result cards.
- Result cards show a 4x2 camera mosaic, caption, chips, frame position, and
  segment count.
- `/frame/<frame_name>` shows a cockpit detail page with camera panes, metadata,
  and a segment scrubber.
- The status line shows the actual count of visual frames indexed.

## Acceptance Criteria

- `intersection` can return a frame only when a real VLM caption/tag contains
  intersection evidence.
- `crash` returns no results when no VLM caption/tag contains crash evidence.
- `left turn` does not return `GO_RIGHT` frames.
- Search results never use placeholder captions as visual evidence.
- Provider failures are visible in `caption_status`.
- The UI and `/api/status` expose the visual indexed frame count.
- The product can be rebuilt locally from a shard and `.env` without committed
  generated data.

## Reliability Requirements

- Captioning must be resumable.
- One failed provider call must not discard successful prior captions.
- Search should prefer no result over a misleading result.
- The UI must remain usable when the index is missing or partially populated.
- Secrets must stay in `.env` and out of git.

## Risks And Blockers

- Gemini key/provider reliability is currently blocking full visual coverage.
- OpenAI account rate limits can block hosted routing/captioning.
- Hosted VLMs may misread camera mosaics; evaluation queries are needed.
- Full-shard captioning cost and latency are still unknown for this environment.
- The current local vector backend is adequate for a small MVP but not the
  final retrieval quality target.

## Milestones

### M0: Local Vertical Slice

Status: implemented.

- scan one local shard
- caption a limited batch
- embed rows
- serve search and detail UI
- prevent unsupported visual searches from returning unrelated rows

### M1: Real One-Shard Visual Index

Status: next.

- fix managed provider reliability
- caption all frames in one shard with `caption_status == "vlm"`
- run acceptance queries
- record a demo only from the active index

### M2: Quality And Scale

Status: planned.

- add evaluation fixtures
- improve text/image embeddings
- add batch progress and failure reporting
- optionally add LanceDB or another persistent vector store

### M3: Dataset Expansion

Status: deferred.

- GCS range reads
- all E2E test shards
- Motion/E2E joins
- richer scenario-level search
