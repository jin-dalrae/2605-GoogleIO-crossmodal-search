from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from crossmodal_search.indexer.embed import VECTOR_FEATURES, tags_to_text, vectorizer
from crossmodal_search.lib.e2e import CAMERA_DISPLAY_ORDER, CROSSMODAL_ROOT


DEFAULT_DATA_DIR = CROSSMODAL_ROOT / "data"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "search_index.json"


INTENT_QUERY_TERMS = {
    "GO_LEFT": ("left turn", "turn left", "turning left", "go left", "left"),
    "GO_RIGHT": ("right turn", "turn right", "turning right", "go right", "right"),
    "GO_STRAIGHT": ("straight", "go straight", "going straight", "forward"),
}

MOTION_QUERY_TERMS = {
    "stopped": ("stopped", "stop", "stationary", "parked"),
    "slow": ("slow", "creeping"),
    "moving": ("moving", "fast", "driving"),
}

UNSUPPORTED_VISUAL_TERMS = (
    "bike",
    "bicycle",
    "bus",
    "construction",
    "crosswalk",
    "cyclist",
    "dawn",
    "day",
    "dusk",
    "fog",
    "highway",
    "intersection",
    "lane",
    "motorcycle",
    "night",
    "parking",
    "pedestrian",
    "person",
    "rain",
    "residential",
    "roundabout",
    "snow",
    "stop sign",
    "traffic light",
    "truck",
    "urban",
    "vehicle",
    "wet road",
)

STOPWORDS = {
    "about",
    "and",
    "are",
    "at",
    "ego",
    "find",
    "for",
    "frame",
    "frames",
    "give",
    "in",
    "is",
    "me",
    "of",
    "on",
    "or",
    "scene",
    "scenes",
    "show",
    "the",
    "to",
    "turn",
    "turns",
    "where",
    "with",
}

SUPPORTED_METADATA_PHRASES = tuple(
    sorted(
        {
            term
            for terms in list(INTENT_QUERY_TERMS.values()) + list(MOTION_QUERY_TERMS.values())
            for term in terms
        },
        key=len,
        reverse=True,
    )
)


def _clean_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [str(item) for item in value.tolist()]
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        return [part.strip() for part in value.split() if part.strip()]
    return [str(value)]


def _has_term(text: str, term: str) -> bool:
    pattern = r"(?<![a-z0-9])" + re.escape(term).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
    return re.search(pattern, text) is not None


def unsupported_visual_terms(query: str) -> list[str]:
    lowered = query.lower()
    return [term for term in UNSUPPORTED_VISUAL_TERMS if _has_term(lowered, term)]


def metadata_only_unsupported_terms(query: str) -> list[str]:
    lowered = query.lower()
    stripped = lowered
    for phrase in SUPPORTED_METADATA_PHRASES:
        pattern = r"(?<![a-z0-9])" + re.escape(phrase).replace(r"\ ", r"\s+") + r"(?![a-z0-9])"
        stripped = re.sub(pattern, " ", stripped)

    leftovers = []
    for token in re.findall(r"[a-z0-9_]+", stripped):
        if len(token) < 3 or token in STOPWORDS:
            continue
        leftovers.append(token)

    explicit = unsupported_visual_terms(query)
    combined = explicit + leftovers
    return list(dict.fromkeys(combined))


def query_vector(text: str, n_features: int = VECTOR_FEATURES) -> np.ndarray:
    vec = vectorizer().transform([text])
    return vec.astype(np.float32).toarray()[0]


def infer_filters(query: str) -> dict[str, str]:
    lowered = query.lower()
    filters: dict[str, str] = {}
    for intent_name, terms in INTENT_QUERY_TERMS.items():
        if any(term in lowered for term in terms):
            filters["intent_name"] = intent_name
            break
    for motion_state, terms in MOTION_QUERY_TERMS.items():
        if any(_has_term(lowered, term) for term in terms):
            filters["motion_state"] = motion_state
            break
    return filters


def filter_bonus(row: pd.Series, filters: dict[str, str]) -> float:
    bonus = 0.0
    for column, expected in filters.items():
        value = str(row.get(column, "") or "")
        if value == expected:
            bonus += 0.18
        elif expected in value:
            bonus += 0.08
    return bonus


def lexical_bonus(row: pd.Series, query: str) -> float:
    lowered = query.lower()
    if not lowered:
        return 0.0
    tokens = {token for token in lowered.replace("_", " ").split() if len(token) >= 3}
    if not tokens:
        return 0.0
    haystack = " ".join(
        [
            str(row.get("search_text", "")),
            tags_to_text(row.get("tags")),
            str(row.get("caption", "")),
        ]
    ).lower()
    hits = sum(1 for token in tokens if token in haystack)
    return min(0.12, hits * 0.025)


@dataclass
class CrossmodalIndex:
    frames: pd.DataFrame
    vectors: np.ndarray
    manifest: dict[str, Any]

    @classmethod
    def load(cls, manifest_path: Path = DEFAULT_MANIFEST) -> "CrossmodalIndex":
        manifest = json.loads(manifest_path.read_text())
        frames_path = Path(manifest["frames_path"])
        vectors_path = Path(manifest["vectors_path"])
        if not frames_path.is_absolute():
            frames_path = manifest_path.parent / frames_path
        if not vectors_path.is_absolute():
            vectors_path = manifest_path.parent / vectors_path
        frames = pd.read_parquet(frames_path)
        vectors = np.load(vectors_path)
        if len(frames) != len(vectors):
            raise ValueError(f"Index mismatch: {len(frames)} rows, {len(vectors)} vectors")
        return cls(frames=frames, vectors=vectors, manifest=manifest)

    def visual_search_available(self) -> bool:
        if "caption_status" not in self.frames.columns:
            return False
        return bool((self.frames["caption_status"] == "vlm").any())

    def blocking_query_terms(self, query: str) -> list[str]:
        if self.visual_search_available():
            return []
        return metadata_only_unsupported_terms(query)

    def frame_row(self, frame_name: str) -> pd.Series:
        matches = self.frames[self.frames["frame_name"] == frame_name]
        if matches.empty:
            raise KeyError(frame_name)
        return matches.iloc[0]

    def segment_frames(self, segment_id: str) -> pd.DataFrame:
        frames = self.frames[self.frames["segment_id"] == segment_id].copy()
        if frames.empty:
            return frames
        return frames.sort_values(
            by=["segment_frame_position", "frame_index", "record_index"],
            ascending=[True, True, True],
            na_position="last",
        )

    def search(self, query: str, *, k: int = 12, candidate_k: int = 80) -> list[dict[str, Any]]:
        if self.frames.empty:
            return []
        if self.blocking_query_terms(query):
            return []

        qvec = query_vector(query, self.manifest.get("vectorizer", {}).get("n_features", VECTOR_FEATURES))
        scores = np.einsum("ij,j->i", self.vectors, qvec)
        scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)
        if not query.strip():
            scores = np.linspace(1.0, 0.0, len(self.frames), dtype=np.float32)

        filters = infer_filters(query)
        candidate_count = min(max(candidate_k, k * 4), len(self.frames))
        candidate_indexes = np.argpartition(-scores, candidate_count - 1)[:candidate_count]

        ranked_rows = []
        for index in candidate_indexes:
            row = self.frames.iloc[int(index)]
            final_score = float(scores[index]) + filter_bonus(row, filters) + lexical_bonus(row, query)
            ranked_rows.append((final_score, int(index), row))
        ranked_rows.sort(key=lambda item: item[0], reverse=True)

        per_segment_hits: dict[str, int] = {}
        for _, _, row in ranked_rows:
            segment_id = str(row.get("segment_id") or row.get("frame_name"))
            per_segment_hits[segment_id] = per_segment_hits.get(segment_id, 0) + 1

        results = []
        seen_segments = set()
        for score, index, row in ranked_rows:
            segment_id = str(row.get("segment_id") or row.get("frame_name"))
            if segment_id in seen_segments:
                continue
            seen_segments.add(segment_id)
            results.append(self._result(row, score, per_segment_hits.get(segment_id, 1), rank_index=index))
            if len(results) >= k:
                break
        return results

    def _result(self, row: pd.Series, score: float, matches_in_segment: int, *, rank_index: int) -> dict[str, Any]:
        frame_position = _clean_scalar(row.get("segment_frame_position"))
        frame_count = _clean_scalar(row.get("segment_frame_count"))
        if frame_position is not None:
            frame_display = int(frame_position) + 1
        else:
            frame_display = _clean_scalar(row.get("frame_index"))
        return {
            "rank_index": rank_index,
            "score": score,
            "frame_name": _clean_scalar(row.get("frame_name")),
            "record_index": _clean_scalar(row.get("record_index")),
            "segment_id": _clean_scalar(row.get("segment_id")),
            "frame_index": _clean_scalar(row.get("frame_index")),
            "segment_frame_position": frame_position,
            "segment_frame_count": frame_count,
            "frame_display": frame_display,
            "caption": _clean_scalar(row.get("caption")),
            "intent_name": _clean_scalar(row.get("intent_name")),
            "time_of_day": _clean_scalar(row.get("time_of_day")),
            "weather": _clean_scalar(row.get("weather")),
            "road_type": _clean_scalar(row.get("road_type")),
            "motion_state": _clean_scalar(row.get("motion_state")),
            "caption_status": _clean_scalar(row.get("caption_status")),
            "tags": _tags(row.get("tags")),
            "matches_in_segment": int(matches_in_segment),
            "camera_order": CAMERA_DISPLAY_ORDER,
        }

    def frame_detail(self, frame_name: str) -> dict[str, Any]:
        row = self.frame_row(frame_name)
        segment = self.segment_frames(str(row["segment_id"]))
        frames = []
        for _, frame in segment.iterrows():
            frames.append(
                {
                    "frame_name": _clean_scalar(frame.get("frame_name")),
                    "frame_index": _clean_scalar(frame.get("frame_index")),
                    "segment_frame_position": _clean_scalar(frame.get("segment_frame_position")),
                    "intent_name": _clean_scalar(frame.get("intent_name")),
                    "caption": _clean_scalar(frame.get("caption")),
                }
            )

        return {
            **self._result(row, 1.0, 1, rank_index=int(row.name)),
            "byte_offset": _clean_scalar(row.get("byte_offset")),
            "byte_length": _clean_scalar(row.get("byte_length")),
            "shard_path": _clean_scalar(row.get("shard_path")),
            "shard_object": _clean_scalar(row.get("shard_object")),
            "past_speed_now": _clean_scalar(row.get("past_speed_now")),
            "past_speed_mean": _clean_scalar(row.get("past_speed_mean")),
            "past_yaw_change": _clean_scalar(row.get("past_yaw_change")),
            "past_displacement_m": _clean_scalar(row.get("past_displacement_m")),
            "image_brightness": _clean_scalar(row.get("image_brightness")),
            "segment_frames": frames,
            "camera_order": CAMERA_DISPLAY_ORDER,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query the local crossmodal search index.")
    parser.add_argument("query", nargs="?", default="", help="Natural-language query.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("-k", type=int, default=8)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = CrossmodalIndex.load(args.manifest)
    blocked = index.blocking_query_terms(args.query)
    if blocked:
        terms = ", ".join(blocked)
        print(f"visual search unavailable for: {terms}")
        return 0
    for result in index.search(args.query, k=args.k):
        print(
            f"{result['score']:.3f} {result['frame_name']} "
            f"{result['intent_name']} frame {result['frame_display']} of {result['segment_frame_count']} "
            f"- {result['caption']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
