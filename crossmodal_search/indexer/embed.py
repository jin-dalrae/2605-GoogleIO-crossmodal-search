from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize

from crossmodal_search.lib.e2e import CROSSMODAL_ROOT


DEFAULT_IN = CROSSMODAL_ROOT / "data" / "frames_tagged.parquet"
DEFAULT_OUT = CROSSMODAL_ROOT / "data" / "frames_embedded.parquet"
DEFAULT_VECTOR_PATH = CROSSMODAL_ROOT / "data" / "text_vectors.npy"
DEFAULT_MANIFEST = CROSSMODAL_ROOT / "data" / "search_index.json"
DEFAULT_LANCEDB_DIR = CROSSMODAL_ROOT / "data" / "vectors.lance"
VECTOR_FEATURES = 2048


def tags_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, np.ndarray):
        return " ".join(str(item) for item in value.tolist())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def build_search_text(row: pd.Series) -> str:
    parts = [
        row.get("caption", ""),
        row.get("intent_name", ""),
        row.get("time_of_day", ""),
        row.get("weather", ""),
        row.get("road_type", ""),
        row.get("motion_state", ""),
        tags_to_text(row.get("tags")),
        row.get("frame_name", ""),
        row.get("segment_id", ""),
    ]
    return " ".join(str(part) for part in parts if part is not None)


def vectorizer() -> HashingVectorizer:
    return HashingVectorizer(
        n_features=VECTOR_FEATURES,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        lowercase=True,
    )


def embed_texts(texts: list[str]) -> np.ndarray:
    matrix = vectorizer().transform(texts)
    matrix = normalize(matrix, norm="l2", copy=False)
    return matrix.astype(np.float32).toarray()


def maybe_write_lancedb(df: pd.DataFrame, vectors: np.ndarray, db_dir: Path) -> bool:
    try:
        import lancedb
    except Exception:
        return False

    db_dir.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(db_dir))
    records = []
    for index, row in df.iterrows():
        records.append(
            {
                "vector": vectors[index].tolist(),
                "frame_name": row["frame_name"],
                "segment_id": row["segment_id"],
                "caption": row.get("caption", ""),
                "intent_name": row.get("intent_name", ""),
                "tags": tags_to_text(row.get("tags")),
            }
        )
    db.create_table("frames", records, mode="overwrite")
    return True


def build_index(
    tagged_path: Path,
    out_path: Path,
    vector_path: Path,
    manifest_path: Path,
    lancedb_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = pd.read_parquet(tagged_path).copy()
    df["search_text"] = df.apply(build_search_text, axis=1)
    vectors = embed_texts(df["search_text"].fillna("").tolist())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    np.save(vector_path, vectors)
    lancedb_written = maybe_write_lancedb(df, vectors, lancedb_dir)

    manifest = {
        "version": 1,
        "backend": "hashing_vectorizer",
        "vectorizer": {
            "type": "sklearn.HashingVectorizer",
            "n_features": VECTOR_FEATURES,
            "alternate_sign": False,
            "ngram_range": [1, 2],
            "norm": "l2",
        },
        "frames_path": str(out_path),
        "vectors_path": str(vector_path),
        "source_path": str(tagged_path),
        "row_count": int(len(df)),
        "lancedb_path": str(lancedb_dir) if lancedb_written else None,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return df, manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Embed tagged captions and build a local text vector index.")
    parser.add_argument("--tagged", type=Path, default=DEFAULT_IN, help="Input frames_tagged.parquet path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output frames_embedded.parquet path.")
    parser.add_argument("--vectors", type=Path, default=DEFAULT_VECTOR_PATH, help="Output NumPy vector matrix path.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Output search index manifest path.")
    parser.add_argument("--lancedb-dir", type=Path, default=DEFAULT_LANCEDB_DIR, help="Optional LanceDB output directory.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    df, manifest = build_index(args.tagged, args.out, args.vectors, args.manifest, args.lancedb_dir)
    print(f"wrote {len(df)} embedded frames to {args.out}")
    print(f"wrote {manifest['backend']} vectors to {args.vectors}")
    if manifest["lancedb_path"]:
        print(f"wrote LanceDB table to {manifest['lancedb_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
