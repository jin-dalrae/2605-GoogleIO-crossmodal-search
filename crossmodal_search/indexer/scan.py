from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from crossmodal_search.lib.e2e import (
    CAMERA_DISPLAY_ORDER,
    CROSSMODAL_ROOT,
    DEFAULT_SHARD_PATH,
    index_tfrecord,
    parse_e2e_payload,
    read_record,
    resolve_shard_path,
)


DEFAULT_OUT = CROSSMODAL_ROOT / "data" / "frames.parquet"


def _add_segment_positions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        df["segment_frame_position"] = []
        df["segment_frame_count"] = []
        return df

    df = df.copy()
    segment_keys = df["segment_id"].fillna(df["frame_name"]).fillna(df["record_index"].astype(str))
    df["_segment_key"] = segment_keys
    positions = pd.Series(index=df.index, dtype="Int64")
    counts = pd.Series(index=df.index, dtype="Int64")

    for _, group in df.groupby("_segment_key", sort=False):
        ordered = group.sort_values(
            by=["frame_index", "record_index"],
            ascending=[True, True],
            na_position="last",
        )
        for position, index in enumerate(ordered.index):
            positions.at[index] = position
            counts.at[index] = len(ordered)

    df["segment_frame_position"] = positions.astype("int64")
    df["segment_frame_count"] = counts.astype("int64")
    return df.drop(columns=["_segment_key"])


def scan_shard(shard_path: Path, *, limit: int | None = None, progress_every: int = 100) -> pd.DataFrame:
    offsets = index_tfrecord(shard_path, limit=limit)
    rows = []
    for record_index, (byte_offset, byte_length) in enumerate(offsets):
        payload = read_record(shard_path, byte_offset, byte_length)
        parsed = parse_e2e_payload(payload, include_images=False)
        past_states = parsed.pop("past_states")
        row = {
            "record_index": record_index,
            "frame_name": parsed["frame_name"],
            "segment_id": parsed["segment_id"],
            "frame_index": parsed["frame_index"],
            "intent": parsed["intent"],
            "intent_name": parsed["intent_name"],
            "past_states_steps": parsed["past_states_steps"],
            "past_speed_mean": parsed["past_speed_mean"],
            "past_speed_now": parsed["past_speed_now"],
            "past_yaw_change": parsed["past_yaw_change"],
            "past_displacement_m": parsed["past_displacement_m"],
            "future_states_present": parsed["future_states_present"],
            "preference_trajectory_count": parsed["preference_trajectory_count"],
            "camera_order": CAMERA_DISPLAY_ORDER,
            "shard_path": str(shard_path.resolve()),
            "shard_object": shard_path.name,
            "byte_offset": int(byte_offset),
            "byte_length": int(byte_length),
        }
        row.update(past_states)
        rows.append(row)

        if progress_every and (record_index + 1) % progress_every == 0:
            print(f"scanned {record_index + 1}/{len(offsets)} records", file=sys.stderr)

    df = pd.DataFrame(rows)
    df = _add_segment_positions(df)
    return df


def write_frames(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan one local Waymo E2E shard into frames.parquet.")
    parser.add_argument("--shard", type=Path, default=DEFAULT_SHARD_PATH, help="Local E2E TFRecord shard path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output parquet path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max records for quick tests.")
    parser.add_argument("--progress-every", type=int, default=100, help="Progress interval; 0 disables progress.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    shard_path = resolve_shard_path(args.shard)
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing shard: {shard_path}")

    df = scan_shard(shard_path, limit=args.limit, progress_every=args.progress_every)
    write_frames(df, args.out)
    segments = df["segment_id"].nunique(dropna=True) if not df.empty else 0
    print(f"wrote {len(df)} frames across {segments} segments to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
