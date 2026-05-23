from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from crossmodal_search.lib.e2e import load_record_by_offset, resolve_shard_path


@lru_cache(maxsize=96)
def camera_jpegs_for_record(shard_path: str, byte_offset: int, byte_length: int) -> dict[str, bytes]:
    record = load_record_by_offset(
        resolve_shard_path(Path(shard_path)),
        int(byte_offset),
        int(byte_length),
        include_images=True,
    )
    return record["image_payloads"]

