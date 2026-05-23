from __future__ import annotations

import io
import math
import struct
from pathlib import Path
from typing import Any

import google_crc32c
from google.protobuf.internal.decoder import _DecodeVarint


CROSSMODAL_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = CROSSMODAL_ROOT.parent
DEFAULT_SHARD_NAME = "test_202504211836-202504220845.tfrecord-00004-of-00266"
DEFAULT_SHARD_PATH = PROJECT_ROOT / DEFAULT_SHARD_NAME

WAYMO_EGO_INTENTS = {
    0: "UNKNOWN",
    1: "GO_STRAIGHT",
    2: "GO_LEFT",
    3: "GO_RIGHT",
}

WAYMO_CAMERA_NAMES = {
    0: "UNKNOWN",
    1: "FRONT",
    2: "FRONT_LEFT",
    3: "FRONT_RIGHT",
    4: "SIDE_LEFT",
    5: "SIDE_RIGHT",
    6: "REAR_LEFT",
    7: "REAR",
    8: "REAR_RIGHT",
}

CAMERA_DISPLAY_ORDER = [
    "FRONT_LEFT",
    "FRONT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "REAR_RIGHT",
    "REAR",
    "REAR_LEFT",
]

CAMERA_COCKPIT_ORDER = [
    "FRONT_LEFT",
    "FRONT",
    "FRONT_RIGHT",
    "SIDE_LEFT",
    "SIDE_RIGHT",
    "REAR_LEFT",
    "REAR",
    "REAR_RIGHT",
]

PAST_STATE_FIELDS = {
    1: "past_pos_x",
    2: "past_pos_y",
    3: "past_pos_z",
    4: "past_vel_x",
    5: "past_vel_y",
    6: "past_accel_x",
    7: "past_accel_y",
}


def masked_crc32c(data: bytes) -> int:
    value = google_crc32c.value(data)
    return (((value >> 15) | ((value << 17) & 0xFFFFFFFF)) + 0xA282EAD8) & 0xFFFFFFFF


def index_tfrecord(path: Path, *, validate_crc: bool = True, limit: int | None = None) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    with path.open("rb") as handle:
        while True:
            offset = handle.tell()
            length_bytes = handle.read(8)
            if not length_bytes:
                break
            if len(length_bytes) != 8:
                raise ValueError(f"{path}: truncated record length at byte {offset}")

            length_crc_bytes = handle.read(4)
            if len(length_crc_bytes) != 4:
                raise ValueError(f"{path}: truncated length CRC at byte {offset}")
            length_crc = struct.unpack("<I", length_crc_bytes)[0]
            if validate_crc and masked_crc32c(length_bytes) != length_crc:
                raise ValueError(f"{path}: invalid length CRC at byte {offset}")

            length = struct.unpack("<Q", length_bytes)[0]
            handle.seek(length + 4, io.SEEK_CUR)
            offsets.append((offset, length))
            if limit is not None and len(offsets) >= limit:
                break
    return offsets


def read_record(path: Path, offset: int, length: int | None = None, *, validate_crc: bool = True) -> bytes:
    with path.open("rb") as handle:
        handle.seek(offset)
        length_bytes = handle.read(8)
        if len(length_bytes) != 8:
            raise ValueError(f"{path}: truncated record length at byte {offset}")
        length_crc_bytes = handle.read(4)
        if len(length_crc_bytes) != 4:
            raise ValueError(f"{path}: truncated length CRC at byte {offset}")
        length_crc = struct.unpack("<I", length_crc_bytes)[0]
        if validate_crc and masked_crc32c(length_bytes) != length_crc:
            raise ValueError(f"{path}: invalid length CRC at byte {offset}")

        actual_length = struct.unpack("<Q", length_bytes)[0]
        if length is not None and actual_length != int(length):
            raise ValueError(f"{path}: expected length {length}, found {actual_length} at byte {offset}")

        payload = handle.read(actual_length)
        if len(payload) != actual_length:
            raise ValueError(f"{path}: truncated payload at byte {offset}")
        data_crc_bytes = handle.read(4)
        if len(data_crc_bytes) != 4:
            raise ValueError(f"{path}: truncated data CRC at byte {offset}")
        data_crc = struct.unpack("<I", data_crc_bytes)[0]
        if validate_crc and masked_crc32c(payload) != data_crc:
            raise ValueError(f"{path}: invalid data CRC at byte {offset}")
        return payload


def proto_fields(buf: bytes, max_fields: int = 200) -> list[dict[str, Any]]:
    pos = 0
    fields: list[dict[str, Any]] = []
    while pos < len(buf) and len(fields) < max_fields:
        key, next_pos = _DecodeVarint(buf, pos)
        field_number = key >> 3
        wire = key & 7
        pos = next_pos
        item: dict[str, Any] = {"field": field_number, "wire": wire}
        if wire == 0:
            value, pos = _DecodeVarint(buf, pos)
            item["value"] = value
        elif wire == 1:
            item["value"] = buf[pos:pos + 8]
            pos += 8
        elif wire == 2:
            length, pos = _DecodeVarint(buf, pos)
            item["value"] = buf[pos:pos + length]
            pos += length
        elif wire == 5:
            item["value"] = buf[pos:pos + 4]
            pos += 4
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire} at byte {pos}")
        fields.append(item)
    return fields


def first_field(fields: list[dict[str, Any]], field_number: int, wire: int | None = None) -> dict[str, Any] | None:
    for field in fields:
        if field["field"] != field_number:
            continue
        if wire is not None and field["wire"] != wire:
            continue
        return field
    return None


def repeated_fields(fields: list[dict[str, Any]], field_number: int, wire: int | None = None) -> list[dict[str, Any]]:
    return [
        field
        for field in fields
        if field["field"] == field_number and (wire is None or field["wire"] == wire)
    ]


def decode_packed_floats(data: bytes) -> list[float]:
    if len(data) % 4 != 0:
        raise ValueError(f"Packed float payload has invalid length {len(data)}")
    return [struct.unpack("<f", data[index:index + 4])[0] for index in range(0, len(data), 4)]


def unwrap_jpeg(blob: bytes) -> bytes:
    start = blob.find(b"\xff\xd8")
    if start == -1:
        raise ValueError("JPEG start marker not found")
    end = blob.rfind(b"\xff\xd9")
    if end == -1 or end < start:
        raise ValueError("JPEG end marker not found")
    return blob[start:end + 2]


def split_frame_name(frame_name: str | None) -> tuple[str | None, int | None]:
    if not frame_name:
        return None, None
    prefix, sep, suffix = frame_name.rpartition("-")
    if not sep:
        return frame_name, None
    try:
        return prefix, int(suffix)
    except ValueError:
        return prefix, None


def parse_frame_name(frame_bytes: bytes) -> str | None:
    frame_fields = proto_fields(frame_bytes, max_fields=200)
    context = first_field(frame_fields, 1, 2)
    if context is None:
        return None
    context_fields = proto_fields(context["value"], max_fields=50)
    name_field = first_field(context_fields, 1, 2)
    if name_field is None:
        return None
    return name_field["value"].decode("utf-8", "replace")


def parse_ego_trajectory_states(traj_bytes: bytes) -> dict[str, list[float]]:
    fields = proto_fields(traj_bytes, max_fields=20)
    parsed: dict[str, list[float]] = {}
    for field in fields:
        if field["field"] in PAST_STATE_FIELDS and field["wire"] == 2:
            parsed[PAST_STATE_FIELDS[field["field"]]] = decode_packed_floats(field["value"])
    return parsed


def parse_camera_image(image_bytes: bytes) -> dict[str, Any]:
    fields = proto_fields(image_bytes, max_fields=30)
    camera_enum = None
    jpeg_payload = None
    for field in fields:
        if field["field"] in (1, 4) and field["wire"] == 0 and camera_enum is None:
            camera_enum = field["value"]
        elif field["field"] == 2 and field["wire"] == 2:
            jpeg_payload = field["value"]

    camera_name = WAYMO_CAMERA_NAMES.get(camera_enum, f"UNKNOWN_{camera_enum}")
    return {
        "camera_enum": camera_enum,
        "camera_name": camera_name,
        "jpeg_payload": unwrap_jpeg(jpeg_payload) if jpeg_payload else None,
    }


def _series_or_empty(states: dict[str, list[float]], key: str) -> list[float]:
    values = states.get(key)
    return values if values is not None else []


def summarize_past_states(past_states: dict[str, list[float]]) -> dict[str, float | int | None]:
    pos_x = _series_or_empty(past_states, "past_pos_x")
    pos_y = _series_or_empty(past_states, "past_pos_y")
    vel_x = _series_or_empty(past_states, "past_vel_x")
    vel_y = _series_or_empty(past_states, "past_vel_y")
    speeds = [
        math.hypot(vx, vy)
        for vx, vy in zip(vel_x, vel_y)
    ]

    yaw_change = None
    if len(vel_x) >= 2 and len(vel_y) >= 2:
        first_yaw = math.atan2(vel_y[0], vel_x[0])
        last_yaw = math.atan2(vel_y[-1], vel_x[-1])
        yaw_change = math.atan2(math.sin(last_yaw - first_yaw), math.cos(last_yaw - first_yaw))

    displacement = None
    if len(pos_x) >= 2 and len(pos_y) >= 2:
        displacement = math.hypot(pos_x[-1] - pos_x[0], pos_y[-1] - pos_y[0])

    return {
        "past_states_steps": max((len(values) for values in past_states.values()), default=0),
        "past_speed_mean": sum(speeds) / len(speeds) if speeds else None,
        "past_speed_now": speeds[-1] if speeds else None,
        "past_yaw_change": yaw_change,
        "past_displacement_m": displacement,
    }


def parse_e2e_payload(payload: bytes, *, include_images: bool = False) -> dict[str, Any]:
    fields = proto_fields(payload, max_fields=50)
    frame_field = first_field(fields, 1, 2)
    if frame_field is None:
        raise ValueError("Missing E2EDFrame.frame field")

    frame_bytes = frame_field["value"]
    frame_name = parse_frame_name(frame_bytes)
    segment_id, frame_index = split_frame_name(frame_name)
    intent_field = first_field(fields, 7, 0)
    past_states_field = first_field(fields, 6, 2)
    future_states_field = first_field(fields, 5, 2)
    preference_trajectory_fields = repeated_fields(fields, 8, 2)

    past_states = parse_ego_trajectory_states(past_states_field["value"]) if past_states_field else {}
    summary = summarize_past_states(past_states)

    parsed: dict[str, Any] = {
        "frame_name": frame_name,
        "segment_id": segment_id,
        "frame_index": frame_index,
        "intent": intent_field["value"] if intent_field else None,
        "intent_name": (
            WAYMO_EGO_INTENTS.get(intent_field["value"], f"UNKNOWN_{intent_field['value']}")
            if intent_field else "UNKNOWN"
        ),
        "past_states": past_states,
        "future_states_present": future_states_field is not None,
        "preference_trajectory_count": len(preference_trajectory_fields),
        **summary,
    }

    if include_images:
        frame_fields = proto_fields(frame_bytes, max_fields=500)
        image_fields = repeated_fields(frame_fields, 4, 2)
        image_payloads = {}
        image_meta = []
        for image_field in image_fields:
            image = parse_camera_image(image_field["value"])
            if image["jpeg_payload"] is not None:
                image_payloads[image["camera_name"]] = image["jpeg_payload"]
            image_meta.append(
                {
                    "camera_name": image["camera_name"],
                    "camera_enum": image["camera_enum"],
                    "jpeg_size_bytes": len(image["jpeg_payload"]) if image["jpeg_payload"] else 0,
                }
            )
        parsed["image_payloads"] = image_payloads
        parsed["images"] = image_meta
        parsed["image_count"] = len(image_payloads)

    return parsed


def resolve_shard_path(shard_path: str | Path | None = None) -> Path:
    if shard_path is None:
        return DEFAULT_SHARD_PATH
    path = Path(shard_path)
    if path.exists():
        return path
    candidate = PROJECT_ROOT / path
    if candidate.exists():
        return candidate
    return path


def load_record_by_offset(shard_path: Path, byte_offset: int, byte_length: int, *, include_images: bool = False) -> dict[str, Any]:
    payload = read_record(shard_path, int(byte_offset), int(byte_length))
    return parse_e2e_payload(payload, include_images=include_images)

