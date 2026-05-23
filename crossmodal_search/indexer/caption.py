from __future__ import annotations

import argparse
import json
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image, ImageDraw, ImageOps, ImageStat

from crossmodal_search.lib.e2e import (
    CAMERA_DISPLAY_ORDER,
    CROSSMODAL_ROOT,
    DEFAULT_SHARD_PATH,
    load_record_by_offset,
    resolve_shard_path,
)


DEFAULT_IN = CROSSMODAL_ROOT / "data" / "frames.parquet"
DEFAULT_OUT = CROSSMODAL_ROOT / "data" / "frames_tagged.parquet"
DEFAULT_COMPOSITE_DIR = CROSSMODAL_ROOT / "data" / "composites"


def _fit_tile(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    image = ImageOps.exif_transpose(image.convert("RGB"))
    return ImageOps.fit(image, size, method=Image.Resampling.LANCZOS)


def create_mosaic(jpegs: dict[str, bytes], *, tile_size: tuple[int, int] = (384, 216)) -> Image.Image:
    width, height = tile_size
    mosaic = Image.new("RGB", (width * 4, height * 2), (20, 22, 24))
    draw = ImageDraw.Draw(mosaic)

    for index, camera_name in enumerate(CAMERA_DISPLAY_ORDER):
        x = (index % 4) * width
        y = (index // 4) * height
        payload = jpegs.get(camera_name)
        if payload:
            with Image.open(BytesIO(payload)) as raw:
                tile = _fit_tile(raw, tile_size)
            mosaic.paste(tile, (x, y))
        else:
            draw.rectangle((x, y, x + width, y + height), fill=(36, 38, 42))

        draw.rectangle((x, y, x + width, y + 24), fill=(0, 0, 0))
        draw.text((x + 8, y + 6), camera_name.replace("_", " "), fill=(255, 255, 255))

    return mosaic


def image_scene_stats(image: Image.Image) -> dict[str, float]:
    sample = image.resize((128, 72)).convert("RGB")
    gray = sample.convert("L")
    stat = ImageStat.Stat(sample)
    gray_stat = ImageStat.Stat(gray)
    r, g, b = stat.mean
    brightness = gray_stat.mean[0]
    contrast = gray_stat.stddev[0]
    saturation_proxy = (max(r, g, b) - min(r, g, b)) / 255.0
    return {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "saturation": float(saturation_proxy),
    }


def intent_phrase(intent_name: str) -> str:
    return {
        "GO_LEFT": "turning left",
        "GO_RIGHT": "turning right",
        "GO_STRAIGHT": "going straight",
        "UNKNOWN": "driving",
    }.get(intent_name or "UNKNOWN", "driving")


def infer_caption_and_tags(row: pd.Series, stats: dict[str, float]) -> dict[str, Any]:
    intent_name = str(row.get("intent_name") or "UNKNOWN")
    speed = row.get("past_speed_now")
    speed_value = None if pd.isna(speed) else float(speed)
    motion_tag = "stopped"
    if speed_value is not None:
        if speed_value >= 8.0:
            motion_tag = "moving"
        elif speed_value >= 1.0:
            motion_tag = "slow"

    tags = [
        intent_name.lower(),
        intent_phrase(intent_name).replace(" ", "_"),
        motion_tag,
        "metadata_only",
        "needs_vlm",
    ]
    tags = list(dict.fromkeys(tag for tag in tags if tag and tag != "unknown"))

    speed_text = "unknown current speed"
    if speed_value is not None:
        speed_text = f"current speed about {speed_value:.1f} m/s"
    caption = (
        f"Metadata-only placeholder: ego intent is {intent_name}; {speed_text}. "
        "Visual caption not generated yet."
    )
    return {
        "caption": caption,
        "tags": tags,
        "time_of_day": "unknown",
        "weather": "unknown",
        "road_type": "unknown",
        "motion_state": motion_tag,
        "traffic_lights_visible": None,
        "agents_json": json.dumps([]),
        "notable": [],
        "caption_model": "metadata-only",
        "caption_status": "needs_vlm",
    }


class LocalBlipCaptioner:
    def __init__(self, model_name: str) -> None:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self.processor = BlipProcessor.from_pretrained(model_name, local_files_only=True)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name, local_files_only=True)

    def caption(self, image: Image.Image) -> str:
        inputs = self.processor(image.convert("RGB"), return_tensors="pt")
        output = self.model.generate(**inputs, max_new_tokens=48)
        return self.processor.decode(output[0], skip_special_tokens=True)


def maybe_load_vlm(model: str, model_name: str) -> LocalBlipCaptioner | None:
    if model not in {"auto", "blip"}:
        return None
    try:
        return LocalBlipCaptioner(model_name)
    except Exception as exc:
        print(f"local VLM unavailable, using heuristic captions: {exc}", file=sys.stderr)
        return None


def caption_frames(
    frames: pd.DataFrame,
    *,
    shard_path: Path,
    out_composite_dir: Path,
    save_composites: bool,
    model: str,
    model_name: str,
    limit: int | None = None,
    progress_every: int = 25,
) -> pd.DataFrame:
    vlm = maybe_load_vlm(model, model_name)
    rows = []
    total = len(frames) if limit is None else min(limit, len(frames))

    for position, (_, row) in enumerate(frames.iterrows()):
        if limit is not None and position >= limit:
            break

        record = load_record_by_offset(
            shard_path,
            int(row["byte_offset"]),
            int(row["byte_length"]),
            include_images=True,
        )
        mosaic = create_mosaic(record["image_payloads"])
        stats = image_scene_stats(mosaic)
        caption_data = infer_caption_and_tags(row, stats)

        if vlm is not None:
            try:
                vlm_caption = vlm.caption(mosaic)
                if vlm_caption:
                    caption_data["caption"] = vlm_caption
                    caption_data["caption_model"] = model_name
                    caption_data["caption_status"] = "vlm"
            except Exception as exc:
                caption_data["caption_status"] = f"vlm_error:{exc.__class__.__name__}"

        composite_path = ""
        if save_composites:
            out_composite_dir.mkdir(parents=True, exist_ok=True)
            frame_name = str(row["frame_name"])
            composite_file = out_composite_dir / f"{frame_name}.jpg"
            mosaic.save(composite_file, quality=88)
            composite_path = str(composite_file)

        merged = row.to_dict()
        merged.update(caption_data)
        merged.update(
            {
                "composite_path": composite_path,
                "image_brightness": stats["brightness"],
                "image_contrast": stats["contrast"],
                "image_saturation": stats["saturation"],
            }
        )
        rows.append(merged)

        if progress_every and (position + 1) % progress_every == 0:
            print(f"captioned {position + 1}/{total} frames", file=sys.stderr)

    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Caption scanned E2E frames with a local VLM or deterministic fallback.")
    parser.add_argument("--frames", type=Path, default=DEFAULT_IN, help="Input frames.parquet path.")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output frames_tagged.parquet path.")
    parser.add_argument("--shard", type=Path, default=DEFAULT_SHARD_PATH, help="Local E2E TFRecord shard path.")
    parser.add_argument("--model", choices=["auto", "heuristic", "blip"], default="auto")
    parser.add_argument("--model-name", default="Salesforce/blip-image-captioning-base")
    parser.add_argument("--save-composites", action="store_true", help="Write cached 4x2 composite JPEGs.")
    parser.add_argument("--composite-dir", type=Path, default=DEFAULT_COMPOSITE_DIR)
    parser.add_argument("--limit", type=int, default=None, help="Optional max records for quick tests.")
    parser.add_argument("--progress-every", type=int, default=25)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    frames = pd.read_parquet(args.frames)
    shard_path = resolve_shard_path(args.shard)
    if not shard_path.exists():
        raise FileNotFoundError(f"Missing shard: {shard_path}")

    tagged = caption_frames(
        frames,
        shard_path=shard_path,
        out_composite_dir=args.composite_dir,
        save_composites=args.save_composites,
        model=args.model,
        model_name=args.model_name,
        limit=args.limit,
        progress_every=args.progress_every,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tagged.to_parquet(args.out, index=False)
    print(f"wrote {len(tagged)} tagged frames to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
