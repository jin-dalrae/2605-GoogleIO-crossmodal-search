from __future__ import annotations

import base64
import json
import os
import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any

from PIL import Image

from crossmodal_search.lib.env import load_repo_env

load_repo_env()
DEFAULT_AGENT_MODEL = os.environ.get("CROSSMODAL_AGENT_MODEL", "gpt-5-mini")
DEFAULT_VISION_AGENT_MODEL = os.environ.get("CROSSMODAL_VISION_AGENT_MODEL", DEFAULT_AGENT_MODEL)
DEFAULT_GEMINI_MODEL = os.environ.get("GEMINI_MODEL", os.environ.get("CROSSMODAL_GEMINI_MODEL", "gemini-2.5-flash"))

QUERY_ROUTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "normalized_query": {"type": "string"},
        "intent_name": {"type": ["string", "null"], "enum": ["GO_LEFT", "GO_RIGHT", "GO_STRAIGHT", "UNKNOWN", None]},
        "motion_state": {"type": ["string", "null"], "enum": ["stopped", "slow", "moving", None]},
        "requires_visual_index": {"type": "boolean"},
        "unsupported_terms": {"type": "array", "items": {"type": "string"}},
        "explanation": {"type": "string"},
    },
    "required": [
        "normalized_query",
        "intent_name",
        "motion_state",
        "requires_visual_index",
        "unsupported_terms",
        "explanation",
    ],
}

VISION_CAPTION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "caption": {"type": "string"},
        "time_of_day": {"type": "string", "enum": ["day", "dusk", "dawn", "night", "unknown"]},
        "weather": {"type": "string", "enum": ["clear", "rain", "snow", "fog", "wet_road", "unknown"]},
        "road_type": {
            "type": "string",
            "enum": ["highway", "urban_street", "intersection", "residential", "parking_lot", "other", "unknown"],
        },
        "traffic_lights_visible": {"type": ["boolean", "null"]},
        "agents": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": ["vehicle", "pedestrian", "cyclist", "truck", "bus", "other"]},
                    "count": {"type": "integer"},
                },
                "required": ["type", "count"],
            },
        },
        "notable": {"type": "array", "items": {"type": "string"}},
        "tags": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "caption",
        "time_of_day",
        "weather",
        "road_type",
        "traffic_lights_visible",
        "agents",
        "notable",
        "tags",
    ],
}


@dataclass(frozen=True)
class ManagedAgentStatus:
    available: bool
    provider: str
    model: str | None
    reason: str | None = None


def _openai_client() -> Any:
    from openai import OpenAI

    return OpenAI()


def _provider_preference() -> str:
    return os.environ.get("CROSSMODAL_AGENT_PROVIDER", "auto").lower()


def managed_agents_enabled() -> bool:
    value = os.environ.get("CROSSMODAL_MANAGED_AGENTS", "auto").lower()
    return value not in {"0", "false", "off", "no"}


def managed_agent_status(model: str = DEFAULT_AGENT_MODEL) -> ManagedAgentStatus:
    if not managed_agents_enabled():
        return ManagedAgentStatus(False, "managed_agent", model, "disabled by CROSSMODAL_MANAGED_AGENTS")

    preference = _provider_preference()
    if preference in {"auto", "gemini"} and os.environ.get("GEMINI_API_KEY"):
        try:
            import requests  # noqa: F401
        except Exception as exc:
            return ManagedAgentStatus(False, "gemini_generate_content", DEFAULT_GEMINI_MODEL, f"requests unavailable: {exc}")
        return ManagedAgentStatus(True, "gemini_generate_content", DEFAULT_GEMINI_MODEL, None)

    if preference in {"auto", "openai"} and os.environ.get("OPENAI_API_KEY"):
        try:
            import openai  # noqa: F401
        except Exception as exc:
            if preference == "openai":
                return ManagedAgentStatus(False, "openai_responses", model, f"openai package unavailable: {exc}")
        else:
            return ManagedAgentStatus(True, "openai_responses", model, None)

    if preference == "gemini":
        return ManagedAgentStatus(False, "gemini_generate_content", DEFAULT_GEMINI_MODEL, "missing GEMINI_API_KEY")
    return ManagedAgentStatus(False, "managed_agent", model, "missing OPENAI_API_KEY or GEMINI_API_KEY")


def _json_schema_text_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": {
            "type": "json_schema",
            "name": name,
            "strict": True,
            "schema": schema,
        }
    }


def _response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    chunks = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(text)
    return "".join(chunks)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    try:
        return value.item()
    except Exception:
        return str(value)


def _bounded_jpeg_b64(image: Image.Image, *, max_width: int = 1024) -> str:
    image = image.convert("RGB")
    if image.width > max_width:
        height = int(image.height * (max_width / image.width))
        image = image.resize((max_width, height), Image.Resampling.LANCZOS)
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=82, optimize=True)
    return base64.b64encode(buffered.getvalue()).decode("ascii")


def _gemini_generate_json(
    *,
    model: str,
    system_prompt: str,
    user_payload: Any,
    image: Image.Image | None = None,
) -> dict[str, Any]:
    import requests

    parts: list[dict[str, Any]] = [{"text": system_prompt + "\n\n" + json.dumps(_json_safe(user_payload))}]
    if image is not None:
        parts.append(
            {
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": _bounded_jpeg_b64(image),
                }
            }
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {"response_mime_type": "application/json"},
    }
    last_exc = None
    for attempt in range(3):
        try:
            response = requests.post(
                url,
                params={"key": os.environ["GEMINI_API_KEY"]},
                json=payload,
                timeout=90,
            )
            break
        except requests.RequestException as exc:
            last_exc = exc
            if attempt == 2:
                raise
            time.sleep(1.5 * (attempt + 1))
    else:
        raise last_exc  # type: ignore[misc]
    if not response.ok:
        detail = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"Gemini HTTP {response.status_code}: {detail}")
    payload = response.json()
    text = payload["candidates"][0]["content"]["parts"][0]["text"]
    return json.loads(text)


class QueryRouterAgent:
    """Hosted query router that decides which parts of a query are indexable."""

    def __init__(self, model: str = DEFAULT_AGENT_MODEL) -> None:
        status = managed_agent_status(model)
        if not status.available:
            raise RuntimeError(status.reason or "managed agent unavailable")
        self.provider = status.provider
        self.model = status.model or model
        self.client = _openai_client() if self.provider == "openai_responses" else None

    def route(self, query: str, *, visual_search_available: bool) -> dict[str, Any]:
        system_prompt = (
            "You are the Crossmodal Search query-router agent. "
            "Route a driving-scene search query to the local index. "
            "The local metadata index supports only ego intent "
            "(GO_LEFT, GO_RIGHT, GO_STRAIGHT) and motion state "
            "(stopped, slow, moving). Visual concepts such as crash, "
            "pedestrian, cyclist, weather, lighting, road type, lanes, "
            "traffic lights, color, and object descriptions require a "
            "visual caption index. Do not invent support for visual terms. "
            "Return JSON with normalized_query, intent_name, motion_state, "
            "requires_visual_index, unsupported_terms, and explanation."
        )
        user_payload = {
            "query": query,
            "visual_search_available": visual_search_available,
        }
        if self.provider == "gemini_generate_content":
            route = _gemini_generate_json(
                model=self.model,
                system_prompt=system_prompt,
                user_payload=user_payload,
            )
            route["provider"] = self.provider
            route["model"] = self.model
            route["agent"] = "query_router"
            return route

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            text=_json_schema_text_format("crossmodal_query_route", QUERY_ROUTE_SCHEMA),
        )
        route = json.loads(_response_text(response))
        route["provider"] = "openai_responses"
        route["model"] = self.model
        route["agent"] = "query_router"
        return route


class VisionCaptionAgent:
    """Hosted vision agent that captions an 8-camera mosaic for indexing."""

    def __init__(self, model: str = DEFAULT_VISION_AGENT_MODEL) -> None:
        status = managed_agent_status(model)
        if not status.available:
            raise RuntimeError(status.reason or "managed agent unavailable")
        self.provider = status.provider
        self.model = status.model or model
        self.client = _openai_client() if self.provider == "openai_responses" else None

    def caption(self, image: Image.Image, *, frame_context: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are the Crossmodal Search vision-indexing agent. "
            "Inspect the 4x2 mosaic of synchronized vehicle cameras. "
            "Return grounded scene tags only when visible; use unknown "
            "or empty lists when the evidence is insufficient. Return JSON with "
            "caption, time_of_day, weather, road_type, traffic_lights_visible, "
            "agents, notable, and tags."
        )
        user_payload = {
            "task": "caption_waymo_e2e_camera_mosaic",
            "camera_order": [
                "FRONT_LEFT",
                "FRONT",
                "FRONT_RIGHT",
                "SIDE_LEFT",
                "SIDE_RIGHT",
                "REAR_RIGHT",
                "REAR",
                "REAR_LEFT",
            ],
            "frame_context": frame_context,
        }
        if self.provider == "gemini_generate_content":
            parsed = _gemini_generate_json(
                model=self.model,
                system_prompt=system_prompt,
                user_payload=user_payload,
                image=image,
            )
            parsed["provider"] = self.provider
            parsed["model"] = self.model
            parsed["agent"] = "vision_captioner"
            return parsed

        buffered = BytesIO()
        image.convert("RGB").save(buffered, format="JPEG", quality=88)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")
        response = self.client.responses.create(
            model=self.model,
            input=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": json.dumps(user_payload),
                        },
                        {"type": "input_image", "image_url": data_url},
                    ],
                },
            ],
            text=_json_schema_text_format("crossmodal_vision_caption", VISION_CAPTION_SCHEMA),
        )
        parsed = json.loads(_response_text(response))
        parsed["provider"] = "openai_responses"
        parsed["model"] = self.model
        parsed["agent"] = "vision_captioner"
        return parsed
