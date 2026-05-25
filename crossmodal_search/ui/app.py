from __future__ import annotations

import io
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from crossmodal_search.lib.e2e import CAMERA_COCKPIT_ORDER, CAMERA_DISPLAY_ORDER, CROSSMODAL_ROOT
from crossmodal_search.search.hydrate import camera_jpegs_for_record
from crossmodal_search.search.query import CrossmodalIndex


APP_ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = CROSSMODAL_ROOT / "data" / "search_index.json"

app = Flask(
    __name__,
    template_folder=str(APP_ROOT / "templates"),
    static_folder=str(APP_ROOT / "static"),
)

SEARCH_INDEX: CrossmodalIndex | None = None
LOAD_ERROR: str | None = None
LOADED_MANIFEST_MTIME: float | None = None


def load_index() -> CrossmodalIndex:
    global SEARCH_INDEX, LOAD_ERROR, LOADED_MANIFEST_MTIME
    manifest_mtime = DEFAULT_MANIFEST.stat().st_mtime if DEFAULT_MANIFEST.exists() else None
    if SEARCH_INDEX is not None and LOADED_MANIFEST_MTIME == manifest_mtime:
        return SEARCH_INDEX
    try:
        SEARCH_INDEX = CrossmodalIndex.load(DEFAULT_MANIFEST)
        LOADED_MANIFEST_MTIME = manifest_mtime
        LOAD_ERROR = None
        return SEARCH_INDEX
    except Exception as exc:
        LOAD_ERROR = str(exc)
        raise


def index_status() -> dict[str, object]:
    try:
        index = load_index()
        visual_frames = int(index.frames["caption_status"].eq("vlm").sum()) if "caption_status" in index.frames else 0
        return {
            "ready": True,
            "row_count": len(index.frames),
            "visual_indexed_frames": visual_frames,
            "manifest": str(DEFAULT_MANIFEST),
            "visual_search_available": index.visual_search_available(),
            "error": None,
        }
    except Exception:
        return {
            "ready": False,
            "row_count": 0,
            "visual_indexed_frames": 0,
            "manifest": str(DEFAULT_MANIFEST),
            "visual_search_available": False,
            "error": LOAD_ERROR,
        }


@app.route("/")
def home():
    return render_template(
        "search.html",
        status=index_status(),
        camera_order=CAMERA_DISPLAY_ORDER,
    )


@app.route("/frame/<frame_name>")
def frame_page(frame_name: str):
    return render_template(
        "detail.html",
        frame_name=frame_name,
        status=index_status(),
        camera_order=CAMERA_COCKPIT_ORDER,
    )


@app.route("/api/status")
def api_status():
    return jsonify(index_status())


@app.route("/api/search")
def api_search():
    try:
        index = load_index()
    except Exception:
        abort(503)
    query = request.args.get("q", "")
    k = min(max(request.args.get("k", default=12, type=int), 1), 40)
    route = index.route_query(query)
    blocked = route["blocked_terms"]
    warnings = []
    if blocked:
        warnings.append(
            "Visual search is not indexed yet. Unsupported terms: " + ", ".join(blocked)
        )
        results = []
    else:
        results = index.search(
            route["normalized_query"],
            k=k,
            filters=route["filters"],
            enforce_blocking=False,
        )
    return jsonify(
        {
            "query": query,
            "normalized_query": route["normalized_query"],
            "results": results,
            "warnings": warnings,
            "mode": "visual" if index.visual_search_available() else "metadata_only",
            "agent": route["agent"],
            "camera_order": CAMERA_DISPLAY_ORDER,
        }
    )


@app.route("/api/frame/<frame_name>")
def api_frame(frame_name: str):
    try:
        index = load_index()
        detail = index.frame_detail(frame_name)
    except KeyError:
        abort(404)
    except Exception:
        abort(503)
    detail["cockpit_order"] = CAMERA_COCKPIT_ORDER
    return jsonify(detail)


@app.route("/api/image/<frame_name>/<camera_name>")
def api_image(frame_name: str, camera_name: str):
    try:
        index = load_index()
        row = index.frame_row(frame_name)
    except KeyError:
        abort(404)
    except Exception:
        abort(503)

    shard_path = str(row["shard_path"])
    images = camera_jpegs_for_record(shard_path, int(row["byte_offset"]), int(row["byte_length"]))
    payload = images.get(camera_name)
    if payload is None:
        abort(404)
    return send_file(io.BytesIO(payload), mimetype="image/jpeg")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
