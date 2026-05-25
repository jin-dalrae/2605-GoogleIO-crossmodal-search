from __future__ import annotations

import os
from pathlib import Path


_LOADED = False

ALIASES = {
    "openai_api_key": "OPENAI_API_KEY",
    "openai_key": "OPENAI_API_KEY",
    "gemini_api_key": "GEMINI_API_KEY",
    "gemini_key": "GEMINI_API_KEY",
    "generative_language_api_key": "GEMINI_API_KEY",
    "gemini_model": "GEMINI_MODEL",
}


def load_repo_env(start: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from the nearest .env without overwriting env."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True

    root = (start or Path.cwd()).resolve()
    candidates = []
    for directory in [root, *root.parents]:
        candidates.append(directory)
        candidates.append(directory / "crossmodal_search")
    for directory in candidates:
        env_path = directory / ".env"
        if not env_path.exists():
            continue
        for raw_line in env_path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
            canonical = ALIASES.get(key.lower())
            if canonical and canonical not in os.environ:
                os.environ[canonical] = value
        return
