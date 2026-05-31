"""Model selection layer.

Every model used by the pipeline is chosen per *role* in ``config/models.yaml`` and reached
through one OpenAI-compatible client. Because Ollama, vLLM, and OpenAI all speak the same API,
swapping "use a bigger model" or "point at the vLLM server on the H100s" is a config edit only —
no code changes anywhere in the pipeline.

Roles:
  classifier  (required)  - assigns each transaction to a category
  extractor   (optional)  - Step 1 LLM fallback that reads a PDF when deterministic parsing fails
                            (defaults to reuse classifier)
  insights    (optional)  - writes the one-paragraph report summary (defaults to reuse classifier)
  embeddings  (optional)  - merchant-similarity / no-LLM fast path (off by default)
  ocr         (optional)  - Step 1 scanned-PDF OCR toggle (off by default; handled in extract.py)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import yaml

from . import paths

DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL = "qwen2.5:7b-instruct"

DEFAULT_MODELS_CONFIG: dict[str, Any] = {
    "classifier": {
        "provider": "ollama",
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "temperature": 0,
    },
    # Reuse the classifier model for the PDF-reading fallback by default.
    "extractor": {"use": "classifier", "enabled": True},
    # Reuse the classifier model for the report summary by default.
    "insights": {"use": "classifier", "enabled": True},
    "embeddings": {"provider": "ollama", "base_url": DEFAULT_BASE_URL,
                   "model": "nomic-embed-text", "enabled": False},
    "ocr": {"enabled": False},
}


@dataclass
class RoleConfig:
    role: str
    provider: str = "ollama"
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    temperature: float = 0.0
    api_key: str = "ollama"  # Ollama ignores it; OpenAI/vLLM read from here or env
    enabled: bool = True
    extra: Optional[dict] = None


def _raw_config() -> dict[str, Any]:
    if paths.MODELS_YAML.exists():
        data = yaml.safe_load(paths.MODELS_YAML.read_text()) or {}
        # shallow-merge over defaults so partial configs still work
        merged = {k: dict(v) if isinstance(v, dict) else v
                  for k, v in DEFAULT_MODELS_CONFIG.items()}
        for k, v in data.items():
            if isinstance(v, dict) and isinstance(merged.get(k), dict):
                merged[k].update(v)
            else:
                merged[k] = v
        return merged
    return {k: dict(v) if isinstance(v, dict) else v
            for k, v in DEFAULT_MODELS_CONFIG.items()}


def write_default_models_config(overwrite: bool = False) -> None:
    """Write config/models.yaml with documented defaults (used during onboarding)."""
    if paths.MODELS_YAML.exists() and not overwrite:
        return
    paths.ensure_dirs()
    header = (
        "# OCD model configuration. Every role speaks the OpenAI-compatible API, so you can\n"
        "# point any of them at Ollama (default), a local vLLM server, or a remote endpoint by\n"
        "# changing base_url + model only.\n"
        "#\n"
        "# To use the H100s with vLLM instead of Ollama, e.g.:\n"
        "#   classifier: { provider: vllm, base_url: http://localhost:8000/v1,\n"
        "#                 model: Qwen/Qwen2.5-14B-Instruct, temperature: 0, api_key: EMPTY }\n"
    )
    paths.MODELS_YAML.write_text(header + yaml.safe_dump(DEFAULT_MODELS_CONFIG, sort_keys=False))


def get_role_config(role: str) -> RoleConfig:
    cfg = _raw_config()
    section = dict(cfg.get(role, {}))

    # 'use: <other-role>' indirection (e.g. insights reusing classifier).
    if "use" in section:
        target = section["use"]
        base = dict(cfg.get(target, {}))
        base.update({k: v for k, v in section.items() if k != "use"})
        section = base

    known = {"provider", "base_url", "model", "temperature", "api_key", "enabled"}
    extra = {k: v for k, v in section.items() if k not in known}
    return RoleConfig(
        role=role,
        provider=section.get("provider", "ollama"),
        base_url=section.get("base_url", DEFAULT_BASE_URL),
        model=section.get("model", DEFAULT_MODEL),
        temperature=float(section.get("temperature", 0.0)),
        api_key=section.get("api_key", "ollama"),
        enabled=bool(section.get("enabled", True)),
        extra=extra or None,
    )


def get_model(role: str) -> str:
    return get_role_config(role).model


def is_enabled(role: str) -> bool:
    return get_role_config(role).enabled


def get_client(role: str):
    """Return an OpenAI-compatible client configured for ``role``.

    Imported lazily so the package can be inspected without the ``openai`` dependency.
    """
    from openai import OpenAI

    rc = get_role_config(role)
    return OpenAI(base_url=rc.base_url, api_key=rc.api_key or "ollama")


def health_check(role: str = "classifier") -> tuple[bool, str]:
    """Best-effort check that the role's endpoint is reachable and the model is available."""
    rc = get_role_config(role)
    try:
        client = get_client(role)
        models = {m.id for m in client.models.list().data}
        # Ollama reports models with and without the ':latest' suffix inconsistently.
        wanted = rc.model
        ok = any(wanted == m or wanted.split(":")[0] == m.split(":")[0] for m in models) or not models
        if ok:
            return True, f"{role}: '{rc.model}' reachable at {rc.base_url}"
        return False, (f"{role}: endpoint up at {rc.base_url} but model '{rc.model}' not found. "
                       f"Available: {sorted(models)}")
    except Exception as e:  # noqa: BLE001 - surface any connection/config error to the user
        return False, f"{role}: cannot reach {rc.base_url} ({e}). Is the server running?"
