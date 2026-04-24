"""Environment-driven settings. No extra dependencies."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _truthy(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    """
    vLLM is expected as an OpenAI-compatible HTTP server, e.g.:
      vllm serve Qwen/Qwen2.5-1.5B-Instruct --host 127.0.0.1 --port 8001
    Then set COMICSTAR_VLLM_BASE_URL=http://127.0.0.1:8001/v1
    (some setups omit /v1; we normalize below).
    """

    vllm_base_url: str
    vllm_model: str
    vllm_api_key: str
    vllm_timeout_s: float
    vllm_max_tokens: int
    vllm_temperature: float
    use_stub: bool


def load_settings() -> Settings:
    base = os.getenv("COMICSTAR_VLLM_BASE_URL", "").strip().rstrip("/")
    # Accept either .../v1 or server root; chat path is always /v1/chat/completions
    force_stub = _truthy("COMICSTAR_USE_STUB")

    return Settings(
        vllm_base_url=base,
        vllm_model=os.getenv("COMICSTAR_VLLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct"),
        vllm_api_key=os.getenv("COMICSTAR_VLLM_API_KEY", "EMPTY"),
        vllm_timeout_s=float(os.getenv("COMICSTAR_VLLM_TIMEOUT", "120")),
        vllm_max_tokens=int(os.getenv("COMICSTAR_VLLM_MAX_TOKENS", "4096")),
        vllm_temperature=float(os.getenv("COMICSTAR_VLLM_TEMPERATURE", "0.1")),
        use_stub=force_stub or not base,
    )
