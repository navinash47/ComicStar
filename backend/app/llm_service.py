from __future__ import annotations

import json
import re
from typing import Any, List, Optional

import httpx
from pydantic import ValidationError

from .schemas import Character, Dialogue, Panel, ParsedScript, Scene
from .settings import load_settings


class ParseScriptLLMError(Exception):
    """Raised when vLLM output cannot be turned into a valid ParsedScript."""

    def __init__(
        self,
        message: str,
        *,
        raw_output: Optional[str] = None,
        validation_errors: Optional[List[Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.raw_output = raw_output
        self.validation_errors = validation_errors


def _stub_parsed_script() -> ParsedScript:
    return ParsedScript(
        title="Draft Comic",
        logline="Stub output (set COMICSTAR_VLLM_BASE_URL to call vLLM).",
        characters=[
            Character(
                id="c1",
                name="Protagonist",
                role="protagonist",
                description="Main lead character",
            ),
        ],
        scenes=[
            Scene(
                id="s1",
                key="opening",
                label="Opening scene",
                mood="neutral",
            ),
        ],
        panels=[
            Panel(
                panel_number=1,
                scene_id="s1",
                description="Wide shot: protagonist enters the first scene.",
                emotion="curious",
                dialogues=[
                    Dialogue(
                        speaker_id="c1",
                        text="Let's begin this story.",
                        emotion="cheerful",
                    )
                ],
            )
        ],
    )


def _system_prompt() -> str:
    schema = json.dumps(ParsedScript.model_json_schema(), separators=(",", ":"))
    return (
        "You are a comic and manhwa script parser. "
        "Convert the user's unstructured story text into ONE JSON object only.\n\n"
        "Rules:\n"
        "- Output raw JSON only. No markdown code fences, no commentary, no text before or after the object.\n"
        "- The object MUST conform to this JSON Schema (field names and types must match):\n"
        f"{schema}\n\n"
        "Referential integrity:\n"
        "- Every panels[].scene_id must equal exactly one scenes[].id.\n"
        "- Every panels[].dialogues[].speaker_id must equal exactly one characters[].id.\n"
        "- Use stable snake_case or slug ids (e.g. c_hero, s_rooftop).\n"
        "- panel_number starts at 1 and increases in reading order.\n"
    )


def _strip_json_fences(text: str) -> str:
    s = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _extract_json_object(text: str) -> str:
    """Best-effort: isolate a single JSON object from model output."""
    s = _strip_json_fences(text)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model output")
    return s[start : end + 1]


def _chat_completion_content(payload: dict[str, Any]) -> str:
    try:
        choice0 = payload["choices"][0]
        msg = choice0["message"]
        content = msg.get("content")
    except (KeyError, IndexError, TypeError) as e:
        raise ParseScriptLLMError(
            "Unexpected vLLM OpenAI response shape",
            raw_output=json.dumps(payload)[:16000],
        ) from e
    if not isinstance(content, str) or not content.strip():
        raise ParseScriptLLMError(
            "Empty or missing message content from vLLM",
            raw_output=json.dumps(payload)[:16000],
        )
    return content


def parse_script_with_llm(raw_script: str) -> ParsedScript:
    settings = load_settings()
    if settings.use_stub:
        return _stub_parsed_script()

    base = settings.vllm_base_url
    if base.endswith("/v1"):
        url = f"{base}/chat/completions"
    else:
        url = f"{base}/v1/chat/completions"

    body = {
        "model": settings.vllm_model,
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {
                "role": "user",
                "content": (
                    "Parse this script into the JSON schema described in your instructions.\n\n"
                    f"--- SCRIPT ---\n{raw_script}"
                ),
            },
        ],
        "temperature": settings.vllm_temperature,
        "max_tokens": settings.vllm_max_tokens,
    }

    headers = {
        "Authorization": f"Bearer {settings.vllm_api_key}",
        "Content-Type": "application/json",
    }

    try:
        with httpx.Client(timeout=settings.vllm_timeout_s) as client:
            resp = client.post(url, headers=headers, json=body)
            raw_http = resp.text
            if resp.status_code >= 400:
                raise ParseScriptLLMError(
                    f"vLLM HTTP {resp.status_code}",
                    raw_output=raw_http[:16000],
                )
            data = resp.json()
    except httpx.RequestError as e:
        raise ParseScriptLLMError(
            f"vLLM request failed: {e!s}",
            raw_output=None,
        ) from e

    content = _chat_completion_content(data)

    try:
        json_text = _extract_json_object(content)
        obj = json.loads(json_text)
    except (json.JSONDecodeError, ValueError) as e:
        raise ParseScriptLLMError(
            f"Model output is not valid JSON: {e!s}",
            raw_output=content,
        ) from e

    try:
        return ParsedScript.model_validate(obj)
    except ValidationError as e:
        raise ParseScriptLLMError(
            "JSON parsed but does not match ParsedScript schema",
            raw_output=content,
            validation_errors=e.errors(),
        ) from e
