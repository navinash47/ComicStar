from __future__ import annotations

from typing import Any, Dict, Union

from fastapi import Body, FastAPI
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from .llm_service import parse_script_with_llm
from .schemas import ParseScriptRequest, ParseScriptResponse, ParsedScript

app = FastAPI(title="ComicStar Backend", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/parse-script", response_model=ParseScriptResponse)
def parse_script(payload: ParseScriptRequest) -> ParseScriptResponse:
    parsed = parse_script_with_llm(payload.raw_script)
    return ParseScriptResponse(data=parsed)


@app.post("/validate-script")
def validate_script(
    body: Dict[str, Any] = Body(..., description="JSON object matching ParsedScript"),
) -> Union[JSONResponse, dict[str, Any]]:
    """
    Validate a comic script document against the ParsedScript model.
    On failure, returns 422 with Pydantic's error list (e.errors()).
    """
    try:
        data = ParsedScript.model_validate(body)
    except ValidationError as e:
        return JSONResponse(
            status_code=422,
            content={"valid": False, "errors": e.errors()},
        )
    return {"valid": True, "data": data.model_dump()}
