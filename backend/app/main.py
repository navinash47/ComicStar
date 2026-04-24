from fastapi import FastAPI
from .schemas import ParseScriptRequest, ParseScriptResponse
from .llm_service import parse_script_with_llm

app = FastAPI(title="ComicStar Backend", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/parse-script", response_model=ParseScriptResponse)
def parse_script(payload: ParseScriptRequest) -> ParseScriptResponse:
    parsed = parse_script_with_llm(payload.raw_script)
    return ParseScriptResponse(data=parsed)