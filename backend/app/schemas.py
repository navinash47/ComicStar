from pydantic import BaseModel, Field
from typing import List, Optional


class Character(BaseModel):
    id: str = Field(..., description="Unique character id")
    name: str
    description: Optional[str] = None


class Dialogue(BaseModel):
    speaker_id: str
    text: str


class Panel(BaseModel):
    panel_number: int
    scene_description: str
    dialogues: List[Dialogue] = []


class ParseScriptRequest(BaseModel):
    raw_script: str = Field(..., min_length=1)


class ParsedScript(BaseModel):
    title: str
    characters: List[Character] = []
    panels: List[Panel] = []


class ParseScriptResponse(BaseModel):
    data: ParsedScript