from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class Character(BaseModel):
    id: str = Field(..., min_length=1, description="Unique character id, stable across the script")
    name: str = Field(..., min_length=1)
    role: Optional[str] = Field(
        default=None, description="e.g. protagonist, supporting, antagonist"
    )
    description: Optional[str] = None


class Scene(BaseModel):
    id: str = Field(..., min_length=1, description="Unique scene id referenced by panels")
    key: str = Field(
        ...,
        min_length=1,
        description="Stable slug for routing/state, e.g. 'rooftop_dusk'",
    )
    label: str = Field(..., min_length=1, description="Human-readable location or beat")
    mood: Optional[str] = Field(
        default=None, description="e.g. tense, comedic, melancholic"
    )


class Dialogue(BaseModel):
    speaker_id: str = Field(..., min_length=1, description="Must match a Character.id")
    text: str = Field(..., min_length=1)
    emotion: Optional[str] = Field(
        default=None, description="Delivery tone, e.g. shout, whisper, deadpan"
    )


class Panel(BaseModel):
    panel_number: int = Field(..., ge=1, description="1-based order in the comic")
    scene_id: str = Field(..., min_length=1, description="Must match a Scene.id")
    description: str = Field(
        ...,
        min_length=1,
        description="Visual / action description for the panel",
    )
    emotion: Optional[str] = Field(
        default=None, description="Emotional beat for the panel, e.g. shock, joy"
    )
    dialogues: List[Dialogue] = Field(default_factory=list)


class ParsedScript(BaseModel):
    title: str = Field(..., min_length=1)
    logline: Optional[str] = Field(
        default=None, description="One-line story summary for tooling/search"
    )
    characters: List[Character] = Field(default_factory=list)
    scenes: List[Scene] = Field(default_factory=list)
    panels: List[Panel] = Field(default_factory=list)

    @model_validator(mode="after")
    def _references(self) -> ParsedScript:
        char_ids = {c.id for c in self.characters}
        scene_ids = {s.id for s in self.scenes}
        for p in self.panels:
            if p.scene_id not in scene_ids:
                raise ValueError(
                    f"Panel {p.panel_number} scene_id {p.scene_id!r} not in scenes: {sorted(scene_ids)}"
                )
            for d in p.dialogues:
                if d.speaker_id not in char_ids:
                    raise ValueError(
                        f"Panel {p.panel_number} dialogue references unknown "
                        f"speaker_id {d.speaker_id!r}; known: {sorted(char_ids)}"
                    )
        return self


class ParseScriptRequest(BaseModel):
    raw_script: str = Field(..., min_length=1)


class ParseScriptResponse(BaseModel):
    data: ParsedScript
