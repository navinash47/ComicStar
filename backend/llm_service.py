from .schemas import ParsedScript, Character, Panel, Dialogue


def parse_script_with_llm(raw_script: str) -> ParsedScript:
    """
    Stub parser for V1.
    Later: replace with vLLM call that returns strict JSON.
    """
    return ParsedScript(
        title="Draft Comic",
        characters=[
            Character(id="c1", name="Protagonist", description="Main lead character"),
        ],
        panels=[
            Panel(
                panel_number=1,
                scene_description="Opening scene based on input script.",
                dialogues=[
                    Dialogue(speaker_id="c1", text="Let's begin this story.")
                ],
            )
        ],
    )