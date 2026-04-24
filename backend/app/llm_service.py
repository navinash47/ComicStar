from .schemas import Character, Dialogue, Panel, ParsedScript, Scene


def parse_script_with_llm(raw_script: str) -> ParsedScript:
    """
    Stub parser for V1.
    Later: replace with vLLM call that returns strict JSON against ParsedScript.
    """
    return ParsedScript(
        title="Draft Comic",
        logline="Stub output until vLLM parsing is connected.",
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
