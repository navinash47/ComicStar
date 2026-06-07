import json


class CustomPromptIterator:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        # path is /workspace/datasets/ayaka_psychic_girl/ayaka_descriptions.json
        descriptions_path = "/workspace/datasets/ayaka_psychic_girl/ayaka_descriptions.json"
        with open(descriptions_path, "r") as f:
            descriptions = json.load(f)
        descriptions = descriptions.get("Descriptions", [])
        return {
            "required": {
                # A multiline text box for your image descriptions
                "image_descriptions": ("STRING", {"multiline": True, "default": "\n".join(descriptions)}),
                # An integer tracker to step through the list
                "run_index": ("INT", {"default": 0, "min": 0, "max": len(descriptions) - 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("selected_description", "current_index")
    FUNCTION = "iterate_descriptions"
    CATEGORY = "Custom Methods"

    def iterate_descriptions(self, image_descriptions, run_index):
        # Clean up lines and filter out empty inputs
        lines = [line.strip() for line in image_descriptions.split('\n') if line.strip()]
        
        if not lines:
            return ("", 0)
        
        # Modulo prevents index errors and causes the list to loop infinitely
        actual_index = run_index % len(lines)
        chosen_description = lines[actual_index]
        
        return (chosen_description, actual_index)

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "CustomPromptIterator": CustomPromptIterator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomPromptIterator": "Custom Prompt Iterator"
}