import os
import re
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import wandb
from transformers import CLIPModel, CLIPProcessor


# -----------------------------
# Global model cache
# -----------------------------

_CLIP_MODEL = None
_CLIP_PROCESSOR = None
_WANDB_RUN = None


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_clip():
    global _CLIP_MODEL, _CLIP_PROCESSOR

    if _CLIP_MODEL is None:
        model_name = "openai/clip-vit-large-patch14"
        _CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)
        _CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(DEVICE)
        _CLIP_MODEL.eval()

    return _CLIP_MODEL, _CLIP_PROCESSOR


def get_wandb_run(project: str, run_name: str):
    global _WANDB_RUN

    if not project:
        return None

    if _WANDB_RUN is None:
        _WANDB_RUN = wandb.init(
            project=project,
            name=run_name if run_name else None,
            config={
                "evaluator": "phase0_prompt_attribute_eval",
                "clip_model": "openai/clip-vit-large-patch14",
            },
        )

    return _WANDB_RUN


# -----------------------------
# Caption parsing
# -----------------------------

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def has_any(text: str, phrases):
    return any(p in text for p in phrases)


def add_check(checks, check_type, expected, weight=1.0, method="clip_contrastive"):
    if expected:
        checks.append({
            "type": check_type,
            "expected": expected,
            "weight": float(weight),
            "method": method
        })


def parse_description_to_checks(description: str, image_index: int):
    """
    Converts one long training description into structured attribute checks.
    This is rule-based and tuned for your Ayaka/Hana-style prompt format.
    """

    text = normalize_text(description)
    parts = [p.strip() for p in description.split(",") if p.strip()]

    trigger_word = parts[0] if parts else ""

    checks = []

    # People count / gender
    if "1girl" in text or "1 girl" in text:
        add_check(checks, "people_count", "one girl", 1.5)
        add_check(checks, "gender", "girl", 1.0)

    if "anime girl" in text:
        add_check(checks, "character_type", "anime girl", 0.8)

    # Age / vibe
    if has_any(text, ["teenage girl", "high school girl", "schoolgirl"]):
        if "high school girl" in text:
            add_check(checks, "age_vibe", "high school girl", 0.8)
        elif "teenage girl" in text:
            add_check(checks, "age_vibe", "teenage girl", 0.8)
        else:
            add_check(checks, "age_vibe", "schoolgirl", 0.8)

    if "gyaru" in text:
        add_check(checks, "fashion_vibe", "slightly gyaru vibe", 0.6, "clip_absolute")

    # Hair
    if "medium-length" in text and "hair" in text:
        add_check(checks, "hair_length", "medium-length hair", 1.1)

    if "dark brown hair" in text:
        add_check(checks, "hair_color", "dark brown hair", 1.3)

    if has_any(text, ["reddish brown tint", "reddish tint", "soft reddish brown tint"]):
        add_check(checks, "hair_tint", "reddish brown hair tint", 0.5, "clip_absolute")

    if "straight bangs" in text:
        add_check(checks, "hair_style", "straight bangs", 0.8)

    if "face-framing strands" in text:
        add_check(checks, "hair_style", "face-framing hair strands", 0.7)

    if "wind blowing hair" in text or "wind gently moving hair" in text:
        add_check(checks, "hair_motion", "wind blowing hair", 0.5, "clip_absolute")

    # Eyes
    if "warm brown eyes" in text or "brown eyes" in text:
        add_check(checks, "eye_color", "brown eyes", 1.2)

    # Outfit
    outfit_map = [
        ("school uniform", "school uniform"),
        ("casual hoodie", "casual hoodie"),
        ("casual sweater", "casual sweater"),
        ("casual jacket", "casual jacket"),
        ("casual clothes", "casual clothes"),
        ("casual top and skirt", "casual top and skirt"),
        ("hoodie and skirt", "hoodie and skirt"),
    ]

    for key, expected in outfit_map:
        if key in text:
            add_check(checks, "outfit", expected, 1.2)
            break

    # Expressions / emotions
    expression_map = [
        ("smiling confidently", "confident smile"),
        ("sad expression", "sad expression"),
        ("thoughtful expression", "thoughtful expression"),
        ("emotional expression", "emotional expression"),
        ("blushing lightly", "blushing"),
        ("determined expression", "determined expression"),
        ("surprised expression", "surprised expression"),
        ("confident expression", "confident expression"),
        ("angry expression", "angry expression"),
        ("laughing brightly", "laughing brightly"),
        ("gentle smile", "gentle smile"),
        ("serious expression", "serious expression"),
        ("calm expression", "calm expression"),
        ("crying softly", "crying softly"),
        ("shy smile", "shy smile"),
        ("reflective mood", "reflective mood"),
        ("playful expression", "playful expression"),
        ("soft smile", "soft smile"),
        ("waiting expression", "waiting expression"),
        ("slightly annoyed expression", "annoyed expression"),
    ]

    for key, expected in expression_map:
        if key in text:
            add_check(checks, "expression", expected, 1.1)
            break

    # Pose / action
    pose_map = [
        ("standing", "standing"),
        ("sitting alone", "sitting alone"),
        ("sitting in", "sitting"),
        ("seated", "seated"),
        ("walking", "walking"),
        ("holding an umbrella", "holding umbrella"),
        ("close-up portrait", "close-up portrait"),
        ("close-up face", "close-up face"),
        ("crossing arms", "crossing arms"),
        ("running", "running"),
        ("hand near face", "hand near face"),
        ("hands on hips", "hands on hips"),
        ("reading a note", "reading a note"),
        ("looking out the window", "looking out the window"),
    ]

    for key, expected in pose_map:
        if key in text:
            if "close-up" in expected:
                add_check(checks, "camera", expected, 0.9)
            else:
                add_check(checks, "pose_action", expected, 1.0)
            break

    # Location / background
    location_map = [
        ("school hallway", "school hallway"),
        ("classroom window", "classroom window"),
        ("classroom background", "classroom"),
        ("classroom setting", "classroom"),
        ("in a classroom", "classroom"),
        ("rainy street", "rainy street"),
        ("warm cafe", "warm cafe"),
        ("cozy cafe", "cozy cafe"),
        ("school rooftop", "school rooftop"),
        ("corridor", "school corridor"),
        ("on a bed", "bedroom"),
        ("school gate", "school gate"),
        ("under a tree", "under a tree"),
        ("city street", "city street at night"),
        ("shrine path", "shrine path"),
        ("school courtyard", "school courtyard"),
        ("train platform", "train platform"),
        ("sitting on stairs", "stairs"),
        ("cherry blossoms", "cherry blossoms"),
    ]

    for key, expected in location_map:
        if key in text:
            add_check(checks, "location", expected, 1.1)
            break

    # Lighting / time
    lighting_map = [
        ("soft daylight", "soft daylight"),
        ("soft afternoon light", "soft afternoon light"),
        ("sunset lighting", "sunset lighting"),
        ("sunset sky", "sunset sky"),
        ("cinematic lighting", "cinematic lighting"),
        ("soft indoor lighting", "soft indoor lighting"),
        ("soft lighting", "soft lighting"),
        ("bright daylight", "bright daylight"),
        ("dramatic lighting", "dramatic lighting"),
        ("dramatic soft lighting", "dramatic soft lighting"),
        ("daylight", "daylight"),
        ("soft natural lighting", "soft natural lighting"),
        ("neon lights", "neon lights"),
        ("evening light", "evening light"),
        ("soft moonlight", "soft moonlight"),
        ("warm lighting", "warm lighting"),
        ("spring light", "spring light"),
    ]

    for key, expected in lighting_map:
        if key in text:
            add_check(checks, "lighting", expected, 0.8)
            break

    # Mood
    mood_map = [
        ("quiet night mood", "quiet night mood"),
        ("emotional scene", "emotional scene"),
        ("subtle supernatural atmosphere", "subtle supernatural atmosphere"),
        ("spring atmosphere", "spring atmosphere"),
        ("evening atmosphere", "evening atmosphere"),
    ]

    for key, expected in mood_map:
        if key in text:
            add_check(checks, "mood", expected, 0.7, "clip_absolute")
            break

    # Props
    prop_map = [
        ("umbrella", "umbrella"),
        ("note", "note"),
        ("desk", "desk"),
    ]

    for key, expected in prop_map:
        if key in text:
            add_check(checks, "prop", expected, 0.7)
            break

    # Style
    if "romance anime style" in text:
        add_check(checks, "style", "romance anime style", 0.8, "clip_absolute")

    if "clean line art" in text:
        add_check(checks, "style", "clean line art", 0.7, "clip_absolute")

    if "detailed eyes" in text:
        add_check(checks, "quality_detail", "detailed eyes", 0.5, "clip_absolute")

    if "detailed face" in text:
        add_check(checks, "quality_detail", "detailed face", 0.5, "clip_absolute")

    negative_checks = [
        {
            "type": "text_or_watermark",
            "expected_absent": True,
            "weight": 0.7,
            "method": "clip_contrastive"
        },
        {
            "type": "extra_people",
            "expected_absent": True,
            "weight": 0.9,
            "method": "clip_contrastive"
        }
    ]

    return {
        "image_index": image_index,
        "image_id": f"image_{image_index:03d}",
        "trigger_word": trigger_word,
        "raw_description": description,
        "checks": checks,
        "negative_checks": negative_checks
    }


def build_structured_eval_json(input_json_path: str, output_json_path: str):
    with open(input_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    descriptions = data.get("Descriptions", [])

    items = []
    for idx, desc in enumerate(descriptions):
        items.append(parse_description_to_checks(desc, idx))

    output = {
        "version": "phase0_eval_v1",
        "description_count": len(items),
        "items": items
    }

    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    return output


# -----------------------------
# CLIP evaluation
# -----------------------------

CANDIDATE_POOLS = {
    "people_count": [
        "one girl",
        "two girls",
        "one boy",
        "multiple people",
        "no people"
    ],
    "gender": [
        "girl",
        "boy",
        "man",
        "woman"
    ],
    "character_type": [
        "anime girl",
        "realistic girl",
        "cartoon animal",
        "landscape with no character"
    ],
    "hair_length": [
        "short hair",
        "medium-length hair",
        "long hair",
        "very long hair"
    ],
    "hair_color": [
        "dark brown hair",
        "black hair",
        "blonde hair",
        "pink hair",
        "blue hair",
        "white hair",
        "red hair"
    ],
    "hair_style": [
        "straight bangs",
        "face-framing hair strands",
        "ponytail",
        "twin tails",
        "braided hair",
        "messy hair"
    ],
    "eye_color": [
        "brown eyes",
        "blue eyes",
        "green eyes",
        "red eyes",
        "purple eyes",
        "black eyes"
    ],
    "outfit": [
        "school uniform",
        "casual hoodie",
        "casual sweater",
        "casual jacket",
        "casual clothes",
        "casual top and skirt",
        "hoodie and skirt",
        "dress"
    ],
    "expression": [
        "confident smile",
        "sad expression",
        "thoughtful expression",
        "emotional expression",
        "blushing",
        "determined expression",
        "surprised expression",
        "confident expression",
        "angry expression",
        "laughing brightly",
        "gentle smile",
        "serious expression",
        "calm expression",
        "crying softly",
        "shy smile",
        "reflective mood",
        "playful expression",
        "soft smile",
        "waiting expression",
        "annoyed expression"
    ],
    "pose_action": [
        "standing",
        "sitting",
        "sitting alone",
        "seated",
        "walking",
        "holding umbrella",
        "crossing arms",
        "running",
        "hand near face",
        "hands on hips",
        "reading a note",
        "looking out the window"
    ],
    "camera": [
        "close-up portrait",
        "close-up face",
        "half body portrait",
        "full body shot",
        "wide shot"
    ],
    "location": [
        "school hallway",
        "classroom",
        "classroom window",
        "rainy street",
        "warm cafe",
        "cozy cafe",
        "school rooftop",
        "school corridor",
        "bedroom",
        "school gate",
        "under a tree",
        "city street at night",
        "shrine path",
        "school courtyard",
        "train platform",
        "stairs",
        "cherry blossoms"
    ],
    "lighting": [
        "soft daylight",
        "soft afternoon light",
        "sunset lighting",
        "sunset sky",
        "cinematic lighting",
        "soft indoor lighting",
        "soft lighting",
        "bright daylight",
        "dramatic lighting",
        "dramatic soft lighting",
        "daylight",
        "soft natural lighting",
        "neon lights",
        "evening light",
        "soft moonlight",
        "warm lighting",
        "spring light"
    ],
    "prop": [
        "umbrella",
        "note",
        "desk",
        "no object"
    ],
}


QUERY_PREFIX = {
    "people_count": "anime illustration of {}",
    "gender": "anime illustration of a {}",
    "character_type": "{}",
    "hair_length": "anime girl with {}",
    "hair_color": "anime girl with {}",
    "hair_style": "anime girl with {}",
    "eye_color": "anime girl with {}",
    "outfit": "anime girl wearing {}",
    "expression": "anime girl with {}",
    "pose_action": "anime girl {}",
    "camera": "{} of an anime girl",
    "location": "anime girl in a {}",
    "lighting": "anime scene with {}",
    "prop": "anime girl with an {}",
}


def comfy_tensor_to_pil(image_tensor):
    """
    ComfyUI IMAGE is usually a torch tensor:
    [batch, height, width, channels], float32, range 0..1
    """

    if isinstance(image_tensor, torch.Tensor):
        img = image_tensor.detach().cpu()

        if img.ndim == 4:
            img = img[0]

        arr = img.numpy()
    else:
        arr = np.array(image_tensor)

    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)


def clip_contrastive_score(pil_image, check_type: str, expected: str):
    model, processor = get_clip()

    candidates = CANDIDATE_POOLS.get(check_type)
    if not candidates or expected not in candidates:
        return clip_absolute_score(pil_image, expected), expected, "clip_absolute_fallback"

    prefix = QUERY_PREFIX.get(check_type, "{}")
    texts = [prefix.format(c) for c in candidates]

    inputs = processor(
        text=texts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image[0].softmax(dim=-1)

    probs_list = probs.detach().cpu().tolist()
    expected_idx = candidates.index(expected)
    pred_idx = int(np.argmax(probs_list))

    return float(probs_list[expected_idx]), candidates[pred_idx], "clip_contrastive"


def clip_absolute_score(pil_image, expected: str):
    model, processor = get_clip()

    texts = [
        expected,
        f"anime illustration of {expected}",
        f"anime girl, {expected}"
    ]

    inputs = processor(
        text=texts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        image_features = model.get_image_features(pixel_values=inputs["pixel_values"])
        text_features = model.get_text_features(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask")
        )

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        cosine_scores = image_features @ text_features.T
        best_cosine = float(cosine_scores[0].max().detach().cpu())

    # CLIP cosine is not naturally 0..1 for this task.
    # This sigmoid makes it more useful for comparing runs.
    score = 1.0 / (1.0 + math.exp(-(best_cosine - 0.23) * 20.0))
    return float(score)


def evaluate_negative_check(pil_image, check):
    """
    Simple negative checks using CLIP contrast.
    This is not perfect, but useful for Phase 0.
    """

    check_type = check["type"]

    if check_type == "text_or_watermark":
        bad_texts = [
            "image with text",
            "image with watermark",
            "image with logo",
            "speech bubble with text"
        ]
        good_texts = [
            "clean anime image without text",
            "clean anime illustration without watermark"
        ]
    elif check_type == "extra_people":
        bad_texts = [
            "multiple people",
            "two girls",
            "group of people",
            "crowd"
        ]
        good_texts = [
            "one girl",
            "single anime girl",
            "solo anime girl"
        ]
    else:
        return 1.0, "unknown"

    model, processor = get_clip()
    texts = good_texts + bad_texts

    inputs = processor(
        text=texts,
        images=pil_image,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image[0].softmax(dim=-1)

    probs = probs.detach().cpu().numpy()
    good_score = float(probs[:len(good_texts)].sum())
    bad_score = float(probs[len(good_texts):].sum())

    # Expected absent: high score means likely clean.
    clean_score = good_score / max(good_score + bad_score, 1e-6)
    pred = "clean" if clean_score >= 0.5 else "artifact_possible"

    return clean_score, pred


def evaluate_image_against_item(pil_image, item):
    rows = []
    weighted_sum = 0.0
    total_weight = 0.0

    for check in item.get("checks", []):
        check_type = check["type"]
        expected = check["expected"]
        weight = float(check.get("weight", 1.0))
        method = check.get("method", "clip_contrastive")

        if method == "clip_contrastive":
            score, predicted, used_method = clip_contrastive_score(
                pil_image,
                check_type,
                expected
            )
        else:
            score = clip_absolute_score(pil_image, expected)
            predicted = expected
            used_method = "clip_absolute"

        weighted_sum += score * weight
        total_weight += weight

        rows.append({
            "type": check_type,
            "expected": expected,
            "predicted": predicted,
            "score": float(score),
            "weight": weight,
            "method": used_method
        })

    for check in item.get("negative_checks", []):
        weight = float(check.get("weight", 1.0))
        score, predicted = evaluate_negative_check(pil_image, check)

        weighted_sum += score * weight
        total_weight += weight

        rows.append({
            "type": check["type"],
            "expected": "absent",
            "predicted": predicted,
            "score": float(score),
            "weight": weight,
            "method": check.get("method", "clip_contrastive")
        })

    overall = weighted_sum / max(total_weight, 1e-6)

    return {
        "image_index": item["image_index"],
        "image_id": item["image_id"],
        "trigger_word": item.get("trigger_word", ""),
        "raw_description": item.get("raw_description", ""),
        "overall_score": float(overall),
        "checks": rows
    }


# -----------------------------
# ComfyUI nodes
# -----------------------------

class Phase0DescriptionJSONBuilder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_json_path": ("STRING", {
                    "default": "/workspace/manhwa_phase0/descriptions.json"
                }),
                "output_json_path": ("STRING", {
                    "default": "/workspace/manhwa_phase0/structured_eval.json"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("structured_json_path",)
    FUNCTION = "build"
    CATEGORY = "phase0/evaluation"

    def build(self, input_json_path, output_json_path):
        build_structured_eval_json(input_json_path, output_json_path)
        return (output_json_path,)


class Phase0ImageAttributeEvaluator:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "structured_json_path": ("STRING", {
                    "default": "/workspace/manhwa_phase0/structured_eval.json"
                }),
                "image_index": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 10000
                }),
                "wandb_project": ("STRING", {
                    "default": "manhwa_phase0_eval"
                }),
                "wandb_run_name": ("STRING", {
                    "default": "ayaka_lora_eval_v1"
                }),
                "save_report_path": ("STRING", {
                    "default": "/workspace/manhwa_phase0/eval_reports"
                }),
                "log_to_wandb": ("BOOLEAN", {
                    "default": True
                }),
            }
        }

    RETURN_TYPES = ("FLOAT", "STRING")
    RETURN_NAMES = ("overall_score", "report_json")
    FUNCTION = "evaluate"
    CATEGORY = "phase0/evaluation"

    def evaluate(
        self,
        image,
        structured_json_path,
        image_index,
        wandb_project,
        wandb_run_name,
        save_report_path,
        log_to_wandb
    ):
        with open(structured_json_path, "r", encoding="utf-8") as f:
            structured = json.load(f)

        items = structured.get("items", [])

        if image_index < 0 or image_index >= len(items):
            raise ValueError(
                f"image_index={image_index} out of range. "
                f"Structured JSON has {len(items)} items."
            )

        item = items[image_index]
        pil_image = comfy_tensor_to_pil(image)

        report = evaluate_image_against_item(pil_image, item)

        save_dir = Path(save_report_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        report_path = save_dir / f"eval_image_{image_index:03d}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        if log_to_wandb:
            run = get_wandb_run(wandb_project, wandb_run_name)

            log_data = {
                "image_index": image_index,
                "overall_score": report["overall_score"],
                "generated_image": wandb.Image(
                    pil_image,
                    caption=report.get("raw_description", "")
                )
            }

            for row in report["checks"]:
                key = f"attr/{row['type']}"
                log_data[key] = row["score"]

            # Also log full table
            table = wandb.Table(
                columns=[
                    "image_index",
                    "type",
                    "expected",
                    "predicted",
                    "score",
                    "weight",
                    "method"
                ]
            )

            for row in report["checks"]:
                table.add_data(
                    image_index,
                    row["type"],
                    row["expected"],
                    row["predicted"],
                    row["score"],
                    row["weight"],
                    row["method"]
                )

            log_data["attribute_table"] = table

            run.log(log_data, step=image_index)

        return (float(report["overall_score"]), json.dumps(report, indent=2))


NODE_CLASS_MAPPINGS = {
    "Phase0DescriptionJSONBuilder": Phase0DescriptionJSONBuilder,
    "Phase0ImageAttributeEvaluator": Phase0ImageAttributeEvaluator,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Phase0DescriptionJSONBuilder": "Phase0 Description JSON Builder",
    "Phase0ImageAttributeEvaluator": "Phase0 Image Attribute Evaluator",
}