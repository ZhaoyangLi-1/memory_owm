"""Prompts and JSON schemas for the GPT-6 Astra reference (spec 12.4, 12.5). Coordinates are [row, column]."""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw

SYSTEM = (
    "You are the high-level decision maker for a robot arm in a tabletop simulation.\n"
    "At each decision point you choose the next high-level action from a fixed list of options.\n"
    "Some actions need a target location. Answer only with JSON that matches the given schema."
)
_HEAD = ("Task instruction: {task_goal}\n\n"
         "The images below are frames from this episode so far, in chronological order (oldest first).\n"
         "The last image is the current front-camera view.")
_SETTING_A = ("In the current view, each object you may target is marked with a numbered circle.\n"
              "If the chosen action needs a target, give the number of the target object as \"candidate_id\";\n"
              "otherwise set \"candidate_id\" to null.")
_SETTING_B = ("If the chosen action needs a target, give its pixel location in the current view as \"point\": [row, column],\n"
              "where the image is 256 x 256 pixels, row 0 is the top edge and column 0 is the left edge.\n"
              "Otherwise set \"point\" to null.")


def options_text(options: list[dict]) -> str:
    lines = [f"- {o['label']}: {o['action']} (needs a target: {'yes' if o['need_parameter'] else 'no'})" for o in options]
    return "Available actions:\n" + "\n".join(lines)


def png_b64(frame: np.ndarray | Image.Image) -> str:
    img = frame if isinstance(frame, Image.Image) else Image.fromarray(np.asarray(frame))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def draw_candidates(frame_rgb: np.ndarray, cand_uv: np.ndarray, size: int = 256):
    """Numbered circle (radius 6) on every GT candidate object; the arm is not marked.
    Returns (image, {number -> candidate index})."""
    img = Image.fromarray(np.asarray(frame_rgb)).convert("RGB")
    draw = ImageDraw.Draw(img)
    mapping = {}
    for k, uv in enumerate(cand_uv, start=1):
        x, y = float(uv[0]) * size, float(uv[1]) * size
        draw.ellipse([x - 6, y - 6, x + 6, y + 6], outline=(255, 255, 0), width=2)
        draw.text((x + 7, y - 12), str(k), fill=(255, 255, 0))
        mapping[k] = k - 1
    return img, mapping


def build_input(task_goal: str, options: list[dict], images: list, setting: str) -> list[dict]:
    """Responses-API `input`: one user message with text + images (oldest first)."""
    content = [{"type": "input_text", "text": _HEAD.format(task_goal=task_goal)}]
    content += [{"type": "input_image", "image_url": f"data:image/png;base64,{png_b64(im)}"} for im in images]
    tail = options_text(options) + "\n\n" + (_SETTING_A if setting == "A" else _SETTING_B) + "\n\nWhich action should the robot take next?"
    content.append({"type": "input_text", "text": tail})
    return [{"role": "user", "content": content}]


def schema(options: list[dict], setting: str, relaxed: bool = False) -> dict:
    labels = [o["label"] for o in options]
    if setting == "A":
        props = {"choice": {"type": "string", "enum": labels}, "candidate_id": {"type": ["integer", "null"]}}
        req = ["choice", "candidate_id"]
    else:
        item = {"type": "integer"} if relaxed else {"type": "integer", "minimum": 0, "maximum": 255}
        point = {"type": ["array", "null"], "items": item}
        if not relaxed:
            point.update(minItems=2, maxItems=2)
        props = {"choice": {"type": "string", "enum": labels}, "point": point}
        req = ["choice", "point"]
    return {"type": "object", "properties": props, "required": req, "additionalProperties": False}
