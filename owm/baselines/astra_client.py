"""One GPT-6 Astra call with structured output, retries, and usage / cost logging (spec 12.6).

Verified against the installed OpenAI SDK (openai 3.16): Responses API, `reasoning={"effort": ...}` (allowed: none,
minimal, low, medium, high, xhigh, max), `text={"format": {"type": "json_schema", "name", "schema", "strict"}}`,
images as {"type": "input_image", "image_url": "data:image/png;base64,..."}; the model id `gpt-6-astra` is in the SDK's
model list. NOT verifiable offline: how image tokens are billed — the cost below is computed from the token usage the
API reports and the list prices in experiment.yaml; check the first trial run against the dashboard.
"""
from __future__ import annotations

import json
import time

from owm.baselines import astra_prompts as P
from owm.config import load_cfg


def call_astra(task_goal: str, options: list[dict], images: list, setting: str, client=None) -> dict:
    cfg = load_cfg().astra
    if client is None:
        from openai import OpenAI
        client = OpenAI()
    inp = P.build_input(task_goal, options, images, setting)
    out = dict(parsed=None, raw=None, usage=None, cost_usd=None, error=None, attempts=0, relaxed_schema=False)
    relaxed = False
    for attempt in range(cfg.retries + 1):
        out["attempts"] = attempt + 1
        try:
            resp = client.responses.create(
                model=cfg.model, instructions=P.SYSTEM, input=inp, reasoning={"effort": cfg.reasoning_effort},
                text={"format": {"type": "json_schema", "name": "decision", "schema": P.schema(options, setting, relaxed),
                                 "strict": True}}, store=False)
            out["raw"] = resp.output_text
            u = resp.usage
            if u is not None:
                out["usage"] = dict(input_tokens=u.input_tokens, output_tokens=u.output_tokens, total_tokens=u.total_tokens)
                out["cost_usd"] = (u.input_tokens * cfg.price_per_mtok.input + u.output_tokens * cfg.price_per_mtok.output) / 1e6
            parsed = json.loads(resp.output_text)
            if parsed.get("choice") in [o["label"] for o in options]:
                out["parsed"], out["error"], out["relaxed_schema"] = parsed, None, relaxed
                return out
            out["error"] = f"invalid choice: {parsed!r}"
        except Exception as e:  # API error, schema rejected, malformed JSON ...
            out["error"] = f"{type(e).__name__}: {e}"
            if "schema" in str(e).lower() and not relaxed:
                relaxed = True   # strict mode may reject minItems / minimum: retry without them
            time.sleep(2.0 * (attempt + 1))
    return out
