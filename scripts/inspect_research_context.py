#!/usr/bin/env python3
"""Repackage one saved research request offline; never call models or simulators."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simex.research_context import ResearchContext, MAX_PROMPT_CHARS, prepare_edit_context
from simex.storage import atomic_json


def inspect(request_path, output):
    request = json.loads(Path(request_path).read_text())
    header, body = request["prompt"].split("INPUT JSON:\n", 1)
    payload, _ = json.JSONDecoder().raw_decode(body)
    header = header.split("\n\nAttached images, in order")[0]
    # Legacy requests put every image filename in their header. New packets use
    # labels and IDs matching only the attachments actually supplied that turn.
    header, pinned_keys, _ = prepare_edit_context(request["role"], header, payload)
    context = ResearchContext(payload, request.get("images", []), instruction=header, pinned_keys=pinned_keys)
    output = Path(output)
    manifest = context.persist(output / "context")
    prompt, images = context.prompt(header)
    atomic_json(output / "preview_request.json", {"role": request["role"], "prompt": prompt, "images": images})
    result = {"source_request": str(Path(request_path).resolve()),
              "old_prompt_chars": len(request["prompt"]), "new_prompt_chars": len(prompt),
              "reduction_percent": round(100*(1-len(prompt)/len(request["prompt"])), 2),
              "old_images": len(request.get("images", [])), "initial_images": len(images),
              "available_images": manifest["image_count"], "assets": len(manifest["assets"]),
              "max_prompt_chars": MAX_PROMPT_CHARS, "model_calls": 0, "rollouts": 0}
    atomic_json(output / "summary.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(inspect(args.request, args.out), ensure_ascii=False, indent=2))
