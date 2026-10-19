"""Bounded, lossless access to evidence already approved by the research harness.

This broker is deliberately not a filesystem tool. It can serve only values in
the supplied public payload and the explicitly supplied image attachments. Raw
artifacts stay on disk; references, short excerpts and selected pages reach the
model. A reference never authorizes opening a path supplied by the model.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

from .storage import atomic_json

VERSION = 1
MAX_PROMPT_CHARS = 64_000
SUMMARY_CHARS = 34_000
READ_CHARS = 16_000
MAX_READ_ROUNDS = 4
MAX_READ_REQUESTS = 4
MAX_IMAGES = 4
MAX_NOTES_CHARS = 2_000
MAX_PINNED_CHARS = 30_000
RESEARCH_ROLES = frozenset(("discover_probe", "discover_focus", "discover_revision",
    "discover_select", "align", "repair_hypotheses", "repair", "screen_tasks", "repair_select"))

ACCESS_INSTRUCTIONS = """
Research evidence access (not the one-shot robot policy protocol):
INPUT JSON contains a bounded overview. Fields with evidence_id are excerpts or
references, NOT complete evidence. All rollouts retain their status and scores;
omitted text is available through the read protocol, not through filesystem tools.
To inspect details instead of making your decision yet, return ONLY:
{"evidence_requests":[{"id":"text_...","offset":0,"limit":8000}],"notes":"concise findings to carry forward"}
For an image use {"id":"image_..."}. For text you may supply "find":"literal
substring" to locate a relevant section; the returned page starts at that match.
Use only listed IDs, at most 4 requests per round and 16000 returned JSON characters
per round. Full code may span multiple pages. Do not reconstruct omitted code from
an excerpt: read the relevant implementation before revising it. Distinguish an
excerpt from a full execution log, and an unviewed image from visual evidence.
There are at most 4 read rounds, then a final decision. The latest notes (at most
2000 characters) and latest read results replace earlier pages; they do not append
unbounded history. Retain your findings in notes. Every call counts toward the
model budget. Reads never generate a policy or execute an experiment.
When ready, return the originally requested decision JSON without evidence_requests.
action_summary describes public commands and robot motion only. A close command,
TCP rise, or subsequent opening does not establish object contact, a grasp, or
placement success. tcp_rise_after_close_m stops at the next opening command;
last_motion_rejection with source=policy_stdout is a reported diagnosis, while
source=action_result is an API failure flag that measured-pose checking may accept.
Use the independent task outcome and phase images to assess the manipulation.
""".strip()

EDIT_INSTRUCTIONS = """
Output override for a capability library too large to include in full:
Read the relevant source pages by evidence_id, then return
{"edits":[{"file":"toolbox.py","old":"exact existing source block","new":"replacement block"}],"hypothesis":"..."}.
You may edit only toolbox.py and skill.md, with at most 16 edits. Each nonempty
old block must match EXACTLY ONCE in the current file after previous edits.
The harness applies these replacements to the complete current files and then
validates the resulting library. Do not return reconstructed whole files from
excerpts. A bad edit is an invalid candidate, not a reason to retry generation.
""".strip()


def prepare_edit_context(role, instruction, payload):
    """Full small libraries; exact block edits for larger libraries."""
    if role not in ("discover_revision", "repair"):
        return instruction, (), False
    if len(dumps(payload.get("toolbox", {}))) <= MAX_PINNED_CHARS:
        return instruction, ("toolbox",), False
    return instruction + "\n\n" + EDIT_INSTRUCTIONS, (), True


def apply_source_edits(response, current):
    """Materialize a bounded edit response without exposing host filesystem APIs."""
    edits = response.get("edits")
    if (not isinstance(edits, list) or not 1 <= len(edits) <= 16 or
            any(name in response for name in ("toolbox.py", "skill.md"))):
        raise ContextError("Expected 1 to 16 source edits, without whole-file replacements")
    files = {name: current[name] for name in ("toolbox.py", "skill.md")}
    for edit in edits:
        if not isinstance(edit, dict) or set(edit) != {"file", "old", "new"}:
            raise ContextError("Each edit requires exactly file, old and new")
        name, old, new = edit["file"], edit["old"], edit["new"]
        if not isinstance(name, str) or name not in files:
            raise ContextError("Edits may change only toolbox.py and skill.md")
        if not isinstance(old, str) or not old or not isinstance(new, str):
            raise ContextError("Source edit requires nonempty old text and string new text")
        if files[name].count(old) != 1:
            raise ContextError("Source edit old text must match exactly once: " + name)
        files[name] = files[name].replace(old, new, 1)
    return {**files, "hypothesis": response.get("hypothesis"), "source_edits": edits}


class ContextError(RuntimeError):
    pass


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def _preview(text, limit=480):
    if len(text) <= limit:
        return text
    head = limit // 4
    return text[:head] + "\n[... excerpt; read by evidence_id ...]\n" + text[-(limit-head):]


def _action_preview(action):
    """Measured endpoints, never an inferred grasp/contact or private score."""
    if not isinstance(action, dict):
        return action
    result = {key: action[key] for key in ("method", "error", "result") if key in action}
    result["arguments"] = action.get("kwargs", action.get("arguments", {}))
    for label, state in (("before", action.get("start", action.get("before", {}))),
                         ("after", action.get("end", action.get("after", {})))):
        result[label] = {key: state[key] for key in ("sim_time", "ee_position", "gripper_width")
                         if key in state}
    return result


class ResearchContext:
    def __init__(self, payload, image_paths=(), *, pinned_keys=(), instruction=""):
        self.assets = {}
        self.image_ids = []
        self._image_by_path = {}
        self._seen_text = {}
        for source in image_paths:
            path = Path(source)
            # The caller supplies an allowlist produced from public evidence.
            # Once snapshotted, requests use IDs, never these source paths.
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            identifier = "image_" + digest[:32]
            self._image_by_path[str(source)] = identifier
            if identifier not in self.assets:
                self.assets[identifier] = {"kind": "image", "sha256": digest,
                    "source": str(path), "suffix": path.suffix, "data": data, "labels": []}
                self.image_ids.append(identifier)
            elif self.assets[identifier]["sha256"] != digest:
                raise ContextError("Evidence ID collision")
        self.raw_chars = len(dumps(payload))
        self.input_sha256 = sha(dumps({"version": VERSION, "payload": payload,
            "pinned_keys": list(pinned_keys),
            "images": [(p, self.assets[i]["sha256"]) for p, i in self._image_by_path.items()]}))
        # Revisions return whole files. Keep their current implementation EXACT
        # in every turn, rather than asking a fresh call to reconstruct code
        # from notes about previously read pages.
        self.pinned = {key: payload[key] for key in pinned_keys if key in payload}
        for key, files in self.pinned.items():
            for name, source in files.items():
                if isinstance(source, str):
                    self._seen_text[sha(source)] = "input." + key + "." + name
        self.summary = self._walk({key: value for key, value in payload.items() if key not in self.pinned}, "input")
        summary_budget = min(SUMMARY_CHARS, MAX_PROMPT_CHARS - len(dumps(self.pinned)) -
                             len(instruction) - len(ACCESS_INSTRUCTIONS) - 12_000)
        if summary_budget < 2_000:
            raise ContextError("Current capability source leaves insufficient room for research evidence")
        # Keep overview structure/outcomes intact. Progressively externalize large
        # leaf strings, rather than silently dropping trials or slicing JSON.
        while len(dumps(self.summary)) > summary_budget:
            candidates = []
            def collect(value, label):
                if isinstance(value, dict) and "evidence_id" not in value:
                    for key, child in value.items():
                        if isinstance(child, str) and len(child) > 350:
                            candidates.append((len(child), value, key, label + "." + key))
                        else:
                            collect(child, label + "." + key)
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        collect(child, f"{label}[{index}]")
            collect(self.summary, "input")
            if not candidates:
                rollouts = []
                def collect_rollouts(value):
                    if isinstance(value, dict):
                        if "full_result" in value and not value.get("brief_overview"):
                            rollouts.append(value)
                        else:
                            for child in value.values():
                                collect_rollouts(child)
                    elif isinstance(value, list):
                        for child in value:
                            collect_rollouts(child)
                collect_rollouts(self.summary)
                if rollouts:
                    value = max(rollouts, key=lambda item: len(dumps(item)))
                    self._brief_rollout(value)
                    continue
                # Excerpts are convenience only; shrink them before refusing an
                # oversized index. The exact blob remains available by its ID.
                def excerpts(value):
                    if isinstance(value, dict):
                        if "evidence_id" in value:
                            for key in ("preview", "last_diagnostic_line", "selected_endpoints"):
                                child = value.get(key)
                                if child is not None and len(dumps(child)) > 180:
                                    candidates.append((len(dumps(child)), value, key, None))
                        else:
                            for child in value.values():
                                excerpts(child)
                    elif isinstance(value, list):
                        for child in value:
                            excerpts(child)
                excerpts(self.summary)
                if not candidates:
                    raise ContextError("Research overview exceeds its character budget; refusing an oversized model request")
            _, parent, key, label = max(candidates, key=lambda item: item[0])
            parent[key] = (self._text_ref(parent[key], label, preview=120) if label is not None else
                           _preview(parent[key] if isinstance(parent[key], str) else dumps(parent[key]), 100))

    def _text_ref(self, text, label, preview=480, **extra):
        digest = sha(text)
        identifier = "text_" + digest[:32]
        asset = self.assets.setdefault(identifier, {"kind": "text", "sha256": digest,
                                                   "text": text, "labels": []})
        if asset["sha256"] != digest:
            raise ContextError("Evidence ID collision")
        if label not in asset["labels"]:
            asset["labels"].append(label)
        result = {"evidence_id": identifier, "chars": len(text), "excerpt": True, **extra}
        if preview:
            result["preview"] = _preview(text, preview)
        return result

    def _walk(self, value, label):
        key = label.rsplit(".", 1)[-1]
        if isinstance(value, str):
            if value in self._image_by_path:
                identifier = self._image_by_path[value]
                self.assets[identifier]["labels"].append(label)
                return {"evidence_id": identifier, "kind": "image", "label": label}
            if len(value) > 800:
                digest = sha(value)
                if digest in self._seen_text:
                    return self._text_ref(value, label, preview=0, same_as=self._seen_text[digest])
                self._seen_text[digest] = label
            threshold = 800 if key == "stdout" else 2_500
            # One full current implementation is useful if it fits. Repeated
            # toolboxes and larger versions are still addressable as exact text.
            if key in ("toolbox.py", "skill.md"):
                threshold = 24_000
            if len(value) > threshold:
                result = self._text_ref(value, label, preview=320)
                if key == "stdout":
                    matches = list(re.finditer(r"(?im)^.*(?:STOP|FAIL|ERROR|ABORT|UNREACHABLE).*$", value))
                    if matches:
                        result["last_diagnostic_line"] = matches[-1].group(0)[:320]
                return result
            return value
        if isinstance(value, list):
            if key in ("prior_attempts", "recent_research_history", "task_history",
                       "historical_replay_anchors"):
                return [self._history_record(child, f"{label}[{index}]")
                        if isinstance(child, dict) and len(dumps(child)) > 10_000 else
                        self._walk(child, f"{label}[{index}]") for index, child in enumerate(value)]
            if key in ("actions", "trace_tail", "robot_trace_tail") and len(dumps(value)) > 1_200:
                indices = sorted(set([0, max(0, len(value)-2), len(value)-1] +
                    [i for i, a in enumerate(value) if isinstance(a, dict) and a.get("error")][:1]))
                selected = [{"index": i, "record": _action_preview(value[i])} for i in indices if i >= 0]
                result = self._text_ref(dumps(value), label, preview=0, count=len(value))
                result["selected_endpoints"] = selected if len(dumps(selected)) < 1_600 else _preview(dumps(selected), 900)
                return result
            return [self._walk(child, f"{label}[{index}]") for index, child in enumerate(value)]
        if isinstance(value, dict):
            if key == "scene" and len(dumps(value)) > 500:
                return self._text_ref(dumps(value), label, preview=0)
            if "status" in value and "evidence" in value and len(dumps(value)) > 4_000:
                return self._rollout(value, label)
            if key == "motion_summary" and len(dumps(value)) > 800:
                result = self._text_ref(dumps(value), label, preview=0)
                result["overview"] = {k: v for k, v in value.items() if k not in ("initial", "final")}
                for state in ("initial", "final"):
                    result["overview"][state] = {k: v for k, v in value.get(state, {}).items()
                        if k in ("sim_time", "ee_position", "gripper_width")}
                return result
            return {key: self._walk(child, label + "." + key) for key, child in value.items()}
        return value

    def _history_record(self, value, label):
        record = {"full_history": self._text_ref(dumps(value), label, preview=0)}
        for key in ("cycle", "revision", "trial", "hypothesis", "weakness", "accepted", "agent_accept",
                    "reason", "gate_reason", "history_reason", "selected", "target_success", "target_status",
                    "validation", "regression", "task_sha256", "successful_toolbox_sha256"):
            if key in value:
                record[key] = self._walk(value[key], label + "." + key)
        outcomes, images = [], []
        def visit(child, location, inside_outcome=False):
            if isinstance(child, dict):
                is_outcome = "success" in child and "status" in child
                if is_outcome and not inside_outcome:
                    outcomes.append({"location": location, **{k: child[k] for k in
                        ("status", "success", "steps") if k in child}})
                for key, nested in child.items():
                    visit(nested, location + "." + key, inside_outcome or is_outcome)
            elif isinstance(child, list):
                for index, nested in enumerate(child):
                    visit(nested, f"{location}[{index}]", inside_outcome)
            elif isinstance(child, str) and child in self._image_by_path:
                images.append(self._walk(child, label + "." + location))
        visit(value, "record")
        if "attempts" not in value:
            record["outcomes"] = outcomes
        if images and "attempts" not in value:
            record["images"] = [{"evidence_id": identifier, "kind": "image"}
                                for identifier in dict.fromkeys(image["evidence_id"] for image in images)]
        if "attempts" in value:
            record["attempts"] = [self._history_record(attempt, f"{label}.attempts[{index}]")
                for index, attempt in enumerate(value["attempts"])]
        return record

    def _rollout(self, value, label):
        """One overview per trial; no score-based selection or lost failures."""
        evidence = value.get("evidence", {})
        result = {key: value[key] for key in ("status", "success", "steps") if key in value}
        result["full_result"] = self._text_ref(dumps(value), label, preview=0)
        visible = {key: self._walk(evidence[key], label + ".evidence." + key)
                   for key in ("reason", "action_summary", "recording_errors", "evidence_errors",
                               "phase_images", "keyframes", "read_images") if key in evidence}
        stdout = evidence.get("stdout", "")
        if stdout:
            visible["stdout"] = (stdout if len(stdout) <= 400 else
                self._text_ref(stdout, label + ".evidence.stdout", preview=240))
            if isinstance(visible['stdout'], dict):
                matches = list(re.finditer(r"(?im)^.*(?:STOP|FAIL|ERROR|ABORT|UNREACHABLE).*$", stdout))
                if matches:
                    visible['stdout']['last_diagnostic_line'] = matches[-1].group(0)[:320]
        motion = evidence.get("motion_summary", {})
        if motion:
            visible["motion_summary"] = {key: motion[key] for key in
                ("samples", "sim_time_range", "ee_position_bounds", "gripper_width_range") if key in motion}
        actions = evidence.get("actions", [])
        if actions:
            visible["actions"] = self._text_ref(dumps(actions), label + ".evidence.actions", preview=0,
                count=len(actions), last_endpoint=_action_preview(actions[-1]))
        result["evidence"] = visible
        if "execution" in value:
            execution = {}
            for key, child in value["execution"].items():
                child_label = label + ".execution." + key
                if isinstance(child, str) and len(child) > 600:
                    execution[key] = self._text_ref(child, child_label, preview=160 if key == "plan" else 80)
                else:
                    execution[key] = self._walk(child, child_label)
            result["execution"] = execution
        return result

    @staticmethod
    def _brief_rollout(value):
        evidence = value["evidence"]
        value["evidence"] = {key: child for key, child in evidence.items() if key in
            ("reason", "action_summary", "phase_images", "keyframes", "read_images",
             "recording_errors", "evidence_errors", "stdout")}
        stdout = value["evidence"].get("stdout")
        if isinstance(stdout, dict):
            value["evidence"].pop("stdout")
            if stdout.get('last_diagnostic_line'):
                value['evidence']['last_diagnostic_line'] = stdout['last_diagnostic_line']
            elif stdout.get("preview"):
                value["evidence"]["stdout_tail_excerpt"] = stdout["preview"][-80:]
        for key in ("phase_images", "keyframes", "read_images"):
            if key in value["evidence"]:
                value["evidence"][key] = [{k: v for k, v in image.items() if k != "label"}
                    if isinstance(image, dict) else image for image in value["evidence"][key]]
        if "execution" in value:
            value["execution"] = {key: child for key, child in value["execution"].items()
                                  if key in ("actual_started", "script")}
            script = value["execution"].get("script")
            if isinstance(script, dict):
                value["execution"]["script"] = {"evidence_id": script["evidence_id"]}
        value["brief_overview"] = True  # omitted fields are in full_result

    def persist(self, directory):
        """Freeze approved blobs independently of the model's temporary sandbox."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        index = {}
        for identifier, asset in self.assets.items():
            suffix = asset.get("suffix", ".txt")
            path = self.directory / (identifier + suffix)
            data = asset["data"] if asset["kind"] == "image" else asset["text"].encode()
            if path.exists() and path.read_bytes() != data:
                raise ContextError("Frozen research evidence blob changed")
            if not path.exists():
                temporary = path.with_name(path.name + ".tmp")
                with temporary.open("wb") as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                temporary.replace(path)
            index[identifier] = {key: value for key, value in asset.items() if key not in ("data", "text")}
            index[identifier].update(file=path.name, chars=len(asset.get("text", "")))
        manifest = {"version": VERSION, "input_sha256": self.input_sha256,
                    "raw_payload_chars": self.raw_chars, "summary_chars": len(dumps(self.summary)),
                    "pinned_chars": len(dumps(self.pinned)),
                    "overview_sha256": sha(dumps({**self.summary, **self.pinned})),
                    "image_count": len(self.image_ids), "assets": index}
        target = self.directory / "manifest.json"
        if target.exists() and json.loads(target.read_text()) != manifest:
            raise ContextError("Research evidence checkpoint input changed")
        atomic_json(target, manifest)
        atomic_json(self.directory / "overview.json", {**self.summary, **self.pinned})
        return manifest

    def image_paths(self, identifiers):
        return [str(self.directory / (identifier + self.assets[identifier]["suffix"]))
                for identifier in identifiers]

    def read(self, requests, budget=READ_CHARS):
        if not isinstance(requests, list) or not 1 <= len(requests) <= MAX_READ_REQUESTS:
            raise ContextError("evidence_requests must contain 1 to 4 reads")
        replies, images = [], []
        for request in requests:
            if not isinstance(request, dict) or set(request) - {"id", "offset", "limit", "find"}:
                raise ContextError("Read accepts only id, offset, limit and find")
            identifier = request.get("id")
            if not isinstance(identifier, str) or identifier not in self.assets:
                raise ContextError("Unknown evidence ID; paths and external files cannot be read")
            asset = self.assets[identifier]
            if asset["kind"] == "image":
                if set(request) != {"id"}:
                    raise ContextError("An image read accepts only id")
                if identifier not in images:
                    images.append(identifier)
                replies.append({"id": identifier, "kind": "image", "attached": True})
                continue
            content = asset["text"]
            offset, limit = request.get("offset", 0), request.get("limit", 8_000)
            if (type(offset) is not int or not 0 <= offset <= len(content) or
                    type(limit) is not int or not 1 <= limit <= READ_CHARS):
                raise ContextError("Text read requires a valid nonnegative offset and limit in [1,16000]")
            if "find" in request:
                query = request["find"]
                if not isinstance(query, str) or not 1 <= len(query) <= 200:
                    raise ContextError("find must be a literal substring of 1 to 200 characters")
                offset = content.find(query, offset)
                if offset < 0:
                    replies.append({"id": identifier, "found": False})
                    continue
            limit = min(limit, len(content) - offset)
            reply = {"id": identifier, "offset": offset, "end": offset,
                     "total_chars": len(content), "text": "", "complete": False}
            # Bound the serialized response too: escaped logs can be much larger
            # than the text slice itself. Leave room for later read metadata.
            allowance = budget - 1_000
            while True:
                reply.update(text=content[offset:offset+limit], end=offset+limit,
                             complete=offset == 0 and offset+limit == len(content))
                if len(dumps(replies + [reply])) <= allowance or limit == 0:
                    break
                limit = max(0, limit - max(1, len(dumps(replies + [reply])) - allowance))
            reply["page_limited"] = limit < min(request.get("limit", 8_000), len(content)-offset)
            replies.append(reply)
        if len(dumps(replies)) > budget:
            raise ContextError("Read metadata exceeds the evidence response budget")
        return replies, images

    def prompt(self, instruction, *, round_index=0, notes="", replies=None,
               reads_so_far=None, image_ids=None):
        image_ids = self.image_ids[:MAX_IMAGES] if image_ids is None else image_ids
        if len(image_ids) > MAX_IMAGES:
            raise ContextError("Too many research image attachments")
        meta = {"version": VERSION, "read_rounds_remaining": MAX_READ_ROUNDS-round_index,
                "attached_images": [{"image": i+1, "id": identifier,
                                     "labels": self.assets[identifier]["labels"][:4],
                                     "other_uses": max(0, len(self.assets[identifier]["labels"])-4)}
                                    for i, identifier in enumerate(image_ids)],
                "notes": notes[:MAX_NOTES_CHARS], "notes_truncated": len(notes) > MAX_NOTES_CHARS,
                "read_results": replies or [], "reads_so_far": reads_so_far or []}
        header = instruction + "\n\n" + ACCESS_INSTRUCTIONS
        if round_index == MAX_READ_ROUNDS:
            header += "\nNo read rounds remain. Return the final requested decision JSON now."
        prompt = header + "\n\nINPUT JSON:\n" + dumps({**self.summary, **self.pinned, "_research_context": meta})
        # Reserve space for the adapter's fixed instruction and JSON format rule.
        if len(prompt) > MAX_PROMPT_CHARS - 1_000:
            raise ContextError("Research prompt exceeds its character budget; no model call was made")
        return prompt, self.image_paths(image_ids)
