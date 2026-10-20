#!/usr/bin/env python3
"""Fetch the pinned, Apache-2.0 Menagerie Panda without hash verification."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

COMMIT = "62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35"
REPOSITORY = "google-deepmind/mujoco_menagerie"
PREFIX = "franka_emika_panda/"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=Path(__file__).resolve().parents[1] / "assets/panda")
    args = parser.parse_args()
    with urllib.request.urlopen(f"https://api.github.com/repos/{REPOSITORY}/git/trees/{COMMIT}?recursive=1") as response:
        tree = json.load(response)["tree"]
    entries = [item for item in tree if item["type"] == "blob" and item["path"].startswith(PREFIX)
               and (item["path"].startswith(PREFIX + "assets/") or item["path"][len(PREFIX):] in ("panda.xml", "README.md", "LICENSE", "CHANGELOG.md"))]

    def download(item: dict) -> tuple[str, str]:
        relative = item["path"][len(PREFIX):]
        destination = args.destination / relative
        content = destination.read_bytes() if destination.is_file() else b""
        if not destination.is_file():
            url = f"https://raw.githubusercontent.com/{REPOSITORY}/{COMMIT}/{item['path']}"
            with urllib.request.urlopen(url, timeout=90) as response:
                content = response.read()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
        return relative, hashlib.sha256(content).hexdigest()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        hashes = dict(executor.map(download, entries))
    provenance = {"repository": f"https://github.com/{REPOSITORY}", "commit": COMMIT,
                  "upstream_directory": PREFIX.rstrip("/"), "license": "Apache-2.0", "sha256": hashes}
    (args.destination / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Available: {len(hashes)} files at {args.destination} ({COMMIT})")


if __name__ == "__main__":
    main()
