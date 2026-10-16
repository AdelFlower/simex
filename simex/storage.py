import hashlib
import json
import os
from pathlib import Path


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def toolbox_hash(directory):
    h = hashlib.sha256()
    for name in ('toolbox.py', 'skill.md'):
        h.update(name.encode())
        h.update((Path(directory) / name).read_bytes())
    return h.hexdigest()
