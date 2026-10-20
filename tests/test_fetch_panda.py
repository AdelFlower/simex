"""Asset downloads use file presence, never upstream blob hashes."""
import io
import json
import sys

import pytest

from scripts import fetch_panda


@pytest.mark.parametrize("existing", [False, True])
def test_fetch_does_not_verify_upstream_hashes(tmp_path, monkeypatch, existing):
    target = tmp_path / "panda.xml"
    if existing:
        target.write_bytes(b"existing asset")
    calls = []

    def urlopen(url, **kwargs):
        calls.append(url)
        if "api.github.com" in url:
            return io.BytesIO(json.dumps({"tree": [{
                "type": "blob", "path": fetch_panda.PREFIX + "panda.xml",
                "sha": "deliberately incorrect upstream hash",
            }]}).encode())
        return io.BytesIO(b"downloaded asset")

    monkeypatch.setattr(fetch_panda.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(sys, "argv", ["fetch_panda", "--destination", str(tmp_path)])
    fetch_panda.main()
    assert target.read_bytes() == (b"existing asset" if existing else b"downloaded asset")
    assert len(calls) == (1 if existing else 2)
    assert (tmp_path / "PROVENANCE.json").exists()
