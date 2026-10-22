#!/usr/local/bin/python3
"""Usage: simex-research '{"op":"status","id":"unique-id"}'"""
import json
import socket
import sys

request = json.loads(sys.argv[1]) if len(sys.argv) == 2 else json.load(sys.stdin)
with socket.socket(socket.AF_UNIX) as connection:
    connection.connect('/control/research.sock')
    stream = connection.makefile('rwb')
    stream.write(json.dumps(request).encode() + b'\n')
    stream.flush()
    response = json.loads(stream.readline())
print(json.dumps(response, ensure_ascii=False))
raise SystemExit(0 if response.get('ok') else 1)
