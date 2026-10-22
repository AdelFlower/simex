"""Container-local HTTP adapter. The sole network exit is the mounted host socket."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import socket
import subprocess
import sys
import threading


class UnixConnection(http.client.HTTPConnection):
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect('/control/model.sock')


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_POST(self):
        conn = UnixConnection('localhost', timeout=3600)
        try:
            body = self.rfile.read(int(self.headers['Content-Length']))
            conn.request('POST', self.path, body, {'Content-Type': 'application/json',
                         'Authorization': 'Bearer ' + os.environ['SIMEX_GATEWAY_TOKEN']})
            response = conn.getresponse()
            data = response.read()
            self.send_response(response.status)
            self.send_header('Content-Type', response.getheader('Content-Type', 'application/json'))
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        finally:
            conn.close()


if __name__ == '__main__':
    server = ThreadingHTTPServer(('127.0.0.1', 8745), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        raise SystemExit(subprocess.call(sys.argv[1:], env=dict(os.environ)))
    finally:
        server.shutdown()
