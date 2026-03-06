"""
HTTP steering server — REST interface for pipeline live-steering.

Runs a lightweight HTTP server alongside the existing TCP socket, feeding
the same input_queue used by callback_tools.py. This lets OpenClaw and
other programmatic clients steer a running pipeline without maintaining a
persistent TCP connection.

Endpoints:
    POST /interrupt          — enqueue "interrupt" (triggers pause)
    POST /instruction        — body: {"text": "...", "type": "m"|"n"}
                               Enqueues the instruction text, a blank line,
                               another blank line (double-enter), then the m/n choice.
    GET  /status             — returns JSON: {"paused": bool, "queue_depth": int}

Usage (from runner.py or wherever the input_queue is created):
    from consortium.interaction.http_steering import add_http_steering
    add_http_steering(input_queue, host="127.0.0.1", port=5002)

OpenClaw usage:
    # Trigger a pause and inject an instruction:
    curl -s -X POST http://127.0.0.1:5002/interrupt
    curl -s -X POST http://127.0.0.1:5002/instruction \\
         -H "Content-Type: application/json" \\
         -d '{"text": "focus on linear case only", "type": "m"}'
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from queue import Queue
from typing import Optional


def add_http_steering(
    input_queue: Queue,
    host: str = "127.0.0.1",
    port: int = 5002,
) -> HTTPServer:
    """
    Start a background HTTP steering server that shares input_queue with
    the TCP socket server in callback_tools.py.

    Args:
        input_queue: The same Queue returned by setup_user_input_socket().
        host:        Bind address (default: localhost).
        port:        Port (default: 5002, one above the TCP socket's 5001).

    Returns:
        The HTTPServer instance (already running in a daemon thread).
    """

    class _Handler(BaseHTTPRequestHandler):
        queue = input_queue
        _paused = False

        def log_message(self, fmt, *args):
            # Suppress default access logs; use our own prefix
            pass

        def do_GET(self):
            if self.path == "/status":
                body = json.dumps({
                    "paused": _Handler._paused,
                    "queue_depth": self.queue.qsize(),
                }).encode()
                self._respond(200, body)
            else:
                self._respond(404, b'{"error":"not found"}')

        def do_POST(self):
            if self.path == "/interrupt":
                self.queue.put("interrupt")
                _Handler._paused = True
                print("[http_steering] Interrupt enqueued.")
                self._respond(200, b'{"ok":true}')

            elif self.path == "/instruction":
                length = int(self.headers.get("Content-Length", 0))
                raw = self.rfile.read(length)
                try:
                    payload = json.loads(raw)
                    text = payload.get("text", "").strip()
                    choice = payload.get("type", "m").strip().lower()
                    if choice not in ("m", "n"):
                        raise ValueError(f"type must be 'm' or 'n', got '{choice}'")
                    if not text:
                        raise ValueError("text must be non-empty")
                except (json.JSONDecodeError, ValueError) as e:
                    self._respond(400, json.dumps({"error": str(e)}).encode())
                    return

                # Enqueue instruction the same way a human would type it:
                # text lines, then two empty lines (double-enter), then m/n choice.
                for line in text.splitlines():
                    self.queue.put(line)
                self.queue.put("")   # first Enter
                self.queue.put("")   # second Enter (double-enter = end of instruction)
                self.queue.put(choice)
                _Handler._paused = False
                print(f"[http_steering] Instruction enqueued (type={choice}): {text[:80]}")
                self._respond(200, b'{"ok":true}')

            else:
                self._respond(404, b'{"error":"not found"}')

        def _respond(self, code: int, body: bytes) -> None:
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer((host, port), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"[Info] HTTP steering server listening on http://{host}:{port}")
    print(f"[Info]   POST /interrupt   — pause the pipeline")
    print(f"[Info]   POST /instruction — inject a steering instruction")
    print(f"[Info]   GET  /status      — check pause/queue state")
    return server
