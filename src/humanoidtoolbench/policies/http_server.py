"""Local HTTP transport shared by automatic model loading and policy examples."""

from __future__ import annotations

import json
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import numpy as np

from humanoidtoolbench.policies.http_client import _decode, _encode


def make_handler(
    policy,
    name: str,
    *,
    diagnostic: bool = False,
    checkpoint: str | None = None,
    metadata: dict | None = None,
):
    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(_encode(payload), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            if self.path != "/info":
                self.send_error(404)
                return
            self.send_json(
                200,
                {
                    "policy": name,
                    "action_schema": "decoupled_v1",
                    "diagnostic": diagnostic,
                    "checkpoint": checkpoint,
                    **(metadata or {}),
                },
            )

        def do_POST(self) -> None:
            if self.path != "/act":
                self.send_error(404)
                return
            try:
                request = _decode(
                    json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                )
                action = np.asarray(policy(request), dtype=np.float32)
                if (
                    action.ndim != 2
                    or action.shape[1] != 36
                    or len(action) == 0
                    or not np.isfinite(action).all()
                ):
                    raise ValueError(
                        "Policy must return finite actions with shape (T, 36)"
                    )
                self.send_json(200, {"action": action})
            except Exception as exc:
                self.send_json(500, {"error": str(exc)})

        def log_message(self, format: str, *args) -> None:
            pass

    return Handler


@contextmanager
def serve_policy(policy, *, seed_start: int):
    """Serve one condition, preserving per-episode inference RNG identities."""
    episode, query = seed_start - 1, 0

    def predict(request):
        nonlocal episode, query
        if request["history"].get("reset", False):
            episode += 1
            query = 0
        request = {
            **request,
            "humanoidtoolbench_episode_seed": episode,
            "humanoidtoolbench_request_index": query,
        }
        query += 1
        return policy(request)

    metadata = policy.metadata
    handler = make_handler(predict, metadata["policy"], metadata=metadata)
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port
        finally:
            server.shutdown()
            thread.join()
