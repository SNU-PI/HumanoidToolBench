"""Local HTTP transport shared by automatic model loading and policy examples."""

from __future__ import annotations

import json
import os
import sysconfig
import time
import traceback
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Lock, Thread
from typing import Any, Callable

import numpy as np

from humanoidtoolbench.policies.http_client import _decode, _encode

# An evaluator that has not sent a request for this long is treated as gone,
# so a server can be reused after an evaluator that exited without releasing
# its session, for example after a crash. A normal exit releases it at once.
SESSION_IDLE_SECONDS = 60.0
ANONYMOUS_SESSION = "anonymous"

_LIBRARY_ROOTS = tuple(
    os.path.abspath(path) + os.sep
    for key in ("stdlib", "platstdlib", "purelib", "platlib")
    if (path := sysconfig.get_paths().get(key))
)
_THIS_FILE = os.path.abspath(__file__)


class ActionContractError(ValueError):
    """The policy returned an array that violates the (T, 36) finite contract."""


def _where(exc: BaseException) -> str:
    """Name the innermost frame in the user's code, for one-line error reports.

    Frames in the standard library, installed packages and this module are
    skipped, so the location points at the adapter rather than at numpy or
    torch internals.
    """
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        path = os.path.abspath(frame.filename)
        if path == _THIS_FILE or path.startswith(_LIBRARY_ROOTS):
            continue
        return f" at {Path(path).name}:{frame.lineno} in {frame.name}"
    return ""


def describe_error(exc: BaseException) -> str:
    """One line naming the exception, where it happened, and its message.

    The location comes first so that it survives truncation of long messages.
    """
    where = "" if isinstance(exc, ActionContractError) else _where(exc)
    return f"{type(exc).__name__}{where}: {exc}"


def _check_action(action: np.ndarray) -> None:
    if action.ndim != 2 or action.shape[1] != 36 or len(action) == 0:
        raise ActionContractError(
            f"Policy must return an array of shape (T, 36) with T >= 1; got shape "
            f"{action.shape}"
        )
    if not np.isfinite(action).all():
        rows = sorted({int(row) for row in np.argwhere(~np.isfinite(action))[:, 0]})
        raise ActionContractError(
            f"Policy returned NaN or infinite values in {len(rows)} of "
            f"{len(action)} rows (first row {rows[0]})"
        )


def make_handler(
    policy,
    name: str,
    *,
    diagnostic: bool = False,
    checkpoint: str | None = None,
    metadata: dict | None = None,
    exclusive: bool = True,
    call: Callable[[Callable, dict], Any] | None = None,
):
    """Build a request handler that serves ``policy`` on /act and /info.

    With ``exclusive`` the server accepts one evaluator at a time. Each
    evaluator process sends a session identifier; a request from a different
    session is refused with HTTP 409 while the current one has sent a request
    in the last ``SESSION_IDLE_SECONDS`` and has not released it through
    ``POST /release``. Requests without a session count as one anonymous
    session. Stateful policies would otherwise have their episode history reset
    by the other evaluator. Policy calls never overlap; ``call(policy, request)``
    lets the caller choose the thread they run on.
    """
    lock = Lock()
    active = {"session": None, "seen": 0.0, "busy": False}
    call = call or (lambda function, request: function(request))

    def owner() -> str | None:
        if active["session"] is None:
            return None
        if active["busy"] or time.monotonic() - active["seen"] < SESSION_IDLE_SECONDS:
            return active["session"]
        return None

    class Handler(BaseHTTPRequestHandler):
        def send_json(self, status: int, payload: dict) -> None:
            data = json.dumps(_encode(payload), allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def read_json(self) -> dict | None:
            try:
                length = int(self.headers.get("Content-Length") or 0)
                request = _decode(json.loads(self.rfile.read(length)))
                if not isinstance(request, dict):
                    raise ValueError("the body must be a JSON object")
            except Exception as exc:
                self.send_json(400, {"error": f"Malformed request: {exc}"})
                return None
            return request

        def do_GET(self) -> None:
            if self.path != "/info":
                self.send_error(404)
                return
            current = owner() if exclusive else None
            self.send_json(
                200,
                {
                    "policy": name,
                    "action_schema": "decoupled_v1",
                    "diagnostic": diagnostic,
                    "checkpoint": checkpoint,
                    **(metadata or {}),
                    # Live state, not provenance; evaluators drop it from results.
                    "server_state": {
                        "exclusive": exclusive,
                        "active_session": current,
                        "request_in_progress": bool(current and active["busy"]),
                        "seconds_since_request": (
                            round(time.monotonic() - active["seen"], 1)
                            if current
                            else None
                        ),
                    },
                },
            )

        def do_POST(self) -> None:
            if self.path not in ("/act", "/release"):
                self.send_error(404)
                return
            request = self.read_json()
            if request is None:
                return
            session = request.get("session") or ANONYMOUS_SESSION
            if self.path == "/release":
                if active["session"] == session and not active["busy"]:
                    active["session"] = None
                self.send_json(200, {"released": active["session"] is None})
                return
            with lock:
                current = owner()
                if exclusive and current not in (None, session):
                    waited = time.monotonic() - active["seen"]
                    self.send_json(
                        409,
                        {
                            "error": (
                                "This policy server is serving another evaluator "
                                f"(last request {waited:.0f} s ago). Run one server "
                                "per evaluator on its own --port; a shared server "
                                "would mix the evaluators' episode state."
                            )
                        },
                    )
                    return
                active.update(session=session, seen=time.monotonic(), busy=True)
                try:
                    action = np.asarray(call(policy, request), dtype=np.float32)
                    _check_action(action)
                except Exception as exc:
                    # The evaluator receives one line; the full traceback stays
                    # with the server (its terminal, or the run log for --policy).
                    traceback.print_exc()
                    self.send_json(500, {"error": describe_error(exc)})
                    return
                finally:
                    active.update(seen=time.monotonic(), busy=False)
            self.send_json(200, {"action": action})

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
    # One request at a time on one thread: shutdown() then returns only after
    # an in-flight policy call has finished, so conditions never overlap.
    with HTTPServer(("127.0.0.1", 0), handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            yield server.server_port
        finally:
            server.shutdown()
            thread.join()
