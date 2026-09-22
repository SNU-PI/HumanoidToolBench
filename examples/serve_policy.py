"""Example HumanoidToolBench HTTP server. Pass --policy module:function or path/to/file.py:function to use your policy."""

from __future__ import annotations

import argparse
import queue
import threading
from concurrent.futures import Future
from http.server import ThreadingHTTPServer

import numpy as np

from humanoidtoolbench.policies.http_server import make_handler
from humanoidtoolbench.policies.loader import PolicyLoadError, load_policy_callable


def load_policy(spec: str):
    """Resolve --policy to a callable and its code digest, or exit with the reason."""
    try:
        return load_policy_callable(spec, with_digest=True)
    except PolicyLoadError as exc:
        raise SystemExit(str(exc)) from exc


class MainThreadCalls:
    """Run policy calls on the main thread, where the policy module was imported.

    The HTTP server answers /info on its own threads so that a busy server can
    still report its state, while predict keeps any thread-local setup made at
    import time, such as torch.set_grad_enabled(False).
    """

    def __init__(self) -> None:
        self.jobs: queue.Queue = queue.Queue()

    def __call__(self, function, request):
        done: Future = Future()
        self.jobs.put((function, request, done))
        return done.result()

    def run_forever(self) -> None:
        while True:
            try:
                function, request, done = self.jobs.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                done.set_result(function(request))
            except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
                done.set_exception(exc)


def predict(request: dict) -> np.ndarray:
    """Hold the observed pose. This is an integration example, not a task policy.

    A learned policy reads request['image']['rgb_head_stereo_left'],
    request['instruction'], and request['state']['states'] (shape 1 x 32).
    Reset recurrent state when request['history'].get('reset', False) is true.
    Return an unnormalized finite float array with shape (T, 36).
    """
    state = np.asarray(request["state"]["states"], dtype=np.float32)
    action = np.zeros((1, 36), dtype=np.float32)
    action[:, :32] = state
    return action


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address; use 0.0.0.0 when the evaluator runs on another machine",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=21000,
        help="Port to listen on. Default: %(default)s",
    )
    parser.add_argument(
        "--policy",
        help="Your predict function as module:function or path/to/file.py:function",
    )
    parser.add_argument(
        "--checkpoint", help="Checkpoint ID or revision to record in results"
    )
    parser.add_argument(
        "--allow-shared",
        action="store_true",
        help=(
            "Accept several evaluators at once. Only safe for a stateless policy; "
            "by default a second evaluator is refused while the first is active"
        ),
    )
    args = parser.parse_args()
    if args.policy:
        policy, digest = load_policy(args.policy)
    else:
        policy, digest = predict, None
    calls = MainThreadCalls()
    handler = make_handler(
        policy,
        args.policy or "example-hold",
        diagnostic=not args.policy,
        checkpoint=args.checkpoint,
        metadata={"adapter_sha256": digest} if args.policy else None,
        exclusive=not args.allow_shared,
        call=calls,
    )
    with ThreadingHTTPServer((args.host, args.port), handler) as server:
        server.daemon_threads = True
        print(f"Policy server: http://{args.host}:{server.server_port}", flush=True)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            calls.run_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.shutdown()


if __name__ == "__main__":
    main()
