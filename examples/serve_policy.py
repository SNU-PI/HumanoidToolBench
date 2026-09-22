"""Example HumanoidToolBench HTTP server. Pass --policy module:function or path/to/file.py:function to use your policy."""

from __future__ import annotations

import argparse
from http.server import HTTPServer

import numpy as np

from humanoidtoolbench.policies.http_server import make_handler
from humanoidtoolbench.policies.loader import PolicyLoadError, load_policy_callable


def load_policy(spec: str):
    """Resolve --policy to a callable, or exit with the reason."""
    try:
        return load_policy_callable(spec)
    except PolicyLoadError as exc:
        raise SystemExit(str(exc)) from exc


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
    args = parser.parse_args()
    policy = load_policy(args.policy) if args.policy else predict
    handler = make_handler(
        policy,
        args.policy or "example-hold",
        diagnostic=not args.policy,
        checkpoint=args.checkpoint,
    )
    with HTTPServer((args.host, args.port), handler) as server:
        print(f"Policy server: http://{args.host}:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
