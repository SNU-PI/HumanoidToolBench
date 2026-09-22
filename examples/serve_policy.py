"""Example HumanoidToolBench HTTP server. Pass --policy module:function or path/to/file.py:function to use your policy."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import os
import sys
from http.server import HTTPServer
from pathlib import Path

import numpy as np

from humanoidtoolbench.policies.http_server import make_handler


def load_policy(spec: str):
    """Resolve ``module:function`` or ``path/to/file.py:function`` to a callable.

    Modules are looked up in the current directory first, so a ``my_policy.py``
    next to the checkout works without setting PYTHONPATH.
    """
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise SystemExit(
            f"--policy expects module:function, for example my_policy:predict (got {spec!r})"
        )
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        if not path.is_file():
            raise SystemExit(f"--policy file does not exist: {path}")
        # Register under the file's own name so that worker processes and
        # pickles can import it, unless that name already resolves elsewhere
        # (for example a user file called json.py); then use a private name
        # rather than shadow the existing module.
        name = path.stem
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            name = f"humanoidtoolbench_policy_{path.stem}"
        else:
            sys.path.insert(0, str(path.parent))
        loader = importlib.util.spec_from_file_location(name, path)
        assert loader is not None and loader.loader is not None
        module = importlib.util.module_from_spec(loader)
        sys.modules[name] = module
        loader.loader.exec_module(module)
    else:
        sys.path.insert(0, os.getcwd())
        top = module_name.split(".")[0]
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            missing = exc.name or ""
            if not (missing == module_name or module_name.startswith(missing + ".")):
                raise  # The module exists; one of its own imports is missing.
            raise SystemExit(
                f"Cannot import {module_name!r}: {exc}. Run from the directory that "
                f"contains {top}.py or {top}/, or pass the file as "
                f"--policy path/to/file.py:{function_name}"
            ) from exc
    try:
        policy = getattr(module, function_name)
    except AttributeError as exc:
        raise SystemExit(f"{module_name} has no attribute {function_name!r}") from exc
    if not callable(policy):
        raise SystemExit(f"{spec} is not callable")
    print(f"Policy: {spec} ({getattr(module, '__file__', module_name)})", flush=True)
    return policy


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
