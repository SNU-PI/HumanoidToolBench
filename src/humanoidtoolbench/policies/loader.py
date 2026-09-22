"""Load a user's ``predict(request)`` callable for in-process evaluation."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Callable


class PolicyLoadError(ValueError):
    """A ``module:function`` or ``path.py:function`` spec could not be resolved."""


def load_policy_callable(spec: str) -> Callable[[dict], Any]:
    """Resolve ``module:function`` or ``path/to/file.py:function`` to a callable.

    Modules are looked up in the current directory first, so a ``my_policy.py``
    next to the checkout works without setting PYTHONPATH. A missing dependency
    inside the user's module propagates as its own ``ModuleNotFoundError``.
    """
    module_name, separator, function_name = spec.partition(":")
    if not separator or not module_name or not function_name:
        raise PolicyLoadError(
            f"--policy expects module:function, for example my_policy:predict (got {spec!r})"
        )
    if module_name.endswith(".py"):
        path = Path(module_name).resolve()
        if not path.is_file():
            raise PolicyLoadError(f"--policy file does not exist: {path}")
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
            raise PolicyLoadError(
                f"Cannot import {module_name!r}: {exc}. Run from the directory that "
                f"contains {top}.py or {top}/, or pass the file as "
                f"--policy path/to/file.py:{function_name}"
            ) from exc
    try:
        policy = getattr(module, function_name)
    except AttributeError as exc:
        raise PolicyLoadError(
            f"{module_name} has no attribute {function_name!r}"
        ) from exc
    if not callable(policy):
        raise PolicyLoadError(f"{spec} is not callable")
    print(f"Policy: {spec} ({getattr(module, '__file__', module_name)})", flush=True)
    return policy


class CallablePolicy:
    """Adapt a ``predict(request)`` callable to the managed-server interface.

    The evaluator serves it on a local port exactly as it serves a native
    ``--model`` checkpoint, so the request and response contract is the one
    documented for ``examples/serve_policy.py``.
    """

    def __init__(
        self,
        predict: Callable[[dict], Any],
        *,
        name: str,
        checkpoint: str | None = None,
    ) -> None:
        self.predict = predict
        self.metadata: dict[str, Any] = {
            "policy": name,
            "action_schema": "decoupled_v1",
            "diagnostic": False,
            "checkpoint": checkpoint,
        }

    def __call__(self, request: dict) -> Any:
        return self.predict(request)


__all__ = ["CallablePolicy", "PolicyLoadError", "load_policy_callable"]
