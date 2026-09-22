"""Load a user's ``predict(request)`` callable for in-process evaluation."""

from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.util
import inspect
import os
import queue
import sys
import sysconfig
import threading
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable

_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_LIBRARY_ROOTS = tuple(
    {
        os.path.join(os.path.abspath(path), "")
        for path in (
            sys.prefix,
            sys.base_prefix,
            *(
                sysconfig.get_paths().get(key)
                for key in ("stdlib", "purelib", "platlib")
            ),
        )
        if path
    }
)


class PolicyLoadError(ValueError):
    """A ``module:function`` or ``path.py:function`` spec could not be resolved."""


def load_policy_callable(spec: str, *, with_digest: bool = False):
    """Resolve ``module:function`` or ``path/to/file.py:function`` to a callable.

    Modules are looked up in the current directory first, so a ``my_policy.py``
    next to the checkout works without setting PYTHONPATH. A missing dependency
    inside the user's module propagates as its own ``ModuleNotFoundError``.
    With ``with_digest`` it returns ``(policy, digest)``, where ``digest`` is
    ``adapter_sha256`` over the user code imported while loading.
    """
    before = set(sys.modules)
    policy = _resolve(spec)
    if not with_digest:
        return policy
    return policy, adapter_sha256(policy, set(sys.modules) - before)


def _resolve(spec: str) -> Callable[[dict], Any]:
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


def _user_file(module: Any) -> Path | None:
    """The source file of a module written by the user, or None for library code."""
    path = getattr(module, "__file__", None)
    if not path or not os.path.isfile(path):
        return None
    resolved = os.path.abspath(path)
    if resolved.startswith(_LIBRARY_ROOTS) or Path(resolved).is_relative_to(
        _PACKAGE_ROOT
    ):
        # Installed packages and this toolkit are covered by other hashes.
        return None
    return Path(resolved)


def _defining_module(policy: Callable) -> Any:
    target = policy
    while isinstance(target, functools.partial):
        target = target.func
    target = inspect.unwrap(target)
    if not (inspect.isfunction(target) or inspect.ismethod(target)):
        target = type(target)  # a callable instance
    return sys.modules.get(getattr(target, "__module__", "") or "")


def adapter_sha256(policy: Callable, imported: set[str] | None = None) -> str | None:
    """SHA-256 over the user's source files behind ``policy``, or None.

    Covers the module that defines ``policy`` (after unwrapping partials,
    decorators and callable instances) and every other user module in
    ``imported``, typically the modules imported while loading the adapter.
    Installed packages, the standard library and this toolkit are excluded.
    Files imported later, environment variables and weights are not covered.
    Recorded with results so that a changed adapter is not mistaken for the
    one that produced an earlier result.
    """
    modules = {name: sys.modules.get(name) for name in imported or ()}
    defining = _defining_module(policy)
    if defining is not None:
        modules[defining.__name__] = defining
    # Content only: the same files give the same digest however the module was
    # named when loaded (which depends on the working directory).
    hashes = sorted(
        {
            hashlib.sha256(path.read_bytes()).hexdigest()
            for module in modules.values()
            if (path := _user_file(module)) is not None
        }
    )
    if not hashes:
        return None
    return hashlib.sha256("\n".join(hashes).encode()).hexdigest()


class PolicyThread:
    """Run the adapter's import and every predict call on one dedicated thread.

    Thread-local setup made when the adapter module is imported, such as
    ``torch.set_grad_enabled(False)``, then still holds when predict runs,
    whichever server thread received the request.
    """

    def __init__(self) -> None:
        self._jobs: queue.Queue = queue.Queue()
        threading.Thread(target=self._run, name="policy", daemon=True).start()

    def submit(self, function: Callable, *args: Any, **kwargs: Any) -> Any:
        done: Future = Future()
        self._jobs.put((function, args, kwargs, done))
        return done.result()

    def _run(self) -> None:
        while True:
            function, args, kwargs, done = self._jobs.get()
            try:
                done.set_result(function(*args, **kwargs))
            except BaseException as exc:  # noqa: BLE001 - re-raised by submit
                done.set_exception(exc)


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
        adapter_digest: str | None = None,
        runner: PolicyThread | None = None,
    ) -> None:
        self.predict = predict
        self.runner = runner
        self.metadata: dict[str, Any] = {
            "policy": name,
            "action_schema": "decoupled_v1",
            "diagnostic": False,
            "checkpoint": checkpoint,
            "adapter_sha256": adapter_digest or adapter_sha256(predict),
        }

    def __call__(self, request: dict) -> Any:
        if self.runner is not None:
            return self.runner.submit(self.predict, request)
        return self.predict(request)


__all__ = [
    "CallablePolicy",
    "PolicyThread",
    "PolicyLoadError",
    "adapter_sha256",
    "load_policy_callable",
]
