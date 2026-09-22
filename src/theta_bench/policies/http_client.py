"""NumPy-aware HTTP client for remote ToolBench policies."""

from __future__ import annotations

from base64 import b64decode, b64encode
from datetime import datetime
from typing import Any

import numpy as np
import requests
from numpy.lib.format import descr_to_dtype, dtype_to_descr


def _encode(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, (np.ndarray, np.generic)):
        array = np.asarray(value)
        return {
            "__numpy__": b64encode(array.tobytes()).decode(),
            "dtype": dtype_to_descr(array.dtype),
            "shape": array.shape,
        }
    return value


def _decode(value: Any) -> Any:
    if isinstance(value, dict) and "__numpy__" in value:
        array = np.frombuffer(
            b64decode(value["__numpy__"]), descr_to_dtype(value["dtype"])
        )
        shape = value["shape"]
        return array.reshape(shape) if shape else array[0]
    if isinstance(value, dict):
        return {key: _decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode(item) for item in value]
    return value


class HttpActionClient:
    """Call the benchmark policy server's ``/act`` endpoint."""

    def __init__(self, server_ip: str, server_port: int) -> None:
        self.url = f"http://{server_ip}:{server_port}/act"

    def query_action(
        self,
        image_dict: dict[str, Any],
        instruction: str,
        state_dict: dict[str, Any],
        condition_dict: dict[str, Any],
        history: dict[str, Any] | None = None,
        dataset: str = "simple",
        gt_action: np.ndarray | None = None,
    ) -> tuple[np.ndarray, Any, np.ndarray | None]:
        payload = {
            "image": image_dict,
            "instruction": instruction,
            "history": history or {key: [] for key in image_dict},
            "state": state_dict,
            "condition": condition_dict,
            "gt_action": [] if gt_action is None else gt_action,
            "dataset_name": dataset,
            "timestamp": datetime.now().isoformat(),
        }
        try:
            response = requests.post(
                self.url, json=_encode(payload), timeout=(5.0, 1200.0)
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            detail = str(exc)
            if exc.response is not None:
                failed = exc.response
                try:
                    body = failed.json()
                except ValueError:
                    body = None
                error = body.get("error") if isinstance(body, dict) else None
                detail = f"HTTP {failed.status_code} {failed.reason}: {error or failed.text}"
            detail = " ".join(detail.split())
            if len(detail) > 1000:
                detail = detail[:1000] + "..."
            raise RuntimeError(
                f"Policy server request failed at {self.url}: {detail}"
            ) from exc

        decoded = _decode(response.json())
        action = np.asarray(decoded["action"])
        error = decoded.get("err", 0.0)
        trajectory = decoded.get("traj_image")
        if not isinstance(trajectory, np.ndarray) or trajectory.ndim != 3:
            trajectory = None
        return action, error, trajectory


__all__ = ["HttpActionClient"]
