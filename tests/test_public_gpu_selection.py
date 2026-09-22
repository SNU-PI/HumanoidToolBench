"""CUDA and EGL must agree on physical devices despite independent enumeration."""

import importlib.util
import os
from pathlib import Path
import subprocess
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "select_mujoco_device", ROOT / "scripts/select_mujoco_device.py"
)
selector = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(selector)
GPU_0 = uuid.UUID(int=1).bytes
GPU_1 = uuid.UUID(int=2).bytes


@pytest.mark.parametrize(
    ("cuda_uuid", "egl_uuids", "expected"),
    [
        (GPU_1, [GPU_0, GPU_1], 1),
        (GPU_1, [GPU_1, GPU_0], 0),
        (GPU_1, [None, GPU_0, GPU_1], 2),
    ],
)
def test_match_physical_gpu_independently_of_egl_order(cuda_uuid, egl_uuids, expected):
    assert selector.select_egl_device(cuda_uuid, egl_uuids, None) == expected


def test_explicit_matching_egl_device_is_preserved():
    assert selector.select_egl_device(GPU_1, [GPU_0, GPU_1], "1") == 1


@pytest.mark.parametrize("override", ["0", "-1", "2", "GPU-89704cb7", ""])
def test_invalid_or_different_egl_override_is_rejected(override):
    with pytest.raises(ValueError, match="MUJOCO_EGL_DEVICE_ID"):
        selector.select_egl_device(GPU_1, [GPU_0, GPU_1], override)


@pytest.mark.parametrize("devices", [[None], [GPU_0], [GPU_1, GPU_1]])
def test_missing_or_ambiguous_device_fails_instead_of_rendering_elsewhere(devices):
    with pytest.raises(RuntimeError, match="Cannot uniquely match"):
        selector.select_egl_device(GPU_1, devices, None)


@pytest.mark.parametrize(
    ("device", "expected"),
    [("auto", 0), ("cpu", 0), ("cuda", 0), ("cuda:0", 0), ("cuda:1", 1)],
)
def test_policy_device_selects_visible_cuda_index(device, expected):
    assert selector.cuda_device_index(device) == expected


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:x", "mps", "1"])
def test_invalid_policy_device_is_rejected(device):
    with pytest.raises(ValueError, match="Policy device"):
        selector.cuda_device_index(device)


@pytest.fixture
def wrapper_environment(tmp_path):
    # This interpreter shim observes what the shell passes to the selector.
    # Its output deliberately differs from the CUDA index to detect shortcuts.
    python = tmp_path / "bin/python"
    python.parent.mkdir()
    python.write_text(
        '#!/usr/bin/env bash\n'
        '[[ "$1" == */select_mujoco_device.py ]] || exit 11\n'
        '[[ "$2" == --device && "$3" == "${HUMANOIDTOOLBENCH_POLICY_DEVICE:-auto}" ]] || exit 12\n'
        '[[ "${HUMANOIDTOOLBENCH_FORCE_CPU:-}" != 1 ]] || exit 13\n'
        'echo 2\n'
    )
    python.chmod(0o755)
    env = {key: value for key, value in os.environ.items() if not key.startswith(
        ("HUMANOIDTOOLBENCH_", "CUDA_", "MUJOCO_", "__EGL_", "LIBGL_", "MESA_", "EGL_")
    )}
    env.update(HUMANOIDTOOLBENCH_MUJOCO_ENV_PREFIX=str(tmp_path), HUMANOIDTOOLBENCH_NICE="0")
    return env


@pytest.mark.parametrize(
    ("selection", "visible"),
    [
        ({}, "0"),
        ({"HUMANOIDTOOLBENCH_GPU": "1"}, "1"),
        ({"CUDA_VISIBLE_DEVICES": "1,0"}, "1,0"),
        ({"HUMANOIDTOOLBENCH_GPU": f"GPU-{uuid.UUID(bytes=GPU_1)}"}, f"GPU-{uuid.UUID(bytes=GPU_1)}"),
        ({"HUMANOIDTOOLBENCH_GPU": "1", "CUDA_VISIBLE_DEVICES": "0"}, "1"),
        ({"CUDA_VISIBLE_DEVICES": "1,0", "HUMANOIDTOOLBENCH_POLICY_DEVICE": "cuda:1"}, "1,0"),
    ],
)
def test_wrapper_uses_selector_instead_of_copying_cuda_index(wrapper_environment, selection, visible):
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_mujoco.sh"), "env"],
        env=wrapper_environment | selection, text=True, capture_output=True, check=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert values["CUDA_VISIBLE_DEVICES"] == visible
    assert values["MUJOCO_EGL_DEVICE_ID"] == "2"


def test_force_cpu_clears_stale_gpu_renderer_and_skips_selector(wrapper_environment):
    vendor = Path(wrapper_environment["HUMANOIDTOOLBENCH_MUJOCO_ENV_PREFIX"]) / "share/glvnd/egl_vendor.d/50_mesa.json"
    vendor.parent.mkdir(parents=True)
    vendor.write_text("{}")
    result = subprocess.run(
        ["bash", str(ROOT / "scripts/run_mujoco.sh"), "env"],
        env=wrapper_environment | {
            "HUMANOIDTOOLBENCH_FORCE_CPU": "1", "HUMANOIDTOOLBENCH_GPU": "1",
            "CUDA_VISIBLE_DEVICES": "1", "MUJOCO_EGL_DEVICE_ID": "1",
        },
        text=True, capture_output=True, check=True,
    )
    values = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    assert values["CUDA_VISIBLE_DEVICES"] == ""
    assert "MUJOCO_EGL_DEVICE_ID" not in values
    assert values["LIBGL_ALWAYS_SOFTWARE"] == "1"
    assert values["MESA_LOADER_DRIVER_OVERRIDE"] == "llvmpipe"
