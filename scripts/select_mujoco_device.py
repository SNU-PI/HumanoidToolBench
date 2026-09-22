"""Match MuJoCo's EGL device to the policy's CUDA device by physical GPU UUID."""

from __future__ import annotations

import argparse
import ctypes
import os
import sys
import uuid


def cuda_device_uuid(index: int) -> bytes:
    # Query the CUDA driver after the wrapper sets CUDA_VISIBLE_DEVICES. CUDA
    # ordinals can differ from both nvidia-smi and EGL enumeration order.
    cuda = ctypes.CDLL("libcuda.so.1")
    device = ctypes.c_int()
    identifier = (ctypes.c_ubyte * 16)()
    for name, arguments in (
        ("cuInit", (0,)),
        ("cuDeviceGet", (ctypes.byref(device), index)),
        ("cuDeviceGetUuid", (ctypes.byref(identifier), device)),
    ):
        status = getattr(cuda, name)(*arguments)
        if status:
            raise RuntimeError(
                f"{name} failed with CUDA error {status}; check "
                "THETA_BENCH_GPU/CUDA_VISIBLE_DEVICES, or set THETA_BENCH_FORCE_CPU=1"
            )
    return bytes(identifier)


def egl_device_uuids() -> list[bytes | None]:
    egl = ctypes.CDLL("libEGL.so.1")
    egl.eglGetProcAddress.argtypes = [ctypes.c_char_p]
    egl.eglGetProcAddress.restype = ctypes.c_void_p

    def function(name, *arguments):
        address = egl.eglGetProcAddress(name.encode())
        if not address:
            raise RuntimeError(f"EGL driver does not support {name}")
        return ctypes.CFUNCTYPE(ctypes.c_uint, *arguments)(address)

    query_devices = function(
        "eglQueryDevicesEXT", ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
        ctypes.POINTER(ctypes.c_int),
    )
    query_uuid = function(
        "eglQueryDeviceBinaryEXT", ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_int),
    )
    count = ctypes.c_int()
    if not query_devices(0, None, ctypes.byref(count)):
        raise RuntimeError("Cannot enumerate EGL devices")
    devices = (ctypes.c_void_p * count.value)()
    if not query_devices(count.value, devices, ctypes.byref(count)):
        raise RuntimeError("Cannot enumerate EGL devices")
    identifiers = []
    for device in devices:
        identifier = (ctypes.c_ubyte * 16)()
        size = ctypes.c_int()
        # EGL_DEVICE_UUID_EXT identifies the physical GPU across APIs. Keep
        # non-GPU devices in the list so MuJoCo receives the original EGL index.
        valid = query_uuid(device, 0x335C, 16, identifier, ctypes.byref(size))
        identifiers.append(bytes(identifier) if valid and size.value == 16 else None)
    return identifiers


def select_egl_device(
    cuda_uuid: bytes, egl_uuids: list[bytes | None], override: str | None
) -> int:
    if override is not None:
        try:
            index = int(override)
        except ValueError as exc:
            raise ValueError("MUJOCO_EGL_DEVICE_ID must be a nonnegative integer") from exc
        if not 0 <= index < len(egl_uuids) or egl_uuids[index] != cuda_uuid:
            raise ValueError(
                f"MUJOCO_EGL_DEVICE_ID={override} does not match the policy's CUDA GPU; "
                "unset it to select the matching renderer automatically"
            )
        return index
    matches = [i for i, identifier in enumerate(egl_uuids) if identifier == cuda_uuid]
    if len(matches) != 1:
        raise RuntimeError(
            f"Cannot uniquely match CUDA GPU {uuid.UUID(bytes=cuda_uuid)} to an EGL "
            "device; check NVIDIA EGL drivers and EGL_EXT_device_persistent_id support"
        )
    return matches[0]


def cuda_device_index(device: str) -> int:
    if device in ("auto", "cpu", "cuda"):
        return 0
    if device.startswith("cuda:") and device[5:].isascii() and device[5:].isdigit():
        return int(device[5:])
    raise ValueError("Policy device must be auto, cpu, cuda or cuda:N")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    try:
        index = select_egl_device(
            cuda_device_uuid(cuda_device_index(args.device)), egl_device_uuids(),
            os.environ.get("MUJOCO_EGL_DEVICE_ID"),
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"[mujoco] {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(index)


if __name__ == "__main__":
    main()
