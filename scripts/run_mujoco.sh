#!/usr/bin/env bash
# Run a HumanoidToolBench MuJoCo-only command inside the repository-owned environment.
#
# The GPU is used by default. This is still a shared lab server, so the wrapper
# keeps the polite defaults: a single pinned GPU, reduced scheduling priority,
# and bounded math thread pools. HUMANOIDTOOLBENCH_FORCE_CPU=1 restores the original
# GPU-free smoke-test envelope.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${HUMANOIDTOOLBENCH_MUJOCO_ENV_PREFIX:-$ROOT_DIR/.venv}"
FORCE_CPU="${HUMANOIDTOOLBENCH_FORCE_CPU:-0}"

usage() {
    cat <<'USAGE'
Usage: run_mujoco.sh <command> [args...]

Environment:
  HUMANOIDTOOLBENCH_MUJOCO_ENV_PREFIX  environment created by scripts/setup_evaluation.py
  HUMANOIDTOOLBENCH_FORCE_CPU=1    hide every GPU and render with Mesa llvmpipe
  HUMANOIDTOOLBENCH_GPU=<id>       CUDA GPU index or UUID for inference and EGL (default: 0)
  MUJOCO_EGL_DEVICE_ID=N    optional EGL index, checked against the selected CUDA GPU
  HUMANOIDTOOLBENCH_CPU_THREADS=N  math thread-pool cap (default: 4 on GPU, 2 on CPU)
  HUMANOIDTOOLBENCH_CPUSET=<list>  taskset cpu-list (default: unpinned on GPU, 2 CPUs on CPU)
  HUMANOIDTOOLBENCH_NICE=N         scheduling priority, 0-19 (default: 10)
USAGE
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 2
fi
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "[mujoco] no environment at $ENV_PREFIX" >&2
    echo "[mujoco] run uv run --no-project scripts/setup_evaluation.py, or point" >&2
    echo "[mujoco] HUMANOIDTOOLBENCH_MUJOCO_ENV_PREFIX at an existing one" >&2
    exit 1
fi

export PATH="$ENV_PREFIX/bin:$PATH"
export HUMANOIDTOOLBENCH_FORCE_CPU="$FORCE_CPU"
export HUMANOIDTOOLBENCH_RUNTIME_WRAPPER=1
export MUJOCO_GL="${MUJOCO_GL:-egl}"
# Only used by windowed (glfw) rendering; EGL ignores it.
export DISPLAY="${DISPLAY:-:1}"

if [[ "$FORCE_CPU" == "1" ]]; then
    # A conda-era prefix ships its own Mesa; the uv .venv relies on system Mesa.
    MESA_EGL_VENDOR_JSON="$ENV_PREFIX/share/glvnd/egl_vendor.d/50_mesa.json"
    if [[ -f "$MESA_EGL_VENDOR_JSON" ]]; then
        # Mesa has to shadow any system/NVIDIA GL, so the env prefix goes first.
        export LD_LIBRARY_PATH="$ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    else
        MESA_EGL_VENDOR_JSON=/usr/share/glvnd/egl_vendor.d/50_mesa.json
        if [[ ! -f "$MESA_EGL_VENDOR_JSON" ]]; then
            echo "[mujoco] no Mesa EGL vendor file (checked the env prefix and /usr/share/glvnd)." >&2
            echo "[mujoco] HUMANOIDTOOLBENCH_FORCE_CPU=1 needs Mesa; install the system package (e.g. libegl-mesa0)." >&2
            exit 1
        fi
    fi
    export CUDA_VISIBLE_DEVICES=""
    unset MUJOCO_EGL_DEVICE_ID
    export LIBGL_ALWAYS_SOFTWARE=1
    export MESA_LOADER_DRIVER_OVERRIDE=llvmpipe
    export EGL_PLATFORM=surfaceless
    export __EGL_VENDOR_LIBRARY_FILENAMES="$MESA_EGL_VENDOR_JSON"
    DEFAULT_THREADS=2
    RENDERER_LABEL=llvmpipe
    GPU_LABEL=hidden
else
    # The NVIDIA EGL vendor lives in the system glvnd directory; keep it ahead
    # of the environment's Mesa build or rendering silently drops to software.
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}$ENV_PREFIX/lib"
    if [[ -n "${HUMANOIDTOOLBENCH_GPU:-}" ]]; then
        export CUDA_VISIBLE_DEVICES="$HUMANOIDTOOLBENCH_GPU"
    else
        export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"
    fi
    unset LIBGL_ALWAYS_SOFTWARE MESA_LOADER_DRIVER_OVERRIDE EGL_PLATFORM
    NVIDIA_EGL_VENDOR_JSON=/usr/share/glvnd/egl_vendor.d/10_nvidia.json
    if [[ -f "$NVIDIA_EGL_VENDOR_JSON" ]]; then
        export __EGL_VENDOR_LIBRARY_FILENAMES="$NVIDIA_EGL_VENDOR_JSON"
    else
        unset __EGL_VENDOR_LIBRARY_FILENAMES
    fi
    # CUDA and EGL enumerate devices independently. Match their physical UUIDs
    # after configuring the EGL vendor instead of assuming their indices agree.
    EGL_DEVICE_ID="$("$ENV_PREFIX/bin/python" "$ROOT_DIR/scripts/select_mujoco_device.py" \
        --device "${HUMANOIDTOOLBENCH_POLICY_DEVICE:-auto}")"
    export MUJOCO_EGL_DEVICE_ID="$EGL_DEVICE_ID"
    DEFAULT_THREADS=4
    RENDERER_LABEL="nvidia-egl:$MUJOCO_EGL_DEVICE_ID"
    GPU_LABEL="${CUDA_VISIBLE_DEVICES:-none}"
fi

CPU_THREADS="${HUMANOIDTOOLBENCH_CPU_THREADS:-$DEFAULT_THREADS}"
if [[ ! "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
    echo "[mujoco] HUMANOIDTOOLBENCH_CPU_THREADS must be a positive integer" >&2
    exit 2
fi
if [[ "$FORCE_CPU" == "1" && "$CPU_THREADS" -gt 2 ]]; then
    echo "[mujoco] HUMANOIDTOOLBENCH_FORCE_CPU=1 caps HUMANOIDTOOLBENCH_CPU_THREADS at 2" >&2
    exit 2
fi
export OMP_NUM_THREADS="$CPU_THREADS"
export MKL_NUM_THREADS="$CPU_THREADS"
export OPENBLAS_NUM_THREADS="$CPU_THREADS"
export NUMEXPR_NUM_THREADS="$CPU_THREADS"

NICE_LEVEL="${HUMANOIDTOOLBENCH_NICE:-10}"
if [[ ! "$NICE_LEVEL" =~ ^([0-9]|1[0-9])$ ]]; then
    echo "[mujoco] HUMANOIDTOOLBENCH_NICE must be an integer between 0 and 19" >&2
    exit 2
fi

# GPU work is bounded by the GPU itself, so only the CPU-only envelope pins a
# core pair by default. An explicit HUMANOIDTOOLBENCH_CPUSET always wins.
CPUSET="${HUMANOIDTOOLBENCH_CPUSET:-}"
if [[ -z "$CPUSET" && "$FORCE_CPU" == "1" ]]; then
    ALLOWED_CPUS="$(awk '/^Cpus_allowed_list:/ {print $2}' /proc/self/status)"
    FIRST_RANGE="${ALLOWED_CPUS%%,*}"
    if [[ "$FIRST_RANGE" == *-* ]]; then
        FIRST_CPU="${FIRST_RANGE%-*}"
        LAST_CPU="${FIRST_RANGE#*-}"
        if (( FIRST_CPU < LAST_CPU )); then
            CPUSET="$FIRST_CPU,$((FIRST_CPU + 1))"
        else
            CPUSET="$FIRST_CPU"
        fi
    else
        CPUSET="$FIRST_RANGE"
    fi
fi

COMMAND=("$@")
if [[ -n "$CPUSET" ]]; then
    COMMAND=(taskset --cpu-list "$CPUSET" "${COMMAND[@]}")
fi
if [[ "$NICE_LEVEL" != "0" ]]; then
    COMMAND=(nice -n "$NICE_LEVEL" "${COMMAND[@]}")
fi

echo "[mujoco] nice=$NICE_LEVEL cpuset=${CPUSET:-inherited} threads=$CPU_THREADS gpu=$GPU_LABEL renderer=$RENDERER_LABEL" >&2
exec "${COMMAND[@]}"
