#!/usr/bin/env bash
# Install or check the canonical public MuJoCo evaluation environment.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="${THETA_BENCH_MUJOCO_ENV_PREFIX:-$ROOT_DIR/.venv}"
MODE="${1:---check}"

if [[ "$MODE" != "--check" && "$MODE" != "--install" ]]; then
    echo "Usage: $0 [--check|--install]" >&2
    exit 2
fi
cd "$ROOT_DIR"

if [[ "$MODE" == "--install" ]]; then
    for command in git uv zstd; do
        if ! command -v "$command" >/dev/null 2>&1; then
            echo "[evaluation] required command is missing: $command" >&2
            exit 1
        fi
    done
    if ! python3 "$ROOT_DIR/scripts/verify_runtime_assets.py" >/dev/null 2>&1; then
        if ! git lfs version >/dev/null 2>&1; then
            echo "[evaluation] Git LFS is required to download benchmark assets." >&2
            echo "[evaluation] Install it from https://git-lfs.com/ and rerun setup." >&2
            exit 1
        fi
        git lfs pull --include="src/theta_bench/resources/benchmark_assets/**"
    fi
    # A release snapshot has no private Git history or gitlinks. Each controller
    # dependency is fetched anonymously from its pinned public repository.
    while IFS=$'\t' read -r path url revision; do
        if [[ ! -e "$ROOT_DIR/$path/.git" ]]; then
            env GIT_LFS_SKIP_SMUDGE=1 nice -n 10 \
                git clone --filter=blob:none --no-checkout --depth 1 "$url" "$ROOT_DIR/$path"
            nice -n 10 git -C "$ROOT_DIR/$path" fetch --depth 1 origin "$revision"
            env GIT_LFS_SKIP_SMUDGE=1 git -C "$ROOT_DIR/$path" checkout --detach "$revision"
        fi
        actual="$(git -C "$ROOT_DIR/$path" rev-parse HEAD)"
        if [[ "$actual" != "$revision" ]]; then
            echo "[evaluation] $path must be at pinned revision $revision (found $actual)" >&2
            exit 1
        fi
    done < <(python3 - <<'PY'
import json
from pathlib import Path
manifest = json.loads(Path("src/theta_bench/resources/evaluation_assets.json").read_text())
for path, dependency in manifest["submodules"].items():
    print(path, dependency["url"], dependency["revision"], sep="\t")
PY
)
    # Older installs can contain conflicting wheels that share the cv2 files.
    # Removing those wheels also removes shared files, so restore headless cv2.
    opencv_sync_args=()
    if [[ -x "$ENV_PREFIX/bin/python" ]] && "$ENV_PREFIX/bin/python" - <<'PY'
from importlib.metadata import distributions
names = {dist.metadata["Name"].lower().replace("_", "-") for dist in distributions()}
conflicts = {"opencv-python", "opencv-contrib-python", "opencv-contrib-python-headless"}
raise SystemExit(0 if names & conflicts else 1)
PY
    then
        uv pip uninstall --python "$ENV_PREFIX/bin/python" \
            opencv-python opencv-contrib-python opencv-contrib-python-headless
        opencv_sync_args=(--reinstall-package opencv-python-headless)
    fi
    UV_PROJECT_ENVIRONMENT="$ENV_PREFIX" nice -n 10 \
        uv sync --frozen --inexact --no-default-groups --project "$ROOT_DIR" \
            "${opencv_sync_args[@]}"
    if [[ "${THETA_BENCH_FORCE_CPU:-0}" == "1" ]]; then
        nice -n 10 uv pip install --python "$ENV_PREFIX/bin/python" \
            --index-url https://download.pytorch.org/whl/cpu --reinstall \
            torch==2.7.0 torchvision==0.22.0
    fi
fi

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
    echo "[evaluation] no Python environment at $ENV_PREFIX; run $0 --install" >&2
    exit 1
fi
"$ENV_PREFIX/bin/python" "$ROOT_DIR/scripts/verify_mujoco_manifest.py"
if [[ "$MODE" == "--install" ]]; then
    "$ENV_PREFIX/bin/python" "$ROOT_DIR/scripts/fetch_evaluation_assets.py"
fi
"$ENV_PREFIX/bin/python" "$ROOT_DIR/scripts/verify_evaluation_assets.py"
"$ROOT_DIR/scripts/run_mujoco.sh" python - <<'PY'
import av
import cv2
import imageio_ffmpeg
import onnxruntime

gui = next(
    line.strip() for line in cv2.getBuildInformation().splitlines()
    if line.strip().startswith("GUI:")
)
if gui.split(":", 1)[1].strip() != "NONE":
    raise RuntimeError(f"Expected headless OpenCV, found {gui}; rerun setup.")

from decoupled_wbc.control.main.teleop.configs.configs import ControlLoopConfig
from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model
from theta_bench.assets.evaluation import REPO_ROOT, load_manifest
from theta_bench.robots.g1_sonic import G1Sonic

instantiate_g1_robot_model()
for relative in load_manifest()["wbc_weights"]:
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    onnxruntime.InferenceSession(
        str(REPO_ROOT / relative), options, providers=["CPUExecutionProvider"]
    )
print("[evaluation] controller, video dependencies and both WBC weights passed")
PY
echo "[evaluation] assets and dependencies passed; no benchmark episodes were run"
