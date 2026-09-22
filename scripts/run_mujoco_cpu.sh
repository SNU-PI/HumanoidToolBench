#!/usr/bin/env bash
# Backwards-compatible entry point for the GPU-free smoke-test envelope.
# Identical to `THETA_BENCH_FORCE_CPU=1 scripts/run_mujoco.sh <command>`.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec env THETA_BENCH_FORCE_CPU=1 "$ROOT_DIR/scripts/run_mujoco.sh" "$@"
