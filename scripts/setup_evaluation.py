# /// script
# requires-python = "==3.10.*"
# dependencies = []
# ///
"""Install HumanoidToolBench with `uv run --no-project scripts/setup_evaluation.py`."""

import argparse
import os
from pathlib import Path
import subprocess
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="Install the HumanoidToolBench evaluation environment.")
    parser.add_argument("--check", action="store_true", help="Check an existing installation.")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(
        ["bash", str(root / "scripts/bootstrap_evaluation.sh"), "--check" if args.check else "--install"],
        cwd=root,
        env=env,
        check=False,
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
