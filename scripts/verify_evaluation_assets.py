#!/usr/bin/env python3
"""Verify all assets required by the canonical public evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from humanoidtoolbench.assets.evaluation import (  # noqa: E402
    EvaluationAssetError,
    verify_evaluation_assets,
)


def main() -> int:
    try:
        receipt = verify_evaluation_assets()
    except EvaluationAssetError as exc:
        print(f"[evaluation-assets] ERROR: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
