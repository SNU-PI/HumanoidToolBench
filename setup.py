"""Setuptools hooks that reject incomplete Git LFS runtime assets."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.egg_info import egg_info
from setuptools.command.sdist import sdist

ROOT = Path(__file__).resolve().parent


def _load_asset_guard() -> ModuleType:
    path = ROOT / "scripts" / "verify_runtime_assets.py"
    spec = importlib.util.spec_from_file_location("theta_bench_asset_guard", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load runtime asset guard: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _verify_runtime_assets() -> None:
    guard = _load_asset_guard()
    guard.verify_runtime_assets(
        ROOT / "src" / "theta_bench" / "resources" / "benchmark_assets"
    )


class VerifiedBuildPy(build_py):
    def run(self) -> None:
        _verify_runtime_assets()
        # Setuptools reuses build/lib, including packages removed from discovery.
        # Rebuild this project's output so old research modules cannot leak.
        build_output = Path(self.build_lib)
        if build_output.exists():
            shutil.rmtree(build_output)
        super().run()


class VerifiedEggInfo(egg_info):
    def run(self) -> None:
        _verify_runtime_assets()
        super().run()


class VerifiedSdist(sdist):
    def run(self) -> None:
        _verify_runtime_assets()
        super().run()


setup(
    cmdclass={
        "build_py": VerifiedBuildPy,
        "egg_info": VerifiedEggInfo,
        "sdist": VerifiedSdist,
    }
)
