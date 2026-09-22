"""Evaluate an external policy on the canonical HumanoidToolBench environments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

from theta_bench.evals.api import EvalConfig

ENV_IDS = tuple(
    f"theta_bench/G1{task}-L{level}-{mode}"
    for mode in ("S", "R")
    for level in range(3)
    for task in ("BallMove", "BallRetrieve", "IceBreak")
)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "env_id", nargs="?", default=ENV_IDS[0], choices=(*ENV_IDS, "all")
    )
    source = result.add_mutually_exclusive_group()
    source.add_argument(
        "--model", help="HumanoidToolBench ACT/DP Hugging Face model ID or local checkpoint"
    )
    source.add_argument(
        "--host", default="127.0.0.1", help="Use an already running policy server"
    )
    result.add_argument("--revision", help="Hugging Face model revision")
    result.add_argument(
        "--device", default="auto", help="Policy device: auto, cpu or cuda:N"
    )
    result.add_argument("--port", type=int, default=21000)
    result.add_argument("--output", type=Path, default=Path("data/evals"))
    result.add_argument("--episodes", type=int, default=100)
    result.add_argument("--seed-start", type=int, default=10000)
    result.add_argument("--max-steps", type=int, default=3000)
    result.add_argument("--dry-run", action="store_true")
    return result


def configurations(args: argparse.Namespace) -> list[EvalConfig]:
    if args.episodes < 1 or args.seed_start < 0:
        raise ValueError("episodes must be positive and seed-start must be nonnegative")
    if not 1 <= args.max_steps <= 3000:
        raise ValueError("max-steps must be between 1 and 3000")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.revision and not args.model:
        raise ValueError("--revision requires --model")
    selected = ENV_IDS if args.env_id == "all" else (args.env_id,)
    if any(env_id not in ENV_IDS for env_id in selected):
        raise ValueError("Select a canonical HumanoidToolBench environment")
    return [
        EvalConfig(
            env_id=env_id,
            policy="http",
            split="test",
            host=args.host,
            port=args.port,
            data_format="seeds",
            headless=True,
            eval_dir=str(args.output / env_id.removeprefix("theta_bench/")),
            max_episode_steps=args.max_steps,
            num_episodes=args.episodes,
            episode_start=args.seed_start,
            save_video=True,
            num_workers=1,
        )
        for env_id in selected
    ]


def main(argv: list[str] | None = None) -> None:
    cli = parser()
    args = cli.parse_args(argv)
    try:
        configs = configurations(args)
    except ValueError as exc:
        cli.error(str(exc))
    if args.dry_run:
        print(json.dumps([asdict(config) for config in configs], indent=2))
        return

    # Plain `uv run humanoidtoolbench-eval` uses the bounded rendering environment of
    # the existing shell entry point. Re-exec before importing the simulator.
    if os.environ.get("THETA_BENCH_RUNTIME_WRAPPER") != "1":
        wrapper = Path(__file__).resolve().parents[3] / "scripts/run_mujoco.sh"
        os.environ["THETA_BENCH_MUJOCO_ENV_PREFIX"] = sys.prefix
        os.environ["THETA_BENCH_POLICY_DEVICE"] = args.device
        arguments = sys.argv[1:] if argv is None else argv
        os.execv(
            "/bin/bash",
            [
                "bash",
                str(wrapper),
                sys.executable,
                "-m",
                "theta_bench.cli.public_eval",
                *arguments,
            ],
        )

    # Set the headless backend before MuJoCo is imported. The runtime wrapper
    # can select NVIDIA EGL or Mesa and bound its CPU pools explicitly.
    os.environ.setdefault("MUJOCO_GL", "egl")
    from theta_bench.assets.evaluation import verify_evaluation_assets
    from theta_bench.cli.eval_decoupled_wbc import run_eval
    from theta_bench.evals.public_validation import source_identity, validate_run

    asset_receipt = verify_evaluation_assets()
    policy = None
    if args.model:
        from theta_bench.policies.model_source import resolve_model
        from theta_bench.policies.pretrained import load_policy

        print(f"Loading model: {args.model}", flush=True)
        try:
            checkpoint, provenance = resolve_model(args.model, args.revision)
            policy = load_policy(checkpoint, args.device)
        except (OSError, ValueError, RuntimeError) as exc:
            cli.exit(2, f"Model loading failed: {exc}\n")
        policy.metadata.update(provenance)
        print(f"Model ready: HumanoidToolBench {policy.metadata['family'].upper()}", flush=True)
    for config in configs:
        source_before = source_identity()
        started = time.monotonic()
        if policy is not None:
            from theta_bench.policies.http_server import serve_policy

            server = serve_policy(policy, seed_start=config.episode_start)
        else:
            server = nullcontext(config.port)
        with server as port:
            config = replace(
                config, host="127.0.0.1" if policy else config.host, port=port
            )
            result = run_eval(config, show_progress=True, notify=False)
        receipt = validate_run(Path(result.eval_dir), config)
        if receipt["source_sha256"] != source_before:
            raise RuntimeError(
                "Evaluation source or packaged assets changed during the run"
            )
        if verify_evaluation_assets() != asset_receipt:
            raise RuntimeError("Evaluation asset identity changed during the run")
        receipt["assets"] = asset_receipt
        receipt["wall_seconds"] = time.monotonic() - started
        receipt["episodes_per_second"] = config.num_episodes / receipt["wall_seconds"]
        path = Path(result.eval_dir) / "benchmark_result.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)
        label = "benchmark" if receipt["reportable"] else "diagnostic"
        print(
            f"{config.env_id}: {receipt['successes']}/{receipt['episodes']} "
            f"({receipt['success_rate']:.1%}), {label}; {path}"
        )


if __name__ == "__main__":
    main()
