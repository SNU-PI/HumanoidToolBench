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

from humanoidtoolbench.evals.api import EvalConfig

ENV_PREFIX = "humanoidtoolbench/"
ENV_IDS = tuple(
    f"{ENV_PREFIX}G1{task}-L{level}-{mode}"
    for mode in ("S", "R")
    for level in range(3)
    for task in ("BallMove", "BallRetrieve", "IceBreak")
)


def canonical_env_id(value: str) -> str:
    """Return the prefixed canonical ID for a bare or prefixed condition name."""
    name = value.strip()
    if name.startswith(ENV_PREFIX):
        name = name[len(ENV_PREFIX) :]
    env_id = ENV_PREFIX + name
    if env_id not in ENV_IDS:
        raise ValueError(
            f"Unknown environment {value!r}. Expected one of the 18 canonical "
            "conditions (G1BallMove|G1BallRetrieve|G1IceBreak)-L(0|1|2)-(S|R), "
            f"with or without the {ENV_PREFIX} prefix, or 'all'. "
            "Run with --list-envs to print them."
        )
    return env_id


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Conditions are described in docs/ENVIRONMENTS.md; the policy "
            "interface for --host servers is in docs/PUBLIC_EVALUATION.md."
        ),
    )
    result.add_argument(
        "env_id",
        nargs="?",
        default=ENV_IDS[0],
        metavar="ENV_ID",
        help=(
            "Condition to evaluate, for example G1BallMove-L0-S (the "
            f"{ENV_PREFIX} prefix is optional), or 'all' for the 18 canonical "
            "conditions. Default: %(default)s"
        ),
    )
    result.add_argument(
        "--list-envs",
        action="store_true",
        help="Print the 18 canonical environment IDs and exit",
    )
    source = result.add_mutually_exclusive_group()
    source.add_argument(
        "--model",
        help=(
            "HumanoidToolBench ACT/DP Hugging Face model ID or local checkpoint; "
            "the evaluator serves it itself"
        ),
    )
    source.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Address of an already running policy server such as "
            "examples/serve_policy.py; the alternative to --model. "
            "Default: %(default)s"
        ),
    )
    result.add_argument(
        "--revision",
        help="Hugging Face branch, tag or commit for --model (local paths do not accept it)",
    )
    result.add_argument(
        "--device",
        default="auto",
        help=(
            "Policy device for --model and the GPU MuJoCo renders on: auto, cpu "
            "or cuda:N. Default: %(default)s"
        ),
    )
    result.add_argument(
        "--port",
        type=int,
        default=21000,
        help="Port of the policy server for --host. Default: %(default)s",
    )
    result.add_argument(
        "--output",
        type=Path,
        default=Path("data/evals"),
        help="Output root; each condition writes <output>/<condition>/run-<id>/. Default: %(default)s",
    )
    result.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Episodes per condition; the standard protocol is %(default)s",
    )
    result.add_argument(
        "--seed-start",
        type=int,
        default=10000,
        help="First episode seed; the standard protocol uses %(default)s through 10099",
    )
    result.add_argument(
        "--max-steps",
        type=int,
        default=3000,
        help="Policy-control steps per episode at 50 Hz; the standard protocol is %(default)s",
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run configuration as JSON and exit without downloading or simulating",
    )
    return result


def policy_server_unreachable(host: str, port: int, timeout: float = 3.0) -> str | None:
    """Return an actionable message when nothing answers at host:port, else None.

    Only connection-level failures count. A server that answers /info with an
    HTTP error is reachable; the endpoint is optional for custom servers.
    """
    import http.client
    import urllib.error
    import urllib.request

    url = f"http://{host}:{port}/info"
    try:
        urllib.request.urlopen(url, timeout=timeout).close()
    except urllib.error.HTTPError:
        return None
    except (
        urllib.error.URLError,
        http.client.HTTPException,
        OSError,
        ValueError,
    ) as exc:
        reason = str(getattr(exc, "reason", exc)).strip() or type(exc).__name__
        return (
            f"No policy server answered at {url} ({reason}).\n"
            "Start one in another terminal, for example:\n"
            f"  uv run python examples/serve_policy.py --policy my_policy:predict --port {port}\n"
            "or pass --model for a HumanoidToolBench ACT/DP checkpoint. "
            "See docs/PUBLIC_EVALUATION.md#policy-interface."
        )
    return None


def configurations(args: argparse.Namespace) -> list[EvalConfig]:
    if args.episodes < 1 or args.seed_start < 0:
        raise ValueError("episodes must be positive and seed-start must be nonnegative")
    if not 1 <= args.max_steps <= 3000:
        raise ValueError("max-steps must be between 1 and 3000")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")
    if args.revision and not args.model:
        raise ValueError("--revision requires --model")
    name = args.env_id.strip()
    selected = ENV_IDS if name == "all" else (canonical_env_id(name),)
    return [
        EvalConfig(
            env_id=env_id,
            policy="http",
            split="test",
            host=args.host.strip(),
            port=args.port,
            data_format="seeds",
            headless=True,
            eval_dir=str(args.output / env_id.removeprefix(ENV_PREFIX)),
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
    if args.list_envs:
        print("\n".join(ENV_IDS))
        return
    try:
        configs = configurations(args)
    except ValueError as exc:
        cli.error(str(exc))
    if args.dry_run:
        print(json.dumps([asdict(config) for config in configs], indent=2))
        return
    # Fail in seconds, not after the simulator boots, when no server answers.
    # The re-exec below inherits the marker, so the check runs once per run.
    if not args.model and os.environ.get("HUMANOIDTOOLBENCH_PREFLIGHT_DONE") != "1":
        problem = policy_server_unreachable(configs[0].host, configs[0].port)
        if problem:
            cli.exit(2, problem + "\n")
        os.environ["HUMANOIDTOOLBENCH_PREFLIGHT_DONE"] = "1"

    # Plain `uv run humanoidtoolbench-eval` uses the bounded rendering environment of
    # the existing shell entry point. Re-exec before importing the simulator.
    if os.environ.get("HUMANOIDTOOLBENCH_RUNTIME_WRAPPER") != "1":
        wrapper = Path(__file__).resolve().parents[3] / "scripts/run_mujoco.sh"
        os.environ["HUMANOIDTOOLBENCH_MUJOCO_ENV_PREFIX"] = sys.prefix
        os.environ["HUMANOIDTOOLBENCH_POLICY_DEVICE"] = args.device
        arguments = sys.argv[1:] if argv is None else argv
        os.execv(
            "/bin/bash",
            [
                "bash",
                str(wrapper),
                sys.executable,
                "-m",
                "humanoidtoolbench.cli.public_eval",
                *arguments,
            ],
        )

    # Set the headless backend before MuJoCo is imported. The runtime wrapper
    # can select NVIDIA EGL or Mesa and bound its CPU pools explicitly.
    os.environ.setdefault("MUJOCO_GL", "egl")
    from humanoidtoolbench.assets.evaluation import verify_evaluation_assets
    from humanoidtoolbench.cli.eval_decoupled_wbc import run_eval
    from humanoidtoolbench.evals.public_validation import source_identity, validate_run

    asset_receipt = verify_evaluation_assets()
    policy = None
    if args.model:
        from humanoidtoolbench.policies.model_source import resolve_model
        from humanoidtoolbench.policies.pretrained import load_policy

        print(f"Loading model: {args.model}", flush=True)
        try:
            checkpoint, provenance = resolve_model(args.model, args.revision)
            policy = load_policy(checkpoint, args.device)
        except (OSError, ValueError, RuntimeError) as exc:
            cli.exit(2, f"Model loading failed: {exc}\n")
        policy.metadata.update(provenance)
        print(
            f"Model ready: HumanoidToolBench {policy.metadata['family'].upper()}",
            flush=True,
        )
    for config in configs:
        source_before = source_identity()
        started = time.monotonic()
        if policy is not None:
            from humanoidtoolbench.policies.http_server import serve_policy

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
