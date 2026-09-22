"""Evaluate an external policy on the canonical HumanoidToolBench environments."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
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
        "--policy",
        help=(
            "Your predict(request) as module:function or path/to/file.py:function, "
            "loaded and served by the evaluator in this environment; the "
            "alternative to --model and --host"
        ),
    )
    source.add_argument(
        "--host",
        default="127.0.0.1",
        help=(
            "Address of an already running policy server such as "
            "examples/serve_policy.py; the alternative to --model and --policy. "
            "Default: %(default)s"
        ),
    )
    result.add_argument(
        "--checkpoint",
        help="Checkpoint ID or revision recorded as provenance for --policy runs",
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
        "--rerun",
        action="store_true",
        help=(
            "Evaluate a condition again even when a validated result with the same "
            "settings and policy already exists under --output"
        ),
    )
    result.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the run configuration as JSON and exit without downloading or simulating",
    )
    return result


RESUME_FIELDS = (
    "env_id",
    "num_episodes",
    "episode_start",
    "max_episode_steps",
    "split",
    "controller",
    "sim_mode",
)


def completed_result(config: EvalConfig, identity: dict) -> Path | None:
    """Return an earlier validated benchmark_result.json for the same run, if any.

    A result is reused only when its settings and its policy identity (name and
    checkpoint) match; a policy without a checkpoint identifier is never reused.
    """
    if not identity.get("checkpoint"):
        return None
    for path in sorted(Path(config.eval_dir).glob("run-*/benchmark_result.json")):
        try:
            receipt = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        recorded = receipt.get("configuration") or {}
        info = receipt.get("policy_info") or {}
        if (
            receipt.get("status") == "validated"
            and all(recorded.get(k) == getattr(config, k) for k in RESUME_FIELDS)
            and info.get("policy") == identity.get("policy")
            and info.get("checkpoint") == identity.get("checkpoint")
        ):
            return path
    return None


def write_summary(output: Path, rows: list[dict]) -> Path:
    """Write <output>/summary.json and summary.md for a multi-condition run."""
    scored = [row for row in rows if row.get("status") in ("completed", "skipped")]
    complete = len(scored) == len(rows) and bool(rows)
    payload = {
        "conditions": rows,
        "completed": sum(row["status"] == "completed" for row in rows),
        "skipped": sum(row["status"] == "skipped" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "reportable": complete and all(row.get("reportable") for row in scored),
        # The unweighted mean over conditions, only once every condition has a result.
        "mean_success_rate": (
            sum(row["success_rate"] for row in scored) / len(scored)
            if complete
            else None
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "summary.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)
    lines = [
        "| Condition | Successes | Episodes | Rate | Reportable | Status | Result |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        rate = "" if row.get("success_rate") is None else f"{row['success_rate']:.1%}"
        lines.append(
            f"| {row['env_id'].removeprefix(ENV_PREFIX)} | {row.get('successes', '')} | "
            f"{row.get('episodes', '')} | {rate} | {row.get('reportable', '')} | "
            f"{row['status']} | {row.get('result') or row.get('error', '')} |"
        )
    mean = payload["mean_success_rate"]
    lines.append("")
    lines.append(
        "Mean success rate over conditions: "
        + (f"{mean:.1%}" if mean is not None else "incomplete")
    )
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return path


def _summary_row(config: EvalConfig, receipt: dict, path: Path, status: str) -> dict:
    return {
        "env_id": config.env_id,
        "status": status,
        "successes": receipt["successes"],
        "episodes": receipt["episodes"],
        "success_rate": receipt["success_rate"],
        "reportable": receipt["reportable"],
        "wall_seconds": receipt.get("wall_seconds"),
        "result": str(path),
    }


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
    if args.checkpoint and not args.policy:
        raise ValueError("--checkpoint requires --policy")
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
    external = not args.model and not args.policy
    if external and os.environ.get("HUMANOIDTOOLBENCH_PREFLIGHT_DONE") != "1":
        problem = policy_server_unreachable(configs[0].host, configs[0].port)
        if problem:
            cli.exit(2, problem + "\n")
        os.environ["HUMANOIDTOOLBENCH_PREFLIGHT_DONE"] = "1"

    # Plain `uv run htb-eval` uses the bounded rendering environment of
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
    elif args.policy:
        from humanoidtoolbench.policies.loader import (
            CallablePolicy,
            PolicyLoadError,
            load_policy_callable,
        )

        try:
            predict = load_policy_callable(args.policy)
        except PolicyLoadError as exc:
            cli.exit(2, f"{exc}\n")
        policy = CallablePolicy(predict, name=args.policy, checkpoint=args.checkpoint)

    if policy is not None:
        identity = {
            "policy": policy.metadata.get("policy"),
            "checkpoint": policy.metadata.get("checkpoint"),
        }
    else:
        from humanoidtoolbench.cli.eval_decoupled_wbc import _fetch_policy_info

        info = _fetch_policy_info(configs[0].host, configs[0].port)
        identity = {"policy": info.get("policy"), "checkpoint": info.get("checkpoint")}

    def evaluate(config: EvalConfig) -> tuple[dict, Path]:
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
        return receipt, path

    rows: list[dict] = []
    for index, config in enumerate(configs, 1):
        print(f"[{index}/{len(configs)}] {config.env_id}", flush=True)
        existing = None if args.rerun else completed_result(config, identity)
        if existing is not None:
            receipt = json.loads(existing.read_text())
            rows.append(_summary_row(config, receipt, existing, "skipped"))
            print(
                f"{config.env_id}: {receipt['successes']}/{receipt['episodes']} "
                f"({receipt['success_rate']:.1%}), already validated; {existing} "
                "(pass --rerun to evaluate again)",
                flush=True,
            )
            continue
        try:
            receipt, path = evaluate(config)
        except Exception as exc:  # noqa: BLE001 - one condition must not sink the rest
            # A failed condition keeps its run directory for inspection; the
            # remaining conditions still run and the exit status reports it.
            traceback.print_exc()
            print(f"{config.env_id}: FAILED: {exc}", file=sys.stderr, flush=True)
            rows.append(
                {"env_id": config.env_id, "status": "failed", "error": str(exc)}
            )
            continue
        rows.append(_summary_row(config, receipt, path, "completed"))
        label = "benchmark" if receipt["reportable"] else "diagnostic"
        print(
            f"{config.env_id}: {receipt['successes']}/{receipt['episodes']} "
            f"({receipt['success_rate']:.1%}), {label}; {path}",
            flush=True,
        )
    if len(configs) > 1:
        summary = write_summary(args.output, rows)
        print(f"Summary: {summary} and {summary.with_suffix('.md')}", flush=True)
    failed = [row["env_id"] for row in rows if row["status"] == "failed"]
    if failed:
        cli.exit(
            1,
            f"{len(failed)} of {len(configs)} conditions failed: {', '.join(failed)}\n",
        )


if __name__ == "__main__":
    main()
