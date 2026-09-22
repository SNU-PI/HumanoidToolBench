"""Canonical benchmark language shared by simulators and dataset pipelines."""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from theta_bench.scenario_names import (
    SCENARIO_ALIASES,
    canonicalize_env_id,
    canonicalize_env_mapping,
)

LEVELS = (0, 1, 2)
MODES = ("S", "R")
SCENARIO_UIDS = {
    "G1BallMove": "g1_ball_move_teleop",
    "G1BallRetrieve": "g1_ball_retrieve_teleop",
    "G1IceBreak": "g1_ice_break_teleop",
}
MULTITASK_PROTOCOL = "theta_multitask_1200"
MULTITASK_SCENARIO_IDS = tuple(
    f"{scenario}-L{level}-{mode}"
    for scenario in SCENARIO_UIDS
    for level in (1, 2)
    for mode in MODES
)
MULTITASK_1800_PROTOCOL = "theta_multitask_1800"
MULTITASK_DERIVED_L0_PROTOCOL = "theta_multitask_l0derived_l1l2"
MULTITASK_L0_L1_PROTOCOL = "theta_multitask_l0derived_l1"
MULTITASK_S_PROTOCOL = "theta_multitask_l0derived_s"
# Scenarios whose correct tool has a function-verified unseen pool
# (`tools.OOD_CORRECT_ASSET_POOLS`), and so a correct-tool OOD twin.
OOD_CORRECT_SCENARIOS = ("G1BallMove", "G1BallRetrieve")
DERIVED_L0_PROTOCOLS = (
    MULTITASK_DERIVED_L0_PROTOCOL,
    MULTITASK_L0_L1_PROTOCOL,
    MULTITASK_S_PROTOCOL,
)
MULTITASK_1800_SCENARIO_IDS = tuple(
    f"{scenario}-L{level}-{mode}"
    for scenario in SCENARIO_UIDS
    for level in LEVELS
    for mode in MODES
)
MULTITASK_PROTOCOL_SCENARIOS = {
    MULTITASK_PROTOCOL: MULTITASK_SCENARIO_IDS,
    MULTITASK_1800_PROTOCOL: MULTITASK_1800_SCENARIO_IDS,
    MULTITASK_DERIVED_L0_PROTOCOL: MULTITASK_1800_SCENARIO_IDS,
    MULTITASK_L0_L1_PROTOCOL: tuple(
        env for env in MULTITASK_1800_SCENARIO_IDS if "-L2-" not in env
    ),
    MULTITASK_S_PROTOCOL: tuple(
        env for env in MULTITASK_1800_SCENARIO_IDS if env.endswith("-S")
    ),
}
CELL = re.compile(r"^(G1\w+?)-L([012])-([SR])(?:-v0)?(?=$|[-.])(?!-v\d)")

INSTRUCTIONS: dict[tuple[str, int, str], str] = {
    (uid, level, mode): wording[0 if level == 0 else 1]
    for uid, wording in {
        "g1_ball_move_teleop": (
            "Pick the tool for moving the ball to the target.",
            "Pick the tool and move the ball to the target.",
        ),
        "g1_ball_retrieve_teleop": (
            "Pick the tool for retrieving the ball to the target.",
            "Pick the tool and retrieve the ball to the target.",
        ),
        "g1_ice_break_teleop": (
            "Pick the tool for breaking the ice blocks.",
            "Pick the tool and break the ice blocks.",
        ),
    }.items()
    for level in LEVELS
    for mode in MODES
}


def multitask_scenario_ids(protocol: str) -> tuple[str, ...]:
    try:
        return MULTITASK_PROTOCOL_SCENARIOS[protocol]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unknown joint training protocol: {protocol!r}") from exc


def multitask_scenario_counts(protocol: str, counts: dict) -> dict[str, int]:
    """Keep human coverage fixed while admitting audited successful L0 prefixes."""
    scenarios = multitask_scenario_ids(protocol)
    counts = canonicalize_env_mapping(counts)
    if protocol not in DERIVED_L0_PROTOCOLS:
        return dict.fromkeys(scenarios, 100)
    if set(counts) != set(scenarios) or any(
        type(count) is not int
        or count <= 0
        or (cell_from_env_id(env)[1] != 0 and count != 100)
        for env, count in counts.items()
    ):
        raise ValueError(
            "Derived L0 training requires all six L0 conditions and 100 human successes per selected execution condition"
        )
    return counts


def cell_from_env_id(value: str) -> tuple[str, int, str] | None:
    match = CELL.match(value.removeprefix("theta_bench/"))
    if match is None:
        return None
    scenario = SCENARIO_ALIASES.get(match[1], match[1])
    if scenario not in SCENARIO_UIDS:
        return None
    return SCENARIO_UIDS[scenario], int(match[2]), match[3]


def dataset_env_id(root: Path, *, env_id: str | None = None) -> str | None:
    """Resolve the output cell before consulting its source provenance.

    Derived L0 datasets can retain metadata from their L1/L2 source. Directory
    identity therefore takes precedence over copied metadata. A preparation's
    source path is usable even after that source has moved to another host.
    """
    root = Path(root)
    if (root / "meta/multitask_training.json").is_file():
        if env_id is not None:
            raise ValueError("A joint dataset cannot be assigned one env_id")
        return None
    candidates = [env_id] if env_id is not None else [root.name, root.parent.name]
    for value in candidates:
        if value is not None and cell_from_env_id(value) is not None:
            return canonicalize_env_id(
                CELL.match(value.removeprefix("theta_bench/"))[0]
            )
    if env_id is not None:
        return None
    for name in ("info.json", "training_provenance.json", "psi0_preparation.json"):
        path = root / "meta" / name
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        value = metadata.get("env_id")
        if isinstance(value, str) and cell_from_env_id(value) is not None:
            return canonicalize_env_id(
                CELL.match(value.removeprefix("theta_bench/"))[0]
            )
        if name == "psi0_preparation.json":
            source = metadata.get("source")
            if isinstance(source, str):
                for value in reversed(Path(source).parts):
                    if cell_from_env_id(value) is not None:
                        return canonicalize_env_id(CELL.match(value)[0])
    return None


def dataset_instruction(root: Path, *, env_id: str | None = None) -> str | None:
    value = dataset_env_id(root, env_id=env_id)
    return INSTRUCTIONS[cell_from_env_id(value)] if value is not None else None


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _instruction_rows(root: Path) -> tuple[list[dict], list[dict]]:
    tasks = _read_jsonl(root / "meta/tasks.jsonl")
    if len(tasks) != 1:
        raise ValueError(
            f"{root}: expected one benchmark task row, found {len(tasks)}; "
            "resolve task indices before migrating instructions"
        )
    episodes_path = root / "meta/episodes.jsonl"
    episodes = _read_jsonl(episodes_path) if episodes_path.is_file() else []
    return tasks, episodes


def validate_dataset_instructions(
    root: Path, *, env_id: str | None = None, instruction: str | None = None
) -> str | None:
    """Reject stale benchmark prompts before publishing or starting a trainer."""
    root = Path(root)
    if (root / "meta/multitask_training.json").is_file():
        if env_id is not None or instruction is not None:
            raise ValueError(
                "A joint dataset cannot be assigned one instruction/env_id"
            )
        _validate_multitask_instructions(root)
        return None
    want = instruction or dataset_instruction(root, env_id=env_id)
    if want is None:
        return None
    tasks, episodes = _instruction_rows(root)
    if tasks[0].get("task") != want or any(
        row.get("tasks") != [want] for row in episodes
    ):
        raise ValueError(
            f"{root}: stale benchmark instruction; expected {want!r}. "
            "Run scripts/fix_dataset_instructions.py on the dataset with --write "
            "after recording has stopped, then rebuild converted datasets."
        )
    return want


def _validate_multitask_instructions(root: Path) -> None:
    """Validate the explicit joint protocol without loading training dependencies."""
    manifest = json.loads((root / "meta/multitask_training.json").read_text())
    manifest["scenario_ids"] = [
        canonicalize_env_id(env) for env in manifest.get("scenario_ids", [])
    ]
    manifest["scenario_counts"] = canonicalize_env_mapping(
        manifest.get("scenario_counts", {})
    )
    protocol = manifest.get("protocol")
    scenario_ids = multitask_scenario_ids(protocol)
    expected_counts = multitask_scenario_counts(protocol, manifest["scenario_counts"])
    total_episodes = sum(expected_counts.values())
    if (
        manifest.get("format_version") != 1
        or manifest.get("total_episodes") != total_episodes
        or manifest.get("scenario_counts") != expected_counts
        or len(manifest.get("scenario_ids", [])) != len(scenario_ids)
        or set(manifest["scenario_ids"]) != set(expected_counts)
    ):
        raise ValueError(f"{root}: invalid joint dataset scenario manifest")
    task_strings = list(
        dict.fromkeys(INSTRUCTIONS[cell_from_env_id(env)] for env in scenario_ids)
    )
    expected_tasks = [
        {"task_index": index, "task": task} for index, task in enumerate(task_strings)
    ]
    if _read_jsonl(root / "meta/tasks.jsonl") != expected_tasks:
        raise ValueError(
            f"{root}: joint task indices must map to {len(task_strings)} canonical prompts"
        )
    episodes = _read_jsonl(root / "meta/episodes.jsonl")
    sources = manifest.get("episode_sources", [])
    if len(episodes) != total_episodes or len(sources) != total_episodes:
        raise ValueError(
            f"{root}: joint episode provenance must cover {total_episodes} episodes"
        )
    counts = dict.fromkeys(scenario_ids, 0)
    for index, (episode, source) in enumerate(zip(episodes, sources, strict=True)):
        env = canonicalize_env_id(source.get("env_id"))
        if env not in counts:
            raise ValueError(f"{root}: unknown joint episode env_id: {env!r}")
        if protocol == MULTITASK_1800_PROTOCOL:
            oracle = cell_from_env_id(env)[1] == 0
            kind = "oracle_l0_fsm" if oracle else "human_teleoperation"
            if (
                episode.get("demonstration_source") != kind
                or source.get("demonstration_source") != kind
                or episode.get("controller_name")
                != ("L0PickAgent" if oracle else "QuestDecoupledAgent")
                or (oracle and episode.get("collection_method") != "oracle_l0_fsm")
            ):
                raise ValueError(
                    f"{root}: inconsistent joint episode source provenance"
                )
        derived = protocol in DERIVED_L0_PROTOCOLS and cell_from_env_id(env)[1] == 0
        if protocol in DERIVED_L0_PROTOCOLS:
            kind = "derived_l0_prefix" if derived else "human_teleoperation"
            if (
                episode.get("demonstration_source") != kind
                or source.get("demonstration_source") != kind
                or episode.get("controller_name") != "QuestDecoupledAgent"
                or episode.get("collection_method") == "oracle_l0_fsm"
                or (
                    derived
                    and (
                        episode.get("success_source") != "derived_l0_prefix"
                        or type(episode.get("derived_from_episode")) is not int
                        or episode["derived_from_episode"] < 0
                        or episode.get("parent_episode_index")
                        != episode["derived_from_episode"]
                        or source.get("parent_episode_index")
                        != episode["derived_from_episode"]
                        or not re.fullmatch(
                            r"[0-9a-f]{40}", episode.get("derived_from_fingerprint", "")
                        )
                        or episode.get("parent_fingerprint")
                        != episode["derived_from_fingerprint"]
                        or source.get("parent_fingerprint")
                        != episode["derived_from_fingerprint"]
                        or type(episode.get("l0_cut_frame")) is not int
                        or episode.get("length") != episode.get("l0_cut_frame", -2) + 1
                    )
                )
            ):
                raise ValueError(f"{root}: inconsistent derived L0/human provenance")
            if derived and protocol == MULTITASK_L0_L1_PROTOCOL:
                parent_cell = cell_from_env_id(
                    Path(episode.get("derived_from", "")).parent.name
                )
                uid, _, mode = cell_from_env_id(env)
                if parent_cell != (uid, 1, mode):
                    raise ValueError(f"{root}: L0/L1 training requires L1-only parents")
        want = INSTRUCTIONS[cell_from_env_id(env)]
        if (
            episode.get("episode_index") != index
            or source.get("episode_index") != index
            or canonicalize_env_id(episode.get("env_id")) != env
            or episode.get("tasks") != [want]
            or episode.get("task_index") != task_strings.index(want)
            or episode.get("episode_success") is not True
            or episode.get("simulator") != "mujoco"
            or episode.get("render_backend") != "mujoco"
            or episode.get("source_episode_index") != source.get("source_episode_index")
            or episode.get("source_index") != source.get("source_index")
            or (not derived and any(key.startswith("derived_") for key in episode))
        ):
            raise ValueError(
                f"{root}: inconsistent joint episode {index} instruction/provenance"
            )
        counts[env] += 1
    if counts != expected_counts:
        raise ValueError(f"{root}: joint episode counts disagree with the manifest")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
            stream.close()
            os.chmod(temporary, path.stat().st_mode)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def normalize_dataset_instructions(
    root: Path, *, env_id: str | None = None, instruction: str | None = None
) -> str | None:
    """Update inactive or staged v2.1 metadata, preserving frame task indices.

    Callers own the dataset while writing. Historical source snapshots should
    be copied to staging before normalization, and live writers must be stopped.
    """
    root = Path(root)
    if (root / "meta/multitask_training.json").is_file():
        return validate_dataset_instructions(
            root, env_id=env_id, instruction=instruction
        )
    want = instruction or dataset_instruction(root, env_id=env_id)
    if want is None:
        return None
    tasks, episodes = _instruction_rows(root)
    if tasks[0].get("task") != want:
        tasks[0]["task"] = want
        _write_jsonl(root / "meta/tasks.jsonl", tasks)
    if any(row.get("tasks") != [want] for row in episodes):
        for row in episodes:
            row["tasks"] = [want]
        _write_jsonl(root / "meta/episodes.jsonl", episodes)
    return want
