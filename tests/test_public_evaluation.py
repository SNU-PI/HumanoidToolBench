"""External policies and canonical score admission without model checkpoints."""

from __future__ import annotations

import importlib.util
import json
import socket
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

from theta_bench.cli import public_eval
from theta_bench.evals import public_validation as validation
from theta_bench.policies.http_client import HttpActionClient
from theta_bench.policies.http_server import serve_policy
from theta_bench.policies.remote_humanoid import make_remote_task_policy


def config(*args):
    return public_eval.configurations(
        public_eval.parser().parse_args([public_eval.ENV_IDS[0], *args])
    )[0]


def test_public_suite_selects_all_18_canonical_conditions():
    configs = public_eval.configurations(public_eval.parser().parse_args(["all"]))
    assert len({item.env_id for item in configs}) == 18
    assert {item.env_id for item in configs} == {
        f"theta_bench/G1{task}-L{level}-{mode}"
        for task in ("BallMove", "BallRetrieve", "IceBreak")
        for level in range(3)
        for mode in ("S", "R")
    }
    for item in configs:
        assert item.policy == "http"
        assert item.split == "test" and item.data_format == "seeds"
        assert (item.num_episodes, item.episode_start, item.max_episode_steps) == (
            100,
            10000,
            3000,
        )
        assert item.save_video and item.num_workers == 1 and item.headless


@pytest.mark.parametrize(
    "name", ["G1BallMoveEasyGap50Attach", "G1IceBreakEasyNearTouch", "G1BallMoveOOD"]
)
def test_public_cli_rejects_noncanonical_environments(name):
    with pytest.raises(SystemExit):
        public_eval.main([f"theta_bench/{name}-L1-S", "--dry-run"])


@pytest.mark.parametrize(
    "args", [("--episodes", "0"), ("--seed-start", "-1"), ("--max-steps", "3001")]
)
def test_public_cli_rejects_invalid_budget(args):
    with pytest.raises(ValueError):
        config(*args)


def test_dry_run_needs_no_server_or_assets(capsys):
    public_eval.main([public_eval.ENV_IDS[0], "--dry-run"])
    recorded = json.loads(capsys.readouterr().out)
    assert recorded[0]["episode_start"] == 10000


def test_model_only_cli_defaults_to_first_canonical_condition(capsys):
    public_eval.main(["--model", "lab/checkpoint", "--dry-run"])
    recorded = json.loads(capsys.readouterr().out)
    assert len(recorded) == 1 and recorded[0]["env_id"] == public_eval.ENV_IDS[0]


def test_model_and_external_server_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        public_eval.parser().parse_args(["--model", "lab/model", "--host", "localhost"])


def test_managed_server_resets_inference_seeds_and_releases_its_port():
    received = []

    def policy(request):
        received.append((request["theta_episode_seed"], request["theta_request_index"]))
        return np.zeros((1, 36), dtype=np.float32)

    policy.metadata = {"policy": "test", "checkpoint": "sha256:test"}
    for start in (10000, 10000):
        with serve_policy(policy, seed_start=start) as port:
            client = HttpActionClient("127.0.0.1", port)
            for reset in (True, False, True):
                action, _, _ = client.query_action(
                    {}, "instruction", {}, {}, history={"reset": reset}
                )
                assert action.shape == (1, 36)
        with socket.socket() as connection:
            assert connection.connect_ex(("127.0.0.1", port)) != 0
    assert received == [(10000, 0), (10000, 1), (10001, 0)] * 2


def test_policy_exception_is_visible_to_the_evaluation_client():
    def policy(request):
        raise ValueError("Expected state shape (1, 32), got (1, 36)")

    policy.metadata = {"policy": "failing-adapter"}
    with serve_policy(policy, seed_start=10000) as port:
        client = HttpActionClient("127.0.0.1", port)
        with pytest.raises(RuntimeError) as caught:
            client.query_action({}, "instruction", {}, {})

    message = str(caught.value)
    assert f"http://127.0.0.1:{port}/act" in message
    assert "HTTP 500" in message
    assert "Expected state shape (1, 32), got (1, 36)" in message


def test_non_json_http_error_includes_bounded_response_details():
    class UnavailableHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.send_response(503)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Policy worker unavailable\n" + b"diagnostic " * 1000)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), UnavailableHandler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = HttpActionClient("127.0.0.1", server.server_port)
        try:
            with pytest.raises(RuntimeError) as caught:
                client.query_action({}, "instruction", {}, {})
        finally:
            server.shutdown()
            thread.join(timeout=5)

    message = str(caught.value)
    assert client.url in message
    assert "HTTP 503" in message and "Policy worker unavailable" in message
    assert "\n" not in message and len(message) < 1200


def test_unreachable_policy_server_preserves_connection_failure():
    with socket.socket() as reserved_port:
        reserved_port.bind(("127.0.0.1", 0))
        client = HttpActionClient("127.0.0.1", reserved_port.getsockname()[1])
        with pytest.raises(RuntimeError) as caught:
            client.query_action({}, "instruction", {}, {})

    message = str(caught.value)
    assert client.url in message
    assert "Connection refused" in message


def test_example_server_supports_custom_http_policy_and_episode_reset():
    path = Path(__file__).resolve().parents[1] / "examples/serve_policy.py"
    spec = importlib.util.spec_from_file_location("example_server", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    requests = []

    def predict(request):
        requests.append(request)
        action = np.zeros((2, 36), dtype=np.float32)
        action[:, 31] = 0.74
        return action

    with HTTPServer(
        ("127.0.0.1", 0), example.make_handler(predict, "custom")
    ) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            policy = make_remote_task_policy(
                "http",
                host="127.0.0.1",
                port=server.server_port,
                action_schema="decoupled_v1",
            )
            observation = {
                "joint_qpos": np.zeros(43),
                "head_stereo_left": np.full((8, 10, 3), 64, np.uint8),
            }
            for _ in range(2):
                chunk = policy.predict(observation, "Pick the tool.")
                assert len(chunk) == 2 and chunk.schema == "decoupled_v1"
            policy.reset()
            policy.predict(observation, "Pick the tool.")
            assert [request["history"].get("reset", False) for request in requests] == [
                True,
                False,
                True,
            ]
            assert requests[0]["instruction"] == "Pick the tool."
            assert requests[0]["state"]["states"].shape == (1, 32)
            np.testing.assert_array_equal(
                requests[0]["image"]["rgb_head_stereo_left"],
                observation["head_stereo_left"],
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)


def make_run(tmp_path, monkeypatch, *, episodes=1):
    selected = config("--episodes", str(episodes))
    (tmp_path / "evaluation_config.json").write_text(
        json.dumps({**asdict(selected), "eval_dir": str(tmp_path)})
    )
    stats = {f"episode_{seed}": False for seed in range(10000, 10000 + episodes)}
    (tmp_path / "evaluation_result.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "episodes": episodes,
                "stats": stats,
                "success_rate": 0.0,
            }
        )
    )
    (tmp_path / "summaries").mkdir()
    for episode in stats:
        (tmp_path / "summaries" / f"{episode}.json").write_text(
            json.dumps(
                {
                    "episode": episode,
                    "step": 3000,
                    "success": False,
                    "terminated": False,
                    "truncated": True,
                    "outcome": "timeout",
                }
            )
        )
        folder = tmp_path / "videos" / episode
        folder.mkdir(parents=True)
        for camera in validation.CAMERAS:
            (folder / f"{camera}_failed.mp4").touch()
    monkeypatch.setattr(
        validation, "validate_video", lambda path, steps: {"decoded_sha256": path.stem}
    )
    return selected


def test_short_run_is_diagnostic(tmp_path, monkeypatch):
    selected = make_run(tmp_path, monkeypatch)
    result = validation.validate_run(tmp_path, selected)
    assert result["reportable"] is False
    assert result["episodes"] == 1 and result["success_rate"] == 0


def test_complete_fixed_budget_is_reportable(tmp_path, monkeypatch):
    selected = make_run(tmp_path, monkeypatch, episodes=100)
    result = validation.validate_run(tmp_path, selected)
    assert result["reportable"] is True and len(result["videos"]) == 400


def test_pose_example_is_not_a_reportable_model_result(tmp_path, monkeypatch):
    selected = make_run(tmp_path, monkeypatch, episodes=100)
    (tmp_path / "policy_info.json").write_text(
        json.dumps({"policy": "example-hold", "diagnostic": True})
    )
    assert validation.validate_run(tmp_path, selected)["reportable"] is False


@pytest.mark.parametrize(
    "fault",
    [
        "missing_seed",
        "missing_camera",
        "duplicate_camera",
        "incomplete",
        "changed_config",
    ],
)
def test_invalid_results_do_not_receive_a_validated_score(tmp_path, monkeypatch, fault):
    selected = make_run(tmp_path, monkeypatch)
    if fault == "missing_seed":
        (tmp_path / "summaries/episode_10000.json").unlink()
    elif fault == "missing_camera":
        (tmp_path / "videos/episode_10000/wrist_left_failed.mp4").unlink()
    elif fault == "duplicate_camera":
        monkeypatch.setattr(
            validation, "validate_video", lambda *args: {"decoded_sha256": "same"}
        )
    elif fault == "changed_config":
        selected = replace(selected, split="train")
    else:
        path = tmp_path / "summaries/episode_10000.json"
        summary = json.loads(path.read_text())
        summary.update(truncated=False, outcome="agent_exhausted")
        path.write_text(json.dumps(summary))
    with pytest.raises(ValueError):
        validation.validate_run(tmp_path, selected)


def test_video_validator_decodes_frames_and_rejects_wrong_count(tmp_path):
    from theta_bench.envs.video_writer import VideoWriter

    path = tmp_path / "camera.mp4"
    writer = VideoWriter(str(path), 50, (640, 360))
    for value in (40, 80, 120):
        writer.write(np.full((360, 640, 3), value, np.uint8))
    writer.release(False)
    path = tmp_path / "camera_failed.mp4"
    assert validation.validate_video(path, 2)["frames"] == 3
    with pytest.raises(ValueError, match="frames"):
        validation.validate_video(path, 3)


def test_video_validator_rejects_black_middle_frame(tmp_path):
    from theta_bench.envs.video_writer import VideoWriter

    writer = VideoWriter(str(tmp_path / "camera.mp4"), 50, (640, 360))
    for value in (100, 0, 100):
        writer.write(np.full((360, 640, 3), value, np.uint8))
    writer.release(False)
    with pytest.raises(ValueError, match="Black camera frame"):
        validation.validate_video(tmp_path / "camera_failed.mp4", 2)
