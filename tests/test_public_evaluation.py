"""External policies and canonical score admission without model checkpoints."""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
from dataclasses import asdict, replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread

import numpy as np
import pytest

from humanoidtoolbench.cli import public_eval
from humanoidtoolbench.evals import public_validation as validation
from humanoidtoolbench.policies.http_client import HttpActionClient
from humanoidtoolbench.policies.http_server import serve_policy
from humanoidtoolbench.policies.remote_humanoid import make_remote_task_policy


def config(*args):
    return public_eval.configurations(
        public_eval.parser().parse_args([public_eval.ENV_IDS[0], *args])
    )[0]


def test_public_suite_selects_all_18_canonical_conditions():
    configs = public_eval.configurations(public_eval.parser().parse_args(["all"]))
    assert len({item.env_id for item in configs}) == 18
    assert {item.env_id for item in configs} == {
        f"humanoidtoolbench/G1{task}-L{level}-{mode}"
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
        assert item.save_video and item.headless


@pytest.mark.parametrize(
    "name", ["G1BallMoveEasyGap50Attach", "G1IceBreakEasyNearTouch", "G1BallMoveOOD"]
)
def test_public_cli_rejects_noncanonical_environments(name):
    with pytest.raises(SystemExit):
        public_eval.main([f"humanoidtoolbench/{name}-L1-S", "--dry-run"])


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


def test_in_process_policy_is_exclusive_with_model_and_host():
    for other in (("--model", "lab/model"), ("--host", "localhost")):
        with pytest.raises(SystemExit):
            public_eval.parser().parse_args(["--policy", "my_policy:predict", *other])
    with pytest.raises(ValueError):
        config("--checkpoint", "sha256:abc")


def test_in_process_policy_dry_run_does_not_import_it(capsys):
    public_eval.main(["--policy", "no_such_module:predict", "--dry-run"])
    assert json.loads(capsys.readouterr().out)[0]["policy"] == "http"


def test_callable_policy_serves_like_a_native_model():
    from humanoidtoolbench.policies.loader import CallablePolicy

    def predict(request):
        return np.zeros((2, 36), dtype=np.float32)

    policy = CallablePolicy(predict, name="my_policy:predict", checkpoint="rev-1")
    assert (
        policy.metadata["checkpoint"] == "rev-1" and not policy.metadata["diagnostic"]
    )
    with serve_policy(policy, seed_start=10000) as port:
        client = HttpActionClient("127.0.0.1", port)
        action, _, _ = client.query_action(
            {}, "instruction", {}, {}, history={"reset": True}
        )
        assert action.shape == (2, 36)
        import urllib.request

        with urllib.request.urlopen(
            f"http://127.0.0.1:{port}/info", timeout=5
        ) as reply:
            info = json.loads(reply.read())
    assert info["policy"] == "my_policy:predict" and info["checkpoint"] == "rev-1"


SOURCE = {"cli/public_eval.py": "a" * 64}
ASSETS = {"manifest_sha256": "m" * 64}


def _receipt(
    config, *, name="run-abc", status="validated", source=SOURCE, assets=ASSETS, **info
):
    run_dir = Path(config.eval_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": status,
        "configuration": asdict(config),
        "policy_info": {"policy": "p", "checkpoint": "c", **info},
        "source_sha256": source,
        "assets": assets,
        "successes": 3,
        "episodes": 100,
        "success_rate": 0.03,
        "reportable": True,
        "wall_seconds": 12.0,
    }
    path = run_dir / "benchmark_result.json"
    path.write_text(json.dumps(receipt))
    return path


def _fingerprint(**info):
    return public_eval.run_fingerprint(
        {
            "policy_info": {"policy": "p", "checkpoint": "c", **info},
            "source_sha256": SOURCE,
            "assets": ASSETS,
        }
    )


def test_completed_result_requires_identical_policy_code_and_assets(tmp_path):
    selected = replace(config(), eval_dir=str(tmp_path / "G1BallMove-L0-S"))
    current = _fingerprint()
    assert public_eval.completed_result(selected, current) == (None, 0, 0)
    path = _receipt(selected)
    assert public_eval.completed_result(selected, current) == (path, 1, 0)
    # Anything that changes what produced the result forces a new evaluation.
    for changed in (
        _fingerprint(checkpoint="d"),
        _fingerprint(adapter_sha256="b" * 64),
        {**current, "source_digest": public_eval.source_digest({"x": "y"})},
        {**current, "assets_manifest_sha256": "n" * 64},
    ):
        assert public_eval.completed_result(selected, changed) == (None, 0, 1)
    assert (
        public_eval.completed_result(replace(selected, num_episodes=1), current)[0]
        is None
    )
    # A policy without any checkpoint identifier is never reused.
    assert (
        public_eval.completed_result(selected, _fingerprint(checkpoint=None))[0] is None
    )


def test_completed_result_prefers_the_newest_match_and_ignores_failed_runs(tmp_path):
    import os

    selected = replace(config(), eval_dir=str(tmp_path / "G1BallMove-L0-S"))
    older = _receipt(selected, name="run-zzz")
    newer = _receipt(selected, name="run-aaa")
    os.utime(older, (1_000, 1_000))
    _receipt(selected, name="run-failed", status="failed")
    assert public_eval.completed_result(selected, _fingerprint()) == (newer, 2, 0)


def test_native_checkpoints_are_identified_by_weight_hash_not_cache_path():
    first = public_eval.policy_identity(
        {
            "policy": "act",
            "checkpoint": "/home/a/cache/model.safetensors",
            "checkpoint_sha256": "h",
        }
    )
    second = public_eval.policy_identity(
        {
            "policy": "act",
            "checkpoint": "/mnt/b/cache/model.safetensors",
            "checkpoint_sha256": "h",
        }
    )
    assert first == second and first["checkpoint"] is None


def _row(env, status="completed", rate=0.1, fingerprint=None, reportable=True):
    return {
        "env_id": f"humanoidtoolbench/{env}",
        "status": status,
        "successes": int(rate * 100),
        "episodes": 100,
        "success_rate": rate,
        "reportable": reportable,
        "wall_seconds": 1.0,
        "result": env,
        "fingerprint": fingerprint or _fingerprint(),
    }


def test_summary_reports_mean_only_when_every_condition_has_a_result(tmp_path):
    rows = [_row("G1BallMove-L0-S", rate=0.1), _row("G1BallMove-L0-R", "skipped", 0.3)]
    payload = json.loads(public_eval.write_summary(tmp_path, rows).read_text())
    assert payload["mean_success_rate"] == pytest.approx(0.2) and payload["reportable"]
    assert payload["fingerprint"] == _fingerprint()
    text = (tmp_path / "summary.md").read_text()
    assert (
        "| G1BallMove-L0-R | 30 | 100 | 30.0% | yes | skipped | G1BallMove-L0-R |"
        in text
    )
    rows.append(
        {
            "env_id": "humanoidtoolbench/G1IceBreak-L2-R",
            "status": "failed",
            "error": "boom",
        }
    )
    payload = json.loads(public_eval.write_summary(tmp_path, rows).read_text())
    assert payload["mean_success_rate"] is None and payload["failed"] == 1
    assert not payload["reportable"] and payload["not_reportable_because"]


def test_summary_is_not_reportable_when_conditions_come_from_different_runs(tmp_path):
    rows = [
        _row("G1BallMove-L0-S"),
        _row("G1BallMove-L0-R", "skipped", fingerprint=_fingerprint(checkpoint="old")),
    ]
    payload = json.loads(public_eval.write_summary(tmp_path, rows).read_text())
    assert payload["mean_success_rate"] == pytest.approx(0.1)
    assert not payload["reportable"] and payload["fingerprint"] is None
    assert "different policies" in " ".join(payload["not_reportable_because"])
    assert "Reportable: no;" in (tmp_path / "summary.md").read_text()


def test_code_identity_records_version_and_commit():
    identity = public_eval.code_identity()
    assert identity["package_version"]
    assert identity["git_commit"] is None or len(identity["git_commit"]) == 40


def test_mesh_directory_does_not_depend_on_the_working_directory():
    from humanoidtoolbench.assets import tools

    if "HUMANOIDTOOLBENCH_MS_ASSETS" not in __import__("os").environ:
        assert tools.MS_ASSETS_DIR.is_absolute()
        assert (
            tools.MS_ASSETS_DIR
            == Path(__file__).resolve().parents[1] / "data/ms_assets"
        )


def test_bare_environment_ids_are_accepted(capsys):
    public_eval.main(["G1BallRetrieve-L2-R", "--dry-run"])
    recorded = json.loads(capsys.readouterr().out)
    assert recorded[0]["env_id"] == "humanoidtoolbench/G1BallRetrieve-L2-R"


def test_unknown_environment_points_at_list_envs(capsys):
    with pytest.raises(SystemExit):
        public_eval.main(["G1BallMove-L3-S", "--dry-run"])
    assert "--list-envs" in capsys.readouterr().err


def test_list_envs_prints_the_18_conditions(capsys):
    public_eval.main(["--list-envs"])
    assert capsys.readouterr().out.split() == list(public_eval.ENV_IDS)


def test_missing_policy_server_fails_before_simulation(capsys, monkeypatch):
    monkeypatch.delenv("HUMANOIDTOOLBENCH_PREFLIGHT_DONE", raising=False)
    monkeypatch.setattr(
        public_eval.os,
        "execv",
        lambda *args: pytest.fail("reached the simulator re-exec"),
    )
    # A bound, non-listening socket refuses connections for as long as it is open.
    with socket.socket() as reserved_port:
        reserved_port.bind(("127.0.0.1", 0))
        port = reserved_port.getsockname()[1]
        with pytest.raises(SystemExit) as caught:
            public_eval.main(
                ["--host", "127.0.0.1", "--port", str(port), "--episodes", "1"]
            )
    assert caught.value.code == 2
    message = capsys.readouterr().err
    assert f"http://127.0.0.1:{port}/info" in message
    assert "examples/serve_policy.py" in message and "--model" in message


def test_preflight_accepts_servers_without_an_info_endpoint():
    class ActOnly(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_error(404)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), ActOnly) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            assert (
                public_eval.policy_server_unreachable("127.0.0.1", server.server_port)
                is None
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)


def _serve(handler):
    from http.server import ThreadingHTTPServer

    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def _post(port, session, reset=False):
    import requests

    from humanoidtoolbench.policies.http_client import _encode

    payload = {
        "image": {},
        "instruction": "x",
        "history": {"reset": reset},
        "state": {"states": np.zeros((1, 32), np.float32)},
        "session": session,
    }
    return requests.post(
        f"http://127.0.0.1:{port}/act", json=_encode(payload), timeout=5
    )


def test_policy_server_refuses_a_second_evaluator_while_the_first_is_active():
    from humanoidtoolbench.policies.http_server import make_handler

    seen = []

    def predict(request):
        seen.append(request["session"])
        return np.zeros((1, 36), np.float32)

    server, thread = _serve(make_handler(predict, "stateful", checkpoint="c"))
    port = server.server_port
    try:
        assert _post(port, "first", reset=True).status_code == 200
        refused = _post(port, "second", reset=True)
        assert refused.status_code == 409
        assert "one server per evaluator" in refused.json()["error"]
        assert _post(port, "first").status_code == 200
        assert seen == ["first", "first"]
        # The preflight reports the busy server before a simulator starts.
        problem = public_eval.policy_server_unreachable("127.0.0.1", port)
        assert problem and "serving another evaluator" in problem
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_released_session_frees_the_server_for_the_next_evaluator():
    import requests

    from humanoidtoolbench.policies.http_server import make_handler

    server, thread = _serve(
        make_handler(lambda request: np.zeros((1, 36), np.float32), "p")
    )
    port = server.server_port
    try:
        assert _post(port, "first", reset=True).status_code == 200
        assert public_eval.policy_server_unreachable("127.0.0.1", port)
        # Another session cannot release the first one.
        requests.post(
            f"http://127.0.0.1:{port}/release", json={"session": "x"}, timeout=5
        )
        assert _post(port, "second").status_code == 409
        requests.post(
            f"http://127.0.0.1:{port}/release", json={"session": "first"}, timeout=5
        )
        assert public_eval.policy_server_unreachable("127.0.0.1", port) is None
        assert _post(port, "second", reset=True).status_code == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_evaluator_process_releases_its_session_at_exit():
    import subprocess

    from humanoidtoolbench.policies.http_server import make_handler

    server, thread = _serve(
        make_handler(lambda request: np.zeros((1, 36), np.float32), "p")
    )
    port = server.server_port
    code = (
        "import numpy as np\n"
        "from humanoidtoolbench.policies.http_client import HttpActionClient\n"
        f"HttpActionClient('127.0.0.1', {port}).query_action({{}}, 'x', "
        "{'states': np.zeros((1, 32), np.float32)}, {}, history={'reset': True})\n"
    )
    try:
        subprocess.run([sys.executable, "-c", code], check=True, timeout=60)
        assert public_eval.policy_server_unreachable("127.0.0.1", port) is None
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_requests_without_a_session_are_one_anonymous_evaluator():
    from humanoidtoolbench.policies.http_server import make_handler

    server, thread = _serve(
        make_handler(lambda request: np.zeros((1, 36), np.float32), "p")
    )
    try:
        assert _post(server.server_port, None).status_code == 200
        assert _post(server.server_port, "new").status_code == 409
        assert _post(server.server_port, None).status_code == 200
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_malformed_requests_get_http_400():
    import requests

    from humanoidtoolbench.policies.http_server import make_handler

    server, thread = _serve(make_handler(lambda request: None, "p"))
    try:
        for body in ("[1, 2]", "not json"):
            reply = requests.post(
                f"http://127.0.0.1:{server.server_port}/act", data=body, timeout=5
            )
            assert reply.status_code == 400 and "Malformed" in reply.json()["error"]
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_preflight_accepts_a_custom_server_whose_info_is_not_json():
    class PlainInfo(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, *args):
            pass

    server, thread = _serve(PlainInfo)
    try:
        assert (
            public_eval.policy_server_unreachable("127.0.0.1", server.server_port)
            is None
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_shared_policy_server_can_be_allowed_for_stateless_policies():
    from humanoidtoolbench.policies.http_server import make_handler

    handler = make_handler(
        lambda request: np.zeros((1, 36), np.float32), "stateless", exclusive=False
    )
    server, thread = _serve(handler)
    try:
        assert _post(server.server_port, "first").status_code == 200
        assert _post(server.server_port, "second").status_code == 200
        assert (
            public_eval.policy_server_unreachable("127.0.0.1", server.server_port)
            is None
        )
    finally:
        server.shutdown()
        thread.join(timeout=5)


def test_evaluator_requests_carry_one_session_per_process():
    from humanoidtoolbench.policies import http_client

    sessions = []

    def policy(request):
        sessions.append(request["session"])
        return np.zeros((1, 36), np.float32)

    policy.metadata = {"policy": "t", "checkpoint": "c"}
    with serve_policy(policy, seed_start=10000) as port:
        for _ in range(2):
            HttpActionClient("127.0.0.1", port).query_action(
                {}, "x", {}, {}, history={"reset": True}
            )
    assert sessions == [http_client.EVALUATOR_SESSION] * 2


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (np.zeros((2, 32), np.float32), "shape (T, 36) with T >= 1; got shape (2, 32)"),
        (np.full((3, 36), np.nan, np.float32), "NaN or infinite values in 3 of 3 rows"),
    ],
)
def test_policy_errors_name_the_problem_and_the_failing_line(action, expected):
    def predict(request):
        return action

    predict.metadata = {"policy": "broken"}
    with serve_policy(predict, seed_start=10000) as port:
        with pytest.raises(RuntimeError) as caught:
            HttpActionClient("127.0.0.1", port).query_action({}, "x", {}, {})
    assert expected in str(caught.value)

    def raising(request):
        raise KeyError("rgb_head_stereo_right")

    raising.metadata = {"policy": "broken"}
    with serve_policy(raising, seed_start=10000) as port:
        with pytest.raises(RuntimeError) as caught:
            HttpActionClient("127.0.0.1", port).query_action({}, "x", {}, {})
    assert "KeyError" in str(caught.value) and "test_public_evaluation.py:" in str(
        caught.value
    )


def test_policy_thread_keeps_thread_local_state_from_import(tmp_path, monkeypatch):
    from humanoidtoolbench.policies.loader import (
        CallablePolicy,
        PolicyThread,
        load_policy_callable,
    )

    monkeypatch.setattr(sys, "path", list(sys.path))
    adapter = tmp_path / "thread_local_adapter_xyz.py"
    adapter.write_text(
        "import threading\n"
        "_state = threading.local()\n"
        "_state.ready = True\n"
        "def predict(request):\n"
        "    return getattr(_state, 'ready', False)\n"
    )
    try:
        runner = PolicyThread()
        predict, _ = runner.submit(
            load_policy_callable, f"{adapter}:predict", with_digest=True
        )
        policy = CallablePolicy(predict, name="t", runner=runner)
        results = []
        worker = Thread(target=lambda: results.append(policy({})))
        worker.start()
        worker.join(timeout=10)
        assert results == [True]  # called from another thread, run on the import thread
        assert predict({}) is False  # a direct call from this thread would not be
    finally:
        sys.modules.pop("thread_local_adapter_xyz", None)


def test_adapter_digest_covers_helper_modules_and_partials(tmp_path, monkeypatch):
    from humanoidtoolbench.policies.loader import CallablePolicy, load_policy_callable

    monkeypatch.setattr(sys, "path", list(sys.path))
    (tmp_path / "digest_helper_xyz.py").write_text("SCALE = 1.0\n")
    adapter = tmp_path / "digest_adapter_xyz.py"
    adapter.write_text(
        "import functools\n"
        "from digest_helper_xyz import SCALE\n"
        "def _predict(request, scale):\n"
        "    return scale * SCALE\n"
        "predict = functools.partial(_predict, scale=2.0)\n"
    )
    names = ("digest_helper_xyz", "digest_adapter_xyz")

    def digest():
        for name in names:
            sys.modules.pop(name, None)
        return load_policy_callable(f"{adapter}:predict", with_digest=True)

    try:
        predict, first = digest()
        assert predict({}) == 2.0 and first
        assert (
            CallablePolicy(predict, name="a", adapter_digest=first).metadata[
                "adapter_sha256"
            ]
            == first
        )
        assert digest()[1] == first  # stable for unchanged files
        (tmp_path / "digest_helper_xyz.py").write_text("SCALE = 3.0\n")
        assert digest()[1] != first  # an imported helper module is covered
    finally:
        for name in names:
            sys.modules.pop(name, None)


def test_example_server_loads_policies_from_cwd_and_file_paths(
    tmp_path, monkeypatch, capsys
):
    path = Path(__file__).resolve().parents[1] / "examples/serve_policy.py"
    spec = importlib.util.spec_from_file_location("example_server_loader", path)
    example = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(example)
    (tmp_path / "cwd_policy.py").write_text(
        "def predict(request):\n    return request\n"
    )
    (tmp_path / "needs_dep.py").write_text("import definitely_missing_dep_xyz\n")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "file_policy.py").write_text("def predict(request):\n    return 3\n")
    monkeypatch.chdir(tmp_path)
    # load_policy inserts directories itself; restore sys.path afterwards.
    monkeypatch.setattr(sys, "path", list(sys.path))
    loaded = (
        "cwd_policy",
        "humanoidtoolbench_policy_cwd_policy",
        "file_policy",
        "needs_dep",
    )
    for name in loaded:
        assert name not in sys.modules
    try:
        assert example.load_policy("cwd_policy:predict")[0]({"a": 1}) == {"a": 1}
        by_path, digest = example.load_policy(f"{tmp_path / 'cwd_policy.py'}:predict")
        assert by_path({"b": 2}) == {"b": 2} and digest
        assert "cwd_policy" in capsys.readouterr().out
        # A file whose name is already importable loads under a private name.
        assert (
            sys.modules["humanoidtoolbench_policy_cwd_policy"]
            is not sys.modules["cwd_policy"]
        )
        # Otherwise it is registered under its own name and its directory is
        # importable, so spawned worker processes can unpickle its functions.
        assert (
            example.load_policy(f"{elsewhere / 'file_policy.py'}:predict")[0]({}) == 3
        )
        assert sys.modules["file_policy"].__file__ == str(elsewhere / "file_policy.py")
        assert str(elsewhere) in sys.path
        for bad in ("cwd_policy", "cwd_policy:missing", "no_such_module:predict"):
            with pytest.raises(SystemExit) as caught:
                example.load_policy(bad)
            assert isinstance(caught.value.code, str) and caught.value.code
        # A missing dependency inside the user's module is not a "module not found" hint.
        with pytest.raises(ModuleNotFoundError, match="definitely_missing_dep_xyz"):
            example.load_policy("needs_dep:predict")
    finally:
        for name in loaded:
            sys.modules.pop(name, None)


def test_managed_server_resets_inference_seeds_and_releases_its_port():
    received = []

    def policy(request):
        received.append(
            (
                request["humanoidtoolbench_episode_seed"],
                request["humanoidtoolbench_request_index"],
            )
        )
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
    from humanoidtoolbench.envs.video_writer import VideoWriter

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
    from humanoidtoolbench.envs.video_writer import VideoWriter

    writer = VideoWriter(str(tmp_path / "camera.mp4"), 50, (640, 360))
    for value in (100, 0, 100):
        writer.write(np.full((360, 640, 3), value, np.uint8))
    writer.release(False)
    with pytest.raises(ValueError, match="Black camera frame"):
        validation.validate_video(tmp_path / "camera_failed.mp4", 2)
