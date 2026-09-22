"""Optional, concise Slack lifecycle notifications for training and evaluation."""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from urllib.parse import urlsplit

import requests
from dotenv import dotenv_values

from humanoidtoolbench.scenario_names import canonicalize_env_id

ENV_FILE = Path(__file__).resolve().parents[3] / ".env"


def send_message(text: str) -> bool:
    """Return Slack's acknowledgement without exposing webhook credentials."""
    try:
        url = os.environ.get("SLACK_WEBHOOK_URL")
        if url is None:
            url = dotenv_values(ENV_FILE).get("SLACK_WEBHOOK_URL")
        url = (url or "").strip()
        if not url:
            return False
        if not re.fullmatch(
            r"https://hooks\.slack\.com/services/[^/?#\s]+/[^/?#\s]+/[^/?#\s]+",
            url,
        ):
            print(
                "Slack notification skipped: invalid SLACK_WEBHOOK_URL.",
                file=sys.stderr,
            )
            return False
        response = requests.post(
            url,
            json={"text": text, "unfurl_links": False, "unfurl_media": False},
            timeout=(3, 5),
            allow_redirects=False,
        )
        if response.status_code == 200 and response.text.strip() == "ok":
            return True
        # Only Slack's short error code is safe to log, never arbitrary bodies.
        body = response.text.strip()
        reason = body if re.fullmatch(r"[a-z_]{1,64}", body) else "unexpected_response"
        print(
            f"Slack notification failed: HTTP {response.status_code} ({reason}).",
            file=sys.stderr,
        )
    except Exception as error:
        # Notification failures must not change the training outcome. Exception
        # messages can contain the secret URL, so report only the exception type.
        print(f"Slack notification failed: {type(error).__name__}.", file=sys.stderr)
    return False


def _read_metadata(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def dataset_identity(dataset: Path | str) -> dict:
    """Read small preparation metadata; task complexity is separate from DR."""
    path = Path(dataset).expanduser()
    preparation = _read_metadata(path / "meta" / "psi0_preparation.json")
    joint = _read_metadata(path / "meta" / "multitask_training.json")
    if not joint:
        real = _read_metadata(path / "meta" / "real_g1_training.json")
        if real.get("protocol") == "humanoidtoolbench_real_g1_selected_demonstrations_v1":
            joint = real
    scenarios = joint.get("scenario_ids", [])
    condition_count = (
        len({canonicalize_env_id(env) for env in scenarios})
        if isinstance(scenarios, list)
        and all(isinstance(value, str) and value for value in scenarios)
        else 0
    )
    source = preparation.get("source") or str(path)
    if not isinstance(source, str):
        source = str(path)
    task, dr_level = "unknown", None
    for candidate in (source, str(path)):
        matches = list(
            re.finditer(
                r"(?:^|/)(G1\w+-L[012]-[SR](?:-v0)?|[^/]+?-v\d+)" r"(?=/|$|-level-)",
                candidate,
            )
        )
        match = matches[-1] if matches else None
        if match:
            if task == "unknown":
                task = canonicalize_env_id(match[1])
            level = re.match(r"(?:/|-)level-(\d+)(?=/|$|-)", candidate[match.end() :])
        else:
            level = re.search(r"(?:^|/)level-(\d+)$", candidate)
        if dr_level is None and level:
            dr_level = int(level[1])
    if condition_count > 1:
        task = f"Joint training: {condition_count} conditions"
        dr_level = None
    elif task == "unknown":
        try:
            with (path / "meta" / "tasks.jsonl").open() as stream:
                entry = json.loads(stream.readline())
            if isinstance(entry, dict) and isinstance(entry.get("task"), str):
                task = entry["task"] or "unknown"
        except (OSError, ValueError):
            pass
    episodes = joint.get("total_episodes") if condition_count > 1 else None
    if episodes is None:
        episodes = preparation.get("episodes")
    if episodes is None:
        episodes = _read_metadata(path / "meta" / "info.json").get("total_episodes")
    details = []
    if condition_count > 1:
        details.append(f"conditions={condition_count}")
    if dr_level is not None:
        details.append(f"DR={dr_level}")
    if isinstance(episodes, int) and episodes >= 0:
        details.append(f"episodes={episodes}")
    label = canonicalize_env_id(path.name) or str(path)
    if details:
        label += f" ({', '.join(details)})"
    return {"task": task, "data": label, "source": source, "dr_level": dr_level}


def _secret_values() -> list[str]:
    # Use values only for redaction, never include the environment in messages.
    values = list(dotenv_values(ENV_FILE).items()) + list(os.environ.items())
    return sorted(
        {
            value
            for key, value in values
            if re.search(r"TOKEN|SECRET|PASSWORD|API_KEY|WEBHOOK", key, re.I)
            and value
            and len(value) >= 6
        },
        key=len,
        reverse=True,
    )


def _safe_text(value: object, secrets: list[str], limit: int = 240) -> str:
    text = str(value)
    for secret in secrets:
        text = text.replace(secret, "[redacted]")
    text = re.sub(r"https://hooks\.slack\.com/[^\s<>]+", "[redacted webhook]", text)
    text = re.sub(r"\b(?:hf_|xox[baprs]-)[\w-]+", "[redacted token]", text)
    text = re.sub(r"(?i)\bBearer\s+[^\s,;]+", "Bearer [redacted]", text)
    text = re.sub(
        r"(?i)([\"']?[\w-]*(?:api[_-]?key|token|secret|password)[\w-]*[\"']?\s*[=:]\s*)"
        r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)",
        r"\1[redacted]",
        text,
    )
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = " ".join(text.split())
    return escape(text if len(text) <= limit else text[: limit - 3] + "...")


def _log_error(log_path: Path | str | None) -> str | None:
    """Extract a diagnostic line from a bounded tail, never upload whole logs."""
    if log_path is None:
        return None
    try:
        with Path(log_path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 8192))
            lines = stream.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        line = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", line).strip()
        if "ChildFailedError" in line or line.startswith(("raise ", "File ")):
            continue
        if re.search(
            r"\b\w*(?:Error|Exception):|out of memory|No space left|"
            r"Permission denied|SIGKILL|^Killed$",
            line,
        ):
            return line
    return None


def _valid_wandb_url(value: object) -> str | None:
    if not isinstance(value, str) or re.search(r"[\s<>\"'\x00-\x1f]", value):
        return None
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme in ("http", "https")
            and parsed.hostname
            and not parsed.username
            and not parsed.password
            and not parsed.query
            and not parsed.fragment
        ):
            return value
    except ValueError:
        pass
    return None


def wandb_url_from_log(log_path: Path | str | None) -> str | None:
    """Read the native run URL from a bounded head and tail of this launch's log."""
    if log_path is None:
        return None
    try:
        with Path(log_path).open("rb") as stream:
            head = stream.read(65536)
            stream.seek(0, 2)
            size = stream.tell()
            stream.seek(max(len(head), size - 65536))
            text = head + b"\n" + stream.read(65536)
        text = re.sub(
            r"\x1b\[[0-?]*[ -/]*[@-~]", "", text.decode("utf-8", errors="replace")
        )
        for line in reversed(text.splitlines()):
            for match in re.finditer(
                r"https?://[^\s<>\"'\x1b]+/runs/[A-Za-z0-9_-]+", line
            ):
                url = _valid_wandb_url(match[0])
                if url is None:
                    continue
                parsed = urlsplit(url)
                if parsed.hostname == "api.wandb.ai":
                    continue
                if not re.fullmatch(r"/[^/]+/[^/]+/runs/[^/]+", parsed.path):
                    continue
                if parsed.hostname in ("wandb.ai", "www.wandb.ai", "app.wandb.ai") or (
                    ("wandb:" in line.lower() and "view run" in line.lower())
                    or "track this run" in line.lower()
                ):
                    return url
    except (OSError, ValueError):
        pass
    return None


def active_wandb_url(output: Path | str | None) -> str | None:
    """Use only an already initialized online SDK run belonging to this RL output."""
    if output is None:
        return None
    try:
        run = getattr(sys.modules.get("wandb"), "run", None)
        if run is None or getattr(run.settings, "mode", None) != "online":
            return None
        log_dir = run.config.get("log_dir")
        if not log_dir or Path(log_dir).resolve() != Path(output).resolve():
            return None
        return _valid_wandb_url(run.url)
    except Exception:
        return None


@dataclass
class RunNotification:
    """One START and one terminal message per parent process execution."""

    kind: str
    model: str
    task: str
    data: str
    output: Path | str | None = None
    checkpoint: Path | str | None = None
    enabled: bool = True
    stage: str = "setup"
    log_path: Path | str | None = None
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    wandb_url: str | None = None
    _started_at: float | None = field(default=None, init=False, repr=False)
    _finished: bool = field(default=False, init=False, repr=False)

    def start(self) -> None:
        if not self.enabled or self._started_at is not None:
            return
        self._started_at = time.monotonic()
        self._emit("START")

    def finish(
        self,
        success: bool,
        error: BaseException | str | None = None,
        summary: str | None = None,
    ) -> None:
        if not self.enabled or self._started_at is None or self._finished:
            return
        self._finished = True
        self._emit("COMPLETED" if success else "FAILED", error, summary)

    def _emit(
        self,
        status: str,
        error: BaseException | str | None = None,
        summary: str | None = None,
    ) -> None:
        try:
            secrets = _secret_values()

            def safe(value: object, limit: int = 240) -> str:
                return _safe_text(value, secrets, limit)

            lines = [
                f"🔥 [{status}] HUMANOIDTOOLBENCH {safe(self.kind, 24)} | "
                f"{safe(self.model, 80)} | {safe(canonicalize_env_id(self.task), 120)}",
                f"Data: {safe(self.data)}",
            ]
            if self.kind == "training" and os.environ.get("HUMANOIDTOOLBENCH_TRAINING_SCOPE"):
                lines.append(f"Scope: {safe(os.environ['HUMANOIDTOOLBENCH_TRAINING_SCOPE'])}")
            run = Path(self.output).name if self.output is not None else "pending"
            timing = ""
            if status != "START" and self._started_at is not None:
                seconds = max(0, int(time.monotonic() - self._started_at))
                timing = (
                    f" | {seconds // 3600:02}:{seconds // 60 % 60:02}:{seconds % 60:02}"
                )
            lines.append(f"Run: {safe(run, 100)} | id={self.run_id}{timing}")
            if status == "FAILED":
                reason = (
                    f"{type(error).__name__}: {error}"
                    if isinstance(error, BaseException)
                    else error or "Execution ended before completion"
                )
                if isinstance(error, BaseException):
                    cause = error.__cause__
                    if cause is None and not error.__suppress_context__:
                        cause = error.__context__
                    if cause is not None:
                        reason += f" (during {type(cause).__name__}: {cause})"
                detail = _log_error(self.log_path)
                if detail and detail not in reason:
                    reason += f"; log: {detail}"
                lines.append(
                    f"Stage: {safe(self.stage, 64)} | Reason: {safe(reason, 400)}"
                )
            elif summary:
                lines.append(f"Result: {safe(summary)}")
            wandb_url = wandb_url_from_log(self.log_path)
            if not wandb_url and self.kind == "training":
                wandb_url = active_wandb_url(self.output)
            wandb_url = wandb_url or _valid_wandb_url(self.wandb_url)
            if wandb_url:
                label = "W&B training run" if self.kind == "evaluation" else "W&B"
                lines.append(f"{label}: {safe(wandb_url, 600)}")
            if self.checkpoint is not None:
                lines.append(f"Checkpoint: {safe(self.checkpoint, 320)}")
            if self.log_path is not None:
                lines.append(f"Log: {safe(self.log_path, 400)}")
            if self.output is not None:
                lines.append(f"Output: {safe(self.output, 400)}")
            send_message("\n".join(lines))
        except Exception as notification_error:
            print(
                f"Slack notification failed: {type(notification_error).__name__}.",
                file=sys.stderr,
            )


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    title = f"HumanoidToolBench direct Slack test {stamp}"
    if send_message(title + "\nThis is a connection test. No training was started."):
        print(f"Slack accepted the test message: {title}")
        return 0
    print(
        "Slack test failed. Check SLACK_WEBHOOK_URL in .env and the error above.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
