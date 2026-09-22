"""Visual tool-use planning and feedback for language-conditioned policies."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

import numpy as np
import requests

DEFAULT_REASONER_MODEL = "nvidia/Cosmos3-Nano"
PROMPT_VERSION = "humanoidtoolbench-embodied-plan-v3"
FEEDBACK_PROMPT_VERSION = "humanoidtoolbench-embodied-feedback-v2"
GUIDANCE_MAX_WORDS = 25
GUIDANCE_MAX_CHARS = 200
SYSTEM_PROMPT = """You choose the right tool for a humanoid robot's task from its
initial RGB view and the supplied task. Tool selection is YOUR responsibility.
Compare the visible candidate tools by how their shape and other visible features
support the required interaction. Choose the most suitable tool and identify it
by visible appearance and location so the downstream policy can find it.
Give a brief evidence-based reason connecting its visible features to the task,
and plan how to grasp and use that tool to accomplish the ENTIRE task. Do not defer
tool selection to the policy with instructions such as "pick the right tool".
Do not invent hidden object properties, coordinates, or simulator information.
When the view is ambiguous, state the uncertainty instead of inventing certainty.
The downstream robot policy receives ONLY your guidance string once and executes
the task without further planning calls. Keep the original goal and
constraints intact. Return only a JSON object with four non-empty strings:
scene_summary (brief visible evidence), selected_tool (chosen tool's appearance
and location), selection_reason (one brief reason why it suits the task), and
guidance (one short, self-contained imperative sentence using the chosen tool
to achieve the original goal). Name the tool by visible appearance or location
and identify the target as needed; the policy cannot see the other JSON fields.
Do not rely on "it" or "that tool" to identify the chosen tool. Put only what to
do in guidance, without rationale, field labels, numbered steps, or line breaks.
Write in the task's language. Guidance must be at most 25 words and 200 characters.
Do not include private reasoning, code, or low-level joint commands.
"""
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "scene_summary": {"type": "string"},
        "selected_tool": {"type": "string"},
        "selection_reason": {"type": "string"},
        "guidance": {"type": "string"},
    },
    "required": ["scene_summary", "selected_tool", "selection_reason", "guidance"],
    "additionalProperties": False,
}
FEEDBACK_SYSTEM_PROMPT = """You supervise a humanoid robot's tool-use task using RGB
views and the supplied original task. Choose the right tool yourself by its
visible appearance, location, and features that support the required interaction.
Plan how to grasp it, position its working end, interact with the target, and
achieve the original goal. Preserve the task's constraints.
On the first call, assess the current view and plan the task. On later calls,
compare the labeled previous and current views with your previous plan. Assess
whether the tool appears grasped, has slipped or been dropped, whether its working
end is positioned for contact, and whether the target has moved as intended.
Distinguish visible evidence from uncertainty: proximity alone does not prove a
secure grasp or effective contact. Do not invent hidden properties, coordinates,
or simulator information. Previous plans are intentions, not proof of execution.
Keep the previous tool choice and guidance verbatim if they are still appropriate.
Otherwise revise them from the CURRENT state: recover a missed grasp or lost
contact, reposition the tool, or advance past completed steps. Do not restart
completed steps unnecessarily or delegate tool selection to the downstream policy.
Return only a JSON object with five non-empty strings: scene_summary (current
visible evidence), selected_tool (chosen tool's appearance and current location),
selection_reason (brief visible features that suit the task), guidance (one short,
self-contained imperative sentence for the next executable action or subgoal),
and progress_assessment (brief observed progress,
failure signs, or uncertainty, including whether the previous plan needs revision).
The downstream policy receives ONLY guidance, not the original task or other JSON
fields. Name the tool by visible appearance or location and identify the target
as needed; do not rely on "it" or "that tool" alone. Include any original task
constraint needed for the next action. Put only what to do in guidance, without
rationale, field labels, numbered steps, or line breaks. Write in the task's
language. Guidance must be at most 25 words and 200 characters.
Keep scene_summary and progress_assessment brief. Your guidance will be followed
until the next visual review. If the goal appears achieved, give a safe holding
instruction; only the environment decides completion. Do not include private
reasoning, code, or low-level joint commands.
"""
FEEDBACK_PLAN_SCHEMA = {
    **PLAN_SCHEMA,
    "properties": {
        **PLAN_SCHEMA["properties"],
        "progress_assessment": {"type": "string"},
    },
    "required": [*PLAN_SCHEMA["required"], "progress_assessment"],
}


def _encode_image(observation: dict[str, Any]) -> tuple[str, str]:
    from PIL import Image

    image = np.asarray(observation["head_stereo_left"])
    if (
        image.ndim != 3
        or image.shape[-1] != 3
        or min(image.shape[:2]) == 0
        or image.dtype != np.uint8
    ):
        raise ValueError("Reasoner head_stereo_left must be a uint8 RGB image")
    buffer = io.BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    image_bytes = buffer.getvalue()
    return (
        base64.b64encode(image_bytes).decode("ascii"),
        hashlib.sha256(image_bytes).hexdigest(),
    )


@dataclass(frozen=True)
class ReasonerConfig:
    model: str = DEFAULT_REASONER_MODEL
    provider: str = "openai"
    base_url: str | None = None
    api_key_env: str | None = None
    timeout: float = 120.0
    max_tokens: int = 4096
    replan_interval_steps: int | None = None

    def __post_init__(self) -> None:
        if self.provider not in {"openai", "gemini"}:
            raise ValueError("Reasoner provider must be 'openai' or 'gemini'")
        if not self.model or self.model != self.model.strip():
            raise ValueError("Reasoner model must be a non-empty model ID")
        if self.provider == "gemini" and self.model == DEFAULT_REASONER_MODEL:
            raise ValueError(
                "Specify a Gemini model explicitly; Cosmos3-Nano uses openai"
            )
        if not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("Reasoner timeout must be finite and positive")
        if isinstance(self.max_tokens, bool) or not isinstance(self.max_tokens, int):
            raise TypeError("Reasoner max_tokens must be a positive integer")
        if self.max_tokens <= 0:
            raise ValueError("Reasoner max_tokens must be a positive integer")
        if self.replan_interval_steps is not None and (
            isinstance(self.replan_interval_steps, bool)
            or not isinstance(self.replan_interval_steps, int)
            or self.replan_interval_steps <= 0
        ):
            raise ValueError(
                "Reasoner replan_interval_steps must be a positive integer"
            )
        base_url = self.base_url or (
            "https://generativelanguage.googleapis.com/v1beta"
            if self.provider == "gemini"
            else "http://127.0.0.1:8000/v1"
        )
        parsed = urlsplit(base_url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "Reasoner base_url must be HTTP(S) without credentials, query or fragment; "
                "use api_key_env for authentication"
            )
        if self.api_key_env is not None and not re.fullmatch(
            r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env
        ):
            raise ValueError(
                "Reasoner api_key_env must be an environment variable name"
            )
        object.__setattr__(self, "base_url", base_url.rstrip("/"))
        if self.provider == "gemini" and self.api_key_env is None:
            object.__setattr__(self, "api_key_env", "GEMINI_API_KEY")


@dataclass(frozen=True)
class ReasoningResult:
    original_instruction: str
    policy_instruction: str
    scene_summary: str
    selected_tool: str
    selection_reason: str
    guidance: str
    provider: str
    model: str
    latency_seconds: float
    image_sha256: str
    response_id: str | None = None
    model_version: str | None = None
    usage: dict[str, Any] = field(default_factory=dict)
    prompt_version: str = PROMPT_VERSION
    image_key: str = "head_stereo_left"
    progress_assessment: str | None = None
    previous_image_sha256: str | None = None


class VisionReasoner:
    """Stateless HTTP client; the evaluator owns planning cadence and context."""

    def __init__(self, config: ReasonerConfig) -> None:
        self.config = config
        self._api_key = (
            os.environ.get(config.api_key_env) if config.api_key_env else None
        )
        if config.api_key_env and not self._api_key:
            raise ValueError(f"Set {config.api_key_env} before using the reasoner")

    def reason(
        self,
        observation: dict[str, Any],
        instruction: str,
        *,
        previous_result: ReasoningResult | None = None,
        previous_observation: dict[str, Any] | None = None,
        steps_since_plan: int = 0,
    ) -> ReasoningResult:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Reasoner requires a non-empty task instruction")
        config = self.config
        feedback = config.replan_interval_steps is not None
        has_previous = previous_result is not None or previous_observation is not None
        if (
            isinstance(steps_since_plan, bool)
            or not isinstance(steps_since_plan, int)
            or steps_since_plan < 0
        ):
            raise ValueError("Reasoner steps_since_plan must be a non-negative integer")
        if has_previous:
            if not feedback or previous_result is None or previous_observation is None:
                raise ValueError(
                    "Replanning requires feedback mode and both previous inputs"
                )
            if steps_since_plan == 0:
                raise ValueError("Replanning requires positive steps_since_plan")
            if previous_result.original_instruction != instruction:
                raise ValueError(
                    "Replanning must preserve the original task instruction"
                )
        elif steps_since_plan != 0:
            raise ValueError("steps_since_plan requires previous planning context")

        encoded, image_sha256 = _encode_image(observation)
        images = []
        previous_image_sha256 = None
        prompt = f"Task: {instruction}"
        if has_previous:
            previous_encoded, previous_image_sha256 = _encode_image(
                previous_observation
            )
            images.append(
                ("Previous view (when the previous plan was issued):", previous_encoded)
            )
            previous_plan = {
                name: getattr(previous_result, name)
                for name in FEEDBACK_PLAN_SCHEMA["required"]
            }
            prompt += (
                f"\nCompleted control steps since previous plan: {steps_since_plan}"
                f"\nPrevious plan and assessment: {json.dumps(previous_plan, ensure_ascii=False)}"
            )
        images.append(("Current view:", encoded))
        schema = FEEDBACK_PLAN_SCHEMA if feedback else PLAN_SCHEMA
        system_prompt = FEEDBACK_SYSTEM_PROMPT if feedback else SYSTEM_PROMPT
        headers = {"Content-Type": "application/json"}
        if config.provider == "gemini":
            headers["x-goog-api-key"] = self._api_key
            url = f"{config.base_url}/models/{quote(config.model, safe='')}:generateContent"
            parts = []
            for label, image_data in images:
                if has_previous:
                    parts.append({"text": label})
                parts.append(
                    {"inlineData": {"mimeType": "image/png", "data": image_data}}
                )
            parts.append({"text": prompt})
            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [
                    {
                        "role": "user",
                        "parts": parts,
                    }
                ],
                "generationConfig": {
                    "maxOutputTokens": config.max_tokens,
                    "responseMimeType": "application/json",
                    "responseJsonSchema": schema,
                },
            }
        else:
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            url = f"{config.base_url}/chat/completions"
            parts = []
            for label, image_data in images:
                if has_previous:
                    parts.append({"type": "text", "text": label})
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_data}"},
                    }
                )
            parts.append({"type": "text", "text": prompt})
            payload = {
                "model": config.model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": parts,
                    },
                ],
                "max_tokens": config.max_tokens,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "embodied_plan",
                        "strict": True,
                        "schema": schema,
                    },
                },
            }
        started = time.perf_counter()
        try:
            response = requests.post(
                url,
                json=payload,
                headers=headers,
                timeout=(min(10.0, config.timeout), config.timeout),
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise RuntimeError(
                "Reasoner request failed; policy rollout cannot continue"
            ) from exc
        if response.status_code != 200:
            raise RuntimeError(f"Reasoner returned HTTP {response.status_code}")
        try:
            data = response.json()
            if config.provider == "gemini":
                candidate = data["candidates"][0]
                if candidate.get("finishReason") != "STOP":
                    raise ValueError("Gemini did not finish the plan; check max_tokens")
                content = "".join(
                    part["text"]
                    for part in candidate["content"]["parts"]
                    if "text" in part and not part.get("thought", False)
                )
                response_id = data.get("responseId")
                model_version = data.get("modelVersion")
                usage = data.get("usageMetadata", {})
            else:
                choice = data["choices"][0]
                if choice.get("finish_reason") != "stop":
                    raise ValueError(
                        "Reasoner did not finish the plan; check max_tokens"
                    )
                content = choice["message"]["content"]
                response_id = data.get("id")
                model_version = data.get("model")
                usage = data.get("usage", {})
            plan = json.loads(content)
            if not isinstance(plan, dict) or set(plan) != set(schema["required"]):
                raise ValueError(
                    f"Expected {', '.join(schema['required'])} in the JSON plan"
                )
            for name in schema["required"]:
                if not isinstance(plan[name], str) or not plan[name].strip():
                    raise ValueError(f"Reasoner {name} must be a non-empty string")
                plan[name] = plan[name].strip()
            guidance = plan["guidance"]
            if (
                len(guidance.split()) > GUIDANCE_MAX_WORDS
                or len(guidance) > GUIDANCE_MAX_CHARS
            ):
                raise ValueError(
                    "Reasoner guidance exceeds the instruction length limit"
                )
            # Decimal points are not sentence boundaries; preserve the actual text.
            sentence_text = re.sub(r"(?<=\d)\.(?=\d)", "", guidance)
            if len(guidance.splitlines()) != 1 or re.search(
                r"[.!?。！？]\s*\S", sentence_text
            ):
                raise ValueError("Reasoner guidance must be one sentence on one line")
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Reasoner returned an invalid or incomplete JSON plan; "
                "check output format, guidance length/sentence count and max_tokens"
            ) from exc
        return ReasoningResult(
            original_instruction=instruction,
            policy_instruction=guidance,
            scene_summary=plan["scene_summary"],
            selected_tool=plan["selected_tool"],
            selection_reason=plan["selection_reason"],
            guidance=plan["guidance"],
            provider=config.provider,
            model=config.model,
            latency_seconds=time.perf_counter() - started,
            image_sha256=image_sha256,
            response_id=response_id,
            model_version=model_version,
            usage=usage,
            prompt_version=FEEDBACK_PROMPT_VERSION if feedback else PROMPT_VERSION,
            progress_assessment=plan.get("progress_assessment"),
            previous_image_sha256=previous_image_sha256,
        )
