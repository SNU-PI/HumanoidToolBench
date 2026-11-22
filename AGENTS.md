# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Antigravity, and others) working in this repository. There is intentionally no `CLAUDE.md`: Claude Code 2.1.277 and later read this file when none exists, so do not add one.

This repository is the **HumanoidToolBench evaluation toolkit**. Most people who open it with an agent want one thing: evaluate **their own policy** on the 18 benchmark conditions and get success rates. Follow the workflow below for that. The full reference is [docs/PUBLIC_EVALUATION.md](docs/PUBLIC_EVALUATION.md) and [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md); read the relevant section there before improvising. Training pipelines are not included here; do not try to build one unless the user asks.

## 1. Ask first, with the choice UI

Before installing anything, ask the user the questions below **with your agent's structured choice UI** (for example `AskUserQuestion` in Claude Code, or the equivalent multiple-choice prompt in Codex or ChatGPT), not as free-form prose. Offer exactly the options listed, mark the recommended one, and let the user type a custom answer when none fits. Ask them together in one prompt where the UI allows it, and skip any question the user already answered. Only ask in plain text when the agent has no choice UI at all.

1. **What to evaluate** (single choice)
   - My own model through a `predict(request)` adapter (Recommended)
   - My own model behind an existing HTTP server that speaks `POST /act`
   - A native HumanoidToolBench ACT or DP checkpoint (Hugging Face ID or local path)
   - The released ACT checkpoint, only to check that the setup works
2. **Where the model runs** (single choice; skip for native checkpoints)
   - Inside this repo's environment: its dependencies install with `uv pip install` without changing `uv.lock` (Recommended)
   - Its own environment on this machine (conflicting torch/numpy/transformers versions)
   - Another machine (ask for `host:port` next)
3. **Evaluation scope** (single choice)
   - Diagnostic: one condition, 1 episode, 100 steps (Recommended first run)
   - One condition, standard 100 episodes (about 2 hours on one GPU)
   - All 18 conditions, standard protocol (about 1.5 days on one GPU)
   - Custom condition list (ask which IDs next)
4. **GPUs to use** (single choice; ask only on multi-GPU machines, check `nvidia-smi` first)
   - One idle GPU (Recommended)
   - All GPUs, one evaluator per GPU
   - A specific subset (ask which next)

## 2. Check the host

- Linux x86_64 with an NVIDIA GPU whose driver supports EGL headless rendering, plus uv, Git and `zstd`. On Ubuntu also `sudo apt install libegl1 libopengl0`. macOS and Windows cannot run the simulator: say so and stop.
- Setup downloads about 4 GB and needs about 9 GB of disk; a full 18-condition run writes about 15 GB more.
- `sudo` is the user's call. Print the `apt` command and ask them to run it instead of running it yourself.

## 3. Install

```bash
uv run --no-project scripts/setup_evaluation.py          # install, fetch controllers and assets, self-check
uv run --no-project scripts/setup_evaluation.py --check  # verify an existing install
```

Do not edit `uv.lock` or `src/humanoidtoolbench/resources/evaluation_assets.json` to make an install pass: both are hashed into every result, and changing them makes scores incomparable.

## 4. Connect the policy

**Native ACT/DP checkpoint:** nothing to write. Use `--model ID_OR_PATH` (add `--revision COMMIT` for a pinned Hugging Face revision).

**Own model:** write `my_policy.py` in the checkout root. Load the model once at import; do inference in `predict`:

```python
import numpy as np

def predict(request: dict) -> np.ndarray:
    image = request["image"]["rgb_head_stereo_left"]   # (360, 640, 3) uint8 RGB
    state = request["state"]["states"]                 # (1, 32) float32
    instruction = request["instruction"]               # one of six fixed sentences
    if request["history"].get("reset", False):         # first query of an episode
        pass                                            # clear recurrent state / action history
    actions = ...                                       # the user's model
    return np.asarray(actions, dtype=np.float32)        # (T, 36), one 50 Hz command per row
```

- The 32-value state and 36-value action layout, joint order, ranges and units are in [docs/PUBLIC_EVALUATION.md, Policy interface](docs/PUBLIC_EVALUATION.md#policy-interface). Read it before writing the adapter; in particular, the left-hand order is thumb/middle/index, slot 34 is a turn flag (not a yaw rate), and a policy that stays still returns zeros in slots 32 to 35.
- Return physical commands: undo training-time normalization inside `predict`. Image resizing and model-specific preprocessing also belong there.
- If the model's dependencies fit this environment, `htb-eval --policy my_policy:predict` serves it automatically. Otherwise start the supplied server in the model's own environment:

  ```bash
  PYTHONPATH=src python examples/serve_policy.py --port 21000 \
    --policy my_policy:predict --checkpoint MODEL_ID_OR_REVISION
  ```

  Use `--host 0.0.0.0` when the evaluator is on another machine. One server serves one evaluator at a time.
- Always pass `--checkpoint` with an immutable model revision or content hash. Without it the result records `checkpoint: null` and is never resumed.

## 5. Run

Diagnostic first (never a score):

```bash
uv run htb-eval G1BallMove-L0-S --policy my_policy:predict --checkpoint MODEL_ID_OR_REVISION \
  --episodes 1 --max-steps 100 --output data/evals/my-policy
```

Swap the policy source for the user's case: `--model ID_OR_PATH`, or `--host HOST --port PORT` for a running server. `--model`, `--policy` and `--host` are alternatives; pass exactly one.

Then the standard run: drop `--episodes` and `--max-steps` (their defaults are the protocol: 100 episodes, seeds 10000 to 10099, 3000 steps). Use a condition ID from `uv run htb-eval --list-envs`, or `all` for the 18 conditions. Re-running the same command resumes and skips conditions that already have a validated result.

- `--dry-run` checks the configuration without starting the simulator.
- Select a GPU with `HUMANOIDTOOLBENCH_GPU=<physical id>`. For several GPUs, follow [Running conditions in parallel](docs/PUBLIC_EVALUATION.md#running-conditions-in-parallel).
- Long runs: start them in the background and tell the user where the log is instead of blocking on them.

## 6. Troubleshoot

- Black camera frames or EGL errors: check the driver and `libegl1 libopengl0`. Diagnose with `HUMANOIDTOOLBENCH_FORCE_CPU=1 uv run htb-eval ... --device cpu` in a new output directory; the software path is slow and only for diagnosis.
- `predict` errors come back as HTTP 500 naming your file and line; the full traceback is in the server terminal or the run's `eval_latest.log`.
- A wrong action shape or NaN is reported with the returned array's description. Fix the adapter, not the evaluator.
- HTTP 409 means another evaluator holds the server's session. Start one server per evaluator on its own `--port`.

## 7. Report results

- A condition counts only when its `benchmark_result.json` exists (written after 100 outcomes and 400 videos pass validation). A process exit or visible videos alone are not a result.
- Report per condition `successes / episodes` with the exact environment ID, laid out as the 3 scenario by 3 level by 2 mode table. For `all`, also give `<output>/summary.md`; a single number must be called the unweighted mean of the 18 condition success rates.
- State whether each result is reportable (`reportable` in the result). Diagnostic and dry runs are never scores; say so when showing them.
- Include the action-chunk length `T` that `predict` returns; it is not recorded for custom policies.
- Compare scores only between results whose `assets.manifest_sha256` and `source_digest` match. The paper's per-condition results are on the [project page](https://snu-pi.github.io/HumanoidToolBench/#results); the released ACT is a floor reference, not a strong baseline.
- Never change evaluator code, thresholds, seeds or assets to improve a score. If the user asks for that, explain that the result would no longer be a benchmark result.

## Working on the toolkit itself

- Tests: `uv run pytest`.
- Commit messages follow Conventional Commits in English, matching the existing history (`docs(readme): ...`, `fix(assets): ...`).
- When changing README content, keep [README.md](README.md), [README.ko.md](README.ko.md) and [README.zh-CN.md](README.zh-CN.md) in sync.
