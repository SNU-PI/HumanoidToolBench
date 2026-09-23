# Public evaluation

Run your own task policy in the 18 canonical HumanoidToolBench environments through
`htb-eval`. MuJoCo simulates the G1 and tools, the supplied whole-body
controller executes your policy's commands, and the evaluator records task
outcomes and four camera streams.

The [ToolBook dataset](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop) and
[dataset/model collection](https://huggingface.co/collections/snupilab/humanoidtoolbench-6aa6d654e29d415b0dafbe61)
are on Hugging Face. Read [data/README.md](../data/README.md) for access, source
revisions, and the difference between simulation and real G1 recordings.

## Installation scope

Follow the [Quick Start](../README.md#quick-start). On Linux with uv, Git,
and `zstd` installed, run from the repository root:

```bash
uv run --no-project scripts/setup_evaluation.py
```

This installs the pinned Python 3.10 environment, initializes the controller
dependencies, and downloads the simulation assets. Check an existing
installation with:

```bash
uv run --no-project scripts/setup_evaluation.py --check
```

The runtime needs the pinned MuJoCo/Python dependencies, G1 MJCF and meshes,
the `gear_sonic` and `decoupled_wbc` runtime code, Balance/Walk controller
weights, and the benchmark tool/distractor meshes with their collision parts.
The balls, ice blocks and ring markers are primitives generated in code.
Every mesh in the scene, the tools and the irrelevant household objects, is
downloaded by the installer from the pinned MolmoSpaces release into
`data/ms_assets`, which is why `zstd` is required, and verified by SHA-256
against `resources/evaluation_assets.json` together with the controller
weights. Controller weights are required even when the task policy runs on another
machine. Demonstration recordings are a separate download; `--model` downloads
supported task-model checkpoints automatically.

This release freezes the MolmoSpaces meshes and colliders listed in
[`evaluation_assets.json`](../src/humanoidtoolbench/resources/evaluation_assets.json).
Locally generated CoACD parts do not replace the frozen colliders. Asset
hashes are checked before evaluation, and `benchmark_result.json` records the
manifest hash as `assets.manifest_sha256`. This asset selection does not
establish parity with the paper's historical runs. Compare scores only between
reportable results whose `assets.manifest_sha256` and `source_digest` match.

Run evaluation with `uv run htb-eval`. The command configures headless EGL
rendering and bounded CPU thread pools automatically. MuJoCo's headless GPU
camera rendering requires an NVIDIA driver with EGL support. GPU policy
inference uses CUDA. For software rendering, install system
Mesa/EGL support (for example, `libegl1` and `libegl-mesa0` on Ubuntu) and run:

```bash
HUMANOIDTOOLBENCH_FORCE_CPU=1 uv run htb-eval --model /path/to/checkpoint --device cpu
```

This hides CUDA devices and uses Mesa software rendering. The policy's
`--device cpu` option alone changes inference placement; it does not select
software rendering. Validate the rendering path on your machine before a full
run. Custom policies with conflicting dependencies can use a separate server
environment, described under [Policy interface](#policy-interface).

If video validation reports black camera frames, check the driver and EGL
rendering on that machine. Use the software-rendering command above in a new
output directory while diagnosing it; failed video checks do not produce a
benchmark score.

The installer verifies asset hashes and the pinned controller revisions. Run a
short diagnostic on each new machine to check simulation, policy inference,
and recording before a complete evaluation. A dry run checks only the command
configuration.

## Evaluate a checkpoint

Pass a Hugging Face model ID or a local path. The evaluator loads the policy
and manages its local server automatically:

```bash
uv run htb-eval --model snupilab/humanoidtoolbench-act-sim-3003
uv run htb-eval --model /path/to/checkpoint
```

The default condition is `humanoidtoolbench/G1BallMove-L0-S`. Set the positional
environment argument to any canonical ID, with or without the
`humanoidtoolbench/` prefix, or use `all` for all 18 conditions. `--list-envs`
prints the IDs; [ENVIRONMENTS.md](ENVIRONMENTS.md) describes each condition.

```bash
uv run htb-eval G1BallRetrieve-L1-R --model /path/to/checkpoint
uv run htb-eval all --model snupilab/humanoidtoolbench-act-sim-3003
```

For a short diagnostic, add `--episodes 1 --max-steps 100`; `--max-steps`
accepts 1 through 3000. Use `--dry-run` to inspect the environment and episode
settings without downloading weights or starting simulation. Neither a
diagnostic nor a dry run is a benchmark score.

Automatic loading supports native **HumanoidToolBench ACT and Diffusion Policy (DP)
simulation exports**. A compatible model has a `run_config.json` containing
its architecture, image transforms, and normalization bounds, plus native
weights. For example:

```text
checkpoint-repository/
  run/
    run_config.json
    checkpoints/
      ckpt_40000/
        model.safetensors
        ema_net.pth          # DP EMA weights, when available
```

A local path may name this repository directory, the `run` directory, a
`ckpt_STEP` directory, or its weight file. Repository/run paths select the
highest numeric checkpoint step. Directory paths prefer DP `ema_net.pth`, matching its native inference
convention; an explicit weight-file path loads that exact file. Keep `run_config.json`
and the checkpoint directory structure together when copying a local model.
A raw weight file without its configuration is insufficient.

The loader restores the saved transforms and bounds. Supported policies use
one egocentric RGB image, the 32-value benchmark state, a language instruction,
and the 36-value simulation action contract. ACT uses full stateless action
chunks; DP uses 16-action chunks with 100 denoising steps. The pinned CLIP text
encoder downloads on first use. Other architectures and real-robot checkpoints
with different action contracts require a [custom adapter](#policy-interface).

Use `--revision` to select a Hugging Face branch, tag, or commit. The resolved
commit and weight/configuration hashes are recorded in the result. Local paths
do not accept `--revision`.

```bash
uv run htb-eval --model snupilab/humanoidtoolbench-act-sim-3003 \
  --revision 6449611a81229e271104b3ab4f41d4873c1b9d4e --device auto
```

`--device auto` selects CUDA when available, otherwise CPU. Explicit values are
`cpu` or `cuda:N`, where N is an index among visible CUDA devices. The runtime
exposes GPU 0 by default. To select another physical GPU, for example GPU 1,
use `HUMANOIDTOOLBENCH_GPU=1 uv run htb-eval --model /path/to/checkpoint --device cuda:0`.
The evaluator matches the EGL renderer to the policy's physical CUDA GPU by
UUID, including when `CUDA_VISIBLE_DEVICES` reorders devices. An explicit
`MUJOCO_EGL_DEVICE_ID` must identify that same GPU. Both selections are recorded
in `benchmark_result.json`.

This distribution provides evaluation. It does not include training pipelines;
use the [dataset](../data/README.md) with your training framework.

## Policy interface

Native ACT/DP users only need `--model`. To connect another architecture, put
its loading and inference code in a `my_policy.py` file in the checkout root
that exposes `predict(request)`. If your model's dependencies can be installed
in this environment, the evaluator loads and serves it itself:

```bash
uv run htb-eval G1BallMove-L0-S --policy my_policy:predict \
  --checkpoint MODEL_ID_OR_REVISION
```

`--policy` takes `module:function`, resolved from the current directory
first, or a file path such as `adapters/my_policy.py:predict`; `--checkpoint`
records the provenance. Install extra packages your model needs with
`uv pip install`. They must not change any version locked in `uv.lock` (for
example torch 2.7.0, numpy 1.26.4, transformers 4.57.1 and huggingface_hub):
`uv run` restores the locked versions before every run. After installing,
`uv sync --inexact --dry-run` lists what the next `uv run` would change.

When the model needs other versions or its own environment, start the
supplied server from the checkout root in that environment instead, with the
same `--policy` forms. It needs Python 3.10 or newer with NumPy and requests:

```bash
PYTHONPATH=src python examples/serve_policy.py --port 21000 --policy my_policy:predict \
  --checkpoint MODEL_ID_OR_REVISION
```

The source path makes the HTTP adapter available without installing the
simulator's dependencies in your model environment; to keep your own
`PYTHONPATH`, use `PYTHONPATH=src:$PYTHONPATH`. From this checkout's
environment, for example on another machine with a GPU for the model, the
same server runs as `uv run python examples/serve_policy.py ...`.
`my_policy.py` must provide `predict(request)`. The server decodes the HTTP payload into Python objects
and NumPy arrays before calling your function. Load your checkpoint once in
your module, then perform inference in `predict`. Use an immutable model
revision or checkpoint content hash for `--checkpoint`. The server reports
this identifier as provenance; it does not load or verify the weights on your
behalf. The server also records `adapter_sha256`, a hash of the file that
defines your `predict` and of the other files of yours it imports while
loading. The flag is optional, but a run started without it records
`checkpoint: null`, is never resumed, and is still marked reportable, so
always pass it for results you intend to publish; the evaluator prints a
warning when it is missing. Server metadata is retained with the evaluation
result.

In another terminal, connect the evaluator to that server:

```bash
uv run htb-eval --host 127.0.0.1 --port 21000
```

Use the server's address with `--host` when it runs on another machine, and
start the server with `--host 0.0.0.0` so that it accepts remote connections.
`--model`, `--policy` and `--host` are alternative policy sources. Before
loading the simulator, the evaluator checks that something is listening at
`http://HOST:PORT/info` and exits with instructions if the connection fails;
a server without an `/info` route still passes, with empty `policy_info`.

The supplied server serves one evaluator at a time, because a shared server
would let one evaluator's episode reset clear another's history. Each
evaluator process sends a `session` identifier with its requests and releases
it with `POST /release` when it exits, so the next evaluator can start at
once. While a session is active, the server refuses requests from another
session with HTTP 409, and the evaluator's pre-check reports this before the
simulator starts. A session that was never released, for example after a
crash, expires 60 seconds after its last request. Start one server per
evaluator on its own `--port`, or pass `--allow-shared` to
`examples/serve_policy.py` only for a stateless policy. Calls to `predict`
never overlap and run on the thread that imported your module: the main
thread of `examples/serve_policy.py`, or one dedicated thread inside
`htb-eval --policy`. Thread-local setup done at import, such as
`torch.set_grad_enabled(False)`, therefore stays in effect.

A `predict` exception is returned to the evaluator as an HTTP 500 whose
message names the exception, the innermost line of your own code (library
frames are skipped) and the message. A wrong action shape or a NaN is reported
by describing the returned array. The full traceback is printed by the
server: on its terminal for `examples/serve_policy.py`, and in the run's
`eval_latest.log` for `--policy`, whose path the evaluator prints when a
condition fails.

| Request field | Value |
| --- | --- |
| `request["image"]["rgb_head_stereo_left"]` | RGB image, shape `(360, 640, 3)`, `uint8` |
| `request["state"]["states"]` | Policy state, shape `(1, 32)`, float32 |
| `request["instruction"]` | Canonical task instruction |
| `request["history"].get("reset", False)` | True on the first policy query of an episode |

Reset your model's recurrent state or action history when `reset` is true.
Image resizing and model-specific normalization belong in your policy adapter.
The evaluator supplies the canonical instruction, including the distinction
between L0 tool selection and L1/L2 task execution.

Return a finite NumPy array shaped `(T, 36)`, with `T >= 1`. Each row is one
50 Hz controller command; the evaluator executes the returned chunk before
asking for another one, unless the episode ends. Return physical commands
after undoing training-time action normalization.

| Slice | State, 32 values | Action, 36 values |
| --- | --- | --- |
| `0:7` | Left hand joints | Left hand joint targets |
| `7:14` | Right hand joints | Right hand joint targets |
| `14:21` | Left arm joints | Left arm joint targets |
| `21:28` | Right arm joints | Right arm joint targets |
| `28:31` | Measured torso joint values in roll/pitch/yaw order | Torso roll/pitch/yaw commands |
| `31:32` | Previous base-height command | Base-height command |
| `32:36` | Not present | Navigation vx, vy, yaw rate, target yaw |

Joint angles and angular commands use radians, base height uses metres, and
navigation velocities use metres/second or radians/second. The flat left-hand
order is thumb/middle/index; the right-hand order is thumb/index/middle. The
definitions are in [actions/g1.py](../src/humanoidtoolbench/actions/g1.py) and the
state builder in [remote_humanoid.py](../src/humanoidtoolbench/policies/remote_humanoid.py).
The policy state excludes the leg joints from the raw 43-value robot
observation and includes the previous commanded height, starting at 0.74 for a
new episode.

State slots 0 through 30 hold the measured angles of the same joints, in the
same order, that action slots 0 through 30 command; state slot 31 is the
base-height value from the last row of the most recent action chunk the
policy returned (0.74 before the first chunk), not a measurement. The table
below lists every action slot with the joint range of the shipped G1 model.
The evaluator does not clip commands, so keep targets inside these ranges;
behaviour outside them is controller-defined. Slots 28 through 30 are
converted to a torso roll/pitch/yaw command for the lower-body controller,
which chooses the actual waist joint targets.

| Slot | Joint | Range (rad) |
| --- | --- | --- |
| 0 | left_hand_thumb_0_joint | -1.047 to 1.047 |
| 1 | left_hand_thumb_1_joint | -0.724 to 1.047 |
| 2 | left_hand_thumb_2_joint | 0 to 1.745 |
| 3 | left_hand_middle_0_joint | -1.571 to 0 |
| 4 | left_hand_middle_1_joint | -1.745 to 0 |
| 5 | left_hand_index_0_joint | -1.571 to 0 |
| 6 | left_hand_index_1_joint | -1.745 to 0 |
| 7 | right_hand_thumb_0_joint | -1.047 to 1.047 |
| 8 | right_hand_thumb_1_joint | -1.047 to 0.724 |
| 9 | right_hand_thumb_2_joint | -1.745 to 0 |
| 10 | right_hand_index_0_joint | 0 to 1.571 |
| 11 | right_hand_index_1_joint | 0 to 1.745 |
| 12 | right_hand_middle_0_joint | 0 to 1.571 |
| 13 | right_hand_middle_1_joint | 0 to 1.745 |
| 14 | left_shoulder_pitch_joint | -3.089 to 2.670 |
| 15 | left_shoulder_roll_joint | -1.588 to 2.252 |
| 16 | left_shoulder_yaw_joint | -2.618 to 2.618 |
| 17 | left_elbow_joint | -1.047 to 2.094 |
| 18 | left_wrist_roll_joint | -1.972 to 1.972 |
| 19 | left_wrist_pitch_joint | -1.614 to 1.614 |
| 20 | left_wrist_yaw_joint | -1.614 to 1.614 |
| 21 | right_shoulder_pitch_joint | -3.089 to 2.670 |
| 22 | right_shoulder_roll_joint | -2.252 to 1.588 |
| 23 | right_shoulder_yaw_joint | -2.618 to 2.618 |
| 24 | right_elbow_joint | -1.047 to 2.094 |
| 25 | right_wrist_roll_joint | -1.972 to 1.972 |
| 26 | right_wrist_pitch_joint | -1.614 to 1.614 |
| 27 | right_wrist_yaw_joint | -1.614 to 1.614 |
| 28 | waist_roll_joint | -0.52 to 0.52 |
| 29 | waist_pitch_joint | -0.52 to 0.52 |
| 30 | waist_yaw_joint | -2.618 to 2.618 |
| 31 | Base height command (m); 0.74 at reset | not bounded |
| 32 | Navigation forward velocity (m/s) | not bounded |
| 33 | Navigation lateral velocity (m/s) | not bounded |
| 34 | Navigation turn flag: a magnitude of at least 0.1 enables heading tracking of slot 35 | not bounded |
| 35 | Navigation target yaw, absolute heading (rad) | not bounded |

For the index and middle finger joints, zero is the open pose and the other
end of the range is closed; `thumb_2` also has zero at one end, while
`thumb_0` and `thumb_1` have zero inside their range. The shipped controller
switches from the standing to the walking policy when the norm of slots 32
through 34 reaches 0.1. Slot 34 is a flag, not a rate: while its magnitude is
at least 0.1 the controller turns towards the absolute heading in slot 35 at
the heading error divided by 0.5 s, capped at 1 rad/s; below 0.1 it does not
turn. A policy that stays still should return zeros in slots 32 through 35.

The image is the left eye of the head-mounted stereo camera, rendered at
640 by 360 pixels with a 110 degree horizontal field of view, delivered as an
RGB `uint8` array. Wrist cameras are recorded but not sent to the policy. The
instruction is one of six fixed sentences listed in
[ENVIRONMENTS.md](ENVIRONMENTS.md#the-18-conditions); it changes with the
scenario and with L0 versus L1/L2, not with the mode.

For a custom HTTP implementation, serve `POST /act` using the NumPy-aware JSON
encoding in [http_client.py](../src/humanoidtoolbench/policies/http_client.py).
The response contains `action` with shape `(T, 36)`. Requests also carry a
`session` string that identifies the evaluator process, and an evaluator
sends `POST /release` with that `session` when it exits; a custom server may
ignore both. `GET /info` is optional: return a JSON object with at least
`policy` and `checkpoint` to have them recorded, and a `server_state` object
with `exclusive` and `active_session` to take part in the evaluator's busy
check. The provided server implements this transport so policy authors can
work with decoded arrays.

Running `examples/serve_policy.py` without `--policy` selects its pose-holding
example. That example verifies integration and does not demonstrate tool use.

## Evaluation protocol

```bash
uv run htb-eval humanoidtoolbench/G1BallRetrieve-L1-R \
  --model /path/to/checkpoint --episodes 100 --seed-start 10000 \
  --max-steps 3000 --output data/evals/my-policy
```

Use `all` as the environment argument to run all 18 conditions. They run one
after another in a single process, and the command prints `[k/18]` before
each. A condition whose run or validation fails is reported with its
traceback, its run directory is kept for inspection, and the remaining
conditions still run; the exit status is then 1. Running the same command
again resumes: a condition that already has a validated
`benchmark_result.json` under `--output` is skipped, unless `--rerun` is
given. A result is reused only when the settings match and so do all of the
following: the policy name; for `--model`, the weight and configuration
hashes, and otherwise the `--checkpoint` string; `adapter_sha256` for
`--policy` and the supplied server; the evaluation source hashes; the asset
manifest; and the Python, MuJoCo, NumPy and PyAV versions and the `uv.lock`
hash. If any of these changed, the command says so and evaluates the
condition again; when several results match, the newest is used. A policy
without a checkpoint identifier is never skipped. Weights, environment
variables and files your adapter imports only after loading are not hashed:
change `--checkpoint` whenever they change, or pass `--rerun`. This applies to
single conditions as well as `all`. After a multi-condition run the evaluator writes
`<output>/summary.json` and `summary.md` with one row per condition and the
unweighted mean success rate once every condition has a result. The summary
is reportable only when every condition has a reportable result from the
same policy, code, assets and runtime; otherwise it lists the reasons. Use `--dry-run`
to inspect the configuration without starting simulation or contacting the
server.

### Running conditions in parallel

`all` runs conditions one after another. To use several GPUs, run one
evaluator per GPU, each working through its share of the conditions with the
same `--output`, then run `all` once more to write the summary; it skips the
finished conditions:

```bash
MODEL=snupilab/humanoidtoolbench-act-sim-3003
OUT=data/evals/act
GPUS=(${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader)})
mapfile -t ENVS < <(uv run htb-eval --list-envs)
for ((g = 0; g < ${#GPUS[@]}; g++)); do
  (for ((i = g; i < ${#ENVS[@]}; i += ${#GPUS[@]})); do
     HUMANOIDTOOLBENCH_GPU=${GPUS[g]} uv run htb-eval "${ENVS[i]}" --model "$MODEL" --output "$OUT"
   done) &
done
wait
uv run htb-eval all --model "$MODEL" --output "$OUT"
```

Set `GPUS="0 1"` to use only some devices. With `--model` or `--policy` each
evaluator starts its own policy server on a free port. With `--host`, start
one `examples/serve_policy.py` per GPU loop, each on its own `--port`; the
conditions in one loop run one after another, so they can share that server.

| Setting | Standard protocol |
| --- | --- |
| Conditions | 3 scenarios, L0/L1/L2, S/R |
| Episodes | 100 for each condition |
| Seeds | 10000 through 10099 |
| Policy-control rate | 50 Hz |
| Episode limit | 3000 policy-control steps, up to 60 simulated seconds |
| Controller/action schema | Decoupled WBC, `decoupled_v1` |
| Recording | Four camera videos per episode |

L0 success requires selecting and lifting the correct tool. L1/L2 require the
scenario's execution success predicate, including both blocks for IceBreak.
Success must hold for 50 consecutive control steps. The predicates, thresholds
and recorded metrics are listed in [ENVIRONMENTS.md](ENVIRONMENTS.md#success-criteria).
The evaluator preserves the task's success timing and termination rules.

The defaults of `--episodes`, `--seed-start` and `--max-steps` are the
standard protocol, so a run without these options is a standard run.
Anything other than exactly 100 episodes, seed start 10000 and 3000 steps is
a diagnostic setting and produces `reportable=false`. The supplied pose-holder
identifies itself as diagnostic, and its runs remain non-reportable even with
the standard episode budget.

## Results

Keep the full output directory. It contains the rollout outcomes, recordings,
configuration, and validation evidence. `benchmark_result.json` is produced
only after the outcome and video checks pass. A standard condition requires
100 outcomes and 400 validated videos. All 18 standard conditions comprise
1800 episodes and 7200 videos.

Report per-condition successes and the actual denominator together with the
exact environment IDs, code revision, asset/controller versions, model
checkpoint identity, action-chunk length, and evaluation settings. Present
the 18 conditions as the 3 by 3 by 2 table; if you summarize with one number,
state that it is the unweighted mean of the 18 condition success rates. Native
`--model` runs record the chunk length as `chunk_size` in `policy_info`; for
`--policy` and custom servers it is not recorded, so note the `T` your
`predict` returns. Each `benchmark_result.json` records the package version,
Git commit and whether tracked files were modified (`code`), the evaluation
source hashes (`source_sha256`, summarized as `source_digest`), the asset
manifest (`assets`), the policy identity (`policy_info`), the completion time
(`finished_at`), the wall time and the GPU and rendering selection. Paths
inside it are absolute paths on the evaluating machine. A process exit or a visible video alone does not establish a
complete benchmark result.

## Licenses

Project code uses the [MIT license](../LICENSE). Bundled inference code retains
its [upstream license](../src/humanoidtoolbench/policies/_vendor/LICENSE). Controller
code, controller weights, robot models, scanned meshes, and model checkpoints
retain their own terms. The [dataset page](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)
specifies recording access and use conditions.
