# Public evaluation

Run your own task policy in the 18 canonical HumanoidToolBench environments through
`humanoidtoolbench-eval`. MuJoCo simulates the G1 and tools, the supplied whole-body
controller executes your policy's commands, and the evaluator records task
outcomes and four camera streams.

The [dataset](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop) and
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
The downloader requires the `zstd` command. A Git checkout containing LFS
pointers also needs Git LFS; source snapshots already contain those files.
Controller weights are required even when the task policy runs on another
machine. Demonstration recordings are a separate download; `--model` downloads
supported task-model checkpoints automatically.

This release freezes the shipped MolmoSpaces colliders in
[`evaluation_assets.json`](../src/theta_bench/resources/evaluation_assets.json).
That selection preserves the canonical asset files present when this branch
was prepared. It does not establish parity with every historical paper run.
Locally generated CoACD parts do not replace the frozen colliders. Asset hashes
are checked before evaluation and the selected geometry is recorded with the
result, so compare scores only across matching geometry protocols.

Run evaluation with `uv run humanoidtoolbench-eval`. The command configures headless EGL
rendering and bounded CPU thread pools automatically. MuJoCo's headless GPU
camera rendering requires an NVIDIA driver with EGL support. GPU policy
inference uses CUDA. For software rendering, install system
Mesa/EGL support (for example, `libegl1` and `libegl-mesa0` on Ubuntu) and run:

```bash
THETA_BENCH_FORCE_CPU=1 uv run humanoidtoolbench-eval --model /path/to/checkpoint --device cpu
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
uv run humanoidtoolbench-eval --model snupilab/humanoidtoolbench-act-sim-3003
uv run humanoidtoolbench-eval --model /path/to/checkpoint
```

The default condition is `theta_bench/G1BallMove-L0-S`. Set the positional
environment argument to any canonical ID, or use `all` for all 18 conditions:

```bash
uv run humanoidtoolbench-eval theta_bench/G1BallRetrieve-L1-R --model /path/to/checkpoint
uv run humanoidtoolbench-eval all --model snupilab/humanoidtoolbench-act-sim-3003
```

For a short diagnostic, add `--episodes 1 --max-steps 100`. Use `--dry-run`
to inspect the run configuration without downloading weights or starting
simulation. Neither a diagnostic nor a dry run is a benchmark score.

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
uv run humanoidtoolbench-eval --model snupilab/humanoidtoolbench-act-sim-3003 \
  --revision 6449611a81229e271104b3ab4f41d4873c1b9d4e --device auto
```

`--device auto` selects CUDA when available, otherwise CPU. Explicit values are
`cpu` or `cuda:N`, where N is an index among visible CUDA devices. The runtime
exposes GPU 0 by default. To select another physical GPU, for example GPU 1,
use `THETA_BENCH_GPU=1 uv run humanoidtoolbench-eval --model /path/to/checkpoint --device cuda:0`.
The evaluator matches the EGL renderer to the policy's physical CUDA GPU by
UUID, including when `CUDA_VISIBLE_DEVICES` reorders devices. An explicit
`MUJOCO_EGL_DEVICE_ID` must identify that same GPU. Both selections are recorded
in `benchmark_result.json`.

This distribution provides evaluation. It does not include training pipelines;
use the [dataset](../data/README.md) with your training framework.

## Policy interface

Native ACT/DP users only need `--model`. To connect another architecture, put
its loading and inference code in a `my_policy.py` file in the checkout root,
then start the supplied server:

```bash
PYTHONPATH=. uv run python examples/serve_policy.py --port 21000 --policy my_policy:predict \
  --checkpoint MODEL_ID_OR_REVISION
```

To use a separate model environment, run
`PYTHONPATH=src:. python examples/serve_policy.py` with the same arguments,
NumPy, and requests installed. The source path makes the HTTP adapter available
without installing the simulator's dependencies in your model environment.
`my_policy.py` must be importable by that Python process and provide
`predict(request)`. The server decodes the HTTP payload into Python objects
and NumPy arrays before calling your function. Load your checkpoint once in
your module, then perform inference in `predict`. Use an immutable model
revision or checkpoint content hash for `--checkpoint`. The server reports
this identifier as provenance; it does not load or verify the weights on your
behalf. Server metadata is retained with the evaluation result.

In another terminal, connect the evaluator to that server:

```bash
uv run humanoidtoolbench-eval --host 127.0.0.1 --port 21000
```

Use the server's address with `--host` when it runs on another machine.
`--model` and `--host` are alternative policy sources.

| Request field | Value |
| --- | --- |
| `request["image"]["rgb_head_stereo_left"]` | RGB image, shape `(H, W, 3)` |
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
order is thumb/middle/index; the right-hand order is thumb/index/middle. Use
the definitions in [actions/g1.py](../src/theta_bench/actions/g1.py) and the
state builder in [remote_humanoid.py](../src/theta_bench/policies/remote_humanoid.py)
when adapting a model. The policy state excludes the leg joints from the raw
43-value robot observation and includes the previous commanded height, starting
at 0.74 for a new episode.

For a custom HTTP implementation, serve `POST /act` using the NumPy-aware JSON
encoding in [http_client.py](../src/theta_bench/policies/http_client.py).
The response contains `action` with shape `(T, 36)`. The provided server
implements this transport so policy authors can work with decoded arrays.

Running `examples/serve_policy.py` without `--policy` selects its pose-holding
example. That example verifies integration and does not demonstrate tool use.

## Evaluation protocol

```bash
uv run humanoidtoolbench-eval theta_bench/G1BallRetrieve-L1-R \
  --model /path/to/checkpoint --episodes 100 --seed-start 10000 \
  --max-steps 3000 --output data/evals/my-policy
```

Use `all` as the environment argument to run all 18 conditions. Use `--dry-run`
to inspect the configuration without starting simulation or contacting the
server.

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
The evaluator preserves the task's success timing and termination rules.

Smaller episode counts, a different seed start, or a different step budget
are diagnostic settings and produce `reportable=false`. The supplied pose-holder
identifies itself as diagnostic, and its runs remain non-reportable even with
the standard episode budget. Historical completed 200-episode results retain their actual denominator
and provenance; they are not truncated to the new 100-episode protocol.

## Results

Keep the full output directory. It contains the rollout outcomes, recordings,
configuration, and validation evidence. `benchmark_result.json` is produced
only after the outcome and video checks pass. A standard condition requires
100 outcomes and 400 validated videos. All 18 standard conditions comprise
1800 episodes and 7200 videos.

Report per-condition successes and the actual denominator together with the
exact environment IDs, code revision, asset/controller versions, model
checkpoint identity, action-chunk length, and evaluation settings. Record wall
time and the CPU/GPU/rendering configuration so runtime measurements can be
interpreted. A process exit or a visible video alone does not establish a
complete benchmark result.

## Source distribution

The release branch contains the same evaluation sources as the source snapshot.
Training, teleoperation, RL, and Easy configuration modules are excluded from
both. Controller repositories are downloaded by setup at their pinned revisions.

Create a standalone release snapshot:

```bash
uv run --no-project scripts/export_public_evaluation.py ../humanoidtoolbench-public
```

The destination must not exist or must be empty. The exporter refuses symlinks
and writes `export_manifest.json` with relative file names, sizes, and SHA256
hashes. It selects evaluation runtime code and omits Git history, local
credentials/configuration, experiment artifacts, paper sources, recordings,
training modules, and third-party working trees. The README's paper overview
image and required third-party inference code/licenses are included.

Install directly from the snapshot directory:

```bash
cd ../humanoidtoolbench-public
uv run --no-project scripts/setup_evaluation.py
```

The installer retrieves the exact public controller dependency revisions from
the release manifest. The exported directory does not require a development
branch checkout. Creating this directory is a local export, with no repository
push or publication.

## Licenses

Project code uses the [MIT license](../LICENSE). Bundled inference code retains
its [upstream license](../src/theta_bench/policies/_vendor/LICENSE). Controller
code, controller weights, robot models, scanned meshes, and model checkpoints
retain their own terms. The [dataset page](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)
specifies recording access and use conditions.
