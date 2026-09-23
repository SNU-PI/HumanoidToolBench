# HumanoidToolBench

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

HumanoidToolBench evaluates whether a Unitree G1 can choose a suitable tool
and use it to complete a task. Three scenarios (BallMove, BallRetrieve,
IceBreak), three levels (L0 tool selection, L1 stationary use, L2 mobile use)
and two tool-set modes (S, R) form **18 conditions**, simulated in MuJoCo with
a whole-body controller. This repository is the evaluation toolkit: it runs
your policy over HTTP, scores each episode and validates the recordings.

Documentation: [Environments](docs/ENVIRONMENTS.md) ·
[Evaluation and policy interface](docs/PUBLIC_EVALUATION.md) ·
[Dataset](data/README.md)

## Requirements

- Linux x86_64 with an NVIDIA GPU and a driver that supports EGL headless
  rendering. The pinned PyTorch wheels are CUDA 12.8 builds. A software
  rendering mode for GPU-free smoke tests is described in the
  [evaluation guide](docs/PUBLIC_EVALUATION.md#installation-scope).
- [uv](https://docs.astral.sh/uv/), Git and `zstd`. uv provisions Python 3.10.
- Setup downloads about 4 GB and uses about 9 GB of disk. The first `--model`
  run downloads about 2 GB more (checkpoint and CLIP text encoder).
- One condition at the standard 100 episodes takes about 2 hours on one GPU
  and writes about 1 GB of videos and logs. All 18 conditions take about a
  day and a half sequentially and about 15 GB. These figures were measured
  with the shipped ACT checkpoint on one workstation GPU; they grow with
  policy inference time.

## Quick Start

```bash
git clone https://github.com/SNU-PI/HumanoidToolBench.git
cd HumanoidToolBench
uv run --no-project scripts/setup_evaluation.py
```

The setup command installs the pinned Python environment, fetches the
controller repositories at their pinned revisions, downloads the benchmark
meshes and runs a self-check. Add `--check` to verify an existing
installation.

Check that simulation, inference and recording work with a short diagnostic
run on the default condition `G1BallMove-L0-S` (well under a minute of
simulation once the model has loaded; not a benchmark score):

```bash
uv run htb-eval --model snupilab/humanoidtoolbench-act-sim-3003 --episodes 1 --max-steps 100
```

Then run the standard protocol (100 episodes, seeds 10000 through 10099, up to
3000 steps). The first argument picks the condition, here BallRetrieve at L1
with the confusable tool (R); `all` runs the 18 conditions:

```bash
uv run htb-eval G1BallRetrieve-L1-R --model snupilab/humanoidtoolbench-act-sim-3003
uv run htb-eval all --model snupilab/humanoidtoolbench-act-sim-3003
```

Each condition writes `data/evals/<condition>/run-<id>/benchmark_result.json`
with the success count and four camera videos per episode. Diagnostic
settings produce `reportable: false`. `all` also writes
`data/evals/summary.json` and keeps going when one condition fails.
Rerunning a command skips conditions that already have a validated result
from the same policy, code, assets and runtime. `--list-envs` prints the condition
IDs and `--dry-run` prints the environment and episode settings without
running anything. To spread the conditions over several GPUs, see
[running conditions in parallel](docs/PUBLIC_EVALUATION.md#running-conditions-in-parallel).

`--model` loads **HumanoidToolBench ACT and Diffusion Policy simulation
checkpoints** from a Hugging Face ID or a local path, such as
`snupilab/humanoidtoolbench-act-sim-3003` and
`snupilab/humanoidtoolbench-dp-sim-3003`. Any other model is evaluated with
`--policy` or the policy server below.

### Reference results

| Policy | Condition | Successes |
| --- | --- | --- |
| `snupilab/humanoidtoolbench-act-sim-3003` | `G1BallMove-L0-S` | 2 / 100 |
| ACT with untrained, randomly initialized weights | `G1BallMove-L0-S` | 0 / 100 |

Both rows use the standard protocol and the asset manifest shipped in this
release (`assets.manifest_sha256` starting `3f5b21bf`). The ACT row used
weights with SHA-256 starting `1e6459cb`, published at model revision
`ffd733de`, and took about 100 minutes. The second row used the released ACT
configuration with freshly initialized weights (SHA-256 starting `66788a3b`,
not published). The shipped ACT checkpoint is a reference point near the
floor, not a strong baseline, so a low score from a new policy does not by
itself mean the setup is broken. The one-episode diagnostic above is expected
to print 0/1. The DP checkpoint and the other conditions have not been
measured yet.

## The benchmark

![Overview figure from the HumanoidToolBench paper: humanoid tool use, task structure, and real-robot evaluation.](docs/assets/paper-overview.png)

**HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile
Execution** introduces the benchmark, 55 tool assets and the ToolBook
demonstrations from simulation and a real robot. This release contains the
canonical environments; the easier training variants are excluded.

## Evaluate your own policy

Write a `my_policy.py` in the checkout root that loads your model once and
exposes `predict(request)`:

```python
import numpy as np

def predict(request: dict) -> np.ndarray:
    image = request["image"]["rgb_head_stereo_left"]   # (360, 640, 3) uint8 RGB
    state = request["state"]["states"]                 # (1, 32) float32
    instruction = request["instruction"]               # e.g. "Pick the tool and move the ball to the target."
    if request["history"].get("reset", False):         # first query of an episode
        pass                                            # clear any recurrent state here
    actions = ...                                       # your model
    return np.asarray(actions, dtype=np.float32)        # (T, 36), one 50 Hz command per row
```

If your model's dependencies can be installed in this environment with
`uv pip install` without changing any version locked in `uv.lock`, such as
torch, numpy and transformers, one command evaluates it:

```bash
uv run htb-eval G1BallMove-L0-S --policy my_policy:predict --checkpoint MODEL_ID_OR_REVISION --episodes 1 --max-steps 100
```

Otherwise start the server in one terminal from the checkout root, in your
own model environment (Python 3.10 or newer with NumPy and requests), and run
the evaluator in another:

```bash
PYTHONPATH=src python examples/serve_policy.py --policy my_policy:predict --checkpoint MODEL_ID_OR_REVISION
uv run htb-eval G1BallMove-L0-S --host 127.0.0.1 --port 21000 --episodes 1 --max-steps 100
```

The defaults of `--episodes` and `--max-steps` are the standard protocol,
so drop both options for a reportable 100-episode run. When the server runs
on another machine, start it with `--host 0.0.0.0` and pass its address to
the evaluator's `--host`. Run one server per evaluator; a server refuses a
second evaluator while the first is active. The 32-value state, the
36-value action with joint names and limits, the image and reset semantics
are specified in the [policy interface](docs/PUBLIC_EVALUATION.md#policy-interface).

## Conditions

| Scenario | Correct tool | L0 | L1 | L2 |
| --- | --- | --- | --- | --- |
| BallMove | Long stick | Pick the tool | Push the ball into the ring | Same, after moving along the bench |
| BallRetrieve | Hook | Pick the tool | Pull the ball into the ring | Same, after moving along the bench |
| IceBreak | Metal hammer | Pick the tool | Break both ice blocks | Same, after moving along the bench |

Mode **S** places the correct tool beside two irrelevant objects; mode **R**
adds one confusable tool (a short stick, a straight stick, or a light,
compliant decoy such as a fly swatter, paint roller or plunger). Success must
hold for one second; L0 requires lifting the correct tool by 8 cm.
Instructions, thresholds and recorded metrics are listed in
[docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md).

## Data and training

**[Download ToolBook on Hugging Face](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)**

Recording downloads currently require an approved access request. See the
[dataset guide](data/README.md) for layouts and observation/action formats.

This release provides **evaluation code**. You can use the demonstrations to
train in your own framework; model training pipelines are not included.

[MIT code license](LICENSE). Third-party assets, models, and data retain their
own terms.
