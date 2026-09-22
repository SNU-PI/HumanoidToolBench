# HumanoidToolBench

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## Quick Start

On Linux, install [uv](https://docs.astral.sh/uv/), Git, [Git LFS](https://git-lfs.com/), and `zstd`. From this repository:

```bash
git clone https://github.com/SNU-PI/HumanoidToolBench.git
cd HumanoidToolBench
uv run --no-project scripts/setup_evaluation.py
uv run humanoidtoolbench-eval --model snupilab/humanoidtoolbench-act-sim-3003
```

The setup command installs dependencies and downloads the required assets. The evaluation command loads the model and evaluates `G1BallMove-L0-S` in headless mode (no simulator window), saving scores and four camera videos per episode under `data/evals/`.

Use a local checkpoint, or evaluate all 18 conditions:

```bash
uv run humanoidtoolbench-eval --model /path/to/checkpoint
uv run humanoidtoolbench-eval all --model snupilab/humanoidtoolbench-act-sim-3003
```

The standard evaluation uses 100 episodes per condition, seeds 10000 through 10099, and up to 3000 steps. For a short connection check, add `--episodes 1 --max-steps 100`; this is a diagnostic, not a benchmark score.

Automatic loading supports **HumanoidToolBench ACT and Diffusion Policy (DP) simulation checkpoints**. Other model formats need a custom policy adapter.

## The benchmark

![Overview figure from the HumanoidToolBench paper: humanoid tool use, task structure, and real-robot evaluation.](docs/assets/paper-overview.png)

**HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile Execution** studies whether a Unitree G1 can choose a suitable tool and use it to complete a task. Three scenarios (BallMove, BallRetrieve, IceBreak), three levels (selection, stationary use, mobile use), and two tool-set modes form **18 conditions**. The paper introduces 55 tool assets and ToolBook demonstrations from simulation and a real robot.

This release contains the canonical environments. Near, Gap, Attach, and other Easy variants are excluded.

## Data and training

**[Download ToolBook on Hugging Face](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)**

Recording downloads currently require an approved access request. See the [dataset guide](data/README.md) for layouts and observation/action formats.

This release provides **evaluation code**. You can use the demonstrations to train in your own framework; model training pipelines are not included.

[MIT code license](LICENSE). Third-party assets, models, and data retain their own terms.
