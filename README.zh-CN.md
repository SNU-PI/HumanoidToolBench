# HumanoidToolBench

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## Quick Start / 快速开始

在 Linux 上安装 [uv](https://docs.astral.sh/uv/)、Git、[Git LFS](https://git-lfs.com/) 和 `zstd`，然后在仓库根目录运行：

```bash
git clone https://github.com/SNU-PI/HumanoidToolBench.git
cd HumanoidToolBench
uv run --no-project scripts/setup_evaluation.py
uv run humanoidtoolbench-eval --model snupilab/humanoidtoolbench-act-sim-3003
```

安装命令安装依赖并下载所需资源。评测命令加载模型，以无窗口（headless）模式在 `G1BallMove-L0-S` 上进行评测，将得分和每个回合的四路相机视频保存到 `data/evals/`。

也可以使用本地检查点，或评测全部 18 个条件：

```bash
uv run humanoidtoolbench-eval --model /path/to/checkpoint
uv run humanoidtoolbench-eval all --model snupilab/humanoidtoolbench-act-sim-3003
```

标准评测为每个条件 100 个回合，随机种子为 10000 至 10099，每个回合最多 3000 步。添加 `--episodes 1 --max-steps 100` 可快速检查连接；该结果仅用于诊断，不能作为基准得分。

自动加载支持 **HumanoidToolBench ACT 和 Diffusion Policy（DP）仿真检查点**。其他模型格式需要自定义策略适配器。

## 基准简介

![HumanoidToolBench 论文概览图：人形机器人工具使用、任务结构和真实机器人评测。](docs/assets/paper-overview.png)

**HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile Execution** 研究 Unitree G1 能否选择合适的工具并完成任务。三个场景（BallMove、BallRetrieve、IceBreak）、三个执行级别（选择、原地使用、移动使用）和两种工具集合模式构成 **18 个条件**。论文介绍了 55 个工具资源，以及包含仿真和真实机器人示范的 ToolBook 数据集。

本版本提供标准环境，不包含 Near、Gap、Attach 或其他 Easy 变体。

## 数据与训练

**[在 Hugging Face 下载 ToolBook](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)**

目前下载录制数据需要申请并获得访问许可。目录结构和观测/动作格式见[数据指南](data/README.md)。

本版本提供**评测代码**。你可以使用示范数据在自己的框架中训练模型；本版本不包含模型训练流程。

[MIT 代码许可证](LICENSE)。第三方资源、模型和数据遵循各自的条款。
