# HumanoidToolBench

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

HumanoidToolBench 评测 Unitree G1 能否选择合适的工具并用它完成任务。三个场景（BallMove、BallRetrieve、IceBreak）、三个级别（L0 工具选择、L1 原地使用、L2 移动使用）和两种工具集合模式（S、R）构成 **18 个条件**，在 MuJoCo 中用全身控制器进行仿真。本仓库是评测工具包：它通过 HTTP 调用你的策略，为每个回合打分，并校验录像。

文档：[环境说明](docs/ENVIRONMENTS.md) · [评测与策略接口](docs/PUBLIC_EVALUATION.md) · [数据集](data/README.md)

## 系统要求

- Linux x86_64，配备 NVIDIA GPU 和支持 EGL 无头渲染的驱动。锁定的 PyTorch wheel 为 CUDA 12.8 版本。无 GPU 的冒烟测试可使用软件渲染模式，见[评测指南](docs/PUBLIC_EVALUATION.md#installation-scope)。
- [uv](https://docs.astral.sh/uv/)、Git 和 `zstd`。Python 3.10 由 uv 自动安装。
- 安装约下载 4 GB，占用磁盘约 9 GB。首次运行 `--model` 时会再下载约 2 GB（检查点和 CLIP 文本编码器）。
- 单个条件按标准 100 回合运行，在一块 GPU 上约需 2 小时，产生约 1 GB 的视频和日志。全部 18 个条件顺序运行约需一天半，占用约 15 GB。以上数字使用随附的 ACT 检查点在一台工作站的 GPU 上测得，随策略推理时间增加。

## 快速开始

```bash
git clone https://github.com/SNU-PI/HumanoidToolBench.git
cd HumanoidToolBench
uv run --no-project scripts/setup_evaluation.py
```

安装命令会创建锁定的 Python 环境，按固定版本拉取控制器仓库，下载基准网格资源并运行自检。加上 `--check` 可检查已有安装。

先在默认条件 `G1BallMove-L0-S` 上用一次简短的诊断运行确认仿真、推理和录像正常（模型加载后仿真用时远少于一分钟，不是基准得分）：

```bash
uv run humanoidtoolbench-eval --model snupilab/humanoidtoolbench-act-sim-3003 --episodes 1 --max-steps 100
```

然后在单个条件或全部 18 个条件上运行标准协议（100 回合，随机种子 10000 至 10099，每回合最多 3000 步）：

```bash
uv run humanoidtoolbench-eval G1BallMove-L0-S --model snupilab/humanoidtoolbench-act-sim-3003
uv run humanoidtoolbench-eval all --model snupilab/humanoidtoolbench-act-sim-3003
```

每个条件会写入 `data/evals/<condition>/run-<id>/benchmark_result.json`，包含成功次数，并保存每回合四路相机视频。诊断设置得到的结果为 `reportable: false`。`--list-envs` 打印条件 ID 列表，`--dry-run` 只打印环境和回合设置而不运行。

`--model` 从 Hugging Face ID 或本地路径加载 **HumanoidToolBench ACT 和 Diffusion Policy 仿真检查点**。其他模型通过下面的策略服务器进行评测。

## 评测你自己的策略

在仓库根目录编写 `my_policy.py`，加载一次模型并提供 `predict(request)`：

```python
import numpy as np

def predict(request: dict) -> np.ndarray:
    image = request["image"]["rgb_head_stereo_left"]   # (360, 640, 3) uint8 RGB
    state = request["state"]["states"]                 # (1, 32) float32
    instruction = request["instruction"]               # 例如 "Pick the tool and move the ball to the target."
    if request["history"].get("reset", False):         # 回合的第一次查询
        pass                                            # 在此清除循环状态
    actions = ...                                       # 你的模型
    return np.asarray(actions, dtype=np.float32)        # (T, 36)，每行一条 50 Hz 指令
```

在一个终端启动服务器。既可以在本仓库的环境中运行，也可以在你自己的模型环境中运行，后者只需要 NumPy 和 requests：

```bash
uv run python examples/serve_policy.py --policy my_policy:predict --checkpoint MODEL_ID_OR_REVISION
# 或者在你自己的环境中：
PYTHONPATH=src python examples/serve_policy.py --policy my_policy:predict --checkpoint MODEL_ID_OR_REVISION
```

在另一个终端先做一次诊断运行，然后去掉 `--episodes` 和 `--max-steps` 运行标准的 100 回合协议：

```bash
uv run humanoidtoolbench-eval G1BallMove-L0-S --host 127.0.0.1 --port 21000 --episodes 1 --max-steps 100
uv run humanoidtoolbench-eval G1BallMove-L0-S --host 127.0.0.1 --port 21000
```

服务器运行在另一台机器上时，用 `--host 0.0.0.0` 启动服务器，并把它的地址传给评测器的 `--host`。32 维状态、带关节名称和限位的 36 维动作、图像和 reset 语义见[策略接口](docs/PUBLIC_EVALUATION.md#policy-interface)。

## 条件

| 场景 | 正确工具 | L0 | L1 | L2 |
| --- | --- | --- | --- | --- |
| BallMove | 长杆 | 拿起工具 | 把球推进圆环 | 同上，先沿工作台移动 |
| BallRetrieve | 钩子 | 拿起工具 | 把球拉进圆环 | 同上，先沿工作台移动 |
| IceBreak | 金属锤 | 拿起工具 | 敲碎两块冰 | 同上，先沿工作台移动 |

模式 **S** 把正确工具与两个无关物体放在一起；模式 **R** 再加入一个易混淆工具（短杆、直杆，或苍蝇拍、油漆滚筒、马桶搋子这类轻而柔软的干扰工具）。成功状态需保持一秒；L0 要求把正确工具抬起 8 cm。指令、阈值和记录的指标见 [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)。

![HumanoidToolBench 论文概览图：人形机器人工具使用、任务结构和真实机器人评测。](docs/assets/paper-overview.png)

**HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile Execution** 介绍了该基准、55 个工具资源以及包含仿真和真实机器人示范的 ToolBook 数据集。本版本只包含标准环境，不包含用于训练的简化变体。

## 数据与训练

**[在 Hugging Face 下载 ToolBook](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)**

目前下载录制数据需要申请并获得访问许可。目录结构和观测/动作格式见[数据指南](data/README.md)。

本版本提供**评测代码**。你可以使用示范数据在自己的框架中训练模型；本版本不包含模型训练流程。

[MIT 代码许可证](LICENSE)。第三方资源、模型和数据遵循各自的条款。
