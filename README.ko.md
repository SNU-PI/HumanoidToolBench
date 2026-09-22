# HumanoidToolBench

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

## Quick Start / 빠른 시작

Linux에서 [uv](https://docs.astral.sh/uv/), Git, [Git LFS](https://git-lfs.com/), `zstd`를 설치하고 저장소 루트에서 실행하세요.

```bash
git clone https://github.com/SNU-PI/HumanoidToolBench.git
cd HumanoidToolBench
uv run --no-project scripts/setup_evaluation.py
uv run humanoidtoolbench-eval --model snupilab/humanoidtoolbench-act-sim-3003
```

설치 명령은 의존성을 설치하고 필요한 에셋을 내려받습니다. 평가 명령은 모델을 불러와 시뮬레이터 창 없이(headless) `G1BallMove-L0-S`를 평가하고, 점수와 에피소드당 네 카메라 영상을 `data/evals/`에 저장합니다.

로컬 체크포인트를 사용하거나 전체 18개 조건을 평가할 수도 있습니다.

```bash
uv run humanoidtoolbench-eval --model /path/to/checkpoint
uv run humanoidtoolbench-eval all --model snupilab/humanoidtoolbench-act-sim-3003
```

정식 평가는 조건당 100개 에피소드, 시드 10000~10099, 에피소드당 최대 3000스텝으로 실행됩니다. 연결만 빠르게 확인하려면 `--episodes 1 --max-steps 100`을 추가하세요. 이 결과는 진단용이며 벤치마크 점수가 아닙니다.

**HumanoidToolBench ACT 및 Diffusion Policy(DP) 시뮬레이션 체크포인트**를 자동으로 불러옵니다. 다른 모델 형식은 별도 정책 어댑터가 필요합니다.

## 벤치마크 소개

![HumanoidToolBench 논문 개요 그림: 휴머노이드 도구 사용, 태스크 구성, 실제 로봇 평가.](docs/assets/paper-overview.png)

**HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile Execution**은 Unitree G1이 적절한 도구를 고르고 이를 사용해 태스크를 완수하는지 평가합니다. 세 시나리오(BallMove, BallRetrieve, IceBreak), 세 단계(선택, 제자리 사용, 이동 사용), 두 도구 구성으로 **18개 조건**을 구성합니다. 논문에서는 55개 도구 에셋과 시뮬레이션 및 실제 로봇 시연을 담은 ToolBook 데이터셋을 소개합니다.

이 배포본은 정규 환경을 제공하며 Near, Gap, Attach 및 다른 Easy 변형은 포함하지 않습니다.

## 데이터와 훈련

**[Hugging Face에서 ToolBook 다운로드](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)**

현재 녹화 데이터 다운로드에는 접근 신청과 승인이 필요합니다. 폴더 구조와 관측/행동 형식은 [데이터 가이드](data/README.md)를 참고하세요.

이 배포본은 **평가 코드**를 제공합니다. 시연 데이터를 사용해 원하는 프레임워크에서 모델을 훈련할 수 있으며, 모델 훈련 파이프라인은 포함되어 있지 않습니다.

[MIT 코드 라이선스](LICENSE). 외부 에셋, 모델, 데이터는 각각의 이용 조건을 따릅니다.
