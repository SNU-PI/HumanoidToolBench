# HumanoidToolBench

[English](README.md) | [中文](README.zh-CN.md) | [한국어](README.ko.md)

HumanoidToolBench는 Unitree G1이 적절한 도구를 고르고 그 도구로 태스크를
완수하는지 평가합니다. 세 시나리오(BallMove, BallRetrieve, IceBreak), 세
단계(L0 도구 선택, L1 제자리 사용, L2 이동 사용), 두 도구 구성 모드(S, R)가
**18개 조건**을 이루며, MuJoCo와 전신 컨트롤러로 시뮬레이션됩니다. 이
저장소는 평가 툴킷입니다. 정책을 HTTP로 호출하고, 에피소드마다 점수를
매기고, 녹화를 검증합니다.

문서: [환경 설명](docs/ENVIRONMENTS.md) ·
[평가 및 정책 인터페이스](docs/PUBLIC_EVALUATION.md) ·
[데이터셋](data/README.md)

## 요구 사항

- NVIDIA GPU와 EGL 헤드리스 렌더링을 지원하는 드라이버가 있는 Linux x86_64.
  고정된 PyTorch 휠은 CUDA 12.8 빌드입니다. GPU 없이 동작 확인(스모크
  테스트)만 할 때 쓰는 소프트웨어 렌더링 모드는
  [평가 가이드](docs/PUBLIC_EVALUATION.md#installation-scope)에 설명되어
  있습니다.
- [uv](https://docs.astral.sh/uv/), Git, `zstd`. Python 3.10은 uv가 직접
  설치합니다.
- 설치 시 약 4 GB를 내려받고 디스크 약 9 GB를 사용합니다. 첫 `--model`
  실행 때 체크포인트와 CLIP 텍스트 인코더 약 2 GB를 추가로 내려받습니다.
- 조건 하나를 정식 100 에피소드로 돌리면 GPU 한 장 기준 약 2시간이 걸리고
  영상과 로그가 약 1 GB 생깁니다. 18개 조건 전체는 순차 실행 시 약 1.5일,
  약 15 GB입니다. 배포된 ACT 체크포인트로 워크스테이션 GPU 한 장에서
  측정한 값이며 정책 추론 시간에 따라 늘어납니다.

## 빠른 시작

```bash
git clone https://github.com/SNU-PI/HumanoidToolBench.git
cd HumanoidToolBench
uv run --no-project scripts/setup_evaluation.py
```

설치 명령은 고정된 Python 환경을 만들고, 컨트롤러 저장소를 고정 리비전으로
받고, 벤치마크 메쉬를 내려받은 뒤 자체 점검을 실행합니다. 기존 설치를
확인하려면 `--check`를 붙이세요.

시뮬레이션, 추론, 녹화가 동작하는지 기본 조건 `G1BallMove-L0-S`에서 짧은
진단 실행으로 확인합니다(모델 로딩 후 시뮬레이션은 1분이 채 걸리지
않으며, 벤치마크 점수가 아닙니다).

```bash
uv run htb-eval --model snupilab/humanoidtoolbench-act-sim-3003 --episodes 1 --max-steps 100
```

그다음 정식 프로토콜(100 에피소드, 시드 10000~10099, 최대 3000스텝)을
실행합니다. 첫 인자로 조건을 고르며, 아래 예시는 헷갈리는 도구가 있는(R)
BallRetrieve L1입니다. `all`은 18개 조건 전체를 실행합니다.

```bash
uv run htb-eval G1BallRetrieve-L1-R --model snupilab/humanoidtoolbench-act-sim-3003
uv run htb-eval all --model snupilab/humanoidtoolbench-act-sim-3003
```

조건마다 `data/evals/<condition>/run-<id>/benchmark_result.json`에 성공
횟수가 기록되고 에피소드당 카메라 영상 네 개가 저장됩니다. 진단 설정으로
돌린 결과는 `reportable: false`가 됩니다. `all`은 `data/evals/summary.json`도
쓰고, 조건 하나가 실패해도 나머지를 계속 실행합니다. 같은 명령을 다시
실행하면 같은 정책, 코드, 에셋, 실행 환경으로 이미 검증된 결과가 있는
조건은 건너뜁니다. `--list-envs`는 조건 ID 목록을, `--dry-run`은 실행 없이 환경과
에피소드 설정만 출력합니다. 여러 GPU에 조건을 나눠 돌리는 방법은
[조건 병렬 실행](docs/PUBLIC_EVALUATION.md#running-conditions-in-parallel)을
참고하세요.

`--model`은 `snupilab/humanoidtoolbench-act-sim-3003`,
`snupilab/humanoidtoolbench-dp-sim-3003` 같은 **HumanoidToolBench ACT 및
Diffusion Policy 시뮬레이션 체크포인트**를 Hugging Face ID나 로컬 경로에서
불러옵니다. 그 외 모델은 `--policy`나 아래 정책 서버로 평가합니다.

### 기준 결과

| 정책 | 조건 | 성공 |
| --- | --- | --- |
| `snupilab/humanoidtoolbench-act-sim-3003` | `G1BallMove-L0-S` | 2 / 100 |
| 학습하지 않은 무작위 초기화 ACT | `G1BallMove-L0-S` | 0 / 100 |

두 결과 모두 정식 프로토콜, 이 배포본의 에셋 매니페스트
(`assets.manifest_sha256`가 `7891d62a`로 시작), 커밋 `bd8abfa`의 평가
코드(`source_digest`가 `10cab839`로 시작)를 사용했습니다. ACT는 SHA-256이
`1e6459cb`로 시작하는 가중치(모델 리비전 `ffd733de`로 공개)를 사용했고 약
2시간이 걸렸습니다. 두 번째 행은 공개된 ACT 설정에 가중치만
새로 초기화한 것입니다(SHA-256이 `66788a3b`로 시작, 비공개). 배포된 ACT
체크포인트는 바닥에 가까운 기준점이지 강한 베이스라인이 아니므로, 새 정책의
점수가 낮다는 것만으로 설치가 잘못되었다고 볼 수는 없습니다. 위의 1
에피소드 진단 실행은 0/1이 나오는 것이 정상입니다. DP 체크포인트와 다른
조건은 아직 측정하지 않았습니다.

## 벤치마크 소개

![HumanoidToolBench 논문 개요 그림: 휴머노이드 도구 사용, 태스크 구성, 실제 로봇 평가.](docs/assets/paper-overview.png)

**HumanoidToolBench: Benchmarking Humanoid Tool Use from Selection to Mobile
Execution**은 이 벤치마크와 55개 도구 에셋, 시뮬레이션 및 실제 로봇 시연을
담은 ToolBook 데이터셋을 소개합니다. 이 배포본은 정규 환경만 포함하며,
훈련용 쉬운 변형은 제외되어 있습니다.

## 내 정책 평가하기

저장소 루트에 `my_policy.py`를 만들고, 모델을 한 번 불러온 뒤
`predict(request)`를 제공하세요.

```python
import numpy as np

def predict(request: dict) -> np.ndarray:
    image = request["image"]["rgb_head_stereo_left"]   # (360, 640, 3) uint8 RGB
    state = request["state"]["states"]                 # (1, 32) float32
    instruction = request["instruction"]               # 예: "Pick the tool and move the ball to the target."
    if request["history"].get("reset", False):         # 에피소드의 첫 질의
        pass                                            # 여기서 순환(recurrent) 상태를 초기화
    actions = ...                                       # 모델 추론
    return np.asarray(actions, dtype=np.float32)        # (T, 36), 행마다 50 Hz 명령 하나
```

모델의 의존성을 `uv.lock`에 고정된 버전(torch, numpy, transformers 등)을
바꾸지 않고 `uv pip install`로 이 환경에 설치할 수 있다면 명령 하나로
평가됩니다.

```bash
uv run htb-eval G1BallMove-L0-S --policy my_policy:predict --checkpoint MODEL_ID_OR_REVISION --episodes 1 --max-steps 100
```

그렇지 않으면 터미널 하나에서 본인의 모델 환경(NumPy와 requests가 있는
Python 3.10 이상)으로 저장소 루트에서 서버를 실행하고, 다른 터미널에서
평가기를 실행합니다.

```bash
PYTHONPATH=src python examples/serve_policy.py --policy my_policy:predict --checkpoint MODEL_ID_OR_REVISION
uv run htb-eval G1BallMove-L0-S --host 127.0.0.1 --port 21000 --episodes 1 --max-steps 100
```

`--episodes`와 `--max-steps`의 기본값이 정식 프로토콜이므로, 두 옵션을
빼면 reportable한 100 에피소드 실행이 됩니다. 서버가 다른 머신에서 돌 때는
서버를 `--host 0.0.0.0`으로 띄우고 그 주소를 평가기의 `--host`에 넘기세요.
서버는 평가기마다 하나씩 띄우세요. 한 평가기가 사용 중인 서버는 두 번째
평가기를 거절합니다. 32차원 state, 36차원 action(조인트 이름과
한계 포함), 이미지, reset 의미는
[정책 인터페이스](docs/PUBLIC_EVALUATION.md#policy-interface)에 명시되어
있습니다.

## 조건

| 시나리오 | 정답 도구 | L0 | L1 | L2 |
| --- | --- | --- | --- | --- |
| BallMove | 긴 막대 | 도구 집기 | 공을 링 안으로 밀기 | 동일, 작업대를 따라 이동한 뒤 |
| BallRetrieve | 갈고리 | 도구 집기 | 공을 링 안으로 끌어오기 | 동일, 작업대를 따라 이동한 뒤 |
| IceBreak | 금속 망치 | 도구 집기 | 얼음 블록 두 개 깨기 | 동일, 작업대를 따라 이동한 뒤 |

모드 **S**는 정답 도구 옆에 무관한 물체 두 개를 두고, 모드 **R**은 여기에
헷갈리는 도구 하나(짧은 막대, 곧은 막대, 또는 파리채·페인트 롤러·플런저
같은 가볍고 유연한 미끼 도구)를 더합니다. 성공 조건은 1초 동안 유지되어야
하며, L0은 정답 도구를 8 cm 들어 올려야 합니다. 지시문, 임계값, 기록되는
지표는 [docs/ENVIRONMENTS.md](docs/ENVIRONMENTS.md)에 있습니다.

## 데이터와 훈련

**[Hugging Face에서 ToolBook 다운로드](https://huggingface.co/datasets/snupilab/humanoidtoolbench-teleop)**

현재 녹화 데이터 다운로드에는 접근 신청과 승인이 필요합니다. 폴더 구조와
관측/행동 형식은 [데이터 가이드](data/README.md)를 참고하세요.

이 배포본은 **평가 코드**를 제공합니다. 시연 데이터를 사용해 원하는
프레임워크에서 모델을 훈련할 수 있으며, 모델 훈련 파이프라인은 포함되어
있지 않습니다.

[MIT 코드 라이선스](LICENSE). 외부 에셋, 모델, 데이터는 각각의 이용 조건을
따릅니다.
