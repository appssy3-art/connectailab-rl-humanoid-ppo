# 🤖 ConnectAI Lab: Full-Stack Reinforcement Learning (PPO) & Robotics Suite

> **ConnectAI Lab SoloBuild 연구 프로젝트**  
> 고전 제어 연속 제어(Continuous Control)부터 **MuJoCo 17-DoF 고차원 휴머노이드 보행**, 그리고 **MicroDuck 2족 보행 및 롤러스케이트 로봇**까지 — **PPO(Proximal Policy Optimization)** 알고리즘을 활용한 강화학습 전과정과 모델 배포, 3D 실시간 관찰기, 미디어 자동화 파이프라인을 집대성한 프로젝트입니다.

---

## 🌟 3대 연구 및 제어 트랙 (Research Tracks)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ConnectAI Lab RL Architecture                         │
├───────────────────────┬──────────────────────────┬──────────────────────────┤
│ Track 1: 고전 연속 제어│ Track 2: MuJoCo 휴머노이드│ Track 3: 오리 로봇 제어  │
│ [뒤집힌 로드스터 착지] │ [17-DoF 5대 진화 질주]   │ [MicroDuck & RollerDuck] │
│ • 0% -> 100% 수직 착지│ • 0세대 뇌진탕 -> 마라톤 │ • 평발 보행 & 롤러 글라이딩│
└───────────────────────┴──────────────────────────┴──────────────────────────┘
```

---

## 🏆 Track 1. 뒤집힌 로드스터 정밀 수직 착지 (Roadster PPO)

상공 10m에서 뒤집힌 채 자유낙하하는 테슬라 로드스터를 지면 중앙 패드에 100% 안전하게 착지시키는 연속 추진 제어 에이전트입니다.

| 단계 | 모델 버전 | 학습 걸음수 | 평균 보상 | 착지 성공률 | 비행 스타일 및 AI 제어 특징 |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **🌱 초보** | Baseline | 10만 걸음 | `-90.9점` | 0.0% | **자세 제어 미숙 & 전원 추락**: 공중에서 회전하다가 지면 충돌 |
| **🛡️ 안전형** | [**Roadster v1**](https://huggingface.co/appssy3/connectailab-rl-solo-build_H-roadster-ppo-v1) | 50만 걸음 | `+94.2점` | **100.0%** | **생존 최우선 소프트 랜딩**: 전복 없이 안전 착지 성공 |
| **👑 마스터** | [**Roadster v2**](https://huggingface.co/appssy3/connectailab-rl-solo-build_H-roadster-ppo-v2) | 100만 걸음 | `+105.2점` | **90~100%** | **연료 절약형 정밀 착지**: 불필요한 역분사를 줄이고 정중앙 칼착지 |

---

## 🏃 Track 2. MuJoCo 17-DoF 휴머노이드 5대 진화 (Humanoid Evolution)

17개 관절 모터를 스스로 제어하여 '0세대 뇌진탕 낙하'부터 '250만 세대 마라톤 질주'까지 인간형 보행을 터득하는 과정입니다.

| 세대 | 모델명 | 학습량 | 평균 점수 | 최고 시속 | 보행 및 제어 특성 |
| :---: | :--- | :---: | :---: | :---: | :--- |
| **v1** | [**Humanoid-v1**](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v1-Humanoid) | 10만 걸음 | 185점 | 0.3 m/s | **기립 균형 유지**: 제자리에서 넘어지지 않고 서 있는 기본 자세 터득 |
| **v2** | [**Humanoid-v2**](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v2-Humanoid) | 40만 걸음 | 420점 | 0.7 m/s | **첫 걸음마**: 중심을 이동하며 비틀거리지만 앞으로 전진 시작 |
| **v3** | [**Humanoid-v3**](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v3-Humanoid) | 100만 걸음 | 1,164점 | 1.8 m/s | **폭풍 질주**: 넘어지지 않고 10초 이상 초고속 단거리 스프린트 |
| **v4** | [**Humanoid-v4**](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v4-Humanoid) | 150만 걸음 | 1,350점 | 1.4 m/s | **자연스러운 인간형 보행**: 불필요한 관절 떨림을 억제하고 넓은 보폭 완성 |
| **v5** | [**Humanoid-v5**](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v5-Humanoid) | 250만 걸음 | **1,600+점** | 1.6 m/s | **완전무결 마라토너**: 넘어짐 확률 0%의 이상적인 장거리 지속 조깅 모델 |

---

## 🦆 Track 3. MicroDuck & RollerDuck 오리 로봇 (Duck Robotics)

| 모델명 | 관절 / 형태 | 학습 규모 | 낙하율 | 특성 및 데모 영상 |
| :--- | :---: | :---: | :---: | :--- |
| [**MicroDuck-100k**](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-MicroDuck-100k) | 2족 보행 (Flat Shoes) | 10,750+ 에포크 | **0%** | **평발 보행 로봇**: 10억 번 넘어져도 다시 일어나는 강건한 2족 보행 터득 |
| [**RollerDuck-Skate**](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-RollerDuck-Skate) | 14-DoF 롤러스케이트 | 4,300만 스텝 | **0%** | **롤러스케이트 장인**: 무동력 휠 바닥 마찰을 이용한 초저저항 글라이딩(Gliding) |

---

## 🤗 Hugging Face 모델 허브 (전체 9종)

공개 배포된 체크포인트(`.pt`, `.onnx`) 및 시연 영상은 Hugging Face Hub에서 즉시 확인하고 다운로드할 수 있습니다:

| 분류 | 모델 레포지토리 | 가중치 포맷 | 주요 특징 |
| :---: | :--- | :---: | :--- |
| **Roadster** | [appssy3/connectailab-rl-solo-build_H-roadster-ppo-v1](https://huggingface.co/appssy3/connectailab-rl-solo-build_H-roadster-ppo-v1) | PyTorch | 50만보 100% 안전 착지 모델 |
| **Roadster** | [appssy3/connectailab-rl-solo-build_H-roadster-ppo-v2](https://huggingface.co/appssy3/connectailab-rl-solo-build_H-roadster-ppo-v2) | PyTorch | 100만보 연료 최적화 마스터 착지 모델 |
| **Humanoid** | [appssy3/connectailab-rl-solo-build-H-v1-Humanoid](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v1-Humanoid) | PyTorch / Video | 10만보 기립 균형 제어 |
| **Humanoid** | [appssy3/connectailab-rl-solo-build-H-v2-Humanoid](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v2-Humanoid) | PyTorch / Video | 40만보 첫 걸음마 전진 |
| **Humanoid** | [appssy3/connectailab-rl-solo-build-H-v3-Humanoid](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v3-Humanoid) | PyTorch / Video | 100만보 고속 스프린트 (1,164점) |
| **Humanoid** | [appssy3/connectailab-rl-solo-build-H-v4-Humanoid](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v4-Humanoid) | PyTorch / Video | 150만보 부드러운 인간형 보행 |
| **Humanoid** | [appssy3/connectailab-rl-solo-build-H-v5-Humanoid](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-v5-Humanoid) | PyTorch / Video | 250만보 무결점 마라톤 러너 (1,600점) |
| **Duck** | [appssy3/connectailab-rl-solo-build-H-MicroDuck-100k](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-MicroDuck-100k) | PyTorch / ONNX | 평발 신발 보행 오리 |
| **Duck** | [appssy3/connectailab-rl-solo-build-H-RollerDuck-Skate](https://huggingface.co/appssy3/connectailab-rl-solo-build-H-RollerDuck-Skate) | ONNX Runtime | 롤러스케이트 무동력 글라이딩 오리 |

---

## 🎮 3D 실시간 물리 관찰기 (Interactive Visualizer)

Windows 환경에서 더블클릭 한 번으로 훈련된 AI 에이전트의 3D 시뮬레이션을 관찰할 수 있습니다:

```bash
# 휴머노이드 5단계 세대별 관찰기 실행
휴머노이드_모델별_3D_관찰기.bat
```
- 번호 선택 (1~5) 시 해당 세대의 MuJoCo 3D 물리 렌더링 창이 즉시 실행되어 실시간 관절 토크와 보행 모습을 관찰할 수 있습니다.

---

## 🎬 미디어 & 유튜브 쇼츠 자동화 파이프라인

- **유튜브 쇼츠 제작**: `make_master_youtube_shorts.py` / `build_clean_consistent_shorts.py`
  - 1080x1920 세로 해상도, 텍스트 하이라이트 오버레이, TTS 나레이션 음성 믹싱
- **채널 최적화 썸네일 생성기**: `generate_final_perfect_thumbnail.py`
  - 유튜브 모바일 및 PC 채널 그리드 1:1.55 크롭 대응 **상단 180px / 하단 210px 안전 마진(Safe Margin)** 적용
  - 3D 히어로 렌더링 프레임 중앙 배치 및 후킹 텍스트 박스 자동 합성

---

## 🛠️ 빠른 시작 (Quick Start)

### 1. 가상환경 세팅 및 의존성 설치
```bash
git clone https://github.com/appssy3-art/connectailab-rl-humanoid-ppo.git
cd connectailab-rl-humanoid-ppo

# 필수 패키지 일괄 설치
pip install -r requirements.txt
```

### 2. PPO 훈련 실행
```bash
# 휴머노이드 v5 마라톤 모델 훈련
python train_humanoid_v5.py

# 로드스터 착지 훈련
python run_ppo_task.py
```

---

## 📁 디렉토리 구조 요약

```text
├── requirements.txt                         # 일괄 의존성 명세서 (PyTorch, MuJoCo, ONNX 등)
├── README.md                                # 종합 프로젝트 문서
├── .gitignore                               # 바이너리/대용량 영상/보안 파일 제외 규칙
│
├── [Track 1: 로드스터 착지]
│   ├── 로드스터_시뮬레이터.py                # 2D 물리 시뮬레이터
│   ├── run_ppo_task.py                     # PPO 훈련 파이프라인
│   └── 강화학습_6강_실습_PPO_뒤집힌로드스터...ipynb
│
├── [Track 2: MuJoCo 휴머노이드]
│   ├── natural_walking_env.py              # 휴머노이드 보행 특화 커스텀 환경
│   ├── train_humanoid_v1~v5.py             # 세대별 훈련 스크립트
│   ├── view_humanoid_models.py             # 3D 관찰 뷰어
│   └── 휴머노이드_모델별_3D_관찰기.bat      # 원클릭 실행 배치 파일
│
├── [Track 3: 오리 로봇]
│   ├── upload_both_ducks_to_hf.py          # 오리 2종 모델/영상 허깅페이스 배포기
│   └── update_hf_repos_clean.py            # 허깅페이스 저장소 최적화 스크립트
│
└── [Track 4: 미디어 & 썸네일]
    ├── generate_final_perfect_thumbnail.py # 안전 마진 유튜브 쇼츠 썸네일 생성기
    ├── build_perfect_thumbnail.py          # 채널 맞춤형 썸네일 빌더
    └── make_master_youtube_shorts.py       # 쇼츠 영상 자동 렌더링 파이프라인
```

---
**ConnectAI Lab · SoloBuild Project** | Developed by **appssy3**
