# 🧠 강화학습 6강 — PPO 뒤집힌 로드스터 착지 프로젝트

> **AI CITY BUILDERS · Connect AI LAB 강화학습 6강 실습 및 고도화 연구**  
> 5강 DQN(착지율 0%)의 파괴적 망각 한계를 극복하고, **PPO(Proximal Policy Optimization)** 를 통해 상공 10m에서 뒤집힌 채 떨어지는 로드스터를 지면에 100% 안전 착지시키는 연속 제어 에이전트 프로젝트입니다.

---

## 🏆 학습 단계별 3단 진화 비교: [초보] vs [안전] vs [마스터]

| 단계 | 모델 버전 | 학습 걸음수 | 평균 보상 점수 | 착지 성공률 | 비행 스타일 및 AI 제어 특징 |
|:---:|:---:|:---:|:---:|:---:|---|
| **🌱 초보 (Beginner)** | Baseline | 10만 걸음 (13초) | **`-90.9점`** | **0.0%** | **자세 제어 미숙 & 전원 추락**: 공중에서 회전 분사기를 시험해보지만 착지 궤적을 완성하지 못하고 지면에 충돌 |
| **🛡️ 안전형 (Safe)** | [**v1 모델**](https://huggingface.co/appssy3/connectailab-rl-solo-build_H-roadster-ppo-v1) | 50만 걸음 (52초) | **`+94.2점`** | **100.0%** | **생존 및 착지 성공 최우선**: 최초 착지(골든 모멘트) 이후 안정적인 소프트 랜딩 성공. 단, 추락이 두려워 연료를 다소 과소비함 |
| **👑 마스터 (Master)** | [**v2 모델**](https://huggingface.co/appssy3/connectailab-rl-solo-build_H-roadster-ppo-v2) | 100만 걸음 (109초) | **`+105.2점`** | **90 ~ 100%** | **연료 절약형 에코 드라이빙**: 불필요한 낭비 추력을 없애고 꼭 필요한 순간에만 정밀 역분사를 뿜어 패드 정중앙에 수직 안착 |

---

## 🚀 허깅페이스 공개 모델 (Hugging Face Models)

* **v1 (안전형 100% 착지)**: [🤗 appssy3/connectailab-rl-solo-build_H-roadster-ppo-v1](https://huggingface.co/appssy3/connectailab-rl-solo-build_H-roadster-ppo-v1)
* **v2 (마스터 연료 절약형)**: [🤗 appssy3/connectailab-rl-solo-build_H-roadster-ppo-v2](https://huggingface.co/appssy3/connectailab-rl-solo-build_H-roadster-ppo-v2)

---

## 📊 PPO 하이퍼파라미터 및 학습 설정

- **클리핑 범위 (`자르기`)**: `0.1` (±10% 정책 변동 제한)
- **병렬 환경 수 (`세상수`)**: `10`
- **롤아웃 수집 길이 (`한번에`)**: `256` (1주기당 2,560걸음 수집)
- **학습률 (`lr`)**: `3e-4`
- **탐험 보너스 (`entropy_coef`)**: `0.01`
- **PPO 에포크 (`반복`)**: `10`회

### 📈 학습 곡선 요약
![학습 곡선](learning_curve.png)

---

## 📁 파일 구조

```text
├── 강화학습_6강_실습_PPO_뒤집힌로드스터_착지_ipynb의_사본.ipynb  # 주피터 노트북 (Colab & Windows 로컬 호환)
├── 로드스터_시뮬레이터.py                                # Gymnasium 기반 2D 물리 시뮬레이터 (이산/연속)
├── run_ppo_task.py                                     # 고속 PPO 훈련 및 평가 파이프라인 스크립트
├── upload_to_hf.py                                     # v1 허깅페이스 자동 업로드 스크립트
├── upload_to_hf_v2.py                                  # v2 허깅페이스 자동 업로드 스크립트
├── make_telemetry.py                                   # 조종석 HUD 계기판 시각화 영상 생성기
├── model.pt                                            # 훈련된 최신 파이토치 모델 가중치
├── learning_curve.png                                  # 학습 보상 및 착지율 추이 시각화
├── roadster_ppo_joystick.mp4                           # 최종 착지 시연 동영상
└── roadster_telemetry.mp4                              # 실시간 추진기 게이지가 포함된 HUD 동영상
```
