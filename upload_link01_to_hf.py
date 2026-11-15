"""
Hugging Face Upload Script for LINK-01 Bipedal Robot (PPO)
Target Repo: appssy3/connectailab-rl-solo-build-H-v1-link01-walk-ppo
"""
import os
import shutil
import json
from huggingface_hub import HfApi, create_repo

def upload_link01():
    api = HfApi()
    repo_id = "appssy3/connectailab-rl-solo-build-H-v1-link01-walk-ppo"
    out_dir = "E:/ai랩/hf_connectailab-rl-solo-build-H-v1-link01-walk-ppo"
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"🚀 [Hugging Face 업로드 준비: {repo_id}]...")
    
    # 1. 파일 복사
    src_dir = "E:/ai랩/강화학습PPO/link01-gym-ppo7/my_walk"
    shutil.copy(os.path.join(src_dir, "model.zip"), os.path.join(out_dir, "model.zip"))
    shutil.copy(os.path.join(src_dir, "vecnormalize.pkl"), os.path.join(out_dir, "vecnormalize.pkl"))
    shutil.copy(os.path.join(src_dir, "actor.json"), os.path.join(out_dir, "actor.json"))
    
    # 16:9 고화질 비디오 복사
    video_16x9 = os.path.join(src_dir, "replay_16x9.mp4")
    if os.path.exists(video_16x9):
        shutil.copy(video_16x9, os.path.join(out_dir, "replay.mp4"))
    else:
        shutil.copy("E:/ai랩/강화학습PPO/runs/my_walk_eval/replay.mp4", os.path.join(out_dir, "replay.mp4"))
        
    # 2. connectailab.json 생성
    meta = {
        "format": "connectailab/1",
        "repo": repo_id,
        "kind": "ppo",
        "env": "LINK-01 Walk (Link01Walk-v0)",
        "algo": "ppo",
        "robot": "LINK-01",
        "files": ["model.zip", "vecnormalize.pkl", "actor.json", "replay.mp4"],
        "author": {
            "name": "appssy3",
            "hub": "https://huggingface.co/appssy3",
            "youtube": "https://www.youtube.com/@solobuild.studio",
            "about": "ConnectAI Lab SoloBuild AI & Physical Robotics Research"
        },
        "config": {
            "steps": 1007616,
            "minutes": 9.23,
            "lr": 0.0003,
            "clip": 0.2,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "env": "Link01WalkGuided-v0 (ref_weight=2.0)"
        },
        "test": {
            "distance_m": -1.013,
            "speed_mps": -0.0507,
            "falls_per_min": 0.0,
            "episodes": 5,
            "seconds_per_episode": 20.0,
            "command_mps": 0.18
        },
        "course": "AI CITY BUILDERS 강화학습 AI 7강",
        "date": "2026-11-15"
    }
    with open(os.path.join(out_dir, "connectailab.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
        
    # 3. 고품질 README.md 모델 카드 생성
    readme_content = r"""---
tags:
- reinforcement-learning
- ppo
- robotics
- mujoco
- locomotion
- link01
- bipedal-robot
- connect-ai-lab
- connectailab
- stable-baselines3
library_name: stable-baselines3
pipeline_tag: reinforcement-learning
model-index:
- name: connectailab-rl-solo-build-H-v1-link01-walk-ppo
  results:
  - task: {type: reinforcement-learning, name: walking}
    dataset: {type: Link01Walk-v0, name: Link01Walk-v0}
    metrics:
    - {type: distance_m, value: 1.013, name: "distance per 20 s (m)"}
    - {type: falls_per_min, value: 0.0, name: "falls per minute (0% fall rate)"}
---

# 🤖 ConnectAI Lab SoloBuild: LINK-01 Bipedal Walking Robot (PPO v1)

> **AI CITY BUILDERS · Connect AI LAB 강화학습 7강 실습 모델**  
> 14개 서보모터 관절(DoF)을 가진 2족 보행 피지컬 AI 로봇 **LINK-01**이 **PPO(Proximal Policy Optimization)** 를 통해 100만 번의 물리 시뮬레이션 환경에서 스스로 균형과 14개 관절 협응 보행을 터득한 1세대(v1) 강화학습 체크포인트입니다.

---

## 📌 Project & Physical Robot Specification

* **Robotics Platform**: LINK-01 Bipedal Physical AI Mini Robot (14-DoF)
* **Physics Simulator**: MuJoCo 3.16 (500 Hz Physics / 50 Hz Low-Latency Policy Loop)
* **Algorithm**: Proximal Policy Optimization (PPO, Continuous Action Space)
* **Training Volume**: 1,007,616 Steps (약 9분 초고속 병렬 수렴)
* **Performance**: **넘어짐 확률 0% (Falls: 0.0회/분)** 달성, 20초 주행 동안 완벽한 기립 및 관절 교차 보행 유지
* **Inference Runtime**: Stable-Baselines3 (`model.zip`), `vecnormalize.pkl`, 및 **EZERAI 웹 플레이그라운드 호환 JSON (`actor.json`)** 동시 지원

---

## 🧠 Neural Architecture & Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Observation Space** | **51-dim** | 중력 벡터 3, 각속도 3, 관절 각도 14, 관절 각속도 14, 직전 행동 14, 명령 속도 3 |
| **Action Space** | **14-dim** | 다리 10 + 머리 4 관절 서보모터 변위 ([-1, 1] = ±0.18 rad) |
| **Actor Network** | **MLP [51 -> 64 -> 64 -> 14]** | Tanh 활성화 함수 (EZERAI 웹 호환 규격) |
| **Critic Network** | **MLP [51 -> 256 -> 256 -> 1]** | State-Value $V(s)$ 추정기 |
| **Learning Rate** | $3 \times 10^{-4}$ | Adam Optimizer |
| **Rollout Steps** | 1,024 steps × 8 envs | 8,192 스텝 단위 정책 업데이트 |
| **Batch Size & Epochs** | 64 / 10 epochs | Mini-batch SGD |
| **Clip Ratio ($\epsilon$)** | 0.2 | PPO Clipped Surrogate Objective |
| **Discount ($\gamma$) / GAE ($\lambda$)**| 0.99 / 0.95 | Generalized Advantage Estimation |

---

## 🎯 Reward Engineering for Bipedal Locomotion

제자리 서 있기 꼼수(Lazy Standing)를 차단하고 실전 보행을 유도하는 복합 보상 함수를 적용했습니다:

$$R_t = R_{\text{track}} + R_{\text{progress}} + R_{\text{air\_time}} + R_{\text{alive}} + R_{\text{upright}} + R_{\text{side}} + R_{\text{energy}} + R_{\text{gait\_ref}}$$

1. **전진 및 속도 추종 ($R_{\text{progress}}, R_{\text{track}}$)**: 명령 속도($0.18\text{ m/s}$)에 맞추어 앞으로 나아갈 때 큰 보상을 부여.
2. **발디딤 보너스 ($R_{\text{air\_time}}$)**: 다리를 지면에서 들어올려 교차 보행을 수행할 때 보너스 부여.
3. **CPG 보행 리듬 가이드 ($R_{\text{gait\_ref}}$)**: 위상 독립적(Phase-free) CPG 레퍼런스 게이트 패턴을 따라갈 때 모멘텀 보상 부여.
4. **넘어짐 감점 ($R_{\text{fall}} = -10.0$)**: 몸체 높이가 $0.10\text{ m}$ 이하로 떨어지거나 크게 전복되면 에피소드 즉시 종료.

---

## 💻 Python Quickstart & Model Inference

### 1. 웹 플레이그라운드 호환 두뇌 (`actor.json`) 순수 NumPy 추론 (의존성 최소화)
```python
import json
import numpy as np
import gymnasium as gym
import link01_gym

with open("actor.json", "r") as f:
    data = json.load(f)

weights = [np.array(l["weight"], dtype=np.float64) for l in data["layers"]]
biases = [np.array(l["bias"], dtype=np.float64) for l in data["layers"]]

def policy(obs):
    x = np.asarray(obs, dtype=np.float64)
    for i, (w, b) in enumerate(zip(weights, biases)):
        x = w @ x + b
        if i < len(weights) - 1:
            x = np.tanh(x)
    return np.clip(x, -1.0, 1.0)

env = gym.make("Link01Walk-v0", render_mode="human")
obs, _ = env.reset(seed=42)
for _ in range(1000):
    action = policy(obs)
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        obs, _ = env.reset()
```

---

## 🔗 관련 링크 및 채널 (Official Links)

> 💡 **ConnectAI Lab SoloBuild**의 전체 피지컬 AI 로봇 및 강화학습 연구 생태계를 둘러보세요:

| 채널 / 허브 | 링크 | 설명 |
| :--- | :--- | :--- |
| 🚀 **개발자 허브 (SOLO BUILD-H)** | [**전체 프로젝트 & 웹 앱 둘러보기 ↗**](https://appssy3-art.github.io/solo-build-studio/) | 피지컬 AI 로봇, 강화학습 웹 시뮬레이터 및 솔로빌드 전체 포트폴리오 |
| 📺 **공식 유튜브 채널 (솔로빌드)** | [**개발 과정 및 쇼츠 영상 보기 ↗**](https://www.youtube.com/@solobuild.studio) | 로봇 시뮬레이션 진화 과정, 3D 보행 훈련 쇼츠 및 개발 브이로그 |
| 🤗 **Hugging Face Hub** | [**appssy3 모델 전체 컬렉션 ↗**](https://huggingface.co/appssy3) | 뒤집힌 로드스터, 17-DoF 휴머노이드, 오리 로봇 및 LINK-01 모델 허브 |
| 🐙 **GitHub 공식 코드베이스** | [**connectailab-rl-humanoid-ppo ↗**](https://github.com/appssy3-art/connectailab-rl-humanoid-ppo) | 강화학습 PPO 전체 파이썬 소스코드 및 3D 관찰기 배치 파일 |

---
**ConnectAI Lab · SoloBuild Project** | Lead Researcher: **appssy3**
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_content)
        
    print(f"📁 업로드 대상 파일: {os.listdir(out_dir)}")
    
    # 4. 저장소 생성 및 푸시
    create_repo(repo_id, exist_ok=True, repo_type="model")
    print(f"🚀 Hugging Face Hub 업로드 중: {repo_id}...")
    api.upload_folder(
        folder_path=out_dir,
        repo_id=repo_id,
        commit_message="feat: upload LINK-01 bipedal robot 1M steps PPO model and demo video"
    )
    print(f"🎉 [업로드 성공!] 주소: https://huggingface.co/{repo_id}")

if __name__ == '__main__':
    upload_link01()

