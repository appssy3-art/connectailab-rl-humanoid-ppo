import os
import shutil
from huggingface_hub import HfApi, create_repo

def prepare_and_upload():
    api = HfApi()
    
    print("🚀 [허깅페이스 오리 로봇 2종 업로드 시작]...")
    
    # ==============================================================
    # 1. MicroDuck-100k 업로드 준비
    # ==============================================================
    repo_id_walk = "appssy3/connectailab-rl-solo-build-H-MicroDuck-100k"
    dir_walk = "e:/ai랩/hf_connectailab-rl-solo-build-H-MicroDuck-100k"
    os.makedirs(dir_walk, exist_ok=True)
    
    # Copy model files
    shutil.copy("E:/ai랩/오리로봇/saved_models/duck_walk_10750.pt", os.path.join(dir_walk, "duck_walk_10750.pt"))
    shutil.copy("E:/ai랩/오리로봇/saved_models/duck_velstand_17500.pt", os.path.join(dir_walk, "duck_velstand_17500.pt"))
    shutil.copy("E:/ai랩/오리로봇/microduck/policies/alpha_walking.onnx", os.path.join(dir_walk, "alpha_walking.onnx"))
    shutil.copy("E:/ai랩/오리로봇/microduck/policies/alpha_stand.onnx", os.path.join(dir_walk, "alpha_stand.onnx"))
    
    # Copy video
    if os.path.exists("E:/ai랩/MuJoCo_예제_10가지/duck_clip3.mp4"):
        shutil.copy("E:/ai랩/MuJoCo_예제_10가지/duck_clip3.mp4", os.path.join(dir_walk, "replay.mp4"))
    
    # Write README.md for MicroDuck-100k
    readme_walk = """---
tags:
- reinforcement-learning
- ppo
- bipedal-robot
- mujoco
- microduck
- robotics
- sim2real
library_name: pytorch
pipeline_tag: reinforcement-learning
---

# 🦆 ConnectAI Lab SoloBuild: MicroDuck-100k Bipedal Robot

<p align="center">
  <video src="https://huggingface.co/appssy3/connectailab-rl-solo-build-H-MicroDuck-100k/resolve/main/replay.mp4" controls width="640"></video>
</p>

## 📌 Project Overview
**MicroDuck-100k**는 10만 번(10,750+ 에포크) 이상의 시뮬레이션 환경에서 스스로 균형과 보행을 터득한 2족 보행 로봇 **MicroDuck**의 강화학습(PPO) 체크포인트입니다.

* **Robotics Platform**: MicroDuck Bipedal Dynamic Walking Robot
* **Physics Simulator**: MuJoCo (Multi-Joint dynamics with Contact)
* **Algorithm**: Proximal Policy Optimization (PPO, Continuous Action Space)
* **Training Iterations**: 10,750+ Iterations (100k+ Total Episode Steps)
* **Framework**: PyTorch (`.pt`) & Exported ONNX Runtime (`.onnx`)

---

## 🧠 Model Architecture & Hyperparameters

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Algorithm** | PPO (Clip) | Continuous Torque Control |
| **Clip Ratio ($\epsilon$)** | 0.2 | Standard PPO clipping |
| **Discount Factor ($\gamma$)** | 0.99 | Temporal credit assignment |
| **GAE Parameter ($\lambda$)** | 0.95 | Generalized Advantage Estimation |
| **Policy Network** | MLP [256, 256, 128] | Actor Network (Gaussian Mean & Std) |
| **Value Network** | MLP [256, 256, 128] | Critic Network (State Value Estimator) |
| **Learning Rate** | $3 \times 10^{-4}$ | Adam Optimizer with Linear Decay |
| **Control Frequency** | 50 Hz (0.02s) | Low-latency servo control loop |

---

## 🎯 Reward Function Engineering

MicroDuck의 직립 및 전진 보행을 위해 다중 목적(Multi-objective) 보상 함수를 구성하였습니다:

$$R = w_{\text{forward}} \cdot v_x - w_{\text{lateral}} \cdot |v_y| + w_{\text{alive}} - w_{\text{torque}} \cdot \|\tau\|^2 - w_{\text{impact}} \cdot F_{\text{contact}}$$

1. **전진 속도 보상 ($v_x$)**: 목표 보행 속도(1.2 m/s) 유지 시 최대 보상 부여
2. **좌우 흔들림 페널티 ($|v_y|$)**: 2족 보행 특유의 불필요한 좌우 진동 최소화
3. **생존 보상 ($w_{\text{alive}}$)**: 넘어지지 않고 직립 자세 유지 시 매 스텝 보상
4. **토크 효율성 페널티 ($\|\tau\|^2$)**: 액추에이터의 과열 방지 및 부드러운 관절 회전 유도
5. **외력 저항 및 복원력 ($F_{\text{push}}$)**: 랜덤 외력(Push) 충격 후 자세 복원 시 추가 보상

---

## 📦 Checkpoint Contents
* `duck_walk_10750.pt`: 10,750 이터레이션 완결 PyTorch 가중치 (Actor-Critic 및 Observation Normalizer 포함)
* `duck_velstand_17500.pt`: 17,500 이터레이션 직립 균형 및 외력 복원 가중치
* `alpha_walking.onnx`: 경량 추론 및 실시간 Sim2Real 배포용 ONNX 모델
* `alpha_stand.onnx`: 정지 및 직립 상태 유지용 ONNX 모델
* `replay.mp4`: 3D MuJoCo 환경에서의 보행 평가 영상

---

## 💻 Quickstart Inference (Python ONNX)

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("alpha_walking.onnx")
input_name = session.get_inputs()[0].name

# Example Observation vector
obs = np.random.randn(1, session.get_inputs()[0].shape[1]).astype(np.float32)
action = session.run(None, {input_name: obs})[0]
print("MicroDuck Joint Torque Action:", action)
```

---

## 📜 License & Citation
* License: MIT License
* Developed by: **ConnectAI Lab (SoloBuild Series)**
* GitHub: [appssy3-art/connectailab-rl-humanoid-ppo](https://github.com/appssy3-art/connectailab-rl-humanoid-ppo)
"""
    with open(os.path.join(dir_walk, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_walk)
        
    print("Created MicroDuck-100k files!")
    
    # ==============================================================
    # 2. RollerDuck-Skate 업로드 준비
    # ==============================================================
    repo_id_skate = "appssy3/connectailab-rl-solo-build-H-RollerDuck-Skate"
    dir_skate = "e:/ai랩/hf_connectailab-rl-solo-build-H-RollerDuck-Skate"
    os.makedirs(dir_skate, exist_ok=True)
    
    # Copy model files
    shutil.copy("E:/ai랩/오리로봇/microduck/policies/roller.onnx", os.path.join(dir_skate, "roller.onnx"))
    shutil.copy("E:/ai랩/오리로봇/microduck/policies/roller_crouch.onnx", os.path.join(dir_skate, "roller_crouch.onnx"))
    
    # Copy video
    if os.path.exists("E:/ai랩/MuJoCo_예제_10가지/duck_roller_shorts_final.mp4"):
        shutil.copy("E:/ai랩/MuJoCo_예제_10가지/duck_roller_shorts_final.mp4", os.path.join(dir_skate, "replay.mp4"))
        
    # Write README.md for RollerDuck-Skate
    readme_skate = """---
tags:
- reinforcement-learning
- ppo
- roller-skating
- bipedal-robot
- mujoco
- rollerduck
- robotics
library_name: pytorch
pipeline_tag: reinforcement-learning
---

# 🛼 ConnectAI Lab SoloBuild: RollerDuck Skate Robot

<p align="center">
  <video src="https://huggingface.co/appssy3/connectailab-rl-solo-build-H-RollerDuck-Skate/resolve/main/replay.mp4" controls width="640"></video>
</p>

## 📌 Project Overview
**RollerDuck-Skate**는 2족 보행 로봇에게 롤러스케이트를 장착시키고, 바닥 마찰력과 관성 모멘텀을 스스로 활용해 미끄러지듯 질주하도록 훈련시킨 AI 강화학습 모델입니다.

* **Robotics Platform**: RollerDuck (14-DoF Servomotor Actuators with Passive Wheel Skates)
* **Physics Simulator**: MuJoCo (Multi-Joint dynamics with Contact & Low Friction)
* **Algorithm**: Proximal Policy Optimization (PPO)
* **Training Scale**: **43,000,000 Steps (4,300만 번의 가상 물리 연산)**
* **Early Exploration**: 0세대 시절 9스텝 만에 447회 넘어짐
* **Final Performance**: **넘어짐 확률 0% 달성**, 부드러운 활주(Gliding) 및 무게중심 이동 제어 완성

---

## 🛹 Physical Modeling & Control Architecture

1. **바퀴 마찰 제어 (Contact Dynamics)**:
   * 발바닥에 무동력 롤러 휠을 장착하여 전후 방향은 극도로 낮은 구름 저항, 좌우 방향은 에지 그립(Edge Grip) 마찰력을 갖도록 모델링.
2. **14개 관절 서보모터 (14-DoF Control)**:
   * 0.02초(50Hz) 제어 루프를 통해 미세한 미끄러짐을 실시간으로 감지하고 골반/무릎/발목 각도를 칼같이 보정.
3. **다운포스 & 크라우칭 (Crouching State)**:
   * 무게중심을 낮추어 고속 주행 시 전복을 방지하는 크라우칭 스케이트 모드(`roller_crouch.onnx`) 동시 학습.

---

## 🎯 Reward Engineering for Skating

기존의 보행 보상과 달리, 발을 떼지 않고 미끄러지며 속도를 내는 **스케이트 특화 보상 함수**를 설계했습니다:

$$R_{\text{skate}} = w_{\text{glide}} \cdot v_{\text{forward}} - w_{\text{fall}} \cdot \mathbb{I}_{\text{fallen}} - w_{\text{slip}} \cdot |v_{\text{lateral}}| + w_{\text{posture}} \cdot \cos(\theta_{\text{torso}})$$

* **활주 속도 보상 ($w_{\text{glide}}$)**: 지면 접촉을 유지한 채 전진 관성 속도가 증가할수록 지수적 보상 부여
* **자세 균형 보상 ($w_{\text{posture}}$)**: 상체 척추 각도($\theta_{\text{torso}}$)를 안정적인 스케이팅 자세로 유지
* **충격 완화**: 급격한 토크 변화로 인한 바퀴 헛돎 방지

---

## 📦 Checkpoint Contents
* `roller.onnx`: 고속 롤러스케이트 질주 주행용 ONNX 가중치 (실시간 인퍼런스 최적화)
* `roller_crouch.onnx`: 무게중심을 낮추어 안정적인 코너링 및 고속 활주를 수행하는 크라우칭 가중치
* `replay.mp4`: 4,300만 번 학습 끝에 완성된 롤러스케이트 완벽 질주 영상

---

## 💻 Quickstart Inference (Python ONNX)

```python
import numpy as np
import onnxruntime as ort

session = ort.InferenceSession("roller.onnx")
input_name = session.get_inputs()[0].name

# 14-DoF Observation state
obs = np.random.randn(1, session.get_inputs()[0].shape[1]).astype(np.float32)
torque_action = session.run(None, {input_name: obs})[0]
print("RollerDuck 14-DoF Servo Torque Action:", torque_action)
```

---

## 📜 License & Citation
* License: MIT License
* Developed by: **ConnectAI Lab (SoloBuild Series)**
* YouTube Shorts: [걸으랬더니 롤러스케이트 장인이 된 인공지능](https://www.youtube.com/@appssy3)
"""
    with open(os.path.join(dir_skate, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme_skate)
        
    print("Created RollerDuck-Skate files!")
    
    # ==============================================================
    # 3. Create Repositories & Upload to Hugging Face
    # ==============================================================
    for repo_id, folder in [(repo_id_walk, dir_walk), (repo_id_skate, dir_skate)]:
        print(f"\n📦 [{repo_id}] 리포지토리 생성 및 업로드 중...")
        create_repo(repo_id, exist_ok=True, repo_type="model")
        api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            repo_type="model"
        )
        print(f"✅ [{repo_id}] 업로드 완료!")

if __name__ == "__main__":
    prepare_and_upload()

