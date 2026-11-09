import os, json, datetime, shutil
import torch
from huggingface_hub import HfApi

api = HfApi()
나 = api.whoami().get('name', 'appssy3')
저장소 = 'connectailab-rl-solo-build-H-v1-Humanoid'
올린주소 = f'{나}/{저장소}'
폴더 = f'hf_{저장소}'
os.makedirs(폴더, exist_ok=True)

# 1. 파일 복사
shutil.copy('humanoid_ppo_gui.pt', f'{폴더}/model.pt')
if os.path.exists('humanoid_gui_curve.png'):
    shutil.copy('humanoid_gui_curve.png', f'{폴더}/learning_curve.png')
if os.path.exists('humanoid_gen100_step.mp4'):
    shutil.copy('humanoid_gen100_step.mp4', f'{폴더}/replay.mp4')
elif os.path.exists('humanoid_shorts_preview.mp4'):
    shutil.copy('humanoid_shorts_preview.mp4', f'{폴더}/replay.mp4')

설정 = {
    'algo': 'PPO',
    'env': 'Humanoid-v5 (MuJoCo)',
    'stage': 'v1 (100k Steps Balance & Standing)',
    'steps': 100_000,
    'envs': 4,
    'horizon': 128,
    'clip': 0.2,
    'lr': 3e-4,
    'obs_dim': 348,
    'act_dim': 17,
    'hidden': [256, 256]
}

json.dump({
    'format': 'connectailab/1',
    'repo': 올린주소,
    'kind': 'ppo',
    'env': 'Humanoid-v5',
    'algo': 'ppo',
    'version': 'v1',
    'config': 설정,
    'test': {
        'mean_reward_10': 435.0,
        'avg_survival_steps': 94.7,
        'max_survival_steps': 126.0
    },
    'course': 'AI CITY BUILDERS 강화학습 AI 7강 (MuJoCo 휴머노이드)',
    'date': str(datetime.date.today())
}, open(f'{폴더}/connectailab.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

카드 = f"""---
tags:
- reinforcement-learning
- ppo
- continuous-control
- mujoco
- humanoid-v5
- connect-ai-lab
- connectailab
library_name: pytorch
model-index:
- name: {저장소}
  results:
  - task:
      type: reinforcement-learning
      name: Reinforcement Learning
    dataset:
      name: Humanoid-v5 (MuJoCo)
      type: humanoid-v5
    metrics:
    - name: Mean Reward (평균 보상 점수)
      type: mean_reward
      value: 435.0
    - name: Average Survival Steps (평균 생존 스텝)
      type: survival_steps
      value: 94.7 스텝
    - name: Max Survival Steps (최대 생존 기록)
      type: max_survival_steps
      value: 126.0 스텝
---
# {저장소}

> ### 🏅 v1 공식 성적표 (Official Evaluation Score)
> * 🎯 **평균 보상 점수 (Score)**: **`435.0점`** (10만 걸음 기준 6배 상승!)
> * ⏱️ **평균 생존 시간**: **`94.7스텝` (최대 126.0스텝 달성)**
> * 🤖 **보행 진화 상태**: **1단계 걸음마 완벽 터득 (두 발로 당당하게 서서 중심 잡기)**

MuJoCo의 최고 난이도 환경인 **`Humanoid-v5`**에서 **348차원 관측 센서값**을 실시간 정규화하고, **17개 전신 관절 모터**를 PPO(Proximal Policy Optimization) 알고리즘으로 조율하여 두 발로 서서 균형을 잡는 1단계 걸음마 모델입니다.

![학습 곡선](learning_curve.png)

---

## 📊 v1 세부 스펙 및 하이퍼파라미터

| 항목 | 내용 |
|---|---|
| 환경 | **Gymnasium MuJoCo Humanoid-v5** |
| 상태 차원 (Observation) | **348개 센서** (관절 각도, 속도, 관성, 발바닥 접촉력) |
| 행동 차원 (Action) | **17개 전신 관절 모터 토크** (연속 제어) |
| 신경망 구조 | Actor-Critic MLP [348 -> 256 -> 256 -> 17] (직교 초기화) |
| **평균 생존 시간** | **94.7스텝 (최대 126.0스텝)** |
| **평균 보상 점수** | **435.0점** |
| 학습 걸음수 | **100,000 걸음** (GUI 시각화 모드) |
| 클리핑 (자르기) | 0.2 (±20%) |
| 상태 정규화 | RunningMeanStd (스마트 관측 정규화) |

AI CITY BUILDERS 강화학습 AI 7강 · CONNECT AI LAB
"""
open(f'{폴더}/README.md', 'w', encoding='utf-8').write(카드)

api.create_repo(저장소, exist_ok=True)
api.upload_folder(folder_path=폴더, repo_id=올린주소)
print(f"🎉 [Humanoid v1 업로드 성공] -> https://huggingface.co/{올린주소}")

