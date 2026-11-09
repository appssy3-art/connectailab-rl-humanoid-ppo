import os, json, datetime, shutil
import torch, numpy as np, gymnasium as gym
from huggingface_hub import HfApi
from train_humanoid_gui import 휴머노이드두뇌, 상태정규화

def upload_v2():
    api = HfApi()
    나 = api.whoami().get('name', 'appssy3')
    저장소 = 'connectailab-rl-solo-build-H-v2-Humanoid'
    올린주소 = f'{나}/{저장소}'
    폴더 = f'hf_{저장소}'
    os.makedirs(폴더, exist_ok=True)

    model_file = 'humanoid_ppo_gen500.pt'
    if not os.path.exists(model_file):
        print(f"⚠️ {model_file} 파일이 아직 생성되지 않았습니다.")
        return

    # 모델 평가
    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    env = gym.make('Humanoid-v5')
    상태수 = env.observation_space.shape[0]
    행동수 = env.action_space.shape[0]

    정규화 = 상태정규화((상태수,))
    정규화.mean = ckpt['obs_mean']
    정규화.var = ckpt['obs_var']

    뇌 = 휴머노이드두뇌(상태수, 행동수, 은닉=256)
    뇌.load_state_dict(ckpt['brain'])
    뇌.eval()

    scores, steps = [], []
    for seed in range(10):
        o, _ = env.reset(seed=200 + seed)
        ep_r, ep_s, done = 0, 0, False
        while not done:
            with torch.no_grad():
                d = 뇌.분포(torch.as_tensor(정규화.normalize(o), dtype=torch.float32))
                a = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
            o, r, term, trunc, _ = env.step(a)
            ep_r += r
            ep_s += 1
            done = term or trunc
        scores.append(ep_r)
        steps.append(ep_s)
    env.close()

    평균보상 = float(np.mean(scores))
    평균생존 = float(np.mean(steps))
    최대생존 = float(np.max(steps))

    # 파일 복사
    shutil.copy(model_file, f'{폴더}/model.pt')
    if os.path.exists('humanoid_gen100_step.mp4'):
        shutil.copy('humanoid_gen100_step.mp4', f'{폴더}/replay.mp4')

    설정 = {
        'algo': 'PPO',
        'env': 'Humanoid-v5 (MuJoCo)',
        'stage': 'v2 (Continual Walking & Stepping)',
        'total_steps': 400_000,
        'envs': 8,
        'horizon': 256,
        'clip': 0.2,
        'lr': 2.5e-4,
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
        'version': 'v2',
        'config': 설정,
        'test': {
            'mean_reward_10': round(평균보상, 1),
            'avg_survival_steps': round(평균생존, 1),
            'max_survival_steps': round(최대생존, 1)
        },
        'course': 'AI CITY BUILDERS 강화학습 AI 7강 (MuJoCo 휴머노이드 v2)',
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
      value: {평균보상:.1f}
    - name: Average Survival Steps (평균 생존 스텝)
      type: survival_steps
      value: {평균생존:.1f} 스텝
    - name: Max Survival Steps (최대 생존 기록)
      type: max_survival_steps
      value: {최대생존:.1f} 스텝
---
# {저장소}

> ### 🏅 v2 공식 성적표 (Official Evaluation Score)
> * 🎯 **평균 보상 점수 (Score)**: **`{평균보상:.1f}점`**
> * ⏱️ **평균 생존 시간**: **`{평균생존:.1f}스텝` (최대 {최대생존:.1f}스텝 돌파)**
> * 🤖 **보행 진화 상태**: **2단계 전진 보행 진입 (100세대 두뇌 전이 학습 기반)**

1단계(v1)에서 배운 균형 감각을 그대로 이어받아, 다리를 교차로 움직이며 앞으로 전진하는 법을 집중 훈련한 **Humanoid-v5 심화 보행 v2 모델**입니다.

## 📊 v1 vs v2 진화 비교

| 버전 | 학습 걸음 | 평균 보상 | 평균 생존 시간 | 보행 특징 |
|:---:|:---:|:---:|:---:|---|
| **v1 모델** | 10만 걸음 | 435.0점 | 94.7스텝 | 1단계: 제자리 서기 및 균형 잡기 |
| **v2 모델** | 누적 40만 걸음 | **{평균보상:.1f}점** | **{평균생존:.1f}스텝** | **2단계: 앞으로 무게중심 이동 & 전진 보행** |

AI CITY BUILDERS 강화학습 AI 7강 · CONNECT AI LAB
"""
    open(f'{폴더}/README.md', 'w', encoding='utf-8').write(카드)

    api.create_repo(저장소, exist_ok=True)
    api.upload_folder(folder_path=폴더, repo_id=올린주소)
    print(f"🎉 [Humanoid v2 업로드 성공] -> https://huggingface.co/{올린주소}")

if __name__ == '__main__':
    upload_v2()

