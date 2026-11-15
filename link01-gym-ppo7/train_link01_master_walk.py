"""
LINK-01 마스터 보행 PPO 초고속 학습 파이프라인
ConnectAI LAB · 강화학습 7강 실습 고도화 모델
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import gymnasium as gym
import link01_gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

import hub_push
from eval_walk import evaluate

# 튜닝된 보상 가중치 (제자리 서 있기 꼼수 차단 + 전진 & 발걸음 극대화)
TUNED_WEIGHTS = dict(
    track=4.0,       # 목표 속도(0.18 m/s) 추종 보상 2배 강화
    progress=3.0,    # 앞으로 전진할 때마다 들어오는 보상 3배 강화
    alive=0.05,      # 제자리 서 있기 꿀 보상 대폭 삭감! (0.2 -> 0.05)
    air_time=3.0,    # 발을 들어올려 디디는 보행 보너스 3배 강화 (1.0 -> 3.0)
    upright=-2.5,    # 몸체 꼿꼿이 세우기
    side_vy=-1.5,    # 좌우 쏠림 억제
    side_wz=-0.5,    # 몸통 회전 억제
    energy=-3e-4,    # 토크 에너지 페널티
    smooth=-0.01,    # 관절 떨림 억제
)

def make_tuned_env(seed=0, ref_weight=2.0):
    def _init():
        # CPG 보행 리듬 가이드(ref_weight=2.0) + 전진 특화 보상 적용
        e = gym.make('Link01WalkGuided-v0', ref_weight=ref_weight, weights=TUNED_WEIGHTS)
        e.reset(seed=seed)
        return e
    return _init

def train_master(
    total_timesteps=1_000_000,
    n_envs=8,
    save_dir='my_walk',
    eval_dir='runs/my_walk_eval',
    lr=3e-4,
    ref_weight=2.0,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    print("=" * 65)
    print(f"🚀 [LINK-01 마스터 보행 PPO 초고속 학습 시작]")
    print(f"  • 목표 스텝: {total_timesteps:,} steps")
    print(f"  • 병렬 환경: {n_envs}개 SubprocVecEnv")
    print(f"  • CPG 리듬 가이드: {ref_weight}x 강화 (발걸음 모멘텀 유도)")
    print(f"  • 보상 튜닝: 전진(progress: 3.0, track: 4.0), 발디딤(air_time: 3.0), 생존(alive: 0.05)")
    print(f"  • 신경망: Actor [51 -> 64 -> 64 -> 14] Tanh (웹 플레이그라운드 호환)")
    print(f"  • Critic: [51 -> 256 -> 256 -> 1]")
    print("=" * 65)
    
    # 1. 8개 병렬 환경 구성
    venv = SubprocVecEnv([make_tuned_env(seed=100 + i, ref_weight=ref_weight) for i in range(n_envs)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 2. PPO 정책망
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )
    
    model = PPO(
        'MlpPolicy',
        venv,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=1,
    )
    
    # 3. 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=max(25_000 // n_envs, 1),
        save_path=os.path.join(save_dir, 'ckpt'),
        name_prefix='link01_master',
        save_vecnormalize=True
    )
    
    # 4. 학습 시작
    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 학습이 중단되었습니다. 현재까지의 모델을 저장합니다...")
        
    elapsed_minutes = round((time.time() - start_time) / 60, 2)
    print(f"\n🎉 [학습 완료] 소요 시간: {elapsed_minutes}분 (총 {model.num_timesteps:,} 걸음)")
    
    # 5. 가중치 저장
    model_path = os.path.join(save_dir, 'model.zip')
    vec_path = os.path.join(save_dir, 'vecnormalize.pkl')
    actor_path = os.path.join(save_dir, 'actor.json')
    
    model.save(model_path)
    venv.save(vec_path)
    print(f"💾 model.zip 저장: {model_path}")
    print(f"💾 vecnormalize.pkl 저장: {vec_path}")
    
    # 6. actor.json 내보내기 (웹 플레이그라운드 호환)
    hub_push.export_actor_sb3(model, venv, actor_path)
    print(f"🌐 actor.json 저장: {actor_path}")
    
    # 루트에도 동기화 복사
    root_my_walk = 'E:/ai랩/강화학습PPO/link01-gym-ppo7/my_walk'
    os.makedirs(root_my_walk, exist_ok=True)
    import shutil
    shutil.copy(model_path, os.path.join(root_my_walk, 'model.zip'))
    shutil.copy(vec_path, os.path.join(root_my_walk, 'vecnormalize.pkl'))
    shutil.copy(actor_path, os.path.join(root_my_walk, 'actor.json'))
    
    # 7. 평가용 행동 함수 및 리플레이 영상 렌더링
    eval_venv = DummyVecEnv([lambda: gym.make('Link01Walk-v0')])
    eval_norm = VecNormalize.load(vec_path, eval_venv)
    eval_norm.training = False
    
    def action_fn(obs):
        normed = eval_norm.normalize_obs(obs[None])
        act, _ = model.predict(normed, deterministic=True)
        return act[0]
        
    print("\n🎬 [마스터 보행 평가 및 리플레이 영상 렌더링 중]...")
    metrics = evaluate(
        action_fn,
        eval_dir,
        env_id='Link01Walk-v0',
        episodes=5,
        seconds=20.0,
        command=0.18,
        label=f"LINK-01 · PPO Master · {model.num_timesteps:,} steps"
    )
    
    # 루트 영상 복사
    shutil.copy(os.path.join(eval_dir, 'replay.mp4'), 'E:/ai랩/강화학습PPO/runs/my_walk_eval/replay.mp4')
    
    print("=" * 65)
    print(f"🏆 [마스터 보행 최종 평가 결과]")
    print(f"  • 전진 거리: {metrics['distance_m_per_episode']} m / 20초")
    print(f"  • 전진 속도: {metrics['speed_mps']} m/s (명령 속도: 0.18 m/s)")
    print(f"  • 분당 넘어짐: {metrics['falls_per_min']} 회/분")
    print(f"  • 리플레이 영상: {os.path.join(eval_dir, 'replay.mp4')}")
    print("=" * 65)
    
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1_000_000)
    parser.add_argument('--envs', type=int, default=8)
    parser.add_argument('--ref', type=float, default=2.0)
    args = parser.parse_args()
    
    # Working dir을 link01-gym-ppo7로 이동하여 실행
    os.chdir('E:/ai랩/강화학습PPO/link01-gym-ppo7')
    train_master(total_timesteps=args.steps, n_envs=args.envs, ref_weight=args.ref)

