"""
LINK-01 2족 보행 로봇 PPO 강화학습 훈련 파이프라인
ConnectAI LAB · 강화학습 7강 실습 고도화 모델
"""
import os
import sys
import time
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

def make_env(env_id, seed=0):
    def _init():
        e = gym.make(env_id)
        e.reset(seed=seed)
        return e
    return _init

def train(
    env_id='Link01WalkGuided-v0',
    total_timesteps=1_000_000,
    n_envs=8,
    save_dir='runs/my_walk',
    eval_dir='runs/my_walk_eval',
    lr=3e-4,
    n_steps=1024,
    batch_size=64,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.005,
):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(eval_dir, exist_ok=True)
    
    print("=" * 65)
    print(f"🤖 [LINK-01 PPO 훈련 시작]")
    print(f"  • 환경: {env_id}")
    print(f"  • 총 학습 스텝: {total_timesteps:,} steps")
    print(f"  • 병렬 환경 수: {n_envs} envs (SubprocVecEnv)")
    print(f"  • Actor 신경망: [51 -> 64 -> 64 -> 14] (웹 플레이그라운드 호환)")
    print(f"  • Critic 신경망: [51 -> 256 -> 256 -> 1]")
    print(f"  • 저장 위치: {save_dir}")
    print("=" * 65)
    
    # 1. 병렬 벡터 환경 생성 (관측치 51차원 정규화)
    if n_envs > 1:
        venv = SubprocVecEnv([make_env(env_id, seed=100 + i) for i in range(n_envs)])
    else:
        venv = DummyVecEnv([make_env(env_id, seed=100)])
    
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # 2. 신경망 아키텍처 (Actor: 64x64 Tanh -> EZERAI 웹 규격)
    policy_kwargs = dict(
        net_arch=dict(pi=[64, 64], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )
    
    # 3. PPO 에이전트 초기화
    model = PPO(
        'MlpPolicy',
        venv,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=clip_range,
        ent_coef=ent_coef,
        verbose=1,
    )
    
    # 4. 체크포인트 콜백
    checkpoint_callback = CheckpointCallback(
        save_freq=max(10_000 // n_envs, 1),
        save_path=os.path.join(save_dir, 'ckpt'),
        name_prefix='link01_step',
        save_vecnormalize=True
    )
    
    # 5. 훈련 시작
    start_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)
    except KeyboardInterrupt:
        print("\n⚠️ 사용자에 의해 학습이 중단되었습니다. 현재 모델을 저장합니다...")
    
    elapsed_minutes = round((time.time() - start_time) / 60, 2)
    print(f"\n✅ 훈련 완료! 소요 시간: {elapsed_minutes}분")
    
    # 6. 최종 모델 및 정규화 파라미터 저장
    model_path = os.path.join(save_dir, 'model.zip')
    vec_path = os.path.join(save_dir, 'vecnormalize.pkl')
    actor_path = os.path.join(save_dir, 'actor.json')
    
    model.save(model_path)
    venv.save(vec_path)
    print(f"💾 모델 가중치 저장 완료: {model_path}")
    print(f"💾 관측 정규화 저장 완료: {vec_path}")
    
    # 7. 웹 플레이그라운드 호환 actor.json 내보내기
    export_err = hub_push.export_actor_sb3(model, venv, actor_path)
    if export_err:
        print(f"⚠️ actor.json 내보내기 경고: {export_err}")
    else:
        print(f"🌐 EZERAI 웹 플레이그라운드 호환 두뇌 저장: {actor_path}")
    
    # 8. 평가용 행동 함수 생성
    eval_venv = DummyVecEnv([lambda: gym.make(env_id)])
    eval_norm = VecNormalize.load(vec_path, eval_venv)
    eval_norm.training = False
    
    def action_fn(obs):
        normed = eval_norm.normalize_obs(obs[None])
        action, _ = model.predict(normed, deterministic=True)
        return action[0]
    
    model_files = [model_path, vec_path]
    if os.path.exists(actor_path):
        model_files.append(actor_path)
        
    train_info = dict(
        steps=int(model.num_timesteps),
        minutes=elapsed_minutes,
        lr=lr,
        clip=clip_range,
        env=env_id
    )
    
    # 9. 시험 평가 및 리플레이 영상 생성
    print("\n🎬 [보행 성능 평가 및 리플레이 영상 렌더링 중]...")
    metrics = evaluate(
        action_fn,
        eval_dir,
        env_id=env_id,
        episodes=5,
        seconds=20.0,
        command=0.18,
        label=f"LINK-01 · PPO · {model.num_timesteps:,} steps"
    )
    print("=" * 65)
    print(f"📊 [평가 결과 요약]")
    print(f"  • 전진 거리: {metrics['distance_m_per_episode']} m / 20초")
    print(f"  • 전진 속도: {metrics['speed_mps']} m/s (명령: 0.18 m/s)")
    print(f"  • 분당 넘어짐: {metrics['falls_per_min']} 회/분")
    print(f"  • 리플레이 영상: {os.path.join(eval_dir, 'replay.mp4')}")
    print("=" * 65)
    
    return action_fn, model_files, train_info, metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LINK-01 PPO Training Script")
    parser.add_argument('--steps', type=int, default=1_000_000, help="Total training timesteps")
    parser.add_argument('--envs', type=int, default=8, help="Number of parallel environments")
    parser.add_argument('--env', type=str, default='Link01WalkGuided-v0', help="Environment ID (Link01WalkGuided-v0 or Link01Walk-v0)")
    parser.add_argument('--dir', type=str, default='runs/my_walk', help="Model save directory")
    args = parser.parse_args()
    
    train(
        env_id=args.env,
        total_timesteps=args.steps,
        n_envs=args.envs,
        save_dir=args.dir
    )

