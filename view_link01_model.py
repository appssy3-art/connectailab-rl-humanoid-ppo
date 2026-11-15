"""
LINK-01 2족 보행 로봇 실시간 3D 뷰어 (MuJoCo)
ConnectAI LAB · 강화학습 7강 실습 모델 관찰기
"""
import os
import sys
import time
import json
import argparse
import numpy as np
import gymnasium as gym
import link01_gym

def load_actor_json_policy(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    weights = [np.array(l['weight'], dtype=np.float64) for l in data['layers']]
    biases = [np.array(l['bias'], dtype=np.float64) for l in data['layers']]
    trained_steps = data.get('trained_steps', 0)

    def policy(obs):
        x = np.asarray(obs, dtype=np.float64)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = w @ x + b
            if i < len(weights) - 1:
                x = np.tanh(x)
        return np.clip(x, -1.0, 1.0)

    return policy, trained_steps

def main():
    parser = argparse.ArgumentParser(description="LINK-01 3D Live Viewer")
    parser.add_argument('--actor', type=str, default='link01-gym-ppo7/my_walk/actor.json', help="Path to actor.json")
    parser.add_argument('--env', type=str, default='Link01Walk-v0', help="Environment ID")
    parser.add_argument('--mode', type=str, default='model', choices=['model', 'cpg', 'zero', 'random'], help="Control policy mode")
    args = parser.parse_args()

    # 경로 보정 (현재 디렉토리 또는 link01-gym-ppo7 하위 탐색)
    candidates = [
        args.actor,
        os.path.join(os.path.dirname(__file__), args.actor),
        'link01-gym-ppo7/my_walk/actor.json',
        'my_walk/actor.json',
        'link01-gym-ppo7/runs/smoke_test/actor.json'
    ]
    actor_file = None
    for c in candidates:
        if os.path.exists(c):
            actor_file = c
            break

    print("=" * 60)
    print("🤖 [LINK-01 2족 보행 로봇 실시간 3D 물리 관찰기]")
    print(f"  • 선택 모드: {args.mode}")
    print(f"  • 시뮬레이터 환경: {args.env}")
    print("=" * 60)

    policy = None
    if args.mode == 'model':
        if not actor_file:
            print(f"⚠️ 학습된 모델({args.actor})을 찾을 수 없습니다.")
            print("👉 CPG 기준 보행(cpg) 모드로 전환하여 시뮬레이터를 실행합니다.\n")
            args.mode = 'cpg'
        else:
            print(f"✅ 학습된 PPO 두뇌 로드 완료: {actor_file}")
            policy, trained_steps = load_actor_json_policy(actor_file)
            print(f"📊 학습 걸음 수: {trained_steps:,} 걸음 (웹 플레이그라운드 호환 신경망)\n")

    # CPG 기준 걸음 패턴 로드
    if args.mode == 'cpg':
        ref_path = os.path.join(os.path.dirname(link01_gym.__file__), 'reference_gait.npz')
        ref_data = np.load(ref_path)
        ref_actions = ref_data['actions']  # (64, 14)
        print(f"✅ CPG 기준 보행(Reference Gait) 패턴 로드 완료: {len(ref_actions)} 프레임 주기\n")

    # 실시간 렌더링 환경 열기
    print("🚀 MuJoCo 3D 물리 시뮬레이션 창을 여는 중...")
    try:
        env = gym.make(args.env, render_mode='human').unwrapped
    except Exception as e:
        print(f"⚠️ 화면 렌더링 초기화 실패: {e}")
        return

    obs, info = env.reset(seed=42)
    step = 0
    falls = 0

    print("🎮 3D 창이 열렸습니다! (종료하려면 터미널에서 Ctrl+C 또는 창 닫기)\n")
    try:
        while True:
            if args.mode == 'model':
                action = policy(obs)
            elif args.mode == 'cpg':
                action = ref_actions[step % len(ref_actions)]
            elif args.mode == 'zero':
                action = np.zeros(14)
            elif args.mode == 'random':
                action = np.random.uniform(-1, 1, 14)

            obs, r, term, trunc, info = env.step(action)
            step += 1

            if term:
                falls += 1
                dist = env.distance()
                print(f"💥 넘어짐 발생! (이동 거리: {dist:.2f}m, 누적 넘어짐: {falls}회) -> 즉시 재기립 리셋...")
                obs, info = env.reset()
                step = 0

    except KeyboardInterrupt:
        print("\n👋 3D 관찰을 종료합니다.")
    finally:
        env.close()

if __name__ == '__main__':
    main()
