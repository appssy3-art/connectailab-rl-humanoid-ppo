# -*- coding: utf-8 -*-
"""
휴머노이드 3D 실시간 뷰어 (버전별 선택 실행)
- v1 (10만 걸음 제자리 균형)
- v2 (40만 걸음 첫 걸음마)
- v3 (100만 걸음 고속 질주)
- v4 (150만 걸음 자연스러운 1.45m/s 인간 보행)
- v5 (250만 걸음 1.6m/s 마라톤 조거)
"""
import sys
import os
import torch
import numpy as np
import gymnasium as gym

from train_humanoid_gui import 휴머노이드두뇌, 상태정규화

MODELS = {
    '1': ('v1 (10만 걸음 서기)', 'humanoid_ppo_gui.pt'),
    '2': ('v2 (40만 걸음 전진)', 'humanoid_ppo_gen500.pt'),
    '3': ('v3 (100만 걸음 질주)', 'humanoid_ppo_gen1000.pt'),
    '4': ('v4 (150만 보폭 확장)', 'humanoid_ppo_natural_v4.pt'),
    '5': ('v5 (250만 마라톤 조깅)', 'humanoid_ppo_gen2500_v5.pt'),
}

def play(choice='5'):
    if choice not in MODELS:
        choice = '5'
    name, path = MODELS[choice]
    
    if not os.path.exists(path):
        print(f"⚠️ {path} 가중치 파일이 없습니다.")
        return

    print("=" * 60)
    print(f"🎮 [Humanoid-v5] {name} 3D 실시간 뷰어 실행 중...")
    print("  • 조작법:")
    print("    - 마우스 좌클릭 드래그: 360도 카메라 회전")
    print("    - 마우스 우클릭 드래그: 줌 인/아웃")
    print("    - 마우스 휠 클릭 드래그: 카메라 시점 평행 이동")
    print("    - [Space] 일시정지, [Tab] 카메라 변경, [ESC] 종료")
    print("=" * 60)

    env = gym.make("Humanoid-v5", render_mode="human")
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    norm = 상태정규화((obs_dim,))
    norm.mean = ckpt['obs_mean']
    norm.var = ckpt['obs_var']

    brain = 휴머노이드두뇌(obs_dim, act_dim, 은닉=256)
    brain.load_state_dict(ckpt['brain'])
    brain.eval()

    episode = 1
    try:
        while True:
            obs, _ = env.reset(seed=100 + episode)
            done = False
            step = 0
            score = 0.0
            print(f"\n▶️ [에피소드 {episode} 시작]")
            
            while not done:
                with torch.no_grad():
                    dist = brain.분포(torch.as_tensor(norm.normalize(obs), dtype=torch.float32))
                    action = np.clip(dist.mean.numpy(), env.action_space.low, env.action_space.high)
                
                obs, reward, term, trunc, info = env.step(action)
                score += reward
                step += 1
                done = term or trunc
                
            print(f"🏁 [에피소드 {episode} 종료] 생존: {step} 스텝 | 점수: {score:.1f}점")
            episode += 1
    except KeyboardInterrupt:
        print("\n👋 뷰어를 종료합니다.")
    finally:
        env.close()

if __name__ == '__main__':
    if len(sys.argv) > 1:
        play(sys.argv[1])
    else:
        print("버전을 선택하세요:")
        for k, (n, _) in MODELS.items():
            print(f"  [{k}] {n}")
        sel = input("선택 (1~5, 기본 5): ").strip()
        play(sel if sel else '5')
