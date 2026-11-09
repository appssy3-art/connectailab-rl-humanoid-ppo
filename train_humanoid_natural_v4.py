# -*- coding: utf-8 -*-
"""
휴머노이드 PPO 4단계: 진짜 사람처럼 걷는 '자연스러운 보행 특화' (v4 Natural Human Walker)
- 100만 걸음 모델(humanoid_ppo_gen1000.pt)의 전신 제어 능력을 승계
- 맞춤 보상 엔지니어링:
  1) 전진 가중치 4배 상향 (forward_reward_weight=4.0)
  2) 제자리 버티기 날먹 차단 (healthy_reward=1.5)
  3) 모터 관절 페널티 80% 완화 (ctrl_cost_weight=0.02) -> 보폭 확장
  4) 타겟 1.5 m/s 리듬 보상 + 좌우 요동 억제 페널티
"""
import os, sys, math, time
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train_humanoid_gui import 휴머노이드두뇌, 상태정규화
from natural_walking_env import NaturalWalkingWrapper

torch.set_num_threads(1)

def 학습_v4(기존체크포인트='humanoid_ppo_gen1000.pt', 추가걸음=500_000, 세상수=8, 한번에=256, 자르기=0.2, 학습률=1.8e-4):
    print("=" * 65)
    print(f"🚶‍♂️ [Humanoid-v5] v4 '자연스러운 인간 보행' 특화 프로젝트 가동!")
    print(f"  • 베이스 두뇌: {기존체크포인트} (누적 1,000,000 스텝 승계)")
    print(f"  • 핵심 변화: 타겟 1.5m/s 리듬 보상 + 보폭 4배 확장 유도")
    print("=" * 65)

    def 환경생성():
        base = gym.make('Humanoid-v5', forward_reward_weight=4.0, healthy_reward=1.5, ctrl_cost_weight=0.02)
        return NaturalWalkingWrapper(base, target_speed=1.5)

    세상들 = [환경생성() for _ in range(세상수)]
    상태수 = 세상들[0].observation_space.shape[0]
    행동수 = 세상들[0].action_space.shape[0]

    정규화 = 상태정규화((상태수,))
    뇌 = 휴머노이드두뇌(상태수, 행동수, 은닉=256)

    if os.path.exists(기존체크포인트):
        ckpt = torch.load(기존체크포인트, map_location='cpu', weights_only=False)
        뇌.load_state_dict(ckpt['brain'])
        정규화.mean = ckpt['obs_mean']
        정규화.var = ckpt['obs_var']
        print(f"✅ 이전 100만 걸음 두뇌 로드 완료! (전신 균형 감각 유지)")
    else:
        print(f"⚠️ {기존체크포인트} 파일이 없습니다.")
        return

    opt = torch.optim.Adam(뇌.parameters(), lr=학습률, eps=1e-5)
    s = np.stack([e.reset(seed=4000 + i)[0] for i, e in enumerate(세상들)])
    
    판점수 = []
    전진속도들 = []
    지금점수 = np.zeros(세상수)
    지금속도 = [[] for _ in range(세상수)]
    진행기록 = []
    걸음 = 0
    t0 = time.time()
    다음보고 = 40_000

    while 걸음 < 추가걸음:
        S, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(한번에):
            s_norm = 정규화.normalize(s)
            st = torch.as_tensor(s_norm, dtype=torch.float32)
            with torch.no_grad():
                a, lp, v = 뇌.고르기(st)
            an = a.numpy()
            s2 = np.empty_like(s)
            r = np.zeros(세상수, np.float32)
            d = np.zeros(세상수, np.float32)

            for i, e in enumerate(세상들):
                act = np.clip(an[i], e.action_space.low, e.action_space.high)
                o, ri, term, trunc, info = e.step(act)
                지금점수[i] += ri
                지금속도[i].append(info.get('x_velocity', 0.0))
                
                if term or trunc:
                    판점수.append(지금점수[i])
                    if len(지금속도[i]) > 0:
                        전진속도들.append(np.mean(지금속도[i]))
                    지금점수[i] = 0.0
                    지금속도[i] = []
                    o, _ = e.reset()
                s2[i] = o
                r[i] = ri * 0.1 # Critic 스케일링
                d[i] = float(term or trunc)

            정규화.update(s2)
            S.append(st)
            A.append(a)
            LP.append(lp)
            R.append(torch.as_tensor(r))
            D.append(torch.as_tensor(d))
            V.append(v)
            s = s2
            걸음 += 세상수

        # GAE Advantage
        with torch.no_grad():
            마지막가치 = 뇌.가치(torch.as_tensor(정규화.normalize(s), dtype=torch.float32)).squeeze(-1)

        어드 = torch.zeros(한번에, 세상수)
        누적 = torch.zeros(세상수)
        for t in reversed(range(한번에)):
            다음가치 = 마지막가치 if t == 한번에 - 1 else V[t + 1]
            델타 = R[t] + 0.99 * 다음가치 * (1 - D[t]) - V[t]
            누적 = 델타 + 0.99 * 0.95 * (1 - D[t]) * 누적
            어드[t] = 누적

        목표가치 = 어드 + torch.stack(V)
        S_ = torch.cat(S)
        A_ = torch.cat(A)
        LP_ = torch.cat(LP)
        어드_ = 어드.reshape(-1)
        목표_ = 목표가치.reshape(-1)

        어드_ = (어드_ - 어드_.mean()) / (어드_.std() + 1e-8)

        # PPO 8 Epochs Update
        n = S_.shape[0]
        for _ in range(8):
            순서 = torch.randperm(n)
            for k in range(0, n, 256):
                i = 순서[k:k + 256]
                d_ = 뇌.분포(S_[i])
                새lp = d_.log_prob(A_[i]).sum(-1)
                비율 = (새lp - LP_[i]).exp()
                a_ = 어드_[i]
                정책손실 = -torch.min(비율 * a_, 비율.clamp(1 - 자르기, 1 + 자르기) * a_).mean()
                가치손실 = ((뇌.가치(S_[i]).squeeze(-1) - 목표_[i]) ** 2).mean()
                엔트 = d_.entropy().sum(-1).mean()
                loss = 정책손실 + 0.5 * 가치손실 - 0.001 * 엔트

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(뇌.parameters(), 0.5)
                opt.step()

        평균점수 = np.mean(판점수[-50:]) if len(판점수) >= 1 else 0.0
        평균속도 = np.mean(전진속도들[-50:]) if len(전진속도들) >= 1 else 0.0
        진행기록.append((걸음, 평균점수, 평균속도))

        if 걸음 >= 다음보고:
            다음보고 += 40_000
            el = time.time() - t0
            print(f"🔥 [{걸음:>7,}스텝 | {int(el):3d}초] 점수: {평균점수:7.1f}점 | 평균 전진 속도: {평균속도:4.2f} m/s", flush=True)

    for e in 세상들:
        e.close()

    최종파일 = 'humanoid_ppo_natural_v4.pt'
    torch.save({
        'brain': 뇌.state_dict(),
        'obs_mean': 정규화.mean,
        'obs_var': 정규화.var,
        'stage': 'v4_natural_human_walker',
        'target_speed': 1.5
    }, 최종파일)
    print(f"\n🏆 [v4 자연스러운 보행 두뇌 완성!] 저장 완료: {최종파일}")

    # 시각화 저장
    if 진행기록:
        xs, ys, vs = zip(*진행기록)
        fig, ax1 = plt.subplots(figsize=(9, 4.5))
        
        ax1.set_xlabel('Steps Trained (k)', fontsize=10)
        ax1.set_ylabel('Mean Reward', color='#1976d2', fontsize=10)
        line1 = ax1.plot([x//1000 for x in xs], ys, color='#1976d2', lw=2.2, label='Custom Reward')
        ax1.tick_params(axis='y', labelcolor='#1976d2')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Forward Speed (m/s)', color='#e64a19', fontsize=10)
        line2 = ax2.plot([x//1000 for x in xs], vs, '--', color='#e64a19', lw=2.0, label='Forward Speed (m/s)')
        ax2.axhline(1.5, color='#4caf50', linestyle=':', label='Target 1.5 m/s')
        ax2.tick_params(axis='y', labelcolor='#e64a19')
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        plt.title('Humanoid-v4: Natural Human Walker Optimization', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig('humanoid_v4_curve.png', dpi=120)
        plt.close()
        print("📈 v4 보행 속도 & 보상 곡선 저장 완료: humanoid_v4_curve.png")

if __name__ == '__main__':
    학습_v4()
