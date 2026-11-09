# -*- coding: utf-8 -*-
"""
휴머노이드 PPO 3단계: 누적 100만 걸음 초고속 질주 모델 (v3 Continual Training)
- 40만 걸음 모델(humanoid_ppo_gen500.pt)을 이어받아 60만 걸음 추가 학습
- 누적 1,000,000 걸음 대기록 달성 및 장거리 전진 질주 완성
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

torch.set_num_threads(1)

def 학습_v3(기존체크포인트='humanoid_ppo_gen500.pt', 추가걸음=600_000, 세상수=8, 한번에=256, 자르기=0.2, 학습률=2.0e-4):
    print("=" * 65)
    print(f"🚀 [Humanoid-v5] v3 100만 걸음 질주 프로젝트 시작!")
    print(f"  • 베이스 두뇌: {기존체크포인트} (누적 400,000 스텝)")
    print(f"  • 추가 학습 목표: {추가걸음:,} 걸음 진행 (총 누적 1,000,000 걸음 돌파)")
    print("=" * 65)

    세상들 = [gym.make('Humanoid-v5') for _ in range(세상수)]
    상태수 = 세상들[0].observation_space.shape[0]
    행동수 = 세상들[0].action_space.shape[0]

    정규화 = 상태정규화((상태수,))
    뇌 = 휴머노이드두뇌(상태수, 행동수, 은닉=256)

    if os.path.exists(기존체크포인트):
        ckpt = torch.load(기존체크포인트, map_location='cpu', weights_only=False)
        뇌.load_state_dict(ckpt['brain'])
        정규화.mean = ckpt['obs_mean']
        정규화.var = ckpt['obs_var']
        print(f"✅ 이전 v2 두뇌 가중치 로드 성공! (기초 전진 보행 탑재 완료)")
    else:
        print(f"⚠️ {기존체크포인트} 파일이 없습니다.")
        return

    opt = torch.optim.Adam(뇌.parameters(), lr=학습률, eps=1e-5)
    s = np.stack([e.reset(seed=3000 + i)[0] for i, e in enumerate(세상들)])
    
    판점수 = []
    지금점수 = np.zeros(세상수)
    생존스텝 = np.zeros(세상수)
    진행기록 = []
    걸음 = 0
    t0 = time.time()
    다음보고 = 50_000

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
                생존스텝[i] += 1
                if term or trunc:
                    판점수.append(지금점수[i])
                    지금점수[i] = 0.0
                    생존스텝[i] = 0
                    o, _ = e.reset()
                s2[i] = o
                r[i] = ri * 0.1 # Critic 안정화를 위한 스케일링
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

        # GAE Advantage 계산
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

        # PPO 업데이트 (8 Epochs)
        n = S_.shape[0]
        for _ in range(8):
            순서 = torch.randperm(n)
            for k in range(0, n, 256):
                i = 순서[k:k + 256]
                d_ = 뇌.분포(S_[i])
                새lp = d_.log_prob(A_[i]).sum(-1)
                비율 = (새lp - LP_[i]).exp()
                a_ = 어드_[i]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                정책손실 = -torch.min(비율 * a_, 비율.clamp(1 - 자르기, 1 + 자르기) * a_).mean()
                가치손실 = ((뇌.가치(S_[i]).squeeze(-1) - 목표_[i]) ** 2).mean()
                엔트 = d_.entropy().sum(-1).mean()
                loss = 정책손실 + 0.5 * 가치손실 - 0.001 * 엔트

                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(뇌.parameters(), 0.5)
                opt.step()

        평균점수 = np.mean(판점수[-50:]) if len(판점수) >= 1 else 0.0
        진행기록.append((걸음, 평균점수))

        if 걸음 >= 다음보고:
            다음보고 += 50_000
            el = time.time() - t0
            print(f"🔥 [누적 {400_000 + 걸음:>8,}스텝 | {int(el):3d}초] 최근 50판 평균 보상: {평균점수:7.1f}점", flush=True)

    for e in 세상들:
        e.close()

    # 100만 걸음 완성 모델 저장
    최종파일 = 'humanoid_ppo_gen1000.pt'
    torch.save({
        'brain': 뇌.state_dict(),
        'obs_mean': 정규화.mean,
        'obs_var': 정규화.var,
        'total_steps': 1_000_000,
        'stage': 'v3_sprint_1M'
    }, 최종파일)
    print(f"\n🏆 [1,000,000 걸음 대기록 달성!] 두뇌 저장 완료: {최종파일}")

    # 학습 곡선 저장
    if 진행기록:
        xs, ys = zip(*진행기록)
        plt.figure(figsize=(9, 4.5))
        plt.plot([400 + x//1000 for x in xs], ys, 'o-', color='#e91e63', lw=2.5, label='Humanoid-v5 v3 (400k -> 1M steps)')
        plt.title('Humanoid-v5 PPO Continuous Control (1,000,000 Total Steps)', fontsize=12, fontweight='bold')
        plt.xlabel('Total Steps (k)', fontsize=10)
        plt.ylabel('Mean Reward', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig('humanoid_v3_curve.png', dpi=120)
        plt.close()
        print("📈 v3 학습 곡선 저장 완료: humanoid_v3_curve.png")

if __name__ == '__main__':
    학습_v3()
