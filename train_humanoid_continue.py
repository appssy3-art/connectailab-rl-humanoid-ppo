# -*- coding: utf-8 -*-
"""
휴머노이드 PPO 이어 학습하기 (Continual Training)
- 10만 걸음 모델(humanoid_ppo_gui.pt)을 불러와서 시작
- 이미 서 있는 법을 알기 때문에 앞으로 전진/달리기에 집중 학습!
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

def 이어_학습하기(기존체크포인트='humanoid_ppo_gui.pt', 추가걸음=300_000, 세상수=8, 한번에=256, 자르기=0.2, 학습률=2.5e-4):
    print("=" * 65)
    print(f"🚀 [Humanoid-v5] 100세대 두뇌({기존체크포인트})를 장착하고 심화 학습 시작!")
    print(f"  • 목표: 추가 {추가걸음:,} 걸음 진행 (앞으로 걷기 & 달리기 모드)")
    print("=" * 65)

    세상들 = [gym.make('Humanoid-v5') for _ in range(세상수)]
    상태수 = 세상들[0].observation_space.shape[0]
    행동수 = 세상들[0].action_space.shape[0]

    정규화 = 상태정규화((상태수,))
    뇌 = 휴머노이드두뇌(상태수, 행동수, 은닉=256)

    # 이전 10만 걸음 가중치 및 정규화 통계 로드
    if os.path.exists(기존체크포인트):
        ckpt = torch.load(기존체크포인트, map_location='cpu', weights_only=False)
        뇌.load_state_dict(ckpt['brain'])
        정규화.mean = ckpt['obs_mean']
        정규화.var = ckpt['obs_var']
        print(f"✅ 이전 100세대 두뇌 가중치 로드 성공! (기초 균형 잡기 탑재 완료)")
    else:
        print("⚠️ 이전 체크포인트가 없어 처음부터 시작합니다.")

    opt = torch.optim.Adam(뇌.parameters(), lr=학습률, eps=1e-5)
    s = np.stack([e.reset(seed=2000 + i)[0] for i, e in enumerate(세상들)])
    
    판점수 = []
    지금점수 = np.zeros(세상수)
    생존스텝 = np.zeros(세상수)
    진행기록 = []
    걸음 = 0
    t0 = time.time()
    다음보고 = 30_000

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
                    d[i] = 1.0

                s2[i] = o
                # 전진 인센티브 강화를 위한 보상 스케일링
                r[i] = ri * 0.1

            정규화.update(s2)
            S.append(st)
            A.append(a)
            LP.append(lp)
            R.append(torch.as_tensor(r))
            D.append(torch.as_tensor(d))
            V.append(v)
            s = s2
            걸음 += 세상수

        # GAE
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

        # PPO 업데이트
        n = S_.shape[0]
        for _ in range(10):
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
            다음보고 += 30_000
            print(f"🔥 [{걸음:>7,}스텝 | {time.time()-t0:3.0f}초] 최근 50판 평균 보상: {평균점수:7.1f}점", flush=True)

    for e in 세상들:
        e.close()

    # 가중치 저장
    torch.save({
        'brain': 뇌.state_dict(),
        'obs_mean': 정규화.mean,
        'obs_var': 정규화.var
    }, 'humanoid_ppo_gen500.pt')
    print("\n💾 [500세대 질주 두뇌 저장 완료]: humanoid_ppo_gen500.pt")

    return 뇌, 정규화

if __name__ == '__main__':
    이어_학습하기(추가걸음=300_000, 세상수=8)

