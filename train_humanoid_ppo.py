import os, sys, math, time, json, collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

torch.set_num_threads(1)

# ==============================================================
# 1. 상태 정규화 래퍼 (Running Mean / Std)
# ==============================================================
class 상태정규화:
    def __init__(self, shape, eps=1e-8):
        self.mean = np.zeros(shape, np.float64)
        self.var = np.ones(shape, np.float64)
        self.count = eps
        self.eps = eps

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        self.var = M2 / tot_count
        self.count = tot_count

    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + self.eps), -10.0, 10.0).astype(np.float32)

# ==============================================================
# 2. 휴머노이드 PPO 정책 신경망 (Actor-Critic)
# ==============================================================
def 직교초기화(m, gain=np.sqrt(2)):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=gain)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

class 휴머노이드두뇌(nn.Module):
    def __init__(self, 상태수=348, 행동수=17, 은닉=256):
        super().__init__()
        self.정책 = nn.Sequential(
            nn.Linear(상태수, 은닉), nn.Tanh(),
            nn.Linear(은닉, 은닉), nn.Tanh(),
            nn.Linear(은닉, 행동수)
        )
        self.가치 = nn.Sequential(
            nn.Linear(상태수, 은닉), nn.Tanh(),
            nn.Linear(은닉, 은닉), nn.Tanh(),
            nn.Linear(은닉, 1)
        )
        self.정책.apply(직교초기화)
        직교초기화(self.정책[-1], gain=0.01) # 초기 무작위성 유지
        self.가치.apply(직교초기화)
        직교초기화(self.가치[-1], gain=1.0)
        
        # 17개 관절 모터의 초기 탐험 표준편차 (log_std = -0.5 -> std ≈ 0.6)
        self.흔들림 = nn.Parameter(torch.full((행동수,), -0.5))

    def 분포(self, s):
        mu = self.정책(s)
        std = self.흔들림.exp().expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def 고르기(self, s):
        d = self.분포(s)
        a = d.sample()
        lp = d.log_prob(a).sum(-1)
        return a, lp, self.가치(s).squeeze(-1)

# ==============================================================
# 3. 휴머노이드 보행 평가 함수
# ==============================================================
def 보행평가(뇌, 정규화, 에피소드=10):
    env = gym.make('Humanoid-v5')
    총점수들, 스텝들 = [], []
    for 씨 in range(1000, 1000 + 에피소드):
        o, _ = env.reset(seed=씨)
        에피_점수 = 0.0
        스텝 = 0
        done = False
        while not done:
            o_norm = 정규화.normalize(o)
            with torch.no_grad():
                d = 뇌.분포(torch.as_tensor(o_norm, dtype=torch.float32))
                a = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
            o, r, term, trunc, info = env.step(a)
            에피_점수 += r
            스텝 += 1
            done = term or trunc
        총점수들.append(에피_점수)
        스텝들.append(스텝)
    env.close()
    return float(np.mean(총점수들)), float(np.mean(스텝들))

# ==============================================================
# 4. 3D 렌더링 주행 동영상 저장
# ==============================================================
def 렌더링영상(뇌, 정규화, 파일='humanoid_walk.mp4', 최대스텝=500):
    env = gym.make('Humanoid-v5', render_mode='rgb_array')
    o, _ = env.reset(seed=42)
    화면들 = []
    done = False
    스텝 = 0
    총보상 = 0.0
    while not done and 스텝 < 최대스텝:
        화면 = env.render()
        if 화면 is not None:
            화면들.append(화면)
        o_norm = 정규화.normalize(o)
        with torch.no_grad():
            d = 뇌.분포(torch.as_tensor(o_norm, dtype=torch.float32))
            a = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
        o, r, term, trunc, info = env.step(a)
        총보상 += r
        스텝 += 1
        done = term or trunc
    env.close()
    if 화면들:
        imageio.mimsave(파일, 화면들, fps=30, macro_block_size=1)
        print(f"🎬 3D 렌더링 비디오 저장 완료: {파일} (생존 스텝: {스텝}, 누적 보상: {총보상:.1f})")

# ==============================================================
# 5. 메인 PPO 훈련 루프
# ==============================================================
def 휴머노이드_학습(총걸음=300_000, 세상수=8, 한번에=256, 자르기=0.2, 학습률=3e-4, 반복=10, 묶음=256, 씨앗=0):
    torch.manual_seed(씨앗)
    np.random.seed(씨앗)

    세상들 = [gym.make('Humanoid-v5') for _ in range(세상수)]
    상태수 = 세상들[0].observation_space.shape[0]
    행동수 = 세상들[0].action_space.shape[0]
    
    print(f"🤖 [Humanoid-v5 PPO 훈련 시작]")
    print(f"  • 상태 관측 차원: {상태수} | 행동 모터 차원: {행동수}")
    print(f"  • 총 훈련 걸음: {총걸음:,} | 병렬 환경: {세상수}개 | 클리핑: {자르기} | 학습률: {학습률}")

    정규화 = 상태정규화((상태수,))
    뇌 = 휴머노이드두뇌(상태수, 행동수, 은닉=256)
    opt = torch.optim.Adam(뇌.parameters(), lr=학습률, eps=1e-5)

    s = np.stack([e.reset(seed=씨앗 * 1000 + i)[0] for i, e in enumerate(세상들)])
    정규화.update(s)
    
    판점수, 지금점수 = [], np.zeros(세상수)
    진행기록 = []
    걸음 = 0
    t0 = time.time()
    다음보고 = 50_000

    while 걸음 < 총걸음:
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
                if term or trunc:
                    판점수.append(지금점수[i])
                    지금점수[i] = 0.0
                    o, _ = e.reset()
                    d[i] = 1.0
                s2[i] = o
                # 안정적 Critic 수렴을 위한 보상 스케일링 (Reward Scaling)
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

        # GAE 계산
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

        # PPO 최적화
        n = S_.shape[0]
        for _ in range(반복):
            순서 = torch.randperm(n)
            for k in range(0, n, 묶음):
                i = 순서[k:k + 묶음]
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
            경과시간 = time.time() - t0
            print(f"[{걸음:>8,}걸음 | {경과시간:4.0f}초] 최근 50판 평균 보상: {평균점수:7.1f}", flush=True)

    for e in 세상들:
        e.close()

    return 뇌, 정규화, 진행기록

if __name__ == '__main__':
    총걸음 = 300_000
    뇌, 정규화, 기록 = 휴머노이드_학습(총걸음=총걸음, 세상수=8, 자르기=0.2, 학습률=3e-4)

    # 1) 가중치 저장
    torch.save({
        'brain': 뇌.state_dict(),
        'obs_mean': 정규화.mean,
        'obs_var': 정규화.var
    }, 'humanoid_ppo.pt')
    print("💾 휴머노이드 가중치 및 정규화 통계량 저장 완료: humanoid_ppo.pt")

    # 2) 학습 곡선 저장
    걸음들, 점수들 = zip(*기록) if 기록 else ([], [])
    plt.figure(figsize=(9, 4))
    plt.plot(걸음들, 점수들, color='#e056fd', lw=2)
    plt.xlabel('걸음 (Steps)')
    plt.ylabel('평균 보상')
    plt.title(f'Humanoid-v5 PPO 학습 곡선 ({총걸음:,} 걸음 초벌 학습)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('humanoid_learning_curve.png', dpi=150)
    plt.close()
    print("📈 학습 곡선 저장: humanoid_learning_curve.png")

    # 3) 평가 및 3D 비디오 렌더링
    평균점수, 평균스텝 = 보행평가(뇌, 정규화, 에피소드=10)
    print(f"📊 최종 10판 평가 -> 평균 보상: {평균점수:.1f}점 | 평균 생존: {평균스텝:.1f} 스텝")

    렌더링영상(뇌, 정규화, 'humanoid_walk.mp4', 최대스텝=300)

