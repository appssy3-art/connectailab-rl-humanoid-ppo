import os, sys, math, time, collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
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
        직교초기화(self.정책[-1], gain=0.01)
        self.가치.apply(직교초기화)
        직교초기화(self.가치[-1], gain=1.0)
        
        # 17개 관절 모터 초기 탐험 표준편차
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
# 3. 실시간 GUI 학습 함수 (MuJoCo 3D 창 오픈)
# ==============================================================
def 학습_및_GUI_시각화(총걸음=100_000, 세상수=4, 한번에=128, 자르기=0.2, 학습률=3e-4, 반복=10, 묶음=128, 씨앗=0):
    torch.manual_seed(씨앗)
    np.random.seed(씨앗)

    print("=" * 65)
    print("🖥️ [Humanoid-v5] 실시간 3D GUI 학습 창을 오픈합니다!")
    print("  • 마우스 좌클릭 + 드래그: 카메라 3D 회전")
    print("  • 마우스 우클릭 + 드래그: 화면 줌인 / 줌아웃")
    print("  • 스크롤 휠: 카메라 이동")
    print("=" * 65)

    # 1번째 환경은 render_mode='human'으로 윈도우 창을 띄움
    세상들 = [gym.make('Humanoid-v5', render_mode='human' if i == 0 else None) for i in range(세상수)]
    상태수 = 세상들[0].observation_space.shape[0]
    행동수 = 세상들[0].action_space.shape[0]

    정규화 = 상태정규화((상태수,))
    뇌 = 휴머노이드두뇌(상태수, 행동수, 은닉=256)
    opt = torch.optim.Adam(뇌.parameters(), lr=학습률, eps=1e-5)

    s = np.stack([e.reset(seed=씨앗 * 1000 + i)[0] for i, e in enumerate(세상들)])
    정규화.update(s)

    판점수 = []
    지금점수 = np.zeros(세상수)
    생존스텝 = np.zeros(세상수)
    진행기록 = []
    걸음 = 0
    t0 = time.time()
    다음보고 = 10_000
    에피소드_카운트 = 0

    try:
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
                    생존스텝[i] += 1
                    
                    if term or trunc:
                        if i == 0:
                            에피소드_카운트 += 1
                            print(f"  [화면 휴머노이드] 에피소드 #{에피소드_카운트} 종료 -> 생존: {int(생존스텝[i])}스텝 | 누적 보상: {지금점수[i]:.1f}점", flush=True)
                        판점수.append(지금점수[i])
                        지금점수[i] = 0.0
                        생존스텝[i] = 0
                        o, _ = e.reset()
                        d[i] = 1.0

                    s2[i] = o
                    r[i] = ri * 0.1 # 보상 스케일링

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

            # PPO 에포크 업데이트
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

            평균점수 = np.mean(판점수[-30:]) if len(판점수) >= 1 else 0.0
            진행기록.append((걸음, 평균점수))

            if 걸음 >= 다음보고:
                다음보고 += 10_000
                경과시간 = time.time() - t0
                print(f"📊 [{걸음:>7,}스텝 | {경과시간:4.0f}초] 최근 30판 평균 보상: {평균점수:7.1f}점", flush=True)

    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 학습이 중단되었습니다. 현재까지의 두뇌를 저장합니다.")
    finally:
        for e in 세상들:
            e.close()

    # 결과 저장
    torch.save({
        'brain': 뇌.state_dict(),
        'obs_mean': 정규화.mean,
        'obs_var': 정규화.var
    }, 'humanoid_ppo_gui.pt')
    print("\n💾 휴머노이드 두뇌 저장 완료: humanoid_ppo_gui.pt")

    # 학습 곡선 저장
    if 진행기록:
        걸음들, 점수들 = zip(*진행기록)
        plt.figure(figsize=(9, 4))
        plt.plot(걸음들, 점수들, color='#00d2d3', lw=2)
        plt.xlabel('걸음 (Steps)')
        plt.ylabel('평균 보상')
        plt.title('Humanoid-v5 실시간 GUI 학습 곡선')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('humanoid_gui_curve.png', dpi=150)
        plt.close()
        print("📈 학습 곡선 그래프 저장 완료: humanoid_gui_curve.png")

    return 뇌, 정규화

# ==============================================================
# 4. 학습 완료 후 결정론적 시연 GUI 모드
# ==============================================================
def 학습된두뇌_시연(뇌, 정규화, 에피소드=5):
    print("\n🎬 [시연 모드] 학습된 두뇌로 3D GUI 시연을 시작합니다!")
    env = gym.make('Humanoid-v5', render_mode='human')
    for ep in range(에피소드):
        o, _ = env.reset(seed=100 + ep)
        done = False
        스텝 = 0
        총보상 = 0.0
        while not done:
            o_norm = 정규화.normalize(o)
            with torch.no_grad():
                d = 뇌.분포(torch.as_tensor(o_norm, dtype=torch.float32))
                a = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
            o, r, term, trunc, _ = env.step(a)
            총보상 += r
            스텝 += 1
            done = term or trunc
            time.sleep(0.01) # 사람이 보기 편하도록 60fps 수준 조절
        print(f"  시연 에피소드 #{ep+1} 종료 -> 총 {스텝}스텝 생존, 보상 {총보상:.1f}점")
    env.close()

if __name__ == '__main__':
    총걸음 = 100_000 # GUI로 보면서 실시간 관찰할 기본 10만 걸음
    뇌, 정규화 = 학습_및_GUI_시각화(총걸음=총걸음, 세상수=4, 한번에=128)
    학습된두뇌_시연(뇌, 정규화, 에피소드=3)

