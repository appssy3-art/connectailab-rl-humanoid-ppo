import os, sys, math, time, json, random, re, datetime, shutil, collections
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
import imageio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from huggingface_hub import HfApi

from 로드스터_시뮬레이터 import 로드스터, 로드스터연속

# 1. 폰트 설정 (Windows 한글 폰트 우선)
폰트후보 = [
    'C:/Windows/Fonts/malgun.ttf',
    'C:/Windows/Fonts/malgunbd.ttf',
    '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf'
]
글꼴경로 = next((p for p in 폰트후보 if os.path.exists(p)), None)
if 글꼴경로:
    matplotlib.font_manager.fontManager.addfont(글꼴경로)
    plt.rcParams['font.family'] = matplotlib.font_manager.FontProperties(fname=글꼴경로).get_name()
plt.rcParams['axes.unicode_minus'] = False

def 글꼴(n):
    try:
        return ImageFont.truetype(글꼴경로, n) if 글꼴경로 else ImageFont.load_default()
    except Exception:
        return ImageFont.load_default()

torch.set_num_threads(1)

# 2. 정책두뇌 모델 정의
class 정책두뇌(nn.Module):
    def __init__(self, 상태수, 행동수, 연속=False, 은닉=64):
        super().__init__()
        self.연속 = 연속
        몸 = lambda 출력: nn.Sequential(nn.Linear(상태수, 은닉), nn.Tanh(), nn.Linear(은닉, 은닉), nn.Tanh(), nn.Linear(은닉, 출력))
        self.정책 = 몸(행동수)
        self.가치 = 몸(1)
        if 연속:
            self.흔들림 = nn.Parameter(torch.full((행동수,), -0.5))
        with torch.no_grad():
            self.정책[-1].weight.mul_(0.01)
            self.정책[-1].bias.zero_()

    def 분포(self, s):
        out = self.정책(s)
        if self.연속:
            return torch.distributions.Normal(out, self.흔들림.exp().expand_as(out))
        return torch.distributions.Categorical(logits=out)

    def 고르기(self, s):
        d = self.분포(s)
        a = d.sample()
        lp = d.log_prob(a).sum(-1) if self.연속 else d.log_prob(a)
        return a, lp, self.가치(s).squeeze(-1)

# 3. 착지율 평가 함수
def 착지율(뇌, 세상만들기, 판=50, 시작=5000, 확정=True):
    e = 세상만들기()
    ok = 0
    for 씨 in range(시작, 시작 + 판):
        o, _ = e.reset(seed=씨)
        끝 = 잘림 = False
        while not (끝 or 잘림):
            with torch.no_grad():
                d = 뇌.분포(torch.as_tensor(o, dtype=torch.float32))
                if 뇌.연속:
                    a = np.clip((d.mean if 확정 else d.sample()).numpy(), e.action_space.low, e.action_space.high)
                else:
                    a = int(d.probs.argmax()) if 확정 else int(d.sample())
            o, r, 끝, 잘림, 정보 = e.step(a)
        ok += (정보.get('결과') == '착지')
    return (ok / 판) * 100

# 4. PPO 학습 루프
def PPO학습(세상만들기, 총걸음, 세상수=10, 한번에=256, 감마=0.99, 람다=0.95, 자르기=0.1, 학습률=3e-4,
           반복=10, 묶음=256, 탐험보너스=0.01, 씨앗=0, 보고=20_000, 평가=None, 은닉=64):
    torch.manual_seed(씨앗)
    np.random.seed(씨앗)
    세상들 = [세상만들기() for _ in range(세상수)]
    연속 = hasattr(세상들[0].action_space, 'shape') and len(세상들[0].action_space.shape or ()) > 0
    상태수 = 세상들[0].observation_space.shape[0]
    행동수 = 세상들[0].action_space.shape[0] if 연속 else 세상들[0].action_space.n
    뇌 = 정책두뇌(상태수, 행동수, 연속, 은닉)
    opt = torch.optim.Adam(뇌.parameters(), lr=학습률, eps=1e-5)
    s = np.stack([e.reset(seed=씨앗 * 1000 + i)[0] for i, e in enumerate(세상들)])
    판번호 = 세상수
    판점수, 지금점수, 결과기록, 평가기록 = [], np.zeros(세상수), [], []
    진행기록 = [] # 시각화용 기록 (걸음, 평균보상, 착지율)
    걸음 = 0
    t0 = time.time()
    다음보고 = 보고
    최고 = (-1, None)

    print(f"▶ PPO 학습 시작: 총 {총걸음:,} 걸음 | 세상수 {세상수} | 한번에 {한번에} | 자르기 {자르기} | 학습률 {학습률} | 탐험보너스 {탐험보너스}")

    while 걸음 < 총걸음:
        S, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(한번에):
            st = torch.as_tensor(s, dtype=torch.float32)
            with torch.no_grad():
                a, lp, v = 뇌.고르기(st)
            an = a.numpy()
            s2 = np.empty_like(s)
            r = np.zeros(세상수, np.float32)
            d = np.zeros(세상수, np.float32)
            for i, e in enumerate(세상들):
                act = np.clip(an[i], e.action_space.low, e.action_space.high) if 연속 else int(an[i])
                o, ri, 끝, 잘림, 정보 = e.step(act)
                지금점수[i] += ri
                if 끝 or 잘림:
                    판점수.append(지금점수[i])
                    지금점수[i] = 0
                    결과기록.append(정보.get('결과', ''))
                    o, _ = e.reset(seed=씨앗 * 1000 + 판번호)
                    판번호 += 1
                    d[i] = 1.0
                s2[i] = o
                r[i] = ri
            S.append(st)
            A.append(a)
            LP.append(lp)
            R.append(torch.as_tensor(r))
            D.append(torch.as_tensor(d))
            V.append(v)
            s = s2
            걸음 += 세상수

        with torch.no_grad():
            마지막가치 = 뇌.가치(torch.as_tensor(s, dtype=torch.float32)).squeeze(-1)

        어드 = torch.zeros(한번에, 세상수)
        누적 = torch.zeros(세상수)
        for t in reversed(range(한번에)):
            다음가치 = 마지막가치 if t == 한번에 - 1 else V[t + 1]
            델타 = R[t] + 감마 * 다음가치 * (1 - D[t]) - V[t]
            누적 = 델타 + 감마 * 람다 * (1 - D[t]) * 누적
            어드[t] = 누적

        목표가치 = 어드 + torch.stack(V)
        S_ = torch.cat(S)
        A_ = torch.cat(A)
        LP_ = torch.cat(LP)
        어드_ = 어드.reshape(-1)
        목표_ = 목표가치.reshape(-1)

        n = S_.shape[0]
        for _ in range(반복):
            순서 = torch.randperm(n)
            for k in range(0, n, 묶음):
                i = 순서[k:k + 묶음]
                d_ = 뇌.분포(S_[i])
                새lp = d_.log_prob(A_[i]).sum(-1) if 연속 else d_.log_prob(A_[i])
                비율 = (새lp - LP_[i]).exp()
                a_ = 어드_[i]
                a_ = (a_ - a_.mean()) / (a_.std() + 1e-8)
                정책손실 = -torch.min(비율 * a_, 비율.clamp(1 - 자르기, 1 + 자르기) * a_).mean()
                가치손실 = ((뇌.가치(S_[i]).squeeze(-1) - 목표_[i]) ** 2).mean()
                엔트 = d_.entropy().sum(-1).mean() if 연속 else d_.entropy().mean()
                loss = 정책손실 + 0.5 * 가치손실 - 탐험보너스 * 엔트
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(뇌.parameters(), 0.5)
                opt.step()

        최근_평균 = float(np.mean(판점수[-50:])) if len(판점수) >= 1 else 0.0
        최근_착지 = float(np.mean([x == '착지' for x in 결과기록[-50:]])) * 100 if len(결과기록) >= 1 else 0.0
        진행기록.append((걸음, 최근_평균, 최근_착지))

        if 걸음 >= 다음보고:
            다음보고 += 보고
            줄 = f"[{걸음:>7,}걸음 | {time.time()-t0:4.0f}초] 최근 50판 점수: {최근_평균:6.1f} · 착지율: {최근_착지:3.0f}%"
            if 평가:
                점수 = 평가(뇌)
                평가기록.append((걸음, 점수))
                줄 += f" | 시험 착지율 {점수:4.0f}%"
                if 점수 >= 최고[0]:
                    최고 = (점수, {k: v.clone() for k, v in 뇌.state_dict().items()})
                    줄 += "  🏅"
            print(줄, flush=True)

    return 뇌, 판점수, 진행기록, 최고

# 5. 시뮬레이션 프레임 렌더링
def 로드스터그림(e, a=None, 제목='', 색=(0, 255, 65), 세기=None):
    W, H, sc = 480, 360, 26
    ox, gy = W / 2, H - 40
    im = Image.new('RGB', (W, H), (7, 8, 10))
    d = ImageDraw.Draw(im)
    d.rectangle([0, gy, W, H], fill=(20, 23, 28))
    d.rectangle([ox - 1.5 * sc, gy - 3, ox + 1.5 * sc, gy + 3], fill=(255, 181, 71))
    x, y, vx, vy, th, w = e.s
    cx, cy = ox + x * sc, gy - y * sc
    def rot(px, py):
        c, s = math.cos(th), math.sin(th)
        return (cx + px * c + py * s, cy - (-px * s + py * c))
    d.polygon([rot(-16, -5), rot(16, -5), rot(16, 3), rot(6, 9), rot(-8, 9), rot(-16, 3)], fill=(225, 40, 50))
    d.polygon([rot(-5, 8), rot(5, 8), rot(2, 12), rot(-3, 12)], fill=(191, 239, 255))
    for wx in (-10, 10):
        p = rot(wx, -7)
        d.ellipse([p[0] - 5, p[1] - 5, p[0] + 5, p[1] + 5], fill=(30, 30, 30), outline=(160, 160, 160))
    아래 = 세기[0] if 세기 is not None else (1.0 if a == 2 else 0.0)
    회전 = 세기[1] if 세기 is not None else (-1.0 if a == 1 else 1.0 if a == 3 else 0.0)
    if 아래 > 0.05:
        d.polygon([rot(-5, -9), rot(5, -9), rot(0, -9 - 20 * 아래)], fill=(255, 181, 71))
    if 회전 < -0.05:
        d.polygon([rot(16, 2), rot(16, -3), rot(16 + 14 * -회전, 0)], fill=(191, 239, 255))
    if 회전 > 0.05:
        d.polygon([rot(-16, 2), rot(-16, -3), rot(-16 - 14 * 회전, 0)], fill=(191, 239, 255))
    thn = math.degrees((th + math.pi) % (2 * math.pi) - math.pi)
    d.text((12, 10), 제목, fill=색, font=글꼴(18))
    d.text((12, 36), f'높이 {y:4.1f} · 세로속도 {vy:+.2f} · 기울기 {thn:+5.0f}°', fill=(200, 210, 220), font=글꼴(13))
    return np.array(im)

def 영상저장(뇌, 만들기, 씨앗, 제목, 파일):
    e = 만들기()
    o, _ = e.reset(seed=씨앗)
    화면들 = []
    끝 = 잘림 = False
    while not (끝 or 잘림):
        with torch.no_grad():
            d = 뇌.분포(torch.as_tensor(o, dtype=torch.float32))
            if 뇌.연속:
                a = np.clip(d.mean.numpy(), e.action_space.low, e.action_space.high)
                화면들.append(로드스터그림(e, None, 제목, 세기=a))
            else:
                a = int(d.probs.argmax())
                화면들.append(로드스터그림(e, a, 제목))
        o, r, 끝, 잘림, 정보 = e.step(a)
    화면들 += [로드스터그림(e, None, f"{제목} -> {정보.get('결과')}", (0, 255, 65))] * 25
    imageio.mimsave(파일, 화면들, fps=30, macro_block_size=1)
    return 정보.get('결과')

# ==============================================================
# 메인 실행부
# ==============================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=500_000, help='총걸음')
    parser.add_argument('--repo', type=str, default=None, help='허깅페이스 저장소명')
    parser.add_argument('--no_upload', action='store_true', help='업로드 보류')
    args = parser.parse_args()

    총걸음 = args.steps
    자르기 = 0.1
    세상수 = 10
    학습률 = 3e-4
    탐험보너스 = 0.01
    한번에 = 256
    반복 = 10
    씨앗 = 0

    t_start = time.time()
    두뇌, 판점수, 진행기록, 최고 = PPO학습(
        로드스터연속,
        총걸음=총걸음,
        세상수=세상수,
        한번에=한번에,
        자르기=자르기,
        학습률=학습률,
        반복=반복,
        탐험보너스=탐험보너스,
        씨앗=씨앗,
        보고=50_000,
        평가=lambda b: 착지율(b, 로드스터연속, 판=20)
    )

    시간 = time.time() - t_start
    착지점수 = 착지율(두뇌, 로드스터연속, 판=50)
    print(f"\n✨ 학습 완료! 소요 시간: {시간:.1f}초")
    print(f"📊 처음 보는 50판 최종 착지율: {착지점수:.1f}%")

    # 1) 과정 시각화 플롯 저장
    fig, ax1 = plt.subplots(figsize=(9, 4))
    걸음들, 점수들, 착지율들 = zip(*진행기록) if 진행기록 else ([], [], [])
    
    ax1.set_xlabel('걸음 (Steps)')
    ax1.set_ylabel('최근 50판 점수', color='#00c832')
    ax1.plot(걸음들, 점수들, color='#00c832', lw=2, label='최근 50판 점수')
    ax1.tick_params(axis='y', labelcolor='#00c832')
    ax1.axhline(100, ls='--', color='orange', alpha=0.7, label='착지 기준 (+100)')
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('착지율 (%)', color='#007acc')
    ax2.plot(걸음들, 착지율들, color='#007acc', lw=2, ls='-.', label='착지율 (%)')
    ax2.tick_params(axis='y', labelcolor='#007acc')
    ax2.set_ylim(-5, 105)
    
    plt.title(f'PPO 로드스터 학습 과정 (총 {총걸음:,}걸음, 착지율 {착지점수:.0f}%)')
    fig.tight_layout()
    plt.savefig('learning_curve.png', dpi=150)
    plt.close()
    print("📈 학습 과정 시각화 그래프 저장 완료: learning_curve.png")

    # 2) 주행 영상 MP4 저장
    결과 = 영상저장(두뇌, 로드스터연속, 7, 'PPO · 로드스터 연속 착지', 'roadster_ppo_joystick.mp4')
    print(f"🎬 착지 시연 영상 저장 완료: roadster_ppo_joystick.mp4 (결과: {결과})")

    # 3) 모델 로컬 저장
    torch.save(두뇌.state_dict(), 'model.pt')
    print("💾 로컬 모델 저장 완료: model.pt")

    if args.no_upload:
        print("⏸️ 사용자의 파일명 입력을 기다리기 위해 허깅페이스 업로드를 대기합니다.")
        sys.exit(0)

    # 4) 허깅페이스 업로드
    print("\n🤗 허깅페이스 업로드 시작...")
    api = HfApi()
    나 = api.whoami().get('name')
    if not 나:
        print("⚠️ 허깅페이스 로그인 정보가 없어 업로드를 건너뜁니다.")
        sys.exit(0)

    저장소 = args.repo if args.repo else f'connectailab-rl-{나}-v1-roadster-ppo'
    올린주소 = f'{나}/{저장소}'
    폴더 = f'hf_{저장소}'
    os.makedirs(폴더, exist_ok=True)

    torch.save(두뇌.state_dict(), f'{폴더}/model.pt')
    shutil.copy('로드스터_시뮬레이터.py', f'{폴더}/로드스터_시뮬레이터.py')
    if os.path.exists('roadster_ppo_joystick.mp4'):
        shutil.copy('roadster_ppo_joystick.mp4', f'{폴더}/replay.mp4')
    if os.path.exists('learning_curve.png'):
        shutil.copy('learning_curve.png', f'{폴더}/learning_curve.png')

    설정 = {
        'algo': 'PPO',
        'steps': 총걸음,
        'envs': 세상수,
        'horizon': 한번에,
        'clip': 자르기,
        'lr': 학습률,
        'entropy_coef': 탐험보너스,
        'epochs': 반복,
        'gamma': 0.99,
        'gae_lambda': 0.95
    }

    json.dump({
        'format': 'connectailab/1',
        'repo': 올린주소,
        'kind': 'ppo',
        'env': 'Inverted Roadster (continuous)',
        'algo': 'ppo',
        'config': 설정,
        'test': {'landing_rate_50': 착지점수},
        'course': 'AI CITY BUILDERS 강화학습 AI 6강',
        'date': str(datetime.date.today())
    }, open(f'{폴더}/connectailab.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    카드 = f"""---
tags: [reinforcement-learning, ppo, continuous-control, connect-ai-lab, connectailab]
library_name: pytorch
---
# {저장소}

뒤집힌 채로 떨어지는 로드스터를 추진기 **세기(0 ~ 1)와 회전(−1 ~ 1)** 으로 제어하여 바퀴부터 안전하게 착지시키는 **PPO(Proximal Policy Optimization)** 에이전트입니다.

![학습 과정 그래프](learning_curve.png)

## 📊 학습 결과 및 하이퍼파라미터

| 항목 | 내용 |
|---|---|
| 환경 | 뒤집힌 로드스터 연속 제어 (`로드스터연속`, Gymnasium) |
| 상태 차원 | 위치 x·y, 속도 x·y, 기울기 sin·cos, 회전 각속도 (총 7차원) |
| 행동 차원 | [주 추진기 세기 (0~1), 회전 추진기 세기 (-1~1)] (연속 2차원) |
| **최종 평가 착지율** | **처음 보는 50판 착지율 {착지점수:.0f}%** |
| 클리핑 (자르기) | {자르기} (±{int(자르기*100)}%) |
| 병렬 세상수 | {세상수}개 |
| 총 학습 걸음수 | {총걸음:,} 걸음 |
| 롤아웃 길이 (한번에) | {한번에} |
| 학습률 (Learning Rate) | {학습률} |
| 탐험 보너스 (Entropy) | {탐험보너스} |
| PPO 업데이트 반복 에포크 | {반복}회 |

AI CITY BUILDERS 강화학습 AI 6강 · CONNECT AI LAB
"""
    open(f'{폴더}/README.md', 'w', encoding='utf-8').write(카드)

    try:
        api.create_repo(저장소, exist_ok=True)
        api.upload_folder(folder_path=폴더, repo_id=올린주소)
        print(f"🎉 성공적으로 업로드되었습니다 -> https://huggingface.co/{올린주소}")
        print(f"   (50판 시험 착지율: {착지점수:.0f}%)")
    except Exception as ex:
        print(f"❌ 업로드 중 오류 발생: {ex}")

