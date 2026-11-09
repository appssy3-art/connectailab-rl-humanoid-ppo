import os, math, numpy as np, torch
import imageio
from PIL import Image, ImageDraw, ImageFont
from 로드스터_시뮬레이터 import 로드스터연속
from run_ppo_task import 정책두뇌, 글꼴

# 1. 100% 학습된 모델 로드
뇌 = 정책두뇌(7, 2, 연속=True)
뇌.load_state_dict(torch.load('model.pt', map_location='cpu'))

# 2. 계기판(HUD) 합성 렌더러
def 계기판_화면(e, a, 세기):
    W, H, sc = 480, 360, 26
    ox, gy = W / 2, H - 40
    im = Image.new('RGB', (W + 240, H), (12, 14, 18))
    d = ImageDraw.Draw(im)
    
    # 지면 및 패드
    d.rectangle([0, gy, W, H], fill=(22, 26, 32))
    d.rectangle([ox - 1.5 * sc, gy - 3, ox + 1.5 * sc, gy + 3], fill=(255, 181, 71))
    
    # 로드스터 기체
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
        
    아래 = float(np.clip(세기[0], 0, 1))
    회전 = float(np.clip(세기[1], -1, 1))
    if 아래 > 0.05:
        d.polygon([rot(-5, -9), rot(5, -9), rot(0, -9 - 22 * 아래)], fill=(255, 181, 71))
    if 회전 < -0.05:
        d.polygon([rot(16, 2), rot(16, -3), rot(16 + 15 * -회전, 0)], fill=(191, 239, 255))
    if 회전 > 0.05:
        d.polygon([rot(-16, 2), rot(-16, -3), rot(-16 - 15 * 회전, 0)], fill=(191, 239, 255))
        
    thn = math.degrees((th + math.pi) % (2 * math.pi) - math.pi)
    
    # 좌측 기체 상태 텍스트
    d.text((14, 12), "🚗 PPO 실시간 비행 상태", fill=(0, 255, 65), font=글꼴(16))
    d.text((14, 38), f"높이: {y:4.1f}m · 세로속도: {vy:+.2f}m/s · 각도: {thn:+4.0f}°", fill=(210, 220, 230), font=글꼴(13))

    # 우측 HUD 계기판 분리선
    d.line([(W, 0), (W, H)], fill=(40, 48, 60), width=2)
    
    # 우측 HUD 계기판
    d.text((W + 15, 14), "🕹 AI 조이스틱 출력 HUD", fill=(255, 181, 71), font=글꼴(16))
    
    # 1) 주 추진기 게이지 (0% ~ 100%)
    d.text((W + 15, 60), "주 추진기 (상승 제동)", fill=(200, 210, 220), font=글꼴(13))
    d.rectangle([W + 15, 82, W + 215, 102], outline=(90, 100, 115), width=1)
    d.rectangle([W + 16, 83, W + 16 + int(198 * 아래), 101], fill=(255, 140, 0))
    d.text((W + 15, 108), f"출력: {아래*100:4.1f}%", fill=(255, 181, 71), font=글꼴(12))

    # 2) 회전 추진기 게이지 (좌 <- 중립 -> 우)
    d.text((W + 15, 145), "자세 제어 (회전 분사)", fill=(200, 210, 220), font=글꼴(13))
    중앙X = W + 115
    d.rectangle([W + 15, 167, W + 215, 187], outline=(90, 100, 115), width=1)
    d.line([(중앙X, 167), (중앙X, 187)], fill=(120, 130, 145), width=2)
    if 회전 < 0:
        d.rectangle([중앙X + int(99 * 회전), 168, 중앙X, 186], fill=(0, 200, 255))
    elif 회전 > 0:
        d.rectangle([중앙X, 168, 중앙X + int(99 * 회전), 186], fill=(0, 200, 255))
    d.text((W + 15, 193), f"방향: {'좌측' if 회전 < -0.05 else '우측' if 회전 > 0.05 else '중립'} ({회전:+.2f})", fill=(191, 239, 255), font=글꼴(12))
    
    # 3) 비행 단계 진단 안내
    if y > 6.0 and abs(thn) > 40:
        단계 = "1단계: 공중 180° 회전 중"
        색 = (0, 200, 255)
    elif y > 2.0:
        단계 = "2단계: 수평 정렬 및 감속"
        색 = (255, 200, 0)
    else:
        단계 = "3단계: 최종 소프트 랜딩!"
        색 = (0, 255, 65)
    d.text((W + 15, 250), "비행 단계 판정:", fill=(160, 170, 180), font=글꼴(13))
    d.text((W + 15, 275), 단계, fill=색, font=글꼴(14))

    return np.array(im)

# 3. 비디오 렌더링
e = 로드스터연속()
o, _ = e.reset(seed=7)
화면들 = []
끝 = 잘림 = False

while not (끝 or 잘림):
    with torch.no_grad():
        d = 뇌.분포(torch.as_tensor(o, dtype=torch.float32))
        a = np.clip(d.mean.numpy(), e.action_space.low, e.action_space.high)
    화면들.append(계기판_화면(e, a, a))
    o, r, 끝, 잘림, 정보 = e.step(a)

결과 = 정보.get('결과')
마지막_화면 = 화면들[-1]
화면들 += [마지막_화면] * 30

파일 = 'roadster_telemetry.mp4'
imageio.mimsave(파일, 화면들, fps=30, macro_block_size=1)
print(f"🎬 [HUD 계기판 시각화 영상 생성 완료]: {파일} (착지 결과: {결과})")

