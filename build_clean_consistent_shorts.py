# -*- coding: utf-8 -*-
"""
휴머노이드 5단계 진화 쇼츠 - '동적 뱃지 규격 & 통일 상단 배너 & 클린 하단' 완제품 생성기
1. 상단: 글자 길이에 맞춰 뱃지 배경색이 정확히 감싸고, 타이틀과 절대 겹치지 않도록 동적 좌표 계산
2. 중앙: 720x720 고화질 MuJoCo 시뮬레이션
3. 하단: 하단 320px 완전 클린 (쇼츠 UI 짤림 방지 및 사용자 직접 자막 추가 공간)
4. 오디오: '현수 보이스' 나래이션 + 8-bit BGM 자동 싱크 믹싱
"""
import os, sys, asyncio, subprocess, shutil
import cv2
import numpy as np
import torch
import gymnasium as gym
from PIL import Image, ImageDraw, ImageFont
import edge_tts

from train_humanoid_gui import 휴머노이드두뇌, 상태정규화
from train_humanoid_v5 import NaturalJoggingWrapperV5

FONT_PATH = "C:/Windows/Fonts/malgunbd.ttf"
FONT_TITLE = ImageFont.truetype(FONT_PATH, 35)
FONT_SUB = ImageFont.truetype(FONT_PATH, 22)

# 5단계 세대별 메타 정보 및 통일된 상단 타이틀
CLIPS_INFO = [
    {
        'id': 'gen0',
        'badge': 'GEN 0',
        'title': '갓난아기 (0초 컷)',
        'sub': '0세대 · 태어나자마자 안면 슬라이딩',
        'color': (230, 60, 60),      # 빨강
        'model': None,
        'voice_text': '아무것도 모르는 AI에게 걷는 법을 맡겨봤습니다. 시작하자마자 꽈당!',
        'voice_rate': '+12%',
        'pitch': '+0Hz'
    },
    {
        'id': 'gen100',
        'badge': 'GEN 100',
        'title': '첫 걸음마 (균형 잡기)',
        'sub': '10만 걸음 · 팔 벌려 필사의 중심잡기',
        'color': (240, 140, 20),     # 주황
        'model': 'humanoid_ppo_gui.pt',
        'voice_text': '10만 번 넘어지더니, 팔을 벌려 필사적으로 중심을 잡기 시작합니다.',
        'voice_rate': '+10%',
        'pitch': '+0Hz'
    },
    {
        'id': 'gen500',
        'badge': 'GEN 500',
        'title': '전진 보행 (첫 스텝)',
        'sub': '50만 걸음 · 다리를 뻗으며 전진 개시',
        'color': (220, 190, 20),     # 노랑/골드
        'model': 'humanoid_ppo_gen500.pt',
        'voice_text': '50만 세대. 다리를 앞으로 뻗으며 드디어 첫 발을 내딛습니다.',
        'voice_rate': '+10%',
        'pitch': '+1Hz'
    },
    {
        'id': 'gen1000',
        'badge': 'GEN 1000',
        'title': '고속 질주 (스프린트)',
        'sub': '100만 걸음 · 10초 연속 폭풍 질주',
        'color': (40, 170, 250),     # 하늘색
        'model': 'humanoid_ppo_gen1000.pt',
        'voice_text': '100만 세대 돌파! 쓰러지지 않고 10초 연속 폭풍 질주를 시작합니다.',
        'voice_rate': '+10%',
        'pitch': '+1Hz'
    },
    {
        'id': 'gen2500',
        'badge': 'GEN 2500',
        'title': '마라톤 러너 (조거)',
        'sub': '250만 걸음 · 시속 5.6km 완벽한 보행 완성',
        'color': (180, 50, 230),     # 보라/핑크
        'model': 'humanoid_ppo_gen2500_v5.pt',
        'voice_text': '그리고 250만 걸음 뒤, 시속 5.6km로 달리는 완벽한 마라토너가 되었습니다!',
        'voice_rate': '+10%',
        'pitch': '+2Hz'
    }
]

def render_unified_frame(sim_frame_rgb, info):
    """
    720x1280 캔버스에:
    - 상단(y: 55~185): 글자 길이에 맞춰 뱃지 박스가 정확히 늘어나고 타이틀과 분리되는 스마트 배너
    - 중앙(y: 260~980): 720x720 3D 시뮬레이션
    - 하단(y: 980~1280): 완전 클린 (쇼츠 UI safe zone)
    """
    canvas_pil = Image.new('RGB', (720, 1280), (14, 18, 24))
    draw = ImageDraw.Draw(canvas_pil)

    # 1. 상단 통일 배너 외곽 카드 (x: 40~680, y: 55~185)
    card_bg = (22, 28, 38)
    draw.rounded_rectangle([40, 55, 680, 185], radius=16, fill=card_bg, outline=info['color'], width=3)
    
    # 뱃지 텍스트 너비 동적 측정
    badge_text = info['badge']
    bb = draw.textbbox((0, 0), badge_text, font=FONT_TITLE)
    badge_w = bb[2] - bb[0]
    
    # 뱃지 배경 박스 (텍스트 양옆 14px 패딩 부여)
    badge_x1 = 60
    badge_x2 = badge_x1 + badge_w + 28
    badge_box = [badge_x1, 72, badge_x2, 126]
    draw.rounded_rectangle(badge_box, radius=8, fill=info['color'])
    
    # 뱃지 텍스트 렌더링
    draw.text((badge_x1 + 14, 78), badge_text, fill=(255, 255, 255), font=FONT_TITLE)
    
    # 타이틀 텍스트 렌더링 (뱃지 박스 끝에서 18px 띄워 절대 겹치지 않음!)
    title_x = badge_x2 + 18
    draw.text((title_x, 78), info['title'], fill=(255, 255, 255), font=FONT_TITLE)
    
    # 서브타이틀
    draw.text((62, 142), info['sub'], fill=(185, 205, 225), font=FONT_SUB)

    # 2. 중앙 720x720 렌더 화면
    sim_resized = cv2.resize(sim_frame_rgb, (720, 720))
    sim_pil = Image.fromarray(sim_resized)
    canvas_pil.paste(sim_pil, (0, 260))

    # 3. 하단 (980~1280)은 100% 클린 세이프존 유지

    return cv2.cvtColor(np.array(canvas_pil), cv2.COLOR_RGB2BGR)

def get_duration(f):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', f]
    return float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())

async def generate_voice_clips():
    print("🎙️ [통일 나래이션 음성 생성 중]...")
    for item in CLIPS_INFO:
        mp3 = f"clean_voice_{item['id']}.mp3"
        comm = edge_tts.Communicate(item['voice_text'], voice='ko-KR-HyunsuMultilingualNeural', rate=item['voice_rate'], pitch=item['pitch'])
        await comm.save(mp3)
        item['audio'] = mp3
        item['dur'] = get_duration(mp3)
        print(f"  • {item['badge']}: {item['dur']:.2f}초 | \"{item['voice_text']}\"")

def record_clean_video_clips():
    print("\n🎬 [동적 뱃지 배너 + 클린 하단 클립 5종 렌더링 시작]...")
    part_files = []

    for idx, item in enumerate(CLIPS_INFO):
        target_dur = item['dur'] + 0.3
        total_frames = int(target_dur * 30)
        temp_raw_avi = f"temp_clean_raw_{item['id']}.avi"
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(temp_raw_avi, fourcc, 30.0, (720, 1280))

        env = gym.make("Humanoid-v5", render_mode="rgb_array")
        
        brain, norm = None, None
        if item['model'] and os.path.exists(item['model']):
            ckpt = torch.load(item['model'], map_location='cpu', weights_only=False)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            norm = 상태정규화((obs_dim,))
            norm.mean = ckpt['obs_mean']
            norm.var = ckpt['obs_var']
            brain = 휴머노이드두뇌(obs_dim, act_dim, 은닉=256)
            brain.load_state_dict(ckpt['brain'])
            brain.eval()

        seed = 100 + idx * 50
        obs, _ = env.reset(seed=seed)
        done = False
        frame_cnt = 0

        while frame_cnt < total_frames:
            rgb_frame = env.render()
            unified_bgr = render_unified_frame(rgb_frame, item)
            out.write(unified_bgr)
            frame_cnt += 1

            if brain is not None and norm is not None:
                with torch.no_grad():
                    d = brain.분포(torch.as_tensor(norm.normalize(obs), dtype=torch.float32))
                    a = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
            else:
                a = env.action_space.sample()

            obs, r, term, trunc, _ = env.step(a)
            if term or trunc:
                seed += 1
                obs, _ = env.reset(seed=seed)

        out.release()
        env.close()

        part_mp4 = f"clean_part_{idx}_{item['id']}.mp4"
        cmd = [
            'ffmpeg', '-y',
            '-i', temp_raw_avi,
            '-i', item['audio'],
            '-t', f"{target_dur:.2f}",
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            part_mp4
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(temp_raw_avi)
        part_files.append(part_mp4)
        print(f"  ✅ {item['badge']} 클린 클립 완성: {part_mp4} ({target_dur:.2f}초)")

    with open('clean_concat_list.txt', 'w', encoding='utf-8') as f:
        for p in part_files:
            f.write(f"file '{p}'\n")

    temp_concat = "temp_clean_concat.mp4"
    subprocess.run([
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'clean_concat_list.txt',
        '-c', 'copy',
        temp_concat
    ], check=True, capture_output=True)

    final_output = "휴머노이드_통합배너_클린쇼츠_최종본.mp4"
    filter_complex = "[1:a]volume=0.18[bgm];[0:a]volume=1.0[voice];[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    cmd_final = [
        'ffmpeg', '-y',
        '-i', temp_concat,
        '-stream_loop', '-1',
        '-i', 'shorts_final_bgm.wav',
        '-filter_complex', filter_complex,
        '-map', '0:v:0',
        '-map', '[aout]',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        final_output
    ]
    subprocess.run(cmd_final, check=True)
    shutil.copy(final_output, "humanoid_shorts_clean_unified.mp4")
    print(f"\n🎉 [동적 뱃지 & 무간섭 레이아웃 최종 완성!]: {final_output}")

if __name__ == '__main__':
    asyncio.run(generate_voice_clips())
    record_clean_video_clips()
