# -*- coding: utf-8 -*-
"""
휴머노이드 보행 진화 유튜브 쇼츠 자동 제작 스크립트 (720x1280, 9:16 세로)
- 0세대 (꽈당), 100세대 (걸음마), 500세대 (질주) 클립 합성
- 상단/하단 자막 바 및 세대 카운터 오버레이
"""
import os, sys, math, time, cv2, numpy as np
import gymnasium as gym
import torch

def create_shorts_frame(render_frame, title="AI HUMANOID EVOLUTION", sub_title="GEN 0: 갓난아기 (꽈당)", step_cnt=0):
    # 720x1280 캔버스 생성 (다크 모던 배경)
    canvas = np.zeros((1280, 720, 3), dtype=np.uint8)
    canvas[:] = (15, 18, 24)

    # 480x480 렌더 프레임 리사이즈 (가로 폭 720에 맞춤 -> 720x720)
    sim_resized = cv2.resize(render_frame, (720, 720))
    # 중앙 배치 (y: 280 ~ 1000)
    canvas[280:1000, 0:720] = sim_resized

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. 상단 후킹 타이틀 바
    cv2.rectangle(canvas, (30, 80), (690, 220), (22, 28, 38), -1)
    cv2.rectangle(canvas, (30, 80), (690, 220), (0, 210, 255), 2)
    cv2.putText(canvas, title, (50, 135), font, 1.05, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "MUJOCO HUMAN-V5 PPO", (50, 185), font, 0.85, (255, 255, 255), 2, cv2.LINE_AA)

    # 2. 하단 자막 바
    cv2.rectangle(canvas, (30, 1050), (690, 1200), (22, 28, 38), -1)
    cv2.rectangle(canvas, (30, 1050), (690, 1200), (50, 255, 120), 2)
    cv2.putText(canvas, sub_title, (50, 1105), font, 0.95, (50, 255, 120), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"SURVIVAL: {step_cnt} STEPS", (50, 1160), font, 0.8, (200, 210, 225), 2, cv2.LINE_AA)

    return canvas

def record_clip(model_path=None, out_mp4="clip_gen0.mp4", title="STAGE 1", subtitle="GEN 0: FIRST STEPS", max_frames=120):
    env = gym.make('Humanoid-v5', render_mode='rgb_array')
    o, _ = env.reset(seed=42)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_mp4, fourcc, 30, (720, 1280))

    frame_idx = 0
    done = False
    while frame_idx < max_frames:
        frame = env.render()
        if frame is not None:
            # RGB -> BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            shorts_frame = create_shorts_frame(frame_bgr, title=title, sub_title=subtitle, step_cnt=frame_idx)
            out.write(shorts_frame)
            frame_idx += 1

        a = env.action_space.sample() # 무작위 또는 모델 행동
        o, r, term, trunc, _ = env.step(a)
        if term or trunc:
            o, _ = env.reset()

    out.release()
    env.close()
    print(f"✅ 클립 녹화 완료: {out_mp4}")

if __name__ == '__main__':
    print("🎬 휴머노이드 쇼츠 생성기 준비 완료!")
    record_clip(out_mp4="humanoid_shorts_preview.mp4", title="AI HUMANOID WALKING", subtitle="GEN 0: 갓난아기 대참사", max_frames=90)

