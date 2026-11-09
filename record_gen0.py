# -*- coding: utf-8 -*-
"""
세대별 클립 자동 녹화기
- clip_gen0.mp4: 아무것도 모르는 0세대 (서자마자 꽈당)
"""
import os, cv2, numpy as np
import gymnasium as gym

def create_shorts_frame(render_frame, title="AI HUMANOID EVOLUTION", sub_title="GEN 0: 갓난아기 (꽈당)", step_cnt=0):
    canvas = np.zeros((1280, 720, 3), dtype=np.uint8)
    canvas[:] = (15, 18, 24)

    # 480x480 화면을 폭 720에 맞춤
    sim_resized = cv2.resize(render_frame, (720, 720))
    canvas[280:1000, 0:720] = sim_resized

    font = cv2.FONT_HERSHEY_SIMPLEX

    # 상단 배너
    cv2.rectangle(canvas, (30, 70), (690, 210), (22, 28, 38), -1)
    cv2.rectangle(canvas, (30, 70), (690, 210), (0, 210, 255), 2)
    cv2.putText(canvas, title, (50, 125), font, 1.0, (0, 220, 255), 3, cv2.LINE_AA)
    cv2.putText(canvas, "MUJOCO HUMANOID-V5 PPO", (50, 175), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # 하단 배너
    cv2.rectangle(canvas, (30, 1050), (690, 1200), (22, 28, 38), -1)
    cv2.rectangle(canvas, (30, 1050), (690, 1200), (255, 75, 75), 2) # 빨간색 경고 배너
    cv2.putText(canvas, sub_title, (50, 1105), font, 0.9, (255, 95, 95), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"SURVIVED: {step_cnt} STEPS (FAIL)", (50, 1160), font, 0.8, (200, 210, 225), 2, cv2.LINE_AA)

    return canvas

def record_gen0(out_mp4="humanoid_gen0_fail.mp4", total_frames=120):
    env = gym.make('Humanoid-v5', render_mode='rgb_array')
    o, _ = env.reset(seed=12)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_mp4, fourcc, 30, (720, 1280))

    frame_idx = 0
    step_in_ep = 0
    while frame_idx < total_frames:
        frame = env.render()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            shorts_frame = create_shorts_frame(
                frame_bgr, 
                title="AI ROBOT LEARNS TO WALK", 
                sub_title="GEN 0: 갓난아기 대참사 (0.5초 컷)", 
                step_cnt=step_in_ep
            )
            out.write(shorts_frame)
            frame_idx += 1

        # 0세대는 무작위 행동으로 바로 넘어짐
        a = env.action_space.sample()
        o, r, term, trunc, _ = env.step(a)
        step_in_ep += 1
        if term or trunc:
            o, _ = env.reset()
            step_in_ep = 0

    out.release()
    env.close()
    print(f"🎬 [0세대 실패 영상 생성 완료]: {out_mp4} (총 {total_frames}프레임, 4초)")

if __name__ == '__main__':
    record_gen0()

