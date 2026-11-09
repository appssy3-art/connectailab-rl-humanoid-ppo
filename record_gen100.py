# -*- coding: utf-8 -*-
"""
100세대 걸음마 클립 녹화기 (humanoid_gen100_step.mp4)
- 10만 걸음 학습된 humanoid_ppo_gui.pt 모델 사용
- 양팔을 휘저으며 3~4초간 버티는 감동의 첫 걸음마 클립
"""
import os, cv2, numpy as np, torch
import gymnasium as gym
from train_humanoid_gui import 휴머노이드두뇌, 상태정규화
from record_gen0 import create_shorts_frame

def record_gen100(model_file="humanoid_ppo_gui.pt", out_mp4="humanoid_gen100_step.mp4", total_frames=180):
    env = gym.make('Humanoid-v5', render_mode='rgb_array')
    o, _ = env.reset(seed=105)
    
    # 모델 및 정규화 로드 (PyTorch 2.6+ 호환 weights_only=False)
    checkpoint = torch.load(model_file, map_location='cpu', weights_only=False)
    상태수 = env.observation_space.shape[0]
    행동수 = env.action_space.shape[0]
    
    정규화 = 상태정규화((상태수,))
    정규화.mean = checkpoint['obs_mean']
    정규화.var = checkpoint['obs_var']

    뇌 = 휴머노이드두뇌(상태수, 행동수, 은닉=256)
    뇌.load_state_dict(checkpoint['brain'])
    뇌.eval()

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
                sub_title="GEN 100: 첫 걸음마 (필사의 중심잡기)", 
                step_cnt=step_in_ep
            )
            out.write(shorts_frame)
            frame_idx += 1

        o_norm = 정규화.normalize(o)
        with torch.no_grad():
            d = 뇌.분포(torch.as_tensor(o_norm, dtype=torch.float32))
            a = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
        o, r, term, trunc, _ = env.step(a)
        step_in_ep += 1
        if term or trunc:
            o, _ = env.reset(seed=106 + frame_idx)
            step_in_ep = 0

    out.release()
    env.close()
    print(f"🎬 [100세대 걸음마 영상 생성 완료]: {out_mp4} (총 {total_frames}프레임, 6초)")

if __name__ == '__main__':
    record_gen100()
