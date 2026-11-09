import torch
import numpy as np
import cv2
import gymnasium as gym
import subprocess
from train_humanoid_gui import 휴머노이드두뇌, 상태정규화

def record_gen1000():
    print("🎬 [Gen 1000 스프린트 질주] 9:16 쇼츠용 영상 캡처 시작...")
    
    model_file = 'humanoid_ppo_gen1000.pt'
    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    env = gym.make("Humanoid-v5", render_mode="rgb_array")
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    norm = 상태정규화((obs_dim,))
    norm.mean = ckpt['obs_mean']
    norm.var = ckpt['obs_var']
    
    brain = 휴머노이드두뇌(obs_dim, act_dim, 은닉=256)
    brain.load_state_dict(ckpt['brain'])
    brain.eval()
    
    w_out, h_out = 720, 1280
    raw_mp4 = 'humanoid_gen1000_raw.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_mp4, fourcc, 30.0, (w_out, h_out))
    
    best_frames = []
    best_score = 0
    
    for seed in [300, 301, 302, 303, 304, 305]:
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0
        total_r = 0
        frames = []
        
        while not done and step < 300:
            frame = env.render()
            
            canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)
            canvas[:, :] = (18, 22, 30)
            
            # 중앙 MuJoCo 렌더
            frame_resized = cv2.resize(frame, (680, 680))
            y_offset = (h_out - 680) // 2
            x_offset = (w_out - 680) // 2
            canvas[y_offset:y_offset+680, x_offset:x_offset+680] = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            
            # 상단 타이틀
            cv2.rectangle(canvas, (40, 90), (w_out - 40, 190), (180, 20, 80), -1)
            cv2.putText(canvas, "[ GEN 1000: SPRINT MASTER ]", (50, 145), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.05, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "1,000,000 Steps High-Speed Running", (70, 175), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 225, 235), 2, cv2.LINE_AA)
            
            # 하단 상태창
            cv2.rectangle(canvas, (40, h_out - 220), (w_out - 40, h_out - 100), (0, 0, 0), -1)
            cv2.putText(canvas, f"Step: {step:03d} | Reward: {total_r:5.1f}", (60, h_out - 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 200), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Status: High-Speed Continuous Sprinting", (60, h_out - 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 215, 0), 2, cv2.LINE_AA)
            
            frames.append(canvas)
            
            with torch.no_grad():
                d = brain.분포(torch.as_tensor(norm.normalize(obs), dtype=torch.float32))
                action = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
            
            obs, r, term, trunc, _ = env.step(action)
            total_r += r
            step += 1
            done = term or trunc
            
        print(f"Seed {seed}: {step} steps survived, Score: {total_r:.1f}")
        if len(frames) > len(best_frames):
            best_frames = frames
            best_score = total_r
            
    for f in best_frames:
        out.write(f)
        
    out.release()
    env.close()
    
    # 웹 표준 H.264로 인코딩
    out_mp4 = 'humanoid_gen1000_sprint.mp4'
    cmd = f'ffmpeg -y -i {raw_mp4} -c:v libx264 -pix_fmt yuv420p -movflags +faststart {out_mp4}'
    subprocess.run(cmd, shell=True, check=True)
    print(f"✅ Gen 1000 영상 생성 완료: {out_mp4} ({len(best_frames)} 프레임)")

if __name__ == '__main__':
    record_gen1000()

