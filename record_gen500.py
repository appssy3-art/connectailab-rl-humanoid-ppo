import torch
import numpy as np
import cv2
import gymnasium as gym
from train_humanoid_gui import 휴머노이드두뇌, 상태정규화

def record_gen500():
    print("🎬 [Gen 500 전진 보행] 9:16 쇼츠용 영상 캡처 시작...")
    
    ckpt = torch.load('humanoid_ppo_gen500.pt', map_location='cpu', weights_only=False)
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
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('humanoid_gen500_run.mp4', fourcc, 30.0, (w_out, h_out))
    
    best_frames = []
    
    # 5개 시드 중 가장 오래 달린 에피소드 탐색
    for seed in [205, 206, 207, 208, 209]:
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0
        frames = []
        
        while not done and step < 240:
            frame = env.render()
            
            # 720x1280 9:16 세로 캔버스 생성 (MuJoCo 480x480 정중앙 배치)
            canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)
            canvas[:, :] = (20, 24, 32)
            
            # MuJoCo 프레임 680x680 확대 리사이즈
            frame_resized = cv2.resize(frame, (680, 680))
            y_offset = (h_out - 680) // 2
            x_offset = (w_out - 680) // 2
            canvas[y_offset:y_offset+680, x_offset:x_offset+680] = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            
            # 쇼츠 상단 타이틀 배너
            cv2.rectangle(canvas, (40, 90), (w_out - 40, 185), (25, 118, 210), -1)
            cv2.putText(canvas, "[ GEN 500: FORWARD WALKING ]", (60, 145), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "PPO Continuous Control Evolution", (80, 175), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 235, 255), 2, cv2.LINE_AA)
            
            # 쇼츠 하단 상태창
            cv2.rectangle(canvas, (40, h_out - 220), (w_out - 40, h_out - 100), (0, 0, 0), -1)
            cv2.putText(canvas, f"Step: {step:03d} / 240", (60, h_out - 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (100, 255, 100), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Balance + Forward Step Acquired!", (60, h_out - 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 215, 0), 2, cv2.LINE_AA)
            
            frames.append(canvas)
            
            with torch.no_grad():
                d = brain.분포(torch.as_tensor(norm.normalize(obs), dtype=torch.float32))
                action = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
            
            obs, r, term, trunc, _ = env.step(action)
            step += 1
            done = term or trunc
            
        print(f"Seed {seed}: {step} steps survived.")
        if len(frames) > len(best_frames):
            best_frames = frames
            
    for f in best_frames:
        out.write(f)
        
    out.release()
    env.close()
    print(f"✅ Gen 500 영상 생성 완료: humanoid_gen500_run.mp4 ({len(best_frames)} 프레임)")

if __name__ == '__main__':
    record_gen500()
