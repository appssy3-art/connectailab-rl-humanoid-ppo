import torch
import numpy as np
import cv2
import gymnasium as gym
import subprocess
from train_humanoid_gui import 휴머노이드두뇌, 상태정규화
from train_humanoid_v5 import NaturalJoggingWrapperV5

def record_v5():
    print("🎬 [v5 초고속 마라톤 조거] 9:16 쇼츠용 영상 캡처 시작...")
    
    model_file = 'humanoid_ppo_gen2500_v5.pt'
    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    
    base_env = gym.make("Humanoid-v5", forward_reward_weight=4.0, healthy_reward=1.5, ctrl_cost_weight=0.02, render_mode="rgb_array")
    env = NaturalJoggingWrapperV5(base_env, target_speed=1.6)
    
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    norm = 상태정규화((obs_dim,))
    norm.mean = ckpt['obs_mean']
    norm.var = ckpt['obs_var']
    
    brain = 휴머노이드두뇌(obs_dim, act_dim, 은닉=256)
    brain.load_state_dict(ckpt['brain'])
    brain.eval()
    
    w_out, h_out = 720, 1280
    raw_mp4 = 'humanoid_v5_raw.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(raw_mp4, fourcc, 30.0, (w_out, h_out))
    
    best_frames = []
    best_score = 0
    
    for seed in [501, 502, 503, 504, 505]:
        obs, _ = env.reset(seed=seed)
        done = False
        step = 0
        total_r = 0
        v_list = []
        frames = []
        
        while not done and step < 360:
            frame = env.render()
            
            canvas = np.zeros((h_out, w_out, 3), dtype=np.uint8)
            canvas[:, :] = (12, 16, 24)
            
            frame_resized = cv2.resize(frame, (680, 680))
            y_offset = (h_out - 680) // 2
            x_offset = (w_out - 680) // 2
            canvas[y_offset:y_offset+680, x_offset:x_offset+680] = cv2.cvtColor(frame_resized, cv2.COLOR_RGB2BGR)
            
            # 상단 타이틀
            cv2.rectangle(canvas, (40, 90), (w_out - 40, 190), (120, 20, 200), -1)
            cv2.putText(canvas, "[ GEN 2500: MARATHON JOGGER ]", (42, 145), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.95, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, "2.5M Steps Deep PPO Continuous Control", (55, 175), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 220, 255), 2, cv2.LINE_AA)
            
            # 하단 상태창
            cur_vx = v_list[-1] if v_list else 0.0
            cv2.rectangle(canvas, (40, h_out - 220), (w_out - 40, h_out - 100), (0, 0, 0), -1)
            cv2.putText(canvas, f"Step: {step:03d} | Speed: {cur_vx:4.2f} m/s", (60, h_out - 170), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 220), 2, cv2.LINE_AA)
            cv2.putText(canvas, "Upright Torso + 1.6 m/s Rhythmic Pace", (60, h_out - 130), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 215, 0), 2, cv2.LINE_AA)
            
            frames.append(canvas)
            
            with torch.no_grad():
                d = brain.분포(torch.as_tensor(norm.normalize(obs), dtype=torch.float32))
                action = np.clip(d.mean.numpy(), env.action_space.low, env.action_space.high)
            
            obs, r, term, trunc, info = env.step(action)
            vx = info.get('x_velocity', 0.0)
            v_list.append(vx)
            total_r += r
            step += 1
            done = term or trunc
            
        avg_vx = np.mean(v_list) if v_list else 0.0
        print(f"Seed {seed}: {step} steps survived, Avg Speed: {avg_vx:.2f} m/s, Score: {total_r:.1f}")
        if len(frames) > len(best_frames):
            best_frames = frames
            best_score = total_r
            
    for f in best_frames:
        out.write(f)
        
    out.release()
    env.close()
    
    out_mp4 = 'humanoid_v5_marathon.mp4'
    cmd = f'ffmpeg -y -i {raw_mp4} -c:v libx264 -pix_fmt yuv420p -movflags +faststart {out_mp4}'
    subprocess.run(cmd, shell=True, check=True)
    print(f"✅ v5 영상 생성 완료: {out_mp4} ({len(best_frames)} 프레임)")

if __name__ == '__main__':
    record_v5()

