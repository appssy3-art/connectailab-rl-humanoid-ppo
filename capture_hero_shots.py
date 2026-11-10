import torch
import numpy as np
import cv2
import gymnasium as gym
from train_humanoid_gui import 휴머노이드두뇌, 상태정규화
from train_humanoid_v5 import NaturalJoggingWrapperV5

def capture_hero_shots():
    print("📸 [히어로 샷 캡처] 캐릭터가 화면에 꽉 차는 고화질 렌더링 시작...")
    
    # 1. Gen 0 (0세대 넘어짐 샷)
    env0 = gym.make("Humanoid-v5", render_mode="rgb_array", width=1080, height=1080)
    # 카메라를 캐릭터 허리/가슴 높이(z=0.6)에 맞추고 거리 2.1m로 초근접 촬영
    env0.unwrapped.mujoco_renderer.default_cam_config = {
        'trackbodyid': 1,
        'distance': 2.1,
        'lookat': np.array([0., 0., 0.55]),
        'elevation': -10.0,
        'azimuth': 90.0
    }
    
    obs, _ = env0.reset(seed=102)
    fall_frames = []
    for step in range(60):
        # random action -> falls comically
        action = np.random.uniform(-1, 1, env0.action_space.shape)
        obs, r, term, trunc, _ = env0.step(action)
        frame = env0.render()
        fall_frames.append(frame)
        if term or trunc:
            # take a few more after fall
            for _ in range(5):
                fall_frames.append(env0.render())
            break
    env0.close()
    
    # Save the best fall frames
    print(f"Gen 0 captured {len(fall_frames)} frames.")
    for idx in [len(fall_frames)//2, len(fall_frames)-1, max(0, len(fall_frames)-5)]:
        cv2.imwrite(f"hero_gen0_{idx}.png", cv2.cvtColor(fall_frames[idx], cv2.COLOR_RGB2BGR))
    
    # 2. Gen 2500 (250만 세대 역동적 질주 샷)
    model_file = 'humanoid_ppo_gen2500_v5.pt'
    ckpt = torch.load(model_file, map_location='cpu', weights_only=False)
    
    base_env = gym.make("Humanoid-v5", forward_reward_weight=4.0, healthy_reward=1.5, ctrl_cost_weight=0.02, 
                        render_mode="rgb_array", width=1080, height=1080)
    base_env.unwrapped.mujoco_renderer.default_cam_config = {
        'trackbodyid': 1,
        'distance': 2.2,
        'lookat': np.array([0., 0., 0.65]),
        'elevation': -8.0,
        'azimuth': 90.0
    }
    env5 = NaturalJoggingWrapperV5(base_env, target_speed=1.6)
    
    obs_dim = env5.observation_space.shape[0]
    act_dim = env5.action_space.shape[0]
    
    norm = 상태정규화((obs_dim,))
    norm.mean = ckpt['obs_mean']
    norm.var = ckpt['obs_var']
    
    brain = 휴머노이드두뇌(obs_dim, act_dim, 은닉=256)
    brain.load_state_dict(ckpt['brain'])
    brain.eval()
    
    run_frames = []
    obs, _ = env5.reset(seed=503)
    for step in range(90):
        sb = torch.tensor(norm.normalize(obs), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            act = brain.분포(sb).mean.squeeze(0).numpy()
        obs, r, term, trunc, _ = env5.step(act)
        frame = env5.render()
        run_frames.append(frame)
        if term or trunc:
            break
    env5.close()
    
    print(f"Gen 2500 captured {len(run_frames)} frames.")
    for idx in range(30, min(80, len(run_frames)), 10):
        cv2.imwrite(f"hero_gen2500_{idx}.png", cv2.cvtColor(run_frames[idx], cv2.COLOR_RGB2BGR))
        
    print("Hero frames generated successfully!")

if __name__ == "__main__":
    capture_hero_shots()
