"""
Open Duck Mini v2 - 학습된 강화학습 모델 3D 실시간 관찰기
ConnectAI LAB · PPO 기립 및 보행 정책 시각화
"""
import os
import sys
import time
import zipfile
import io
import numpy as np
import gymnasium as gym
import torch
from stable_baselines3 import PPO

import open_duck_env

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

def load_open_duck_model(env):
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )
    model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=0)

    # 1. Direct PyTorch .pt 파일 우선 확인
    pt_paths = [
        os.path.join("models", "open_duck_best_balance.pt"),
        os.path.join("models", "open_duck_final.pt"),
    ]
    for pt in pt_paths:
        if os.path.exists(pt):
            try:
                state_dict = torch.load(pt, map_location="cpu", weights_only=True)
                model.policy.load_state_dict(state_dict)
                print(f"✅ 모델 로드 성공 (PyTorch .pt): {pt}")
                return model
            except Exception as e:
                print(f"⚠️ {pt} 로드 재시도 중: {e}")

    # 2. .zip 파일 내부 policy.pth 로드
    zip_paths = [
        os.path.join("models", "open_duck_best_balance.zip"),
        os.path.join("models", "open_duck_final.zip"),
    ]
    for zp in zip_paths:
        if os.path.exists(zp):
            try:
                with zipfile.ZipFile(zp, "r") as z:
                    st = torch.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=False)
                    model.policy.load_state_dict(st)
                    print(f"✅ 모델 로드 성공 (Zip StateDict): {zp}")
                    return model
            except Exception as e:
                print(f"⚠️ {zp} 로드 재시도 중: {e}")

    raise FileNotFoundError("models 폴더에서 학습된 모델 파일을 찾을 수 없습니다.")

def main():
    print("=" * 70)
    print(" 🦆 [Open Duck Mini v2] 학습된 기립 & 보행 모델 3D 실시간 관찰기")
    print("=" * 70)
    print(" • 모터 부하 경감: 눈 사이 60g + 상체 중심 250g 무게추 반영")
    print(" • 토크 과다 처벌 + 좌우 체중이동(Sway) + 발 교대 체공(Air-time) 반영")
    print(" • 50 FPS 1:1 실시간 물리 시뮬레이션")
    print(" • 조작법:")
    print("    - 마우스 좌클릭 드래그: 360도 카메라 회전")
    print("    - 마우스 휠: 줌 인 / 줌 아웃")
    print("    - 창 닫기(X): 종료")
    print("=" * 70)

    env = gym.make("OpenDuckWalk-v0", render_mode="human")
    try:
        model = load_open_duck_model(env)
    except Exception as e:
        print(f"❌ {e}")
        env.close()
        return

    episode = 0
    try:
        while True:
            episode += 1
            obs, info = env.reset()
            ep_reward = 0.0
            step = 0
            start_x = float(env.unwrapped.data.qpos[0])

            print(f"\n[에피소드 {episode}] 시작! (기립 자세)")

            done = False
            while not done:
                t0 = time.time()

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += reward
                step += 1
                done = terminated or truncated

                # 50 FPS (0.02초) 실시간 동기화
                elapsed = time.time() - t0
                if elapsed < 0.02:
                    time.sleep(0.02 - elapsed)

                # 카메라 트래킹 (오픈덕 위치 추종)
                viewer = env.unwrapped.viewer
                if viewer is not None:
                    if not viewer.is_running():
                        print("[종료] 3D 창이 닫혔습니다.")
                        return
                    cur_x = env.unwrapped.data.qpos[0]
                    cur_y = env.unwrapped.data.qpos[1]
                    viewer.cam.lookat[0] = cur_x + 0.1
                    viewer.cam.lookat[1] = cur_y
                    viewer.cam.lookat[2] = 0.16

            end_x = float(env.unwrapped.data.qpos[0])
            dist = end_x - start_x
            fallen = info.get("fallen", False)
            status_text = "💥 넘어짐" if fallen else "🎉 1,000스텝 완주"
            print(f"  └─ 결과: {status_text} | 생존: {step}스텝 ({step*0.02:.2f}초) | 전진 거리: {dist:.2f}m | 보상: {ep_reward:.1f}")

    except KeyboardInterrupt:
        print("\n[사용자 중단] 관찰을 종료합니다.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
