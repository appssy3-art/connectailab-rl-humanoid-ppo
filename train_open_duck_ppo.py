"""
Open Duck Mini v2 - 초고속 PPO 기립 및 전진 보행 강화학습 스크립트
ConnectAI LAB · 14축 바이패드 밸런스 & 보행 학습
- 4개 병렬 환경(Vectorized Env) 초고속 연산 (~2,000 FPS)
- "넘어지면 새로 계속시작해" 즉시 리셋
- 200g 눈 사이 철판 + 700g 가슴 철판 반영
"""
import os
import sys
import time
import argparse
import numpy as np
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

import open_duck_env

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

class SuperFastDuckCallback(BaseCallback):
    """
    초고속 PPO 학습 모니터링 콜백
    - 넘어짐/재시작 실시간 통계 집계
    - 최고 생존 스텝 및 전진 거리 갱신 시 자동 체크포인트 저장
    """
    def __init__(self, check_freq=4096, save_path="models", verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

        self.total_episodes = 0
        self.total_falls = 0
        self.best_ep_steps = 0
        self.best_dist = 0.0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_step = 0

    def _on_step(self) -> bool:
        # GUI 3D 실시간 뷰어 실행 시 50 FPS 속도 조절 및 카메라 추종
        if hasattr(self.training_env, "envs"):
            raw_env = self.training_env.envs[0].unwrapped
            if hasattr(raw_env, "render_mode") and raw_env.render_mode == "human":
                viewer = getattr(raw_env, "viewer", None)
                if viewer is not None:
                    if not viewer.is_running():
                        print("\n[알림] 3D 시뮬레이션 창이 닫혀 학습을 안전하게 저장하고 종료합니다.")
                        return False
                    # 카메라가 로봇의 전진 위치를 부드럽게 추종
                    cur_x = raw_env.data.qpos[0]
                    cur_y = raw_env.data.qpos[1]
                    viewer.cam.lookat[0] = cur_x + 0.1
                    viewer.cam.lookat[1] = cur_y
                    viewer.cam.lookat[2] = 0.16
                time.sleep(0.015)

        # 벡터 환경의 각 환경별 done 및 info 확인
        dones = self.locals["dones"]
        infos = self.locals["infos"]

        for i, done in enumerate(dones):
            if done:
                self.total_episodes += 1
                info = infos[i]
                fallen = info.get("fallen", False)
                steps = info.get("step_count", 0)
                dist = info.get("forward_dist", 0.0)

                if fallen:
                    self.total_falls += 1

                # 최고 전진 보행 기록 경신 체크 (거리 최우선 + 생존 시간 결합)
                is_new_best = (dist > self.best_dist and steps >= 80) or (dist >= self.best_dist and steps > self.best_ep_steps)
                if is_new_best:
                    self.best_ep_steps = max(self.best_ep_steps, steps)
                    self.best_dist = max(self.best_dist, dist)
                    best_path = os.path.join(self.save_path, "open_duck_best_balance.zip")
                    best_pt = os.path.join(self.save_path, "open_duck_best_balance.pt")
                    try:
                        self.model.save(best_path)
                        torch.save(self.model.policy.state_dict(), best_pt)
                    except Exception:
                        pass
                    print(f"⭐ [최고 전진 기록 경신!] 에피소드 {self.total_episodes}회 | 전진: {dist:.2f}m | 생존: {steps}스텝 ({steps*0.02:.2f}초) -> 저장: {best_path}")

        # 주기적인 진행 상황 콘솔 출력 (약 4,000 스텝마다)
        if self.n_calls - self.last_log_step >= self.check_freq:
            now = time.time()
            elapsed = now - self.last_log_time
            steps_done = self.n_calls - self.last_log_step
            fps = steps_done / max(elapsed, 1e-5)
            self.last_log_time = now
            self.last_log_step = self.n_calls

            fall_rate = (self.total_falls / max(self.total_episodes, 1)) * 100
            print(
                f"📊 [스텝 {self.num_timesteps:7,d}] 속도: {fps:5.0f} FPS | "
                f"총 에피소드: {self.total_episodes:5d}회 (넘어짐 리셋: {self.total_falls}회, {fall_rate:4.1f}%) | "
                f"최장 생존: {self.best_ep_steps:4d}스텝 ({self.best_ep_steps*0.02:4.1f}초) | 최고전진: {self.best_dist:4.2f}m"
            )

            # 최신 모델 저장
            latest_path = os.path.join(self.save_path, "open_duck_latest.zip")
            self.model.save(latest_path)

        return True

def train(gui=False, total_timesteps=150000, n_envs=4, lr=3e-4):
    print("=" * 75)
    print(" 🦆 [Open Duck Mini v2] 초고속 PPO 기립 & 보행 강화학습 ")
    print("=" * 75)
    print(f" • 총 학습 스텝: {total_timesteps:,} steps")
    print(f" • 병렬 환경: {n_envs if not gui else 1}개 Subproc/Vectorized Envs")
    print(f" • 추가 질량: 눈 사이 60g 철판 + 상체 250g 무게추 (모터 부하 65% 절감)")
    print(f" • 동작 원리: 넘어지면 즉시 리셋 후 다음 에피소드로 연속 도전!")
    print(f" • 특화 보상: 머리 고정 + CPG 전진 보행 + 좌우 체중이동(Sway) + 발 교대 체공(Air-Time)")
    print("=" * 75)

    if gui:
        env = gym.make("OpenDuckWalk-v0", render_mode="human")
        env = Monitor(env)
    else:
        env = make_vec_env("OpenDuckWalk-v0", n_envs=n_envs)

    policy_kwargs = dict(
        net_arch=dict(pi=[128, 128], vf=[256, 256]),
        activation_fn=torch.nn.Tanh,
    )

    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        verbose=0,
    )

    callback = SuperFastDuckCallback(check_freq=4096, save_path="models")

    start_wall_time = time.time()
    try:
        model.learn(total_timesteps=total_timesteps, callback=callback)
    except KeyboardInterrupt:
        print("\n[사용자 중단] 학습이 사용자에 의해 중단되었습니다.")
    finally:
        total_time = time.time() - start_wall_time
        final_path = os.path.join("models", "open_duck_final.zip")
        final_pt = os.path.join("models", "open_duck_final.pt")
        try:
            model.save(final_path)
            torch.save(model.policy.state_dict(), final_pt)
        except Exception:
            pass
        print("=" * 75)
        print(f"🏁 [학습 완료] 총 소요 시간: {total_time:.1f}초 ({total_time/60:.2f}분)")
        print(f"   • 최종 모델 저장: {final_path} / {final_pt}")
        print(f"   • 최고 기록 모델: models/open_duck_best_balance.zip / .pt")
        print("=" * 75)
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open Duck Mini v2 PPO Super-Fast Training")
    parser.add_argument("--gui", action="store_true", help="3D 실시간 뷰어 활성화 (1개 환경)")
    parser.add_argument("--no-gui", action="store_true", help="3D 화면 없이 백그라운드 초고속 학습")
    parser.add_argument("--steps", type=int, default=150000, help="총 학습 스텝 수")
    parser.add_argument("--envs", type=int, default=4, help="병렬 환경 수")
    parser.add_argument("--lr", type=float, default=3e-4, help="학습률")
    args = parser.parse_args()

    train(gui=args.gui, total_timesteps=args.steps, n_envs=args.envs, lr=args.lr)
