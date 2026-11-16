import os
import sys
import time
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from rolling_ball_env import RollingBallEnv

# 콘솔 실시간 출력
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

class AdvancedEpisodeTracker(BaseCallback):
    """
    고성능 1,000 에피소드 실시간 모니터링 콜백
    """
    def __init__(self, target_episodes=1000, verbose=1):
        super().__init__(verbose)
        self.target_episodes = target_episodes
        self.episode_count = 0
        self.curr_reward = 0.0
        self.curr_steps = 0
        self.success_count = 0
        self.oob_count = 0
        self.recent_rewards = []
        self.recent_success = []
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        print("\n" + "=" * 72)
        print(" 🚀 [개선된 PPO 강화학습 1,000 에피소드 3D 실시간 렌더 학습]")
        print("=" * 72)
        print(" [개선된 보상 & 처벌 시스템 적용 완료]")
        print("  1. 목적지 도착: 기본 +15점 + 신속 도달 보너스(최대 +5점) -> 최대 +20.0점")
        print("  2. 장외 이탈:   -15.0점 (강력 처벌로 장외 추락 원천 억제)")
        print("  3. 전진 가이드:  목표 방향 전진 속도 비례 보상 (Velocity Alignment)")
        print("  4. 경계 경고:    테두리 4.0m 이상 접근 시 감점 (낙하 방지 버퍼존)")
        print("=" * 72 + "\n")

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos")

        if rewards is not None:
            self.curr_reward += float(rewards[0])
            self.curr_steps += 1

        # 시각적 관찰을 위한 미세 렌더 딜레이 (약 90 FPS)
        time.sleep(0.004)

        # 3D 뷰어 닫힘 감지
        env = self.training_env.envs[0]
        if hasattr(env, "viewer") and env.viewer is not None:
            if not env.viewer.is_running():
                print("\n[알림] 3D 뷰어 창이 닫혀 학습을 조기 종료합니다.")
                return False

        # 에피소드 종료 처리
        if dones is not None and dones[0]:
            self.episode_count += 1
            info = infos[0] if (infos and len(infos) > 0) else {}

            is_success = info.get("is_success", False)
            out_of_bounds = info.get("out_of_bounds", False)

            if is_success:
                self.success_count += 1
                self.recent_success.append(1)
                status_str = "★ [도착 성공!]"
            elif out_of_bounds:
                self.oob_count += 1
                self.recent_success.append(0)
                status_str = "X [장외 이탈!]"
            else:
                self.recent_success.append(0)
                status_str = "- [시간 초과]"

            self.recent_rewards.append(self.curr_reward)
            if len(self.recent_rewards) > 30:
                self.recent_rewards.pop(0)
                self.recent_success.pop(0)

            avg_reward = np.mean(self.recent_rewards)
            recent_acc = np.mean(self.recent_success) * 100.0
            elapsed = time.time() - self.start_time

            print(
                f"[EP {self.episode_count:4d}/{self.target_episodes}] {status_str:<12} "
                f"보상: {self.curr_reward:6.2f} (최근평균: {avg_reward:6.2f}) | "
                f"최근 성공률: {recent_acc:5.1f}% | 스텝: {self.curr_steps:3d} | "
                f"경과: {elapsed:.0f}s"
            )

            self.curr_reward = 0.0
            self.curr_steps = 0

            if self.episode_count >= self.target_episodes:
                print(f"\n🎉 [1,000 에피소드 학습 완료!] 누적 성공 에피소드: {self.success_count}회")
                return False

        return True

def main():
    # 3D human 렌더 모드
    env = RollingBallEnv(render_mode="human", max_episode_steps=500)

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
    )

    callback = AdvancedEpisodeTracker(target_episodes=1000)

    try:
        model.learn(total_timesteps=1_000_000, callback=callback)
        save_path = "ppo_rolling_ball_1000ep.zip"
        model.save(save_path)
        print("=" * 72)
        print(f"💾 최적화된 모델 가중치가 저장되었습니다: {save_path}")
        print("=" * 72)
    except KeyboardInterrupt:
        print("\n[중단] 모델을 임시 저장합니다.")
        model.save("ppo_rolling_ball_interrupted.zip")
    finally:
        env.close()

if __name__ == "__main__":
    main()
