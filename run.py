import os
import sys
import time
import numpy as np
import gymnasium as gym
from rolling_ball_env import RollingBallEnv

# 윈도우 한글/특수문자 및 표준 출력 flush 설정
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

def main():
    print("=" * 68)
    print(" [Gymnasium] MuJoCo 공모양 에이전트 강화학습 환경 (무한 실시간 시뮬레이션)")
    print("=" * 68)
    print(" * 보상 규칙: 목적지 도착 시 [+10점] (성공 & 새 에피소드)")
    print(" * 처벌 규칙: 경기장 밖 이탈 시 [-5점] (처벌 & 리셋)")
    print(" * 뷰어 종료: 3D 창의 [X] 버튼을 누르거나 터미널에서 Ctrl+C")
    print("=" * 68)

    # 1. Gymnasium 환경 생성 (human 렌더 모드)
    env = RollingBallEnv(render_mode="human", max_episode_steps=600)

    episode = 1
    total_score = 0.0

    try:
        # 창이 닫힐 때까지 무한 반복 실행
        while True:
            obs, info = env.reset()
            episode_reward = 0.0
            step_count = 0
            start_dist = info.get("initial_dist", 0.0)

            print(f"\n[Episode {episode}] 시작 | 초기 목표 거리: {start_dist:.2f}m")

            while True:
                step_start = time.time()

                # 사용자가 3D 창의 X 버튼을 눌러 닫았는지 확인
                if env.viewer is not None and not env.viewer.is_running():
                    print("\n3D 뷰어 창이 닫혔습니다.")
                    return

                # 에이전트 액션 결정 (지능형 내비게이션 토크)
                rel_x = obs[0]
                rel_y = obs[1]
                dist = obs[2]

                # 목표 지점을 향한 비례 제어 액션 ([-1, 1] 범위)
                if dist > 0.01:
                    # 목적지 방향 단위 벡터
                    action = np.array([rel_x / dist, rel_y / dist], dtype=np.float32)
                else:
                    action = np.zeros(2, dtype=np.float32)

                # 5% 확률로 약한 노이즈 (자연스러운 주행 및 장외 탈락 시험)
                if np.random.rand() < 0.03:
                    action += np.random.uniform(-0.4, 0.4, size=2).astype(np.float32)
                    action = np.clip(action, -1.0, 1.0)

                # 2. Gymnasium 표준 step 호출
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                step_count += 1

                # 3. 보상 및 처벌 이벤트 처리
                if terminated:
                    if info.get("is_success"):
                        total_score += 10.0
                        print(f"  ★ [목적지 도착!] 보상 +10.0점 획득! (누적 점수: {total_score:.0f}점, 스텝: {step_count})")
                    elif info.get("out_of_bounds"):
                        total_score -= 5.0
                        print(f"  X [경기장 밖 이탈!] 처벌 -5.0점 적용! (누적 점수: {total_score:.0f}점, 스텝: {step_count})")
                    
                    # 도착/이탈 시 시각적 확인을 위해 0.25초 동안 잠시 대기
                    for _ in range(15):
                        if env.viewer is not None and env.viewer.is_running():
                            env.viewer.sync()
                            time.sleep(1.0 / 60.0)
                    break

                if truncated:
                    print(f"  - [시간 초과] 600스텝 도달. 에피소드 재시작.")
                    break

                # 60 FPS 프레임 레이트 동기화
                elapsed = time.time() - step_start
                sleep_time = (1.0 / 60.0) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            episode += 1

    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.")
    finally:
        env.close()
        print("\n환경이 정상 종료되었습니다.")

if __name__ == "__main__":
    main()
