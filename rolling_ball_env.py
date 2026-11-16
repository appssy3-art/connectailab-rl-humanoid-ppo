import os
import sys
import math
import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

class RollingBallEnv(gym.Env):
    """
    MuJoCo 기반 공모양 에이전트 목표 추적 강화학습 환경 (고속 고효율 PPO 보상 체계)
    
    [개선된 보상 및 처벌 체계]
    1. 목적지 도착 보상:
       - 기본 성공 보상: +15.0점
       - 신속 도달 보너스: 최대 +5.0점 (스텝 수에 반비례) -> 총 최대 +20점
    2. 장외 이탈 및 낙하 처벌:
       - 장외 탈락 처벌: -15.0점 (탈락 억제 강화)
       - 경계 접근 위험 감점: 테두리 4.0m 이상 접근 시 감점 (낙하 방지 유도)
    3. 연속 진행 보상 (Dense Shaping):
       - 목표 지향 전진 속도 보상: v_vector와 goal_dir의 내적 비례
       - 거리 단축 보상: (prev_dist - curr_dist) * 3.0
       - 시간 효율 페널티: 매 스텝 -0.01 (최단 시간 경로 학습)
       - 액션 부드러움 페널티: -0.002 * ||action||^2 (과도한 떨림 방지)
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        xml_path=None,
        render_mode=None,
        max_episode_steps=500,
        arena_size=5.0,
        danger_zone=4.0,
        goal_threshold=0.38,
        max_torque=6.0,
    ):
        super().__init__()

        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.arena_size = arena_size
        self.danger_zone = danger_zone
        self.goal_threshold = goal_threshold
        self.max_torque = max_torque

        # XML 파일 경로 확인
        if xml_path is None:
            curr_dir = os.path.dirname(os.path.abspath(__file__))
            xml_path = os.path.join(curr_dir, "world.xml")
        self.xml_path = xml_path

        # Windows 비ASCII 경로 대응
        model_dir = os.path.dirname(os.path.abspath(self.xml_path))
        orig_cwd = os.getcwd()
        try:
            os.chdir(model_dir)
            self.model = mujoco.MjModel.from_xml_path(os.path.basename(self.xml_path))
        finally:
            os.chdir(orig_cwd)

        self.data = mujoco.MjData(self.model)

        # Mocap 타겟 ID 확인
        self.mocap_id = self.model.body("target").mocapid[0]

        # 1. Action Space: 2차원 연속 토크 제어 [-1.0, 1.0]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )

        # 2. Observation Space: 10차원 상태 벡터
        low_obs = np.array([-15.0, -15.0, 0.0, -15.0, -15.0, -50.0, -50.0, -100.0, -100.0, -100.0], dtype=np.float32)
        high_obs = np.array([15.0, 15.0, 30.0, 15.0, 15.0, 50.0, 50.0, 100.0, 100.0, 100.0], dtype=np.float32)
        self.observation_space = spaces.Box(
            low=low_obs, high=high_obs, dtype=np.float32
        )

        # 내부 추적 변수
        self.step_count = 0
        self.target_pos = np.zeros(3, dtype=np.float64)
        self.prev_dist = 0.0

        # 렌더링 뷰어
        self.viewer = None

    def _get_obs(self):
        ball_pos = self.data.qpos[0:3]
        ball_linvel = self.data.qvel[0:3]
        ball_angvel = self.data.qvel[3:6]

        rel_pos = self.target_pos[0:2] - ball_pos[0:2]
        dist = np.linalg.norm(rel_pos)

        obs = np.array([
            rel_pos[0],
            rel_pos[1],
            dist,
            ball_pos[0],
            ball_pos[1],
            ball_linvel[0],
            ball_linvel[1],
            ball_angvel[0],
            ball_angvel[1],
            ball_angvel[2],
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0

        if seed is not None:
            np.random.seed(seed)
        
        # 에이전트 초기 위치: 중앙 부근 (-0.5 ~ +0.5)
        init_x = float(np.random.uniform(-0.5, 0.5))
        init_y = float(np.random.uniform(-0.5, 0.5))
        self.data.qpos[0] = init_x
        self.data.qpos[1] = init_y
        self.data.qpos[2] = 0.26
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # 목표물 위치 무작위 배치 (반경 1.8m ~ 3.6m)
        angle = float(np.random.uniform(0, 2 * math.pi))
        radius = float(np.random.uniform(1.8, 3.6))
        self.target_pos[0] = radius * math.cos(angle)
        self.target_pos[1] = radius * math.sin(angle)
        self.target_pos[2] = 0.20
        self.data.mocap_pos[self.mocap_id] = self.target_pos

        mujoco.mj_forward(self.model, self.data)

        self.prev_dist = float(np.linalg.norm(self.target_pos[0:2] - np.array([init_x, init_y])))

        obs = self._get_obs()
        info = {
            "target_pos": self.target_pos.copy(),
            "initial_dist": self.prev_dist,
        }

        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action):
        self.step_count += 1

        # 액션 클리핑 및 롤링 토크 적용
        act = np.clip(action, self.action_space.low, self.action_space.high)
        torque_x = -float(act[1]) * self.max_torque
        torque_y = float(act[0]) * self.max_torque

        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[3] = torque_x
        self.data.qfrc_applied[4] = torque_y

        # 물리 스텝 진행
        for _ in range(2):
            mujoco.mj_step(self.model, self.data)

        # 상태 추출
        obs = self._get_obs()
        rel_x, rel_y = obs[0], obs[1]
        curr_dist = float(obs[2])
        ball_x, ball_y = float(obs[3]), float(obs[4])
        ball_vx, ball_vy = float(obs[5]), float(obs[6])
        ball_z = float(self.data.qpos[2])

        reward = 0.0
        terminated = False
        truncated = False
        info = {
            "dist": curr_dist,
            "step": self.step_count,
            "is_success": False,
            "out_of_bounds": False,
        }

        # -----------------------------------------------------------------
        # 1. 목적지 도착 보상 (대형 보너스 + 시간 효율성 보너스)
        # -----------------------------------------------------------------
        if curr_dist <= self.goal_threshold:
            # 기본 +15점 + 빠르게 도착할수록 최대 +5점 추가 (총 최대 +20점)
            speed_bonus = max(0.0, (1.0 - (self.step_count / self.max_episode_steps))) * 5.0
            reward = 15.0 + speed_bonus
            terminated = True
            info["is_success"] = True

        # -----------------------------------------------------------------
        # 2. 장외 이탈 및 추락 처벌 (강력한 페널티)
        # -----------------------------------------------------------------
        elif abs(ball_x) > self.arena_size or abs(ball_y) > self.arena_size or ball_z < 0.1:
            reward = -15.0
            terminated = True
            info["out_of_bounds"] = True

        # -----------------------------------------------------------------
        # 3. 진행 중 보상 (학습 속도 10배 향상을 위한 인공지능 네비게이션 가이드)
        # -----------------------------------------------------------------
        else:
            # (1) 목표 지향 속도 보상 (Velocity Alignment)
            # 공의 속도 벡터가 목표를 향하고 있으면 큰 보상 부여
            if curr_dist > 0.05:
                dir_x = rel_x / curr_dist
                dir_y = rel_y / curr_dist
                forward_speed = ball_vx * dir_x + ball_vy * dir_y
                reward += forward_speed * 0.4  # 목표 방향 전진 시 비례 보상

            # (2) 거리 단축 보상 (Potential-based Distance Shaping)
            dist_improvement = self.prev_dist - curr_dist
            reward += dist_improvement * 2.5

            # (3) 경계선 접근 경고 감점 (낙하 방지 세이프티 가이드)
            max_coord = max(abs(ball_x), abs(ball_y))
            if max_coord > self.danger_zone:
                # 4.0m 넘어가면 경계에 가까워질수록 점진적 감점
                danger_depth = (max_coord - self.danger_zone) / (self.arena_size - self.danger_zone)
                reward -= danger_depth * 0.15

            # (4) 시간 소모 미세 페널티 (최단 거리 주행 유도)
            reward -= 0.01

            # (5) 과도한 토크 진동 억제 (부드러운 주행 유도)
            reward -= 0.002 * float(np.sum(np.square(act)))

        self.prev_dist = curr_dist

        # 최대 스텝 도달 시 중단 (truncated)
        if self.step_count >= self.max_episode_steps and not terminated:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return

        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 11.0
                self.viewer.cam.elevation = -35.0
                self.viewer.cam.azimuth = 90.0
                self.viewer.cam.lookat[:] = [0.0, 0.0, 0.25]

            if self.viewer.is_running():
                target_x = float(np.clip(self.data.qpos[0] * 0.4 + self.target_pos[0] * 0.6, -2.5, 2.5))
                target_y = float(np.clip(self.data.qpos[1] * 0.4 + self.target_pos[1] * 0.6, -2.5, 2.5))
                self.viewer.cam.lookat[0] = target_x
                self.viewer.cam.lookat[1] = target_y
                self.viewer.cam.lookat[2] = 0.25
                self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None

gym.register(
    id="RollingBall-v0",
    entry_point="rolling_ball_env:RollingBallEnv",
    max_episode_steps=500,
)
