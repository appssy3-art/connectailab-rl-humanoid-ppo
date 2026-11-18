import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco

try:
    import mujoco.viewer
except ImportError:
    pass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
XML_PATH = os.path.join(CURRENT_DIR, "Open_Duck_Playground", "playground", "open_duck_mini_v2", "xmls", "scene_flat_terrain.xml")

class OpenDuckEnv(gym.Env):
    """
    Open Duck Mini v2 Gymnasium 강화학습 환경
    - 200g 눈 사이 철판 + 700g 가슴 철판 물리 모델 반영
    - 좌우 체중 이동(Lateral Sway) + 발 교대 체공(Air-Time) + 머리 기립 제어
    - "넘어지면 새로 계속시작해" 즉시 리셋 로직
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None, sim_dt=0.002, decimation=10, max_episode_steps=1000):
        super().__init__()
        self.render_mode = render_mode
        self.sim_dt = sim_dt
        self.decimation = decimation
        self.ctrl_dt = sim_dt * decimation  # 0.02초 (50 Hz)
        self.max_episode_steps = max_episode_steps

        orig_cwd = os.getcwd()
        xml_dir = os.path.dirname(XML_PATH)
        xml_file = os.path.basename(XML_PATH)
        try:
            os.chdir(xml_dir)
            self.model = mujoco.MjModel.from_xml_path(xml_file)
            self.data = mujoco.MjData(self.model)
        finally:
            os.chdir(orig_cwd)

        self.model.opt.timestep = self.sim_dt

        # 관절 및 액추에이터 정보
        self.nu = self.model.nu  # 14
        self.nq = self.model.nq  # 21
        self.nv = self.model.nv  # 20

        # 기본 홈 자세 (keyframe 0: 'home')
        self.home_qpos = self.model.key_qpos[0].copy()
        self.home_ctrl = self.model.key_ctrl[0].copy()

        # 관절 제어 범위 및 스케일
        self.action_scale = 0.25
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.nu,), dtype=np.float32
        )

        # 관측 공간 (55차원)
        # 1. 기립 방향 벡터 upvector (3)
        # 2. 로컬 각속도 gyro (3)
        # 3. 전진/횡방향 로컬 선속도 linvel (3)
        # 4. 홈 자세 대비 관절 각도 오차 qpos_err (14)
        # 5. 관절 각속도 qvel (14)
        # 6. 이전 행동 last_action (14)
        # 7. 보행 위상 sin/cos (2)
        # 8. 양발 지면 접촉 센서 (2)
        obs_dim = 3 + 3 + 3 + 14 + 14 + 14 + 2 + 2
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(obs_dim,), dtype=np.float32
        )

        # 센서 및 바디 인덱스 조회
        self.sensor_gyro_adr = self.model.sensor_adr[0]
        self.sensor_local_linvel_adr = self.model.sensor_adr[1]
        self.sensor_upvector_adr = self.model.sensor_adr[3]

        self.floor_body_id = self.model.body("floor").id
        self.left_foot_body_id = self.model.body("foot_assembly").id
        self.right_foot_body_id = self.model.body("foot_assembly_2").id

        self.viewer = None
        self.step_count = 0
        self.phase = 0.0
        self.gait_freq = 1.25  # 1.25 Hz 보행 클럭
        self.last_action = np.zeros(self.nu, dtype=np.float32)
        self.prev_ctrl = self.home_ctrl.copy()

        # 발 접촉 및 체공 시간 추적 (Air-time)
        self.feet_air_time = np.zeros(2, dtype=np.float32)  # [left, right]
        self.last_feet_contact = np.ones(2, dtype=bool)
        self.prev_x = 0.0

    def _get_feet_contacts(self):
        left_contact = False
        right_contact = False
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            if (b1 == self.floor_body_id and b2 == self.left_foot_body_id) or \
               (b2 == self.floor_body_id and b1 == self.left_foot_body_id):
                left_contact = True
            elif (b1 == self.floor_body_id and b2 == self.right_foot_body_id) or \
                 (b2 == self.floor_body_id and b1 == self.right_foot_body_id):
                right_contact = True
        return left_contact, right_contact

    def _get_obs(self, left_contact, right_contact):
        upvector = self.data.sensordata[self.sensor_upvector_adr:self.sensor_upvector_adr+3].astype(np.float32)
        gyro = (self.data.sensordata[self.sensor_gyro_adr:self.sensor_gyro_adr+3] * 0.25).astype(np.float32)
        local_linvel = self.data.sensordata[self.sensor_local_linvel_adr:self.sensor_local_linvel_adr+3].astype(np.float32)

        joint_pos_err = (self.data.qpos[7:21] - self.home_qpos[7:21]).astype(np.float32)
        joint_vel = (self.data.qvel[6:20] * 0.05).astype(np.float32)

        phase_obs = np.array([math.sin(self.phase), math.cos(self.phase)], dtype=np.float32)
        contact_obs = np.array([1.0 if left_contact else -1.0, 1.0 if right_contact else -1.0], dtype=np.float32)

        obs = np.concatenate([
            upvector,
            gyro,
            local_linvel,
            joint_pos_err,
            joint_vel,
            self.last_action,
            phase_obs,
            contact_obs
        ]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)

        # 자연스러운 초기 기립 섭동 노이즈
        if self.np_random is not None:
            self.data.qpos[0] += self.np_random.uniform(-0.01, 0.01)
            self.data.qpos[1] += self.np_random.uniform(-0.01, 0.01)
            noise_joint = self.np_random.uniform(-0.04, 0.04, size=self.nu)
            self.data.qpos[7:21] += noise_joint
            self.data.ctrl[:] = self.home_ctrl + noise_joint

        mujoco.mj_forward(self.model, self.data)

        self.step_count = 0
        self.phase = 0.0
        self.last_action = np.zeros(self.nu, dtype=np.float32)
        self.prev_ctrl = self.data.ctrl.copy()
        self.feet_air_time = np.zeros(2, dtype=np.float32)
        self.last_feet_contact = np.ones(2, dtype=bool)
        self.prev_x = float(self.data.qpos[0])

        left_contact, right_contact = self._get_feet_contacts()

        if self.render_mode == "human":
            self.render()

        return self._get_obs(left_contact, right_contact), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        self.step_count += 1
        self.phase = (self.phase + 2.0 * math.pi * self.gait_freq * self.ctrl_dt) % (2.0 * math.pi)

        # 1. CPG 보행 리듬 (기본 오리 발걸음 및 좌우 체중이동 모멘텀)
        cpg = np.zeros(self.nu, dtype=np.float32)
        sway = 0.10 * math.sin(self.phase)
        cpg[1] = -sway   # left_hip_roll (좌우 체중이동)
        cpg[10] = -sway  # right_hip_roll

        # 전진 스윙: 왼쪽 +가 전진, 오른쪽 -가 전진 (물리 기하학 정확 일치!)
        swing = 0.22 * math.sin(self.phase)
        cpg[2] = swing                      # left_hip_pitch (+ 전진)
        cpg[11] = -swing                    # right_hip_pitch (- 전진)
        cpg[3] = 0.28 * max(0.0, swing)     # left_knee lift (확실한 발 들어올림 ~6cm)
        cpg[12] = 0.28 * max(0.0, -swing)   # right_knee lift
        cpg[4] = -0.12 * max(0.0, swing)    # left_ankle
        cpg[13] = -0.12 * max(0.0, -swing)  # right_ankle

        # 2. 강화학습 잔차 제어 (Residual RL) + 저역통과 필터 (EMA 65:35, 다리 후들거림 원천 차단!)
        raw_target = self.home_ctrl + cpg + self.action_scale * action
        # 머리/목 관절 꼿꼿이 고정 ("고개만 까닥까닥" 꼼수 원천 봉쇄!)
        raw_target[5:9] = self.home_ctrl[5:9]

        filtered_ctrl = 0.65 * self.prev_ctrl + 0.35 * raw_target

        # 3. STS3215 모터 각속도 한계(스텝당 0.09 rad) 클리핑 (급격한 관절 떨림 방지)
        filtered_ctrl = np.clip(filtered_ctrl, self.prev_ctrl - 0.09, self.prev_ctrl + 0.09)
        self.prev_ctrl = filtered_ctrl.copy()
        self.data.ctrl[:] = filtered_ctrl

        # 50Hz 제어를 위해 10번의 물리 서브스텝 (sim_dt = 0.002s)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

        # 발 접촉 상태 확인
        left_contact, right_contact = self._get_feet_contacts()
        current_contacts = np.array([left_contact, right_contact], dtype=bool)

        # 상태 및 센서 데이터 추출
        obs = self._get_obs(left_contact, right_contact)
        upvector_z = float(self.data.sensordata[self.sensor_upvector_adr + 2])
        trunk_z = float(self.data.qpos[2])  # 몸체 높이
        current_x = float(self.data.qpos[0])
        forward_progress = max(0.0, current_x - self.prev_x)
        self.prev_x = current_x

        local_vx = float(self.data.sensordata[self.sensor_local_linvel_adr])
        local_vy = float(self.data.sensordata[self.sensor_local_linvel_adr + 1])
        gyro_z = float(self.data.sensordata[self.sensor_gyro_adr + 2])

        neck_pitch = float(self.data.qpos[12])
        head_pitch = float(self.data.qpos[13])

        # =====================================================================
        # 🎯 밸런스 & 보행 보상 함수 (1. 기립 + 2. 좌우 체중이동 + 3. 적극적 전진 보행)
        # =====================================================================
        # 1. 꼿꼿한 기립 보상 (Upright bonus)
        r_upright = 3.0 * max(0.0, upvector_z)

        # 2. 적정 높이 유지 보상 (z ≈ 0.16m)
        r_height = 2.0 * math.exp(-40.0 * (trunk_z - 0.16) ** 2)

        # 3. 생존 보너스 대폭 축소 (제자리 서 있기 꼼수 차단!)
        r_alive = 0.05

        # 4. 적극적 전진 보행 속도 & 거리 전진 보상 (Target speed 0.22 m/s)
        # - 목표 속도 0.22 m/s 추종 가우시안 보상
        r_vel = 4.0 * math.exp(-25.0 * (local_vx - 0.22) ** 2)
        # - 앞으로 나아갈 때마다 전진 거리에 비례하여 강력한 전진 보너스 (+120점/m)
        r_progress = 120.0 * forward_progress
        r_forward = r_vel + r_progress

        # 5. 좌우 체중 이동 유도 (Lateral Sway, 오리 걸음 리듬 1.25 Hz)
        target_vy = 0.08 * math.sin(self.phase)
        r_sway = 1.5 * math.exp(-30.0 * (local_vy - target_vy) ** 2)

        # 6. 발 교대 체공 보상 (Air-time / Stepping reward)
        # 발을 번갈아 들어올려 딛을 때 큰 보너스 부여
        r_air_time = 0.0
        for i in range(2):
            if current_contacts[i]:
                # 방금 착지한 경우: 적절한 체공 시간(0.06초 ~ 0.35초)이었을 때 보상
                if not self.last_feet_contact[i] and self.feet_air_time[i] > 0.06:
                    r_air_time += 6.0 * min(self.feet_air_time[i], 0.35)
                self.feet_air_time[i] = 0.0
            else:
                # 공중에 떠 있는 동안 체공 시간 누적
                self.feet_air_time[i] += self.ctrl_dt
        self.last_feet_contact = current_contacts.copy()

        # - 양발 모두 땅에 붙이고 가만히 서 있을 때 감점 (제자리 정지 페널티)
        p_stand_still = -0.8 if (left_contact and right_contact) else 0.0

        # =====================================================================
        # ⚠️ 감점 요인 (페널티: 후진 방지 + 모터 부하 + 관절 떨림)
        # =====================================================================
        # - 뒤로 후진하거나 뒤로 밀리는 것 강력 처벌 (뒤로 넘어짐 방지)
        p_backward = -15.0 * max(0.0, -local_vx)

        # - 모터 과다 사용 처벌 (토크 부하 페널티)
        motor_torque_sq = float(np.sum(np.square(self.data.actuator_force)))
        p_torque = -0.002 * motor_torque_sq

        # - 관절 급격한 떨림 억제 (모터 기어 마모 방지)
        action_diff = action - self.last_action
        p_smooth = -0.05 * float(np.sum(action_diff ** 2))

        # - 머리가 앞으로 꺾여 쏠리는 것 억제 (목/머리 피치 제어)
        p_head = -1.8 * (max(0.0, -neck_pitch) ** 2 + max(0.0, -head_pitch) ** 2)

        # - 제자리 회전 억제
        p_spin = -0.5 * (gyro_z ** 2)

        # - 불필요한 과도한 액션 억제
        p_energy = -0.01 * float(np.sum(action ** 2))

        reward = (
            r_upright + r_height + r_alive + r_forward + r_sway + r_air_time +
            p_stand_still + p_backward + p_torque + p_smooth + p_head + p_spin + p_energy
        )
        self.last_action = action.copy()

        # =====================================================================
        # 💥 넘어짐 체크 ("넘어지면 새로 계속시작해")
        # =====================================================================
        fallen = (trunk_z < 0.12) or (upvector_z < 0.65)
        terminated = bool(fallen)
        truncated = bool(self.step_count >= self.max_episode_steps)

        if terminated:
            reward -= 10.0  # 넘어짐 강한 페널티 (-10점)

        info = {
            "upvector_z": upvector_z,
            "trunk_z": trunk_z,
            "forward_vx": local_vx,
            "forward_dist": current_x,
            "fallen": fallen,
            "step_count": self.step_count,
            "terms": {
                "upright": r_upright,
                "height": r_height,
                "alive": r_alive,
                "forward": r_forward,
                "sway": r_sway,
                "air_time": r_air_time,
                "head": p_head,
                "smooth": p_smooth
            }
        }

        if self.render_mode == "human":
            self.render()

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                self.viewer.cam.distance = 1.8
                self.viewer.cam.elevation = -18.0
                self.viewer.cam.azimuth = 135.0
                self.viewer.cam.lookat[:] = [0.2, 0.0, 0.16]
            if self.viewer.is_running():
                self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

# Gymnasium 환경 등록
gym.register(
    id="OpenDuckWalk-v0",
    entry_point=OpenDuckEnv,
    max_episode_steps=1000,
)
