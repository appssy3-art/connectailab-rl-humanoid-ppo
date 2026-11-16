import os
import sys
import time
import math
import numpy as np
import mujoco
import mujoco.viewer

# 윈도우 표준 출력 플러시
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_DIR = os.path.join(CURRENT_DIR, "Open_Duck_Mini", "mini_bdx", "robots", "open_duck_mini_v2")

# 디즈니 BDX 오리 로봇 표준 기립 관절 각도 (16개 서보 모터)
INIT_POS = np.array([
    0.002,   # 0: left_hip_yaw
    0.053,   # 1: left_hip_roll
    -0.630,  # 2: left_hip_pitch
    1.368,   # 3: left_knee
    -0.784,  # 4: left_ankle
    0.000,   # 5: neck_pitch
    0.000,   # 6: head_pitch
    0.000,   # 7: head_yaw
    0.000,   # 8: head_roll
    0.000,   # 9: left_antenna
    0.000,   # 10: right_antenna
    -0.003,  # 11: right_hip_yaw
    -0.065,  # 12: right_hip_roll
    0.635,   # 13: right_hip_pitch
    1.379,   # 14: right_knee
    -0.796,  # 15: right_ankle
], dtype=np.float64)

def main():
    print("=" * 68)
    print(" 🦆 [Open Duck Mini v2] 디즈니 BDX 오리 로봇 3D 시뮬레이터 실행 ")
    print("=" * 68)
    print(" * 모델: Open Duck Mini v2 (Disney BDX 42cm Bipedal Droid)")
    print(" * 모터 수: 16 DOF (서보 모터 위치 제어)")
    print(" * 조작법:")
    print("    - 마우스 좌클릭 드래그: 360도 카메라 회전")
    print("    - 마우스 휠 스크롤 / 우클릭: 줌 인 / 줌 아웃")
    print("    - 마우스 휠 클릭 드래그: 시점 이동 (Pan)")
    print("    - 스페이스바 (Space): 일시정지 / 재생")
    print("=" * 68)

    # 1. 모델 로드 (Windows 한글 경로 대응)
    orig_cwd = os.getcwd()
    try:
        os.chdir(ROBOT_DIR)
        model = mujoco.MjModel.from_xml_path("scene.xml")
        data = mujoco.MjData(model)
    finally:
        os.chdir(orig_cwd)

    # 2. 로봇 초기 위치 및 직립 자세 설정
    data.qpos[0] = 0.0     # x
    data.qpos[1] = 0.0     # y
    data.qpos[2] = 0.285   # z 높이 (발바닥이 바닥에 자연스럽게 접촉)
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # 쿼터니언 회전 (수직)
    data.qpos[7:7+16] = INIT_POS.copy()
    data.ctrl[:16] = INIT_POS.copy()

    mujoco.mj_forward(model, data)

    # 3. 3D 패시브 뷰어 실행
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.35
        viewer.cam.elevation = -18.0
        viewer.cam.azimuth = 135.0
        viewer.cam.lookat[:] = [0.0, 0.0, 0.24]

        start_time = time.time()

        while viewer.is_running():
            step_start = time.time()
            t = time.time() - start_time

            # 4. 살아있는 오리 로봇의 귀여운 아이들링(Idling) 모션 (머리 회전 & 안테나 갸우뚱)
            target_ctrl = INIT_POS.copy()
            target_ctrl[6] = 0.12 * math.sin(t * 1.8)              # head_pitch (끄덕이기)
            target_ctrl[7] = 0.25 * math.sin(t * 0.9)              # head_yaw (좌우 두리번)
            target_ctrl[8] = 0.15 * math.cos(t * 1.2)              # head_roll (갸우뚱)
            target_ctrl[9] = 0.35 * math.sin(t * 4.0)              # left_antenna (파닥이기)
            target_ctrl[10] = 0.35 * math.sin(t * 4.0 + math.pi)   # right_antenna (교차 파닥)

            data.ctrl[:16] = target_ctrl

            # 물리 엔진 한 스텝 전진
            mujoco.mj_step(model, data)

            # 카메라가 로봇 몸체를 부드럽게 주시
            viewer.cam.lookat[0] = data.qpos[0]
            viewer.cam.lookat[1] = data.qpos[1]
            viewer.cam.lookat[2] = data.qpos[2]

            viewer.sync()

            # 60 FPS 제어
            elapsed = time.time() - step_start
            sleep_time = (1.0 / 60.0) - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    print("\n[알림] 오픈덕 3D 뷰어 창이 닫혔습니다.")

if __name__ == "__main__":
    main()
