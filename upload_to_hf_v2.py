import os, json, datetime, shutil
import torch
from huggingface_hub import HfApi
from 로드스터_시뮬레이터 import 로드스터연속
from run_ppo_task import 정책두뇌, 착지율

api = HfApi()
나 = api.whoami().get('name', 'appssy3')
저장소 = 'connectailab-rl-solo-build_H-roadster-ppo-v2'
올린주소 = f'{나}/{저장소}'
폴더 = f'hf_{저장소}'
os.makedirs(폴더, exist_ok=True)

뇌 = 정책두뇌(7, 2, 연속=True)
뇌.load_state_dict(torch.load('model.pt', map_location='cpu'))
착지점수 = 착지율(뇌, 로드스터연속, 판=50)

torch.save(뇌.state_dict(), f'{폴더}/model.pt')
shutil.copy('로드스터_시뮬레이터.py', f'{폴더}/로드스터_시뮬레이터.py')
if os.path.exists('roadster_ppo_joystick.mp4'):
    shutil.copy('roadster_ppo_joystick.mp4', f'{폴더}/replay.mp4')
if os.path.exists('learning_curve.png'):
    shutil.copy('learning_curve.png', f'{폴더}/learning_curve.png')

설정 = {
    'algo': 'PPO',
    'steps': 1_000_000,
    'envs': 10,
    'horizon': 256,
    'clip': 0.1,
    'lr': 3e-4,
    'entropy_coef': 0.01,
    'epochs': 10,
    'gamma': 0.99,
    'gae_lambda': 0.95
}

json.dump({
    'format': 'connectailab/1',
    'repo': 올린주소,
    'kind': 'ppo',
    'env': 'Inverted Roadster (continuous)',
    'algo': 'ppo',
    'version': 'v2',
    'config': 설정,
    'test': {'landing_rate_50': 착지점수},
    'course': 'AI CITY BUILDERS 강화학습 AI 6강',
    'date': str(datetime.date.today())
}, open(f'{폴더}/connectailab.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

카드 = f"""---
tags: [reinforcement-learning, ppo, continuous-control, connect-ai-lab, connectailab]
library_name: pytorch
---
# {저장소}

뒤집힌 채로 떨어지는 로드스터를 추진기 **세기(0 ~ 1)와 회전(−1 ~ 1)** 으로 제어하여 바퀴부터 안전하게 착지시키는 **PPO(Proximal Policy Optimization) 100만 걸음 (v2)** 에이전트입니다.

![학습 과정 그래프](learning_curve.png)

## 📊 v2 (100만 걸음 마스터) 학습 결과 및 하이퍼파라미터

| 항목 | 내용 |
|---|---|
| 버전 | **v2 (1,000,000 걸음 심화 학습)** |
| 환경 | 뒤집힌 로드스터 연속 제어 (`로드스터연속`, Gymnasium) |
| 상태 차원 | 위치 x·y, 속도 x·y, 기울기 sin·cos, 회전 각속도 (총 7차원) |
| 행동 차원 | [주 추진기 세기 (0~1), 회전 추진기 세기 (-1~1)] (연속 2차원) |
| **최종 평가 착지율** | **처음 보는 50판 착지율 {착지점수:.0f}%** |
| 클리핑 (자르기) | 0.1 (±10%) |
| 병렬 세상수 | 10개 |
| 총 학습 걸음수 | **1,000,000 걸음** (약 1분 49초 소요) |
| 롤아웃 길이 (한번에) | 256 |
| 학습률 (Learning Rate) | 3e-4 |
| 탐험 보너스 (Entropy) | 0.01 |
| PPO 업데이트 반복 에포크 | 10회 |

AI CITY BUILDERS 강화학습 AI 6강 · CONNECT AI LAB
"""
open(f'{폴더}/README.md', 'w', encoding='utf-8').write(카드)

api.create_repo(저장소, exist_ok=True)
api.upload_folder(folder_path=폴더, repo_id=올린주소)
print(f"🎉 [v2 업로드 완료] -> https://huggingface.co/{올린주소}")
print(f"   (50판 시험 착지율: {착지점수:.0f}%)")
