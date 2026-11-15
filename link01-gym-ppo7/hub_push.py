"""강화학습 7강 · 내가 학습한 LINK-01 걷기 두뇌를 허깅페이스에 올리기 (6강과 같은 규칙).

노트북에서:
    import hub_push
    folder, repo = hub_push.build(model_files=['my_walk.zip', 'my_vecnorm.pkl'], metrics=m,
                                  train=dict(steps=2_000_000, minutes=25), name='jay', about='...')
    hub_push.push(folder, repo)                        # 토큰은 이때 Secrets / 입력창으로만 받는다

터미널에서 (stable-baselines3 학습 폴더):
    python hub_push.py --run runs/my_walk --name jay            # 올릴 폴더만 만든다 (기본, 안전)
    python hub_push.py --run runs/my_walk --name jay --push     # 진짜 올리기

저장소 이름 = connectailab-rl-<이름>-<버전>-link01-walk-ppo  (앞부분 connectailab-rl 은 꼭 유지)
올라가는 것:
  두뇌 파일들         model.zip / vecnormalize.pkl / model.pt / actor.json … 내가 고른 파일
  replay.mp4          시험 첫 판 리플레이 영상
  connectailab.json   format connectailab/1 · 시험 숫자(거리 · 속도 · 넘어짐/분) · 학습 정보 · 소개
  README.md           모델 카드 (tags: connectailab · link01 · ppo …)
토큰은 어떤 파일에도 쓰지 않습니다.
"""
import argparse, datetime, json, os, re, shutil

TAGS = ['reinforcement-learning', 'ppo', 'robotics', 'mujoco', 'locomotion', 'link01', 'connect-ai-lab', 'connectailab']


def repo_name(name, version='v1'):
    clean = lambda s: re.sub(r'[^a-z0-9.]+', '-', str(s).lower()).strip('-.')
    n, v = clean(name), clean(version) or 'v1'
    if not n:
        raise ValueError('이름은 영문 · 숫자로 적어 주세요 (예: jay, minji, team3)')
    return f'connectailab-rl-{n}-{v}-link01-walk-ppo'


def build(model_files, metrics, train, name, version='v1', about='', owner='<내-허깅페이스-이름>',
          replay=None, out_dir='hf_upload', smoke=False, library='stable-baselines3'):
    """model_files: 올릴 두뇌 파일 경로들. metrics: eval_walk.evaluate() 결과. train: 학습 정보 dict."""
    repo = repo_name(name, version)
    folder = os.path.join(out_dir, repo)
    os.makedirs(folder, exist_ok=True)
    names = []
    for f in model_files:
        shutil.copy(f, os.path.join(folder, os.path.basename(f))); names.append(os.path.basename(f))
    replay = replay or os.path.join(os.path.dirname(metrics.get('_path', '')) or '.', 'replay.mp4')
    if os.path.exists(replay):
        shutil.copy(replay, os.path.join(folder, 'replay.mp4'))
    about = '\n'.join(l.rstrip() for l in (about or '').strip().splitlines())[:3000]
    env_id = metrics.get('env', 'Link01Walk-v0')
    test = dict(distance_m=metrics['distance_m_per_episode'], speed_mps=metrics['speed_mps'],
                falls_per_min=metrics['falls_per_min'], episodes=metrics['episodes'],
                seconds_per_episode=metrics['seconds_per_episode'], command_mps=metrics['command_mps'])
    info = {'format': 'connectailab/1', 'repo': f'{owner}/{repo}', 'kind': 'ppo', 'env': f'LINK-01 Walk ({env_id})',
            'algo': 'ppo', 'robot': 'LINK-01', 'files': names, 'author': {'about': about} if about else None,
            'config': train, 'test': test, 'course': 'AI CITY BUILDERS 강화학습 AI 7강', 'date': str(datetime.date.today())}
    if smoke:
        info['label'] = 'smoke test (not a result)'
    json.dump(info, open(os.path.join(folder, 'connectailab.json'), 'w'), ensure_ascii=False, indent=1)
    tag = 'smoke test · ' if smoke else ''
    warn = (f"> **smoke test** · 아래 숫자는 {train.get('minutes', '?')}분 · {int(train.get('steps', 0)):,}걸음짜리 "
            f"동작 확인용 짧은 실행에서 나온 값입니다. 학습 결과가 아닙니다.\n\n") if smoke else ''
    maker = f'## 만든 사람\n\n{about}\n\n<sub>CONNECT AI LAB 공동 연구 참여 모델</sub>\n\n' if about else ''
    tags = TAGS + (['stable-baselines3'] if library == 'stable-baselines3' else [])
    card = f"""---
tags: [{', '.join(tags)}]
library_name: {library}
model-index:
- name: {repo}
  results:
  - task: {{type: reinforcement-learning, name: walking}}
    dataset: {{type: {env_id}, name: {env_id}}}
    metrics:
    - {{type: distance_m, value: {test['distance_m']}, name: "{tag}distance per {test['seconds_per_episode']:.0f} s (m)"}}
    - {{type: speed_mps, value: {test['speed_mps']}, name: "{tag}mean forward speed (m/s)"}}
    - {{type: falls_per_min, value: {test['falls_per_min']}, name: "{tag}falls per minute"}}
---
# {repo}

{warn}{maker}LINK-01 두 발 로봇이 **PPO로 처음부터** 배우는 걷기 두뇌입니다.

<video src="replay.mp4" controls width="640"></video>

| 항목 | 내용 |
|---|---|
| 세상 | `{env_id}` (link01-gym, MuJoCo) |
| 관측 · 행동 | 51 · 14 (관절 14개, 기본 자세 ±0.18 rad) |
| 시험 | {tag}명령 {test['command_mps']} m/s · {test['episodes']}판 × {test['seconds_per_episode']:.0f}초 |
| 거리 | **{test['distance_m']} m / 판** |
| 속도 | **{test['speed_mps']} m/s** |
| 넘어짐 | **{test['falls_per_min']} 회/분** |
| 학습 | {', '.join(f'{k} {v}' for k, v in train.items())} |
| 파일 | {', '.join(f'`{n}`' for n in names)} |

AI CITY BUILDERS 강화학습 AI 7강 · CONNECT AI LAB
"""
    open(os.path.join(folder, 'README.md'), 'w').write(card)
    return folder, repo


def export_actor_sb3(model, vecnorm, path, task='walk'):
    """stable-baselines3 PPO -> actor.json (EZERAI LINK-01 웹 플레이그라운드 '내 학습 정책 불러오기' 형식, schema 1).
    VecNormalize 관측 정규화는 첫 층에 접어 넣는다. 웹 규칙: 51 -> 64 -> 64 -> 14, tanh/tanh/linear."""
    import numpy as np
    import link01_gym
    sd = model.policy.state_dict()
    layers = []
    for i, n in enumerate(['mlp_extractor.policy_net.0', 'mlp_extractor.policy_net.2', 'action_net']):
        W = sd[n + '.weight'].cpu().double().numpy(); b = sd[n + '.bias'].cpu().double().numpy()
        if i == 0 and vecnorm is not None:
            mu = vecnorm.obs_rms.mean.astype(np.float64); sig = np.sqrt(vecnorm.obs_rms.var + vecnorm.epsilon)
            b = b - W @ (mu / sig); W = W / sig[None, :]
        layers.append({'weight': W.tolist(), 'bias': b.tolist(), 'activation': 'linear' if i == 2 else 'tanh'})
    man = json.load(open(os.path.join(os.path.dirname(link01_gym.__file__), 'manifest.json')))
    actor = {'schema': 1, 'algorithm': 'PPO', 'task': task, 'trained_steps': int(model.num_timesteps),
             'model_sha256': man['model_sha256'], 'observation_size': 51 if task == 'walk' else 48,
             'action_size': 14, 'action_clip': [-1, 1], 'layers': layers}
    problems = check_actor(actor)
    json.dump(actor, open(path, 'w'))
    return problems


def check_actor(actor):
    """웹 플레이그라운드와 같은 검사. 빈 리스트면 통과."""
    out = []
    size = 51 if actor.get('task') == 'walk' else 48
    if actor.get('schema') != 1 or actor.get('observation_size') != size or actor.get('action_size') != 14 or len(actor.get('layers', [])) != 3:
        out.append('계약(schema/관측/행동/층 수)이 다릅니다')
    inputs = size
    for i, l in enumerate(actor.get('layers', [])[:3]):
        outputs = 14 if i == 2 else 64
        if len(l['weight']) != outputs or any(len(r) != inputs for r in l['weight']):
            out.append(f'{i + 1}번째 층 크기가 {outputs} 가 아닙니다 (웹은 51-64-64-14 만 받습니다)'); break
        inputs = outputs
    return out


def get_token():
    """Colab Secrets HF_TOKEN → 환경변수 HF_TOKEN → 입력창. 파일에는 절대 쓰지 않는다."""
    tok = None
    try:
        from google.colab import userdata
        tok = userdata.get('HF_TOKEN')
    except Exception:
        pass
    tok = tok or os.environ.get('HF_TOKEN')
    if not tok:
        from getpass import getpass
        tok = getpass('허깅페이스 Write 토큰을 붙여넣으세요 (화면에 안 보입니다): ').strip()
    return tok or None


def push(folder, repo):
    from huggingface_hub import HfApi
    tok = get_token()
    if not tok:
        print('토큰이 없어 올리기는 건너뜁니다.'); return None
    api = HfApi(token=tok)
    me = api.whoami()['name']
    p = os.path.join(folder, 'connectailab.json')
    info = json.load(open(p)); info['repo'] = f'{me}/{repo}'
    json.dump(info, open(p, 'w'), ensure_ascii=False, indent=1)
    api.create_repo(repo, exist_ok=True)
    api.upload_folder(folder_path=folder, repo_id=f'{me}/{repo}')
    print(f'올렸습니다 → https://huggingface.co/{me}/{repo}')
    print(f'허깅페이스 저장소: {me}/{repo}')
    return f'{me}/{repo}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', required=True, help='model.zip · vecnormalize.pkl · status.json 이 있는 학습 폴더')
    p.add_argument('--eval', default=None, help='eval_walk.py 출력 폴더 (기본 <run>/eval)')
    p.add_argument('--name', required=True, help='영문 이름 (예: jay)')
    p.add_argument('--version', default='v1')
    p.add_argument('--about', default='', help='나를 알리는 글')
    p.add_argument('--smoke', action='store_true', help='짧은 동작 확인 실행이면 숫자에 smoke test 표시')
    p.add_argument('--push', action='store_true', help='진짜 올리기 (없으면 폴더만 만든다)')
    a = p.parse_args()
    ev = a.eval or os.path.join(a.run, 'eval')
    m = json.load(open(os.path.join(ev, 'metrics.json')))
    st = json.load(open(os.path.join(a.run, 'status.json')))
    train = {k: st[k] for k in ('env', 'algo', 'envs', 'n_steps', 'lr', 'gamma', 'gae_lambda', 'clip', 'seed') if k in st}
    train.update(steps=int(st.get('final_steps', st.get('steps', 0))), minutes=st.get('final_wall_min', st.get('wall_min')))
    files = [os.path.join(a.run, f) for f in ('model.zip', 'vecnormalize.pkl', 'actor.json') if os.path.exists(os.path.join(a.run, f))]
    folder, repo = build(files, m, train, a.name, a.version, a.about, replay=os.path.join(ev, 'replay.mp4'),
                         out_dir=os.path.join(a.run, 'hf_upload'), smoke=a.smoke)
    print('올릴 폴더:', folder)
    for f in sorted(os.listdir(folder)):
        print(f'  {f:20s} {os.path.getsize(os.path.join(folder, f)):>10,d} B')
    if a.push:
        push(folder, repo)
    else:
        print('(--push 를 붙이면 올립니다)')


if __name__ == '__main__':
    main()
