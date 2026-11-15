"""강화학습 7강 · 학습한 LINK-01 걷기 시험 + 리플레이 영상.

    python eval_walk.py --run runs/scratch                 # model.zip + vecnormalize.pkl
    python eval_walk.py --run runs/scratch --ckpt runs/scratch/ckpt/5000000.zip
    python eval_walk.py --policy zero --out runs/baseline_zero     # 학습 전 비교: 가만히 서 있기
    python eval_walk.py --policy random --out runs/baseline_random # 학습 전 비교: 아무렇게나

시험 방식 (모든 숫자는 이 방식 그대로):
  명령 속도 0.18 m/s 앞으로, 판마다 --seconds 초, --episodes 판, 씨앗 1000+i (학습 때 안 본 씨앗).
  넘어지면 그 자리에서 다시 세우고 시간은 계속 흐름 → 넘어짐/분 = 넘어진 횟수 / 전체 분.
  거리 = 출발 방향으로 앞으로 간 거리 합 / 판 수, 속도 = 거리 합 / 전체 시간.
출력: <out>/metrics.json, <out>/replay.mp4 (첫 판, 25 fps)
"""
import argparse, json, os
import numpy as np
import gymnasium as gym
import link01_gym  # noqa: F401


def load_policy(a):
    if a.policy == 'zero':
        return lambda o: np.zeros(14), {'policy': 'zero (stand still, before learning)'}
    if a.policy == 'random':
        rng = np.random.default_rng(0)
        return lambda o: rng.uniform(-1, 1, 14), {'policy': 'random actions'}
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    ck = a.ckpt or os.path.join(a.run, 'model.zip')
    vnp = ck.replace('.zip', '_vecnormalize.pkl') if a.ckpt else os.path.join(a.run, 'vecnormalize.pkl')
    model = PPO.load(ck, device='cpu')
    vn = VecNormalize.load(vnp, DummyVecEnv([lambda: gym.make(a.env)]))
    vn.training = False
    pol = lambda o: model.predict(vn.normalize_obs(o[None]), deterministic=True)[0][0]
    return pol, {'policy': 'PPO', 'checkpoint': ck, 'trained_steps': int(model.num_timesteps)}


def hud(frame, lines):
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return frame
    im = Image.fromarray(frame); d = ImageDraw.Draw(im)
    font = None
    for f in ['~/Library/Fonts/Pretendard-SemiBold.otf', '/Library/Fonts/Pretendard-SemiBold.otf',
              '/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf', '/System/Library/Fonts/Supplemental/AppleGothic.ttf',
              'C:/Windows/Fonts/malgunbd.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf']:
        try:
            font = ImageFont.truetype(os.path.expanduser(f), 18); break
        except Exception:
            pass
    font = font or ImageFont.load_default()
    for i, (t, green) in enumerate(lines):
        d.text((18, 14 + 26 * i), t, fill=(0, 255, 65) if green else (255, 255, 255), font=font)
    return np.array(im)


def evaluate(policy, out, env_id='Link01Walk-v0', episodes=5, seconds=20.0, command=0.18, video=True,
             label='LINK-01 · PPO', meta=None):
    """policy(obs 51) -> action 14. 시험 숫자 dict 를 돌려주고 out/metrics.json, out/replay.mp4 를 쓴다."""
    os.makedirs(out, exist_ok=True)
    steps = int(seconds / 0.02)
    per_ep, frames = [], []
    for ep in range(episodes):
        e = gym.make(env_id, command=command, horizon=10 ** 9,
                     render_mode='rgb_array' if (video and ep == 0) else None).unwrapped
        o, _ = e.reset(seed=1000 + ep)
        dist, falls, vx = 0.0, 0, []
        for t in range(steps):
            o, r, term, trunc, info = e.step(np.asarray(policy(o), np.float64))
            vx.append(info['vx'])
            if ep == 0 and video and t % 2 == 0:
                frames.append(hud(e.render(), [(label, False), (f'{dist + e.distance():.2f} m', True),
                                               (f'{t * 0.02:4.1f} s   넘어짐 {falls}', False)]))
            if term:
                falls += 1
                dist += e.distance()
                o, _ = e.reset(seed=1000 + ep + 100 * falls)
        dist += e.distance()
        per_ep.append(dict(distance_m=round(dist, 3), falls=falls, mean_vx=round(float(np.mean(vx)), 4)))
        e.close()
    total_min = episodes * seconds / 60
    D = sum(x['distance_m'] for x in per_ep)
    m = dict(meta or {}, env=env_id, command_mps=command, episodes=episodes, seconds_per_episode=seconds,
             distance_m_per_episode=round(D / episodes, 3),
             speed_mps=round(D / (episodes * seconds), 4),
             falls_per_min=round(sum(x['falls'] for x in per_ep) / total_min, 3),
             episodes_detail=per_ep)
    json.dump(m, open(os.path.join(out, 'metrics.json'), 'w'), indent=1, ensure_ascii=False)
    if frames:
        import imageio
        imageio.mimwrite(os.path.join(out, 'replay.mp4'), frames, fps=25, codec='libx264', quality=7,
                         macro_block_size=16)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--env', default='Link01Walk-v0')
    p.add_argument('--run', default=None)
    p.add_argument('--ckpt', default=None)
    p.add_argument('--policy', default='ppo', choices=['ppo', 'zero', 'random'])
    p.add_argument('--episodes', type=int, default=5)
    p.add_argument('--seconds', type=float, default=20)
    p.add_argument('--command', type=float, default=0.18)
    p.add_argument('--out', default=None)
    p.add_argument('--no_video', action='store_true')
    p.add_argument('--label', default='LINK-01 · PPO')
    a = p.parse_args()
    out = a.out or os.path.join(a.run or 'runs/eval', 'eval')
    pol, meta = load_policy(a)
    m = evaluate(pol, out, a.env, a.episodes, a.seconds, a.command, not a.no_video, a.label, meta)
    print(json.dumps({k: v for k, v in m.items() if k != 'episodes_detail'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
