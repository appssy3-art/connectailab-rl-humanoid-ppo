"""SB3 VecEnv with P worker processes x K envs each (fewer processes, bigger batches for the policy).
Keeps CPU use at P cores while giving PPO P*K parallel episodes."""
import multiprocessing as mp
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, CloudpickleWrapper


def _worker(remote, fns, keep_info):
    envs = [f() for f in fns.var]
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            obs, rews, dones, infos = [], [], [], []
            for e, a in zip(envs, data):
                o, r, term, trunc, info = e.step(a)
                done = term or trunc
                inf = {k: info[k] for k in keep_info if k in info}
                inf['TimeLimit.truncated'] = bool(trunc and not term)
                if done:
                    inf['terminal_observation'] = o
                    o, _ = e.reset()
                obs.append(o); rews.append(r); dones.append(done); infos.append(inf)
            remote.send((np.stack(obs), np.array(rews, np.float32), np.array(dones), infos))
        elif cmd == 'reset':
            out = []
            for i, e in enumerate(envs):
                o, _ = e.reset(seed=None if data is None else data + i)
                out.append(o)
            remote.send(np.stack(out))
        elif cmd == 'getattr':
            remote.send([getattr(e, data) for e in envs])
        elif cmd == 'method':
            name, args, kw = data
            remote.send([getattr(e, name)(*args, **kw) for e in envs])
        elif cmd == 'close':
            remote.close(); break


class BatchVecEnv(VecEnv):
    def __init__(self, env_fns, n_workers, keep_info=()):
        self.n = len(env_fns)
        assert self.n % n_workers == 0
        k = self.n // n_workers
        ctx = mp.get_context('spawn')
        self.remotes, self.procs = [], []
        for w in range(n_workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, CloudpickleWrapper(env_fns[w * k:(w + 1) * k]), tuple(keep_info)), daemon=True)
            p.start(); b.close()
            self.remotes.append(a); self.procs.append(p)
        self.k = k
        self.remotes[0].send(('getattr', 'observation_space'))
        osp = self.remotes[0].recv()[0]
        self.remotes[0].send(('getattr', 'action_space'))
        asp = self.remotes[0].recv()[0]
        super().__init__(self.n, osp, asp)
        self._seed = None

    def step_async(self, actions):
        for i, r in enumerate(self.remotes):
            r.send(('step', actions[i * self.k:(i + 1) * self.k]))

    def step_wait(self):
        res = [r.recv() for r in self.remotes]
        obs = np.concatenate([x[0] for x in res]); rew = np.concatenate([x[1] for x in res])
        done = np.concatenate([x[2] for x in res]); infos = sum([x[3] for x in res], [])
        return obs, rew, done, infos

    def reset(self):
        for i, r in enumerate(self.remotes):
            r.send(('reset', None if self._seed is None else self._seed + i * self.k))
        self._seed = None
        return np.concatenate([r.recv() for r in self.remotes])

    def seed(self, seed=None):
        self._seed = seed
        return [seed]

    def close(self):
        for r in self.remotes:
            try:
                r.send(('close', None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()

    def get_attr(self, attr_name, indices=None):
        out = []
        for r in self.remotes:
            r.send(('getattr', attr_name)); out += r.recv()
        return out if indices is None else [out[i] for i in self._get_indices(indices)]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError

    def env_method(self, method_name, *args, indices=None, **kw):
        out = []
        for r in self.remotes:
            r.send(('method', (method_name, args, kw))); out += r.recv()
        return out if indices is None else [out[i] for i in self._get_indices(indices)]

    def env_is_wrapped(self, wrapper_class, indices=None):
        return [False] * (self.n if indices is None else len(self._get_indices(indices)))

    def pids(self):
        return [p.pid for p in self.procs]
