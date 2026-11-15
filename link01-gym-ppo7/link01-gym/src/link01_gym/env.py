"""CPU MuJoCo balance task, original LINK-01 observation/action contract.

Simulation gains and torque limits are provisional, NOT a calibrated servo model.
SI units. 50 Hz policy, 500 Hz physics. No hardware dependencies.
"""
from pathlib import Path
import hashlib
import os
import xml.etree.ElementTree as ET
import gymnasium as gym
from gymnasium import spaces
import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / 'model/scene.xml'
MODELS = {'base': MODEL}

def model_fingerprint(path):
    files=set()
    def visit(xml):
        xml=Path(xml).resolve(); files.add(xml)
        for element in ET.parse(xml).getroot().iter():
            if element.tag=='include': visit(xml.parent/element.get('file'))
            elif element.tag in ('mesh','texture') and element.get('file'): files.add((xml.parent/element.get('file')).resolve())
    visit(path)
    digest=hashlib.sha256()
    for file in sorted(files,key=str):
        digest.update(str(file.relative_to(ROOT)).encode());digest.update(file.read_bytes())
    return digest.hexdigest()

HOME = np.array([.002,.053,-.63,1.368,-.784,0,0,0,0,0,0,-.003,-.065,.635,1.379,-.796])
CONTROL = np.array([0,1,2,3,4,5,6,7,8,11,12,13,14,15])

class BalanceEnv(gym.Env):
    metadata = {'render_modes': []}
    def __init__(self, horizon=250, disturbances=True, model_path=None, task="balance", body="base"):
        if body not in MODELS: raise ValueError("Unknown body")
        mpath = Path(model_path or MODELS[body]).resolve()
        orig_cwd = os.getcwd()
        try:
            os.chdir(mpath.parent)
            self.model = mujoco.MjModel.from_xml_path(mpath.name)
        finally:
            os.chdir(orig_cwd)
        self.model.opt.timestep = .002
        self.data = mujoco.MjData(self.model)
        joints = self.model.actuator_trnid[:,0]
        self.qadr = self.model.jnt_qposadr[joints]
        self.vadr = self.model.jnt_dofadr[joints]
        self.limits = self.model.jnt_range[joints]
        self.action_space = spaces.Box(-1, 1, (14,), np.float32)
        # local gravity, local angular velocity, q-home, qvel, previous action
        self.observation_space = spaces.Box(-np.inf, np.inf, (48,), np.float32)
        self.task = task
        if task not in ("balance", "walk"): raise ValueError("Unknown task")
        self.commands = np.zeros(3)
        if task == "walk": self.observation_space = spaces.Box(-np.inf,np.inf,(51,),np.float32)
        self.horizon, self.disturbances = horizon, disturbances
        self.previous = np.zeros(14)
        self.targets = HOME.copy()
        self.steps = 0
        self.trunk = self.model.body('trunk_assembly').id

    def observation(self):
        rotation = self.data.xmat[self.trunk].reshape(3,3)
        gravity = rotation.T @ np.array([0,0,-1.])
        obs = np.concatenate([gravity, self.data.qvel[3:6],
            (self.data.qpos[self.qadr]-HOME)[CONTROL],
            self.data.qvel[self.vadr][CONTROL]*.05, self.previous]).astype(np.float32)
        return np.concatenate([obs,self.commands]).astype(np.float32) if self.task == "walk" else obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[2] = .17
        self.data.qpos[self.qadr] = HOME
        self.targets = HOME.copy()
        self.previous[:] = 0
        self.steps = 0
        self.commands[:] = [self.np_random.uniform(.05,.20),0,0] if self.task == "walk" else 0
        # Settle on the actual source feet before perturbing the reset state.
        for _ in range(150): self._physics()
        self.data.qvel[:2] = self.np_random.uniform(-.035,.035,2)
        self.data.qvel[3:5] = self.np_random.uniform(-.12,.12,2)
        self.data.time = 0
        mujoco.mj_forward(self.model, self.data)
        return self.observation(), {}

    def _physics(self):
        self.data.ctrl[:] = np.clip(12*(self.targets-self.data.qpos[self.qadr])
                                   -.35*self.data.qvel[self.vadr], -1.85, 1.85)
        mujoco.mj_step(self.model, self.data)

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('Expected 14 finite action values')
        action = np.clip(action,-1,1)
        requested = HOME.copy()
        requested[CONTROL] += .18*action
        self.targets += np.clip(requested-self.targets,-5.24*.02,5.24*.02)
        self.targets = np.clip(self.targets,self.limits[:,0],self.limits[:,1])
        self.data.xfrc_applied[:] = 0
        # Repeatable short pushes, randomized independently from the policy.
        if self.disturbances and self.steps % 75 == 50:
            self.push = self.np_random.uniform(-2,2,2)
        if self.disturbances and 50 <= self.steps % 75 < 55:
            self.data.xfrc_applied[self.trunk,:2] = self.push
        for _ in range(10): self._physics()
        mujoco.mj_forward(self.model,self.data)
        self.steps += 1
        upright = float(self.data.xmat[self.trunk].reshape(3,3)[2,2])
        height = float(self.data.qpos[2])
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        terminated = bool(not finite or height < .10 or upright < .55)
        reward = (1.0 + 1.5*upright + np.exp(-2000*(height-.16)**2)
                  - .7*np.square(self.data.qvel[:2]).sum()
                  - .03*np.square(action).mean()
                  - .03*np.square(action-self.previous).mean())
        if self.task == "walk":
            rotation = self.data.xmat[self.trunk].reshape(3,3)
            local_velocity = rotation.T @ self.data.qvel[:3]
            tracking = np.exp(-np.square(local_velocity[:2]-self.commands[:2]).sum()/.015)
            reward = 2*tracking + .5*upright - .02*np.square(action).mean() - .03*np.square(action-self.previous).mean()
        if terminated: reward = -10.
        self.previous = action.copy()
        return self.observation(), float(reward), terminated, self.steps >= self.horizon, {
            'upright': upright, 'height': height, 'fallen': terminated,
            'elapsed_s': self.steps*.02}
