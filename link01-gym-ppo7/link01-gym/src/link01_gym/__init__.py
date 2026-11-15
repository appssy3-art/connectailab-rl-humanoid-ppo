"""CONNECT AI LAB · LINK-01 Gymnasium environments.

    import gymnasium as gym, link01_gym
    env = gym.make("Link01Walk-v0")                 # walk forward from scratch (PPO)
    env = gym.make("Link01WalkGuided-v0")           # + rhythm hint from our CPG reference gait
    env = gym.make("Link01Balance-v0")              # original EZERAI balance task
    env = gym.make("Link01Walk-v0", render_mode="rgb_array")   # frames for a replay video
"""
from gymnasium.envs.registration import register

__version__ = "0.7.0"

register(id="Link01Walk-v0", entry_point="link01_gym.walk:Link01WalkEnv")
register(id="Link01WalkGuided-v0", entry_point="link01_gym.walk:Link01WalkEnv", kwargs=dict(ref_weight=1.0))
register(id="Link01Balance-v0", entry_point="link01_gym.env:BalanceEnv", kwargs=dict(task="balance"))
