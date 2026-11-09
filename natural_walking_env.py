# -*- coding: utf-8 -*-
import math
import gymnasium as gym
import numpy as np

class NaturalWalkingWrapper(gym.Wrapper):
    """
    자연스러운 인간 보행(Natural Human Walking)을 유도하는 맞춤 보상 래퍼
    1. 전진 속도 인센티브 (타겟 속도 1.5 m/s 리듬 보상)
    2. 좌우 흔들림(y_velocity) 억제 페널티
    3. 에너지 낭비 줄이고 다리를 힘차게 뻗도록 유도
    """
    def __init__(self, env, target_speed=1.5):
        super().__init__(env)
        self.target_speed = target_speed

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        
        vx = info.get('x_velocity', 0.0)
        vy = info.get('y_velocity', 0.0)
        
        # 1. 이상적인 보행 속도 (1.5 m/s 산책/조깅 리듬) 가우시안 보상
        speed_bonus = 3.0 * math.exp(-((vx - self.target_speed) ** 2) / 0.6)
        
        # 2. 좌우 기우뚱 흔들림 페널티
        lateral_penalty = 1.0 * abs(vy)
        
        # 3. 전진 방향 직진성 추가 보상 (vx가 양수일 때)
        forward_progress = 1.5 * max(0.0, vx)
        
        custom_reward = reward + speed_bonus - lateral_penalty + forward_progress
        
        return obs, custom_reward, term, trunc, info

if __name__ == '__main__':
    base_env = gym.make('Humanoid-v5', forward_reward_weight=4.0, healthy_reward=1.5, ctrl_cost_weight=0.02)
    env = NaturalWalkingWrapper(base_env)
    o, _ = env.reset()
    o, r, tm, tr, info = env.step(env.action_space.sample())
    print("Custom Step Reward:", r)
    env.close()

