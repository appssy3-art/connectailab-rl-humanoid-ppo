# 뒤집힌 로드스터 착지 시뮬레이터 — Gymnasium 틀 (Connect AI LAB · 강화학습 5강)
import math, numpy as np, gymnasium as gym
from gymnasium import spaces

class 로드스터(gym.Env):
    """뒤집힌 채로 떨어지는 차를 냉가스 추진기로 돌려 바퀴부터 착지시키기."""
    metadata = {'render_modes': ['rgb_array']}
    G, DT, MAIN, SIDE, MAXT = 1.0, 0.05, 2.2, 1.8, 400            # 중력 · 한 걸음 시간 · 아래 추진기 힘 · 냉가스 힘 · 최대 걸음
    def __init__(self, 시작각=math.pi, 보상=None):
        self.시작각 = 시작각
        self.P = dict(착지=100.0, 추락=-100.0, 연료_가운데=0.3, 연료_옆=0.03, 다듬기=1.0); self.P.update(보상 or {})
        self.action_space = spaces.Discrete(4)                    # 0 쉬기 · 1 왼쪽 냉가스 · 2 아래 추진기 · 3 오른쪽 냉가스
        self.observation_space = spaces.Box(-np.inf, np.inf, (7,), np.float32)
    def _obs(self):
        x, y, vx, vy, th, w = self.s
        return np.array([x / 5, y / 10, vx / 2, vy / 2, math.sin(th), math.cos(th), w / 2], np.float32)
    def _모양(self):                                              # 착지대 · 느림 · 똑바름에 가까울수록 높은 점수
        x, y, vx, vy, th, w = self.s; thn = (th + math.pi) % (2 * math.pi) - math.pi
        return -10 * math.hypot(x / 5, y / 10) - 10 * math.hypot(vx / 2, vy / 2) - 10 * abs(thn)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed); r = self.np_random
        self.s = np.array([r.uniform(-2.5, 2.5), 10.0, r.uniform(-0.8, 0.8), 0.0, math.pi, r.uniform(-0.3, 0.3)], np.float64)
        self.s[4] = self.시작각 * (1 if r.random() < 0.5 else -1) + r.uniform(-0.15, 0.15)   # 뒤집힌 채로 (왼쪽 · 오른쪽 어느 쪽으로든)
        self.t = 0; self.prev = self._모양(); return self._obs(), {}
    def step(self, a):
        x, y, vx, vy, th, w = self.s; P = self.P; r = 0.0
        ax, ay, aw = 0.0, -self.G, 0.0
        if a == 2: ax += -math.sin(th) * self.MAIN; ay += math.cos(th) * self.MAIN; r -= P['연료_가운데']   # 차 바닥 쪽으로 미는 힘
        elif a == 1: aw -= self.SIDE; r -= P['연료_옆']
        elif a == 3: aw += self.SIDE; r -= P['연료_옆']
        vx += ax * self.DT; vy += ay * self.DT; w += aw * self.DT; w *= 0.96
        x += vx * self.DT; y += vy * self.DT; th += w * self.DT
        self.s = np.array([x, y, vx, vy, th, w]); self.t += 1
        sh = self._모양(); r += P['다듬기'] * (sh - self.prev); self.prev = sh
        끝, 결과 = False, ''
        thn = (th + math.pi) % (2 * math.pi) - math.pi
        if y <= 0:
            끝 = True
            if abs(x) <= 1.5 and abs(thn) < 0.35 and vy > -1.5 and abs(vx) < 1.2: r += P['착지']; 결과 = '착지'
            else: r += P['추락']; 결과 = '추락'
        elif abs(x) > 8 or y > 16: 끝 = True; r += P['추락']; 결과 = '밖으로'
        잘림 = self.t >= self.MAXT and not 끝
        return self._obs(), r, 끝, 잘림, {'결과': 결과 or ('시간 끝' if 잘림 else '')}

class 로드스터연속(로드스터):
    """6강 · 버튼이 아니라 조이스틱 — 아래 추진기 세기(0~1)와 회전 세기(-1~1)를 직접 고른다."""
    def __init__(self, 시작각=math.pi, 보상=None):
        super().__init__(시작각, 보상)
        self.action_space = spaces.Box(np.array([0.0, -1.0], np.float32), np.array([1.0, 1.0], np.float32))
    def step(self, a):
        세기 = float(np.clip(a[0], 0, 1)); 회전 = float(np.clip(a[1], -1, 1))
        x, y, vx, vy, th, w = self.s; P = self.P; r = 0.0
        ax = -math.sin(th) * self.MAIN * 세기; ay = -self.G + math.cos(th) * self.MAIN * 세기; aw = self.SIDE * 회전
        r -= P['연료_가운데'] * 세기 + P['연료_옆'] * abs(회전)
        vx += ax * self.DT; vy += ay * self.DT; w += aw * self.DT; w *= 0.96
        x += vx * self.DT; y += vy * self.DT; th += w * self.DT
        self.s = np.array([x, y, vx, vy, th, w]); self.t += 1
        sh = self._모양(); r += P['다듬기'] * (sh - self.prev); self.prev = sh
        끝, 결과 = False, ''
        thn = (th + math.pi) % (2 * math.pi) - math.pi
        if y <= 0:
            끝 = True
            if abs(x) <= 1.5 and abs(thn) < 0.35 and vy > -1.5 and abs(vx) < 1.2: r += P['착지']; 결과 = '착지'
            else: r += P['추락']; 결과 = '추락'
        elif abs(x) > 8 or y > 16: 끝 = True; r += P['추락']; 결과 = '밖으로'
        잘림 = self.t >= self.MAXT and not 끝
        return self._obs(), r, 끝, 잘림, {'결과': 결과 or ('시간 끝' if 잘림 else '')}

