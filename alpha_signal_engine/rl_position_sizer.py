"""
Reinforcement Learning based position sizing (stub using PPO API pattern).
"""

from dataclasses import dataclass
from typing import Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from stable_baselines3 import PPO
    from gymnasium import Env
    SB3_AVAILABLE = True
except Exception:
    SB3_AVAILABLE = False


@dataclass
class RLActionScale:
    min_position: float = 0.0
    max_position: float = 0.2


class DummyEnv:
    """Minimal env placeholder. Replace with proper market env."""
    def __init__(self):
        pass


class RLPositionSizer:
    def __init__(self, env: Any = None, action_scale: RLActionScale = RLActionScale()):
        self.env = env or DummyEnv()
        self.action_scale = action_scale
        self.agent = None
        if SB3_AVAILABLE:
            try:
                self.agent = PPO('MlpPolicy', self.env, verbose=0)
            except Exception as e:
                logger.warning(f"PPO init failed: {e}")

    def scale_action_to_position_size(self, action: float) -> float:
        a = (action + 1) / 2  # assume action in [-1,1]
        return float(self.action_scale.min_position + a * (self.action_scale.max_position - self.action_scale.min_position))

    def get_position_size(self, market_state: np.ndarray, portfolio_state: np.ndarray, signal_strength: float) -> float:
        state = np.concatenate([market_state, portfolio_state, np.array([signal_strength])])
        action = 0.0
        if self.agent is not None:
            try:
                action = float(self.agent.predict(state, deterministic=True)[0])
            except Exception:
                action = float(np.clip(signal_strength, -1, 1))
        else:
            action = float(np.clip(signal_strength, -1, 1))
        return self.scale_action_to_position_size(action)



