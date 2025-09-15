"""
Regime-aware transaction cost model for Alpha Signal Engine.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np


@dataclass
class SmartTransactionCostModel:
    base_cost: float = 0.001  # 10 bps
    volatility_multiplier: float = 2.0

    def load_volume_impact_model(self):
        return None

    def get_regime_cost_multiplier(self, regime: str) -> float:
        mapping = {
            'trending': 1.0,
            'ranging': 1.1,
            'volatile': 1.5
        }
        return mapping.get(regime, 1.0)

    def calculate_transaction_cost(self, trade_size: float, market_conditions: Dict[str, Any]) -> float:
        cost = self.base_cost
        vol = float(market_conditions.get('volatility', 0.2))
        vol_adjustment = 1 + self.volatility_multiplier * vol
        avg_volume = float(market_conditions.get('avg_volume', 1e6))
        volume_ratio = max(0.0, trade_size / max(avg_volume, 1.0))
        market_impact = 0.001 * np.sqrt(volume_ratio)
        regime_multiplier = self.get_regime_cost_multiplier(market_conditions.get('regime', 'ranging'))
        total_cost = cost * vol_adjustment * regime_multiplier + market_impact
        return float(total_cost)



