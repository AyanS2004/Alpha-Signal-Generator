"""
Factor-based strategy using PCA for Alpha Signal Engine.
"""

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np
import pandas as pd
import logging

try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class FactorSignals:
    factor_momentum: float
    factor_mean_reversion: float
    combined_signal: float


class FactorStrategy:
    def __init__(self, n_components: int = 5):
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for PCA factor strategy")
        self.pca = PCA(n_components=n_components)

    def extract_factors(self, price_matrix: pd.DataFrame) -> np.ndarray:
        returns = price_matrix.pct_change().dropna()
        return self.pca.fit_transform(returns.values)

    def calculate_factor_momentum(self, factors: np.ndarray) -> float:
        # Momentum on first principal component
        pc1 = factors[:, 0]
        if len(pc1) < 10:
            return 0.0
        return float((pc1[-1] - pc1[-10]) / (abs(pc1[-10]) + 1e-9))

    def calculate_factor_mean_reversion(self, factors: np.ndarray) -> float:
        pc2 = factors[:, 1] if factors.shape[1] > 1 else factors[:, 0]
        if len(pc2) < 20:
            return 0.0
        z = (pc2 - np.mean(pc2)) / (np.std(pc2) + 1e-9)
        return float(-z[-1])

    def combine_factor_signals(self, momentum: float, mean_reversion: float) -> float:
        return float(0.6 * momentum + 0.4 * mean_reversion)

    def generate(self, price_matrix: pd.DataFrame) -> FactorSignals:
        try:
            factors = self.extract_factors(price_matrix)
            mom = self.calculate_factor_momentum(factors)
            mr = self.calculate_factor_mean_reversion(factors)
            combined = self.combine_factor_signals(mom, mr)
            return FactorSignals(factor_momentum=mom, factor_mean_reversion=mr, combined_signal=combined)
        except Exception as e:
            logger.error(f"Factor strategy error: {str(e)}")
            return FactorSignals(0.0, 0.0, 0.0)



