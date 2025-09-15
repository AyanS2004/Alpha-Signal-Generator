"""
Bayesian Optimization with Gaussian Processes for Alpha Signal Engine.
Advanced parameter optimization using scikit-optimize.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer, Categorical
    from skopt.utils import use_named_args
    from skopt.acquisition import gaussian_ei
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    logging.warning("scikit-optimize not available. Install with: pip install scikit-optimize")

from .config import Config
from .backtester import Backtester
from .performance_analyzer import PerformanceAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class OptimizationResult:
    """Results from Bayesian optimization."""
    best_params: Dict[str, Any]
    best_score: float
    optimization_history: List[Dict[str, Any]]
    convergence_plot_data: Dict[str, List[float]]
    parameter_importance: Dict[str, float]

class BayesianOptimizer:
    """
    Bayesian Optimization using Gaussian Processes for parameter tuning.
    
    This optimizer uses Gaussian Process regression to model the objective function
    and Expected Improvement acquisition function to guide the search.
    """
    
    def __init__(self, engine, n_calls: int = 100, n_initial_points: int = 10):
        """
        Initialize Bayesian optimizer.
        
        Args:
            engine: AlphaSignalEngine instance
            n_calls: Total number of optimization iterations
            n_initial_points: Number of random initial points
        """
        if not SKOPT_AVAILABLE:
            raise ImportError("scikit-optimize is required for Bayesian optimization")
        
        self.engine = engine
        self.n_calls = n_calls
        self.n_initial_points = n_initial_points
        
        # Define search space
        self.search_space = [
            Real(0.005, 0.05, name='momentum_threshold', prior='log-uniform'),
            Integer(5, 50, name='momentum_lookback'),
            Real(0.05, 0.3, name='position_size'),
            Real(0.5, 3.0, name='mean_reversion_std_multiplier'),
            Real(0.01, 0.1, name='mean_reversion_threshold'),
            Integer(10, 100, name='mean_reversion_lookback'),
            Real(0.1, 0.5, name='final_signal_threshold'),
            Real(0.5, 2.0, name='transaction_cost_bps'),
            Real(25.0, 200.0, name='stop_loss_bps'),
            Real(50.0, 300.0, name='take_profit_bps')
        ]
        
        self.optimization_history = []
        self.best_score = -np.inf
        self.best_params = None
        
    def objective_function(self, params: List[float]) -> float:
        """
        Objective function for optimization.
        Returns negative Sharpe ratio to minimize.
        """
        try:
            # Create config from parameters
            config = self._create_config_from_params(params)
            
            # Run backtest with current parameters
            results = self._run_backtest_with_config(config)
            
            # Extract Sharpe ratio as the objective
            sharpe_ratio = results.get('sharpe_ratio', 0.0)
            
            # Store optimization history
            param_dict = dict(zip([dim.name for dim in self.search_space], params))
            self.optimization_history.append({
                'params': param_dict.copy(),
                'sharpe_ratio': sharpe_ratio,
                'total_return': results.get('total_return', 0.0),
                'max_drawdown': results.get('max_drawdown', 0.0),
                'total_trades': results.get('total_trades', 0)
            })
            
            # Update best if improved
            if sharpe_ratio > self.best_score:
                self.best_score = sharpe_ratio
                self.best_params = param_dict.copy()
            
            logger.info(f"Optimization step: Sharpe={sharpe_ratio:.4f}, Params={param_dict}")
            
            # Return negative for minimization
            return -sharpe_ratio
            
        except Exception as e:
            logger.error(f"Error in objective function: {str(e)}")
            return 1.0  # Return high value for failed evaluations
    
    def _create_config_from_params(self, params: List[float]) -> Config:
        """Create Config object from parameter list."""
        param_dict = dict(zip([dim.name for dim in self.search_space], params))
        
        return Config(
            momentum_threshold=param_dict['momentum_threshold'],
            momentum_lookback=int(param_dict['momentum_lookback']),
            position_size=param_dict['position_size'],
            mean_reversion_std_multiplier=param_dict['mean_reversion_std_multiplier'],
            mean_reversion_threshold=param_dict['mean_reversion_threshold'],
            mean_reversion_lookback=int(param_dict['mean_reversion_lookback']),
            final_signal_threshold=param_dict['final_signal_threshold'],
            transaction_cost_bps=param_dict['transaction_cost_bps'],
            stop_loss_bps=param_dict['stop_loss_bps'],
            take_profit_bps=param_dict['take_profit_bps']
        )
    
    def _run_backtest_with_config(self, config: Config) -> Dict[str, float]:
        """Run backtest with given configuration."""
        try:
            # Create temporary engine with new config
            temp_engine = type(self.engine)(config)
            
            # Get the data from original engine
            signals = self.engine.get_signals()
            if signals is None or signals.empty:
                raise ValueError("No signals data available for optimization")
            
            # Run backtest
            backtest_results = temp_engine.backtester.run_backtest(
                signals, config
            )
            
            # Calculate performance metrics
            performance_metrics = temp_engine.performance_analyzer.calculate_metrics(
                backtest_results, signals
            )
            
            return {
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0.0),
                'total_return': performance_metrics.get('total_return', 0.0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0.0),
                'total_trades': backtest_results.get('total_trades', 0)
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return {'sharpe_ratio': 0.0, 'total_return': 0.0, 'max_drawdown': 0.0, 'total_trades': 0}
    
    def optimize(self, csv_file_path: str = None) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            csv_file_path: Optional CSV file for optimization data
            
        Returns:
            OptimizationResult with best parameters and history
        """
        logger.info("Starting Bayesian optimization...")
        
        # Ensure we have data
        if csv_file_path:
            self.engine.run_complete_analysis(csv_file_path, plot_results=False, save_plots=False)
        elif self.engine.get_signals() is None:
            raise ValueError("No data available for optimization. Provide csv_file_path or run analysis first.")
        
        # Run optimization
        result = gp_minimize(
            func=self.objective_function,
            dimensions=self.search_space,
            n_calls=self.n_calls,
            n_initial_points=self.n_initial_points,
            acq_func='EI',  # Expected Improvement
            acq_optimizer='lbfgs',
            random_state=42
        )
        
        # Extract results
        best_params = dict(zip([dim.name for dim in self.search_space], result.x))
        best_score = -result.fun  # Convert back from negative
        
        # Calculate parameter importance (based on optimization history)
        param_importance = self._calculate_parameter_importance()
        
        # Create convergence plot data
        convergence_data = self._create_convergence_data()
        
        logger.info(f"Optimization completed. Best Sharpe: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            optimization_history=self.optimization_history,
            convergence_plot_data=convergence_data,
            parameter_importance=param_importance
        )
    
    def _calculate_parameter_importance(self) -> Dict[str, float]:
        """Calculate parameter importance based on optimization history."""
        if len(self.optimization_history) < 2:
            return {}
        
        # Extract parameter values and scores
        param_names = [dim.name for dim in self.search_space]
        param_values = {name: [] for name in param_names}
        scores = []
        
        for record in self.optimization_history:
            scores.append(record['sharpe_ratio'])
            for name in param_names:
                param_values[name].append(record['params'][name])
        
        # Calculate correlation between parameter changes and score improvements
        importance = {}
        for name in param_names:
            if len(param_values[name]) > 1:
                # Calculate correlation between parameter and score
                correlation = np.corrcoef(param_values[name], scores)[0, 1]
                importance[name] = abs(correlation) if not np.isnan(correlation) else 0.0
            else:
                importance[name] = 0.0
        
        return importance
    
    def _create_convergence_data(self) -> Dict[str, List[float]]:
        """Create data for convergence plots."""
        if not self.optimization_history:
            return {}
        
        scores = [record['sharpe_ratio'] for record in self.optimization_history]
        best_scores = [max(scores[:i+1]) for i in range(len(scores))]
        
        return {
            'iterations': list(range(1, len(scores) + 1)),
            'current_scores': scores,
            'best_scores': best_scores
        }
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results."""
        if not self.optimization_history:
            return {}
        
        scores = [record['sharpe_ratio'] for record in self.optimization_history]
        
        return {
            'total_evaluations': len(self.optimization_history),
            'best_sharpe_ratio': max(scores),
            'worst_sharpe_ratio': min(scores),
            'mean_sharpe_ratio': np.mean(scores),
            'std_sharpe_ratio': np.std(scores),
            'improvement': max(scores) - scores[0] if scores else 0,
            'convergence_rate': self._calculate_convergence_rate(scores)
        }
    
    def _calculate_convergence_rate(self, scores: List[float]) -> float:
        """Calculate how quickly the optimization converged."""
        if len(scores) < 10:
            return 0.0
        
        # Find when we reached 95% of final best score
        final_best = max(scores)
        target = 0.95 * final_best
        
        for i, score in enumerate(scores):
            if score >= target:
                return 1.0 - (i / len(scores))
        
        return 0.0

class AdvancedParameterOptimizer:
    """
    Advanced parameter optimization with multiple strategies.
    """
    
    def __init__(self, engine):
        self.engine = engine
        self.optimizers = {
            'bayesian': BayesianOptimizer(engine) if SKOPT_AVAILABLE else None,
            'grid_search': None,  # Can be implemented
            'random_search': None  # Can be implemented
        }
    
    def optimize_with_strategy(self, strategy: str = 'bayesian', **kwargs) -> OptimizationResult:
        """
        Optimize parameters using specified strategy.
        
        Args:
            strategy: 'bayesian', 'grid_search', or 'random_search'
            **kwargs: Additional arguments for the optimizer
            
        Returns:
            OptimizationResult
        """
        if strategy == 'bayesian':
            if self.optimizers['bayesian'] is None:
                raise ImportError("Bayesian optimization requires scikit-optimize")
            return self.optimizers['bayesian'].optimize(**kwargs)
        else:
            raise ValueError(f"Unknown optimization strategy: {strategy}")
    
    def compare_optimization_strategies(self, csv_file_path: str = None) -> Dict[str, OptimizationResult]:
        """Compare different optimization strategies."""
        results = {}
        
        if self.optimizers['bayesian'] is not None:
            try:
                results['bayesian'] = self.optimizers['bayesian'].optimize(csv_file_path)
            except Exception as e:
                logger.error(f"Bayesian optimization failed: {str(e)}")
        
        return results


