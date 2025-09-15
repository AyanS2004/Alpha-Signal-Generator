"""
Ensemble Methods with Model Stacking for Alpha Signal Engine.
Advanced ML ensemble with multiple base models and meta-learning.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ML imports
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not available. Install with: pip install scikit-learn")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logging.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Install with: pip install torch")

from .config import Config

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score: float
    feature_importance: Dict[str, float]

@dataclass
class EnsembleResult:
    """Ensemble prediction result."""
    final_prediction: float
    base_predictions: Dict[str, float]
    meta_prediction: float
    confidence: float
    model_weights: Dict[str, float]
    feature_importance: Dict[str, float]

class LSTMClassifier:
    """
    LSTM-based classifier for time series prediction.
    """
    
    def __init__(self, input_size: int = 20, hidden_size: int = 50, num_layers: int = 2):
        """
        Initialize LSTM classifier.
        
        Args:
            input_size: Number of input features
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for LSTM classifier")
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_model(self):
        """Create LSTM model architecture."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers):
                super(LSTMModel, self).__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                                  batch_first=True, dropout=0.2)
                self.fc = nn.Linear(hidden_size, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
                
                out, _ = self.lstm(x, (h0, c0))
                out = self.fc(out[:, -1, :])
                out = self.sigmoid(out)
                return out
        
        return LSTMModel(self.input_size, self.hidden_size, self.num_layers)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Fit LSTM model."""
        try:
            if len(X.shape) != 3:
                raise ValueError("LSTM requires 3D input (samples, timesteps, features)")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y.reshape(-1, 1))
            
            # Create model
            self.model = self._create_model()
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logger.info(f"LSTM Epoch {epoch}, Loss: {loss.item():.4f}")
            
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting LSTM: {str(e)}")
            self.is_fitted = False
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        try:
            if not self.is_fitted or self.model is None:
                return np.zeros((len(X), 2))
            
            # Scale features
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            X_tensor = torch.FloatTensor(X_scaled)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).numpy()
            
            # Convert to 2-class probabilities
            prob_positive = predictions.flatten()
            prob_negative = 1 - prob_positive
            
            return np.column_stack([prob_negative, prob_positive])
            
        except Exception as e:
            logger.error(f"Error predicting with LSTM: {str(e)}")
            return np.zeros((len(X), 2))

class TransformerClassifier:
    """
    Transformer-based classifier for time series prediction.
    """
    
    def __init__(self, input_size: int = 20, d_model: int = 64, nhead: int = 8, num_layers: int = 2):
        """
        Initialize Transformer classifier.
        
        Args:
            input_size: Number of input features
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for Transformer classifier")
        
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def _create_model(self):
        """Create Transformer model architecture."""
        class TransformerModel(nn.Module):
            def __init__(self, input_size, d_model, nhead, num_layers):
                super(TransformerModel, self).__init__()
                self.d_model = d_model
                
                # Input projection
                self.input_projection = nn.Linear(input_size, d_model)
                
                # Positional encoding
                self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
                
                # Transformer encoder
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
                    dropout=0.1, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                
                # Output layers
                self.fc = nn.Linear(d_model, 1)
                self.sigmoid = nn.Sigmoid()
                
            def forward(self, x):
                # Project input
                x = self.input_projection(x)
                
                # Add positional encoding
                seq_len = x.size(1)
                x = x + self.pos_encoding[:seq_len].unsqueeze(0)
                
                # Transformer encoding
                x = self.transformer(x)
                
                # Global average pooling
                x = x.mean(dim=1)
                
                # Output
                x = self.fc(x)
                x = self.sigmoid(x)
                return x
        
        return TransformerModel(self.input_size, self.d_model, self.nhead, self.num_layers)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32):
        """Fit Transformer model."""
        try:
            if len(X.shape) != 3:
                raise ValueError("Transformer requires 3D input (samples, timesteps, features)")
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            
            # Convert to tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y.reshape(-1, 1))
            
            # Create model
            self.model = self._create_model()
            criterion = nn.BCELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=0.001)
            
            # Training loop
            for epoch in range(epochs):
                optimizer.zero_grad()
                outputs = self.model(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
                
                if epoch % 20 == 0:
                    logger.info(f"Transformer Epoch {epoch}, Loss: {loss.item():.4f}")
            
            self.is_fitted = True
            
        except Exception as e:
            logger.error(f"Error fitting Transformer: {str(e)}")
            self.is_fitted = False
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        try:
            if not self.is_fitted or self.model is None:
                return np.zeros((len(X), 2))
            
            # Scale features
            X_scaled = self.scaler.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
            X_tensor = torch.FloatTensor(X_scaled)
            
            self.model.eval()
            with torch.no_grad():
                predictions = self.model(X_tensor).numpy()
            
            # Convert to 2-class probabilities
            prob_positive = predictions.flatten()
            prob_negative = 1 - prob_positive
            
            return np.column_stack([prob_negative, prob_positive])
            
        except Exception as e:
            logger.error(f"Error predicting with Transformer: {str(e)}")
            return np.zeros((len(X), 2))

class EnsembleSignalGenerator:
    """
    Ensemble signal generator with model stacking.
    
    Features:
    - Multiple base models (RF, XGBoost, LSTM, Transformer)
    - Meta-learner for combining predictions
    - Dynamic model weighting
    - Feature importance analysis
    - Cross-validation for model selection
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize ensemble signal generator.
        
        Args:
            config: Configuration object
        """
        self.config = config or Config()
        self.models = {}
        self.meta_learner = None
        self.model_weights = {}
        self.feature_names = []
        self.is_fitted = False
        
        # Initialize base models
        self._initialize_models()
        
        # Performance tracking
        self.model_performance = {}
        
    def _initialize_models(self):
        """Initialize base models."""
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available. Some models will be disabled.")
            return
        
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        )
        
        # Support Vector Machine
        self.models['svm'] = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        
        # Neural Network
        self.models['mlp'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='adaptive',
            max_iter=500,
            random_state=42
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        
        # LSTM (if available)
        if TORCH_AVAILABLE:
            self.models['lstm'] = LSTMClassifier(input_size=20, hidden_size=50, num_layers=2)
            self.models['transformer'] = TransformerClassifier(
                input_size=20, d_model=64, nhead=8, num_layers=2
            )
        
        # Meta-learner
        self.meta_learner = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )
        
        logger.info(f"Initialized {len(self.models)} base models")
    
    def _create_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create features for ML models.
        
        Args:
            data: Price data with technical indicators
            
        Returns:
            Feature matrix and feature names
        """
        try:
            features = []
            feature_names = []
            
            # Price-based features
            if 'Close' in data.columns:
                # Returns
                returns = data['Close'].pct_change().dropna()
                features.append(returns.values)
                feature_names.append('returns')
                
                # Volatility
                volatility = returns.rolling(20).std().dropna()
                features.append(volatility.values)
                feature_names.append('volatility_20')
                
                # Price momentum
                momentum_5 = data['Close'].pct_change(5).dropna()
                momentum_10 = data['Close'].pct_change(10).dropna()
                momentum_20 = data['Close'].pct_change(20).dropna()
                
                features.extend([momentum_5.values, momentum_10.values, momentum_20.values])
                feature_names.extend(['momentum_5', 'momentum_10', 'momentum_20'])
            
            # Technical indicators
            for col in data.columns:
                if col in ['momentum_signal', 'mean_reversion_signal', 'rsi', 'volume_ratio']:
                    if not data[col].isna().all():
                        features.append(data[col].fillna(0).values)
                        feature_names.append(col)
            
            # Moving averages
            if 'Close' in data.columns:
                for window in [5, 10, 20, 50]:
                    ma = data['Close'].rolling(window).mean()
                    ma_ratio = data['Close'] / ma
                    features.append(ma_ratio.fillna(1).values)
                    feature_names.append(f'ma_ratio_{window}')
            
            # Volume features
            if 'Volume' in data.columns:
                volume_ma = data['Volume'].rolling(20).mean()
                volume_ratio = data['Volume'] / volume_ma
                features.append(volume_ratio.fillna(1).values)
                feature_names.append('volume_ratio_ma')
            
            # Combine features
            if features:
                # Find minimum length to avoid NaN issues
                min_length = min(len(f) for f in features)
                feature_matrix = np.column_stack([f[:min_length] for f in features])
                
                # Remove rows with NaN
                valid_rows = ~np.isnan(feature_matrix).any(axis=1)
                feature_matrix = feature_matrix[valid_rows]
                
                return feature_matrix, feature_names
            else:
                return np.array([]), []
                
        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            return np.array([]), []
    
    def _create_targets(self, data: pd.DataFrame, lookforward: int = 5) -> np.ndarray:
        """
        Create target variables for ML models.
        
        Args:
            data: Price data
            lookforward: Number of periods to look forward
            
        Returns:
            Target array (1 for positive return, 0 for negative)
        """
        try:
            if 'Close' not in data.columns:
                return np.array([])
            
            # Calculate future returns
            future_returns = data['Close'].shift(-lookforward) / data['Close'] - 1
            
            # Create binary targets (1 for positive return, 0 for negative)
            targets = (future_returns > 0).astype(int)
            
            # Remove NaN values
            valid_targets = ~targets.isna()
            targets = targets[valid_targets].values
            
            return targets
            
        except Exception as e:
            logger.error(f"Error creating targets: {str(e)}")
            return np.array([])
    
    def fit(self, data: pd.DataFrame, lookforward: int = 5):
        """
        Fit ensemble models.
        
        Args:
            data: Training data
            lookforward: Number of periods to look forward for targets
        """
        try:
            logger.info("Fitting ensemble models...")
            
            # Create features and targets
            X, feature_names = self._create_features(data)
            y = self._create_targets(data, lookforward)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid features or targets created")
                return
            
            # Align features and targets
            min_length = min(len(X), len(y))
            X = X[:min_length]
            y = y[:min_length]
            
            self.feature_names = feature_names
            
            # Fit base models
            base_predictions = {}
            model_scores = {}
            
            for name, model in self.models.items():
                try:
                    logger.info(f"Fitting {name} model...")
                    
                    if name in ['lstm', 'transformer']:
                        # Reshape for LSTM/Transformer (samples, timesteps, features)
                        X_reshaped = X.reshape(-1, 1, X.shape[1])
                        model.fit(X_reshaped, y)
                    else:
                        # Standard sklearn models
                        model.fit(X, y)
                    
                    # Cross-validation score
                    if hasattr(model, 'predict_proba'):
                        cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                        model_scores[name] = cv_scores.mean()
                    else:
                        model_scores[name] = 0.5  # Default score
                    
                    # Get predictions for meta-learner
                    if name in ['lstm', 'transformer']:
                        X_reshaped = X.reshape(-1, 1, X.shape[1])
                        pred_proba = model.predict_proba(X_reshaped)
                    else:
                        pred_proba = model.predict_proba(X)
                    
                    base_predictions[name] = pred_proba[:, 1]  # Probability of positive class
                    
                    logger.info(f"{name} model fitted with CV score: {model_scores[name]:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error fitting {name} model: {str(e)}")
                    base_predictions[name] = np.full(len(y), 0.5)  # Default prediction
                    model_scores[name] = 0.5
            
            # Create meta-features
            if base_predictions:
                meta_features = np.column_stack(list(base_predictions.values()))
                
                # Fit meta-learner
                self.meta_learner.fit(meta_features, y)
                
                # Calculate model weights based on performance
                total_score = sum(model_scores.values())
                if total_score > 0:
                    self.model_weights = {name: score / total_score 
                                        for name, score in model_scores.items()}
                else:
                    # Equal weights if no scores
                    self.model_weights = {name: 1.0 / len(model_scores) 
                                        for name in model_scores.keys()}
                
                self.is_fitted = True
                logger.info("Ensemble models fitted successfully")
                logger.info(f"Model weights: {self.model_weights}")
            else:
                logger.error("No base models fitted successfully")
                
        except Exception as e:
            logger.error(f"Error fitting ensemble: {str(e)}")
            self.is_fitted = False
    
    def predict(self, data: pd.DataFrame) -> EnsembleResult:
        """
        Generate ensemble predictions.
        
        Args:
            data: Input data
            
        Returns:
            EnsembleResult with predictions and confidence
        """
        try:
            if not self.is_fitted:
                logger.warning("Ensemble not fitted. Returning default predictions.")
                return EnsembleResult(
                    final_prediction=0.0,
                    base_predictions={},
                    meta_prediction=0.0,
                    confidence=0.0,
                    model_weights={},
                    feature_importance={}
                )
            
            # Create features
            X, _ = self._create_features(data)
            
            if len(X) == 0:
                return EnsembleResult(
                    final_prediction=0.0,
                    base_predictions={},
                    meta_prediction=0.0,
                    confidence=0.0,
                    model_weights=self.model_weights,
                    feature_importance={}
                )
            
            # Get base model predictions
            base_predictions = {}
            for name, model in self.models.items():
                try:
                    if name in ['lstm', 'transformer']:
                        X_reshaped = X.reshape(-1, 1, X.shape[1])
                        pred_proba = model.predict_proba(X_reshaped)
                    else:
                        pred_proba = model.predict_proba(X)
                    
                    # Use latest prediction
                    base_predictions[name] = pred_proba[-1, 1] if len(pred_proba) > 0 else 0.5
                    
                except Exception as e:
                    logger.error(f"Error predicting with {name}: {str(e)}")
                    base_predictions[name] = 0.5
            
            # Meta-learner prediction
            if base_predictions and self.meta_learner is not None:
                meta_features = np.array(list(base_predictions.values())).reshape(1, -1)
                meta_prediction = self.meta_learner.predict_proba(meta_features)[0, 1]
            else:
                meta_prediction = 0.5
            
            # Weighted ensemble prediction
            if base_predictions and self.model_weights:
                weighted_prediction = sum(
                    base_predictions[name] * self.model_weights.get(name, 0.0)
                    for name in base_predictions.keys()
                )
            else:
                weighted_prediction = meta_prediction
            
            # Final prediction (average of meta-learner and weighted ensemble)
            final_prediction = (meta_prediction + weighted_prediction) / 2
            
            # Calculate confidence based on agreement
            if base_predictions:
                predictions = list(base_predictions.values())
                confidence = 1.0 - np.std(predictions)  # Higher agreement = higher confidence
                confidence = max(0.0, min(1.0, confidence))
            else:
                confidence = 0.0
            
            # Feature importance (from Random Forest if available)
            feature_importance = {}
            if 'rf' in self.models and hasattr(self.models['rf'], 'feature_importances_'):
                rf_importance = self.models['rf'].feature_importances_
                feature_importance = dict(zip(self.feature_names, rf_importance))
            
            return EnsembleResult(
                final_prediction=final_prediction,
                base_predictions=base_predictions,
                meta_prediction=meta_prediction,
                confidence=confidence,
                model_weights=self.model_weights,
                feature_importance=feature_importance
            )
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            return EnsembleResult(
                final_prediction=0.0,
                base_predictions={},
                meta_prediction=0.0,
                confidence=0.0,
                model_weights={},
                feature_importance={}
            )
    
    def get_model_performance(self) -> Dict[str, ModelPerformance]:
        """Get performance metrics for all models."""
        return self.model_performance
    
    def update_model_weights(self, performance_scores: Dict[str, float]):
        """Update model weights based on recent performance."""
        try:
            if not performance_scores:
                return
            
            # Normalize scores
            total_score = sum(performance_scores.values())
            if total_score > 0:
                self.model_weights = {name: score / total_score 
                                    for name, score in performance_scores.items()}
                
                logger.info(f"Updated model weights: {self.model_weights}")
            
        except Exception as e:
            logger.error(f"Error updating model weights: {str(e)}")
    
    def get_ensemble_summary(self) -> Dict[str, Any]:
        """Get ensemble summary information."""
        return {
            'is_fitted': self.is_fitted,
            'num_models': len(self.models),
            'model_names': list(self.models.keys()),
            'model_weights': self.model_weights,
            'feature_names': self.feature_names,
            'meta_learner': type(self.meta_learner).__name__ if self.meta_learner else None
        }


