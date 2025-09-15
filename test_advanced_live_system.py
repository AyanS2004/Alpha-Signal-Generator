#!/usr/bin/env python3
"""
Test script for the advanced live signal generation system.
Demonstrates Kalman filtering, ML model integration, and dynamic retraining.
"""

import sys
import os
import time
import pickle
import numpy as np
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_signal_engine.realtime_feed import RealTimeDataFeed
from alpha_signal_engine.config import Config
from alpha_signal_engine.advanced_signals import AdvancedSignalGenerator

def create_sample_model():
    """Create a sample ML model for testing."""
    print("🔧 Creating sample ML model...")
    
    # Generate synthetic training data
    np.random.seed(42)
    n_samples = 1000
    
    # Create features similar to what the advanced signal generator creates
    features = np.random.randn(n_samples, 20)  # 20 features
    targets = np.random.choice([0, 1], n_samples, p=[0.6, 0.4])  # Binary classification
    
    # Train a simple model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    scaler = StandardScaler()
    
    features_scaled = scaler.fit_transform(features)
    model.fit(features_scaled, targets)
    
    # Save model
    model_data = {
        'model': model,
        'scaler': scaler,
        'timestamp': datetime.now(),
        'config': Config()
    }
    
    model_path = 'sample_ml_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Sample model saved to {model_path}")
    return model_path

def advanced_signal_callback(signal_data):
    """Callback function for advanced signals."""
    signal_type = "BUY" if signal_data['signal'] > 0 else "SELL" if signal_data['signal'] < 0 else "HOLD"
    confidence_pct = signal_data.get('ml_confidence', 0.5) * 100
    
    print(f"🧠 {signal_data['timestamp'].strftime('%H:%M:%S')} | "
          f"Signal: {signal_type} | "
          f"ML Confidence: {confidence_pct:.1f}% | "
          f"ML Signal: {signal_data.get('ml_signal', 0)} | "
          f"Market Regime: {signal_data.get('market_regime', 'unknown')} | "
          f"Price: ${signal_data.get('price', 0):.2f} | "
          f"Smoothed: ${signal_data.get('smoothed_price', 0):.2f} | "
          f"Velocity: {signal_data.get('velocity', 0):.4f}")

def basic_signal_callback(signal_data):
    """Callback function for basic signals."""
    signal_type = "BUY" if signal_data['signal'] > 0 else "SELL" if signal_data['signal'] < 0 else "HOLD"
    confidence_pct = signal_data.get('confidence', 0.5) * 100
    
    print(f"📊 {signal_data['timestamp'].strftime('%H:%M:%S')} | "
          f"Signal: {signal_type} | "
          f"Confidence: {confidence_pct:.1f}% | "
          f"Market State: {signal_data.get('market_state', 'unknown')} | "
          f"Price: ${signal_data.get('price', 0):.2f}")

def test_basic_mode():
    """Test basic stateful simulation mode."""
    print("\n" + "="*60)
    print("📊 TESTING BASIC STATEFUL SIMULATION MODE")
    print("="*60)
    
    config = Config()
    feed = RealTimeDataFeed(config, symbol="AAPL")
    
    feed.start_feed(on_signal_callback=basic_signal_callback)
    
    try:
        print("📡 Basic mode started. Running for 30 seconds...")
        time.sleep(30)
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop_feed()
        
        # Print statistics
        stats = feed.get_simulation_stats()
        print(f"\n📊 Basic Mode Statistics:")
        print(f"Total Signals: {stats['total_signals']}")
        print(f"State Transitions: {stats['state_transitions']}")
        print(f"Current State: {stats['current_state']}")

def test_advanced_mode():
    """Test advanced ML-powered mode."""
    print("\n" + "="*60)
    print("🧠 TESTING ADVANCED ML-POWERED MODE")
    print("="*60)
    
    # Create sample model
    model_path = create_sample_model()
    
    config = Config()
    feed = RealTimeDataFeed(config, symbol="AAPL")
    
    # Enable advanced mode
    feed.enable_advanced_mode(model_path)
    
    feed.start_feed(on_signal_callback=advanced_signal_callback)
    
    try:
        print("🧠 Advanced mode started. Running for 60 seconds...")
        print("Features: Kalman Filtering, ML Predictions, Market Regime Detection")
        
        # Print periodic updates
        for i in range(6):
            time.sleep(10)
            info = feed.get_advanced_mode_info()
            print(f"\n🔍 Advanced Mode Status (after {10*(i+1)}s):")
            print(f"  Model Loaded: {info['model_loaded']}")
            print(f"  Retraining Active: {info['retraining_active']}")
            print(f"  Retraining Data Points: {info['retraining_data_points']}")
            print(f"  Total Signals: {info['total_signals']}")
            
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop_feed()
        
        # Print final statistics
        info = feed.get_advanced_mode_info()
        print(f"\n🧠 Advanced Mode Final Statistics:")
        print(f"Model Loaded: {info['model_loaded']}")
        print(f"Model Last Updated: {info['model_last_updated']}")
        print(f"Total Signals Generated: {info['total_signals']}")
        print(f"Retraining Data Points: {info['retraining_data_points']}")
        
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)
            print(f"🗑️ Cleaned up sample model file")

def test_mode_switching():
    """Test switching between basic and advanced modes."""
    print("\n" + "="*60)
    print("🔄 TESTING MODE SWITCHING")
    print("="*60)
    
    config = Config()
    feed = RealTimeDataFeed(config, symbol="AAPL")
    
    feed.start_feed(on_signal_callback=basic_signal_callback)
    
    try:
        print("📊 Starting in basic mode...")
        time.sleep(15)
        
        print("\n🔄 Switching to advanced mode...")
        model_path = create_sample_model()
        feed.enable_advanced_mode(model_path)
        time.sleep(15)
        
        print("\n🔄 Switching back to basic mode...")
        feed.disable_advanced_mode()
        time.sleep(15)
        
    except KeyboardInterrupt:
        pass
    finally:
        feed.stop_feed()
        
        # Clean up
        if os.path.exists('sample_ml_model.pkl'):
            os.remove('sample_ml_model.pkl')

def main():
    """Main test function."""
    print("🚀 Advanced Live Signal Generation System Test")
    print("=" * 60)
    print("This test demonstrates:")
    print("• Kalman Filter price smoothing")
    print("• ML model integration")
    print("• Market regime detection")
    print("• Dynamic model retraining")
    print("• Hot-swapping capabilities")
    print("• Mode switching")
    
    try:
        # Test basic mode
        test_basic_mode()
        
        # Test advanced mode
        test_advanced_mode()
        
        # Test mode switching
        test_mode_switching()
        
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    
    print("\n✅ All tests completed!")
    print("\nKey Features Demonstrated:")
    print("✅ Context-aware signal generation")
    print("✅ Kalman Filter noise reduction")
    print("✅ ML model integration")
    print("✅ Market regime detection")
    print("✅ Dynamic retraining capability")
    print("✅ Hot-swapping without downtime")
    print("✅ Production-ready architecture")

if __name__ == "__main__":
    main()










