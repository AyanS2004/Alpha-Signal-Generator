#!/usr/bin/env python3
"""
Test script for the stateful simulated signal generator.
Demonstrates the market state transitions and signal generation behavior.
"""

import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from alpha_signal_engine.realtime_feed import RealTimeDataFeed
from alpha_signal_engine.config import Config

def signal_callback(signal_data):
    """Callback function to handle new signals."""
    signal_type = "BUY" if signal_data['signal'] > 0 else "SELL" if signal_data['signal'] < 0 else "HOLD"
    confidence_pct = signal_data['confidence'] * 100
    state_duration = signal_data['state_duration']
    
    print(f"📊 {signal_data['timestamp'].strftime('%H:%M:%S')} | "
          f"Signal: {signal_type} | "
          f"Confidence: {confidence_pct:.1f}% | "
          f"State: {signal_data['market_state']} | "
          f"Price: ${signal_data['price']:.2f} | "
          f"State Duration: {state_duration:.1f}s")

def data_callback(data):
    """Callback function to handle new data (optional)."""
    # Uncomment to see all data updates
    # print(f"📈 {data['timestamp'].strftime('%H:%M:%S')} | Price: ${data['price']:.2f}")
    pass

def main():
    """Main test function."""
    print("🚀 Starting Stateful Signal Generator Test")
    print("=" * 60)
    
    # Create configuration
    config = Config()
    
    # Create real-time feed
    feed = RealTimeDataFeed(config, symbol="AAPL")
    
    # Start the feed with callbacks
    feed.start_feed(
        on_data_callback=data_callback,
        on_signal_callback=signal_callback
    )
    
    try:
        print("📡 Feed started. Watching for signals...")
        print("Press Ctrl+C to stop and see statistics.\n")
        
        # Run for a while to see state transitions
        start_time = time.time()
        while time.time() - start_time < 300:  # Run for 5 minutes
            time.sleep(10)  # Check every 10 seconds
            
            # Print current state info
            state_info = feed.get_current_market_state()
            print(f"🔄 Current State: {state_info['state']} "
                  f"(Duration: {state_info['time_in_state']:.1f}s/{state_info['state_duration']}s)")
    
    except KeyboardInterrupt:
        print("\n🛑 Test stopped by user")
    
    finally:
        # Stop the feed
        feed.stop_feed()
        
        # Print final statistics
        print("\n📊 Final Statistics:")
        print("=" * 40)
        
        stats = feed.get_simulation_stats()
        print(f"Total Signals Generated: {stats['total_signals']}")
        print(f"State Transitions: {stats['state_transitions']}")
        print(f"Current State: {stats['current_state']}")
        
        print("\nState Distribution:")
        for state, count in stats['state_distribution'].items():
            print(f"  {state}: {count} signals")
        
        print("\nSignal Distribution:")
        for signal_type, count in stats['signal_distribution'].items():
            print(f"  {signal_type}: {count} signals")
        
        print("\n✅ Test completed!")

if __name__ == "__main__":
    main()








