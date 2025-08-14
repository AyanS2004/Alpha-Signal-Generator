#!/usr/bin/env python3
"""
Test script for custom backtesting functionality.
Demonstrates how to use the new custom backtesting features.
"""

import requests
import json
from datetime import datetime, timedelta

def test_custom_backtest_api():
    """Test the custom backtest API endpoint."""
    print("🧪 Testing Custom Backtest API")
    print("=" * 50)
    
    # Test data
    test_cases = [
        {
            "symbol": "AAPL",
            "startDate": "2024-01-01",
            "endDate": "2024-06-30",
            "description": "Apple 6 months"
        },
        {
            "symbol": "TSLA",
            "startDate": "2023-01-01",
            "endDate": "2023-12-31",
            "description": "Tesla full year 2023"
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📊 Testing: {test_case['description']}")
        print(f"   Symbol: {test_case['symbol']}")
        print(f"   Period: {test_case['startDate']} to {test_case['endDate']}")
        
        try:
            # Make API request
            response = requests.post(
                "http://localhost:5000/api/backtest/custom",
                json=test_case,
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                print(f"   ✅ Success!")
                print(f"   Total Return: {data['totalReturn']:.2f}%")
                print(f"   Sharpe Ratio: {data['sharpeRatio']:.3f}")
                print(f"   Total Trades: {data['totalTrades']}")
                print(f"   Data Points: {data['dataPoints']}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                print(f"   Response: {response.text}")
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")

def test_stock_search_api():
    """Test the stock search API endpoint."""
    print("\n🔍 Testing Stock Search API")
    print("=" * 50)
    
    search_queries = ["AAPL", "TS", "GOOG", "MS"]
    
    for query in search_queries:
        print(f"\nSearching for: '{query}'")
        try:
            response = requests.get(
                f"http://localhost:5000/api/stocks/search?q={query}",
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                stocks = data.get('stocks', [])
                print(f"   Found {len(stocks)} stocks: {', '.join(stocks)}")
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {str(e)}")

def test_health_check():
    """Test the health check endpoint."""
    print("\n🏥 Testing Health Check")
    print("=" * 50)
    
    try:
        response = requests.get("http://localhost:5000/api/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Backend is healthy!")
            print(f"   Status: {data['status']}")
            print(f"   Version: {data['version']}")
            print(f"   Timestamp: {data['timestamp']}")
        else:
            print(f"   ❌ Health check failed: {response.status_code}")
            
    except Exception as e:
        print(f"   ❌ Exception: {str(e)}")

def main():
    """Run all tests."""
    print("🚀 Alpha Signal Engine - Custom Backtest Testing")
    print("=" * 60)
    
    # Test health check first
    test_health_check()
    
    # Test stock search
    test_stock_search_api()
    
    # Test custom backtest
    test_custom_backtest_api()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    main()

