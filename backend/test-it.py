"""
Test script for Fibonacci RSI MACD Strategy.

This script demonstrates how to test the strategy via the API endpoint.
The actual strategy implementation is in: app/strategies/fibonacci_rsi_macd.py

Usage:
    python test-it.py

Or test via frontend/API:
    POST /api/backtest/strategy
    {
      "symbol": "BTC/USDT:USDT",
      "timeframe": "1h",
      "limit": 500,
      "engine": "vectorbt",
      "strategy_file": "app/strategies/fibonacci_rsi_macd.py",
      "initial_capital": 10000.0,
      "strategy_params": {
        "swing_window": 20,
        "fib_level": 0.618,
        "rsi_threshold": 50.0
      }
    }
"""

import requests
import json
import os
from pathlib import Path

# Load environment variables if .env exists
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass


def test_strategy_via_api():
    """Test the strategy via the API endpoint."""
    
    # API endpoint (adjust if needed)
    api_base = os.getenv("API_BASE_URL", "http://localhost:8000/api")
    endpoint = f"{api_base}/backtest/strategy"
    
    # Strategy parameters
    request_data = {
        "symbol": "BTC/USDT:USDT",
        "timeframe": "15m",
        "limit": 1000,
        "engine": "vectorbt",
        "strategy_file": "app/strategies/fibonacci_rsi_macd.py",
        "initial_capital": 10000.0,
        "strategy_params": {
            "swing_window": 20,
            "fib_level": 0.618,
            "rsi_threshold": 50.0
        },
        "use_database": True  # Use MongoDB instead of API
    }
    
    print("Testing Fibonacci RSI MACD Strategy")
    print("=" * 60)
    print(f"Endpoint: {endpoint}")
    print(f"Request data:")
    print(json.dumps(request_data, indent=2))
    print("=" * 60)
    
    try:
        response = requests.post(endpoint, json=request_data, timeout=60)
        
        if response.status_code == 200:
            result = response.json()
            print("\n✅ Backtest completed successfully!")
            print("\nResults:")
            print(f"  Total Return: {result.get('total_return', 0):.2%}")
            print(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {result.get('max_drawdown', 0):.2%}")
            print(f"  Win Rate: {result.get('win_rate', 0):.2%}")
            print(f"  Total Trades: {result.get('total_trades', 0)}")
            print(f"  Total Profit: {result.get('total_profit', 0):.2f}")
            print(f"  Final Value: {result.get('final_value', 0):.2f}")
            
            if result.get('sortino_ratio'):
                print(f"  Sortino Ratio: {result.get('sortino_ratio', 0):.2f}")
            if result.get('calmar_ratio'):
                print(f"  Calmar Ratio: {result.get('calmar_ratio', 0):.2f}")
            if result.get('profit_factor'):
                print(f"  Profit Factor: {result.get('profit_factor', 0):.2f}")
        else:
            print(f"\n❌ Error: HTTP {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to API server.")
        print("Make sure the backend server is running on http://localhost:8000")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    test_strategy_via_api()
