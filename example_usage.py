#!/usr/bin/env python3
"""Example usage of the Alpaca Trading Pipeline."""

import asyncio
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import pipeline components
from alpaca_pipeline.config import Config
from alpaca_pipeline.data.collector import DataCollector
from alpaca_pipeline.data.preprocessor import DataPreprocessor
from alpaca_pipeline.data.storage import DataStorage
from alpaca_pipeline.strategies.examples import MovingAverageStrategy, RSIStrategy
from alpaca_pipeline.strategies.manager import StrategyManager
from alpaca_pipeline.backtesting.engine import BacktestEngine, BacktestConfig
from alpaca_pipeline.utils.logging import setup_logging


async def main():
    """Main example function."""
    # Setup logging
    setup_logging()
    
    print("Alpaca Trading Pipeline Example")
    print("=" * 50)
    
    # Load configuration
    config = Config()
    print(f"Configuration loaded (Paper Trading: {config.alpaca.paper_trading})")
    
    # Test data collector initialization (without API calls)
    print("Testing data collector initialization...")
    try:
        if config.alpaca.api_key and config.alpaca.secret_key:
            data_collector = DataCollector()
            data_preprocessor = DataPreprocessor()
            data_storage = DataStorage(config.database)
            print("Components initialized successfully")
            
            # Example 1: Data Collection and Processing
            print("\nExample 1: Data Collection and Processing")
            print("-" * 40)
            
            symbols = ['AAPL', 'MSFT', 'GOOGL']
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            
            # Collect historical data
            print(f"Collecting data for {symbols} from {start_date.date()} to {end_date.date()}...")
            
            for symbol in symbols:
                # Get historical bars
                bars = await data_collector.get_historical_bars(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    timeframe='1Day'
                )
                
                if bars:
                    print(f"  {symbol}: {len(bars)} bars collected")
                    
                    # Process data
                    processed_data = data_preprocessor.process_bars(bars)
                    print(f"  {symbol}: Data processed with {len(processed_data.columns)} features")
                    
                    # Store data (optional - requires database setup)
                    # await data_storage.store_bars(symbol, bars)
                    # print(f"  {symbol}: Data stored")
                else:
                    print(f"  {symbol}: No data collected")
        else:
            print("Alpaca API keys not configured - skipping data collection test")
            print("To test data collection, set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
            
    except Exception as e:
        print(f"Data collection test failed: {e}")
        print("This is expected without valid API credentials")
    
    # Example 2: Strategy Development
    print("\nExample 2: Strategy Development")
    print("-" * 40)
    
    # Create strategies
    ma_strategy = MovingAverageStrategy(
        symbols=['AAPL', 'GOOGL'],
        short_window=10,
        long_window=20,
        position_size=0.1
    )
    
    rsi_strategy = RSIStrategy(
        symbols=['AAPL', 'GOOGL'],
        rsi_period=14,
        oversold_threshold=30,
        overbought_threshold=70,
        position_size=0.1
    )
    
    print(f"Created strategies: {ma_strategy.strategy_id}, {rsi_strategy.strategy_id}")
    
    # Strategy manager
    strategy_manager = StrategyManager()
    strategy_manager.add_strategy(ma_strategy)
    strategy_manager.add_strategy(rsi_strategy)
    
    print(f"Strategy manager initialized with {len(strategy_manager.strategies)} strategies")
    
    # Example 3: Backtesting
    print("\nExample 3: Backtesting")
    print("-" * 40)
    
    try:
        # Define date range for backtesting
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        # Create backtest configuration
        backtest_config = BacktestConfig(
            initial_capital=100000.0,
            start_date=start_date,
            end_date=end_date,
            commission=0.005,
            slippage=0.001
        )
        
        # Initialize backtest engine
        backtest_engine = BacktestEngine(backtest_config)
        
        # Add strategies
        backtest_engine.add_strategy(ma_strategy)
        
        print("Backtest engine configured")
        print(f"  Initial Capital: ${backtest_config.initial_capital:,.2f}")
        print(f"  Period: {start_date.date()} to {end_date.date()}")
        print(f"  Commission: {backtest_config.commission*100:.3f}%")
        
        # Note: Actual backtesting would require market data
        print("\nNote: Full backtesting requires historical market data")
        print("   To run complete backtest, ensure data collection is working")
        
    except Exception as e:
        print(f"Backtesting setup error: {e}")
    
    # Example 4: Configuration Check
    print("\nExample 4: Configuration Check")
    print("-" * 40)
    
    print("Current configuration:")
    print(f"  API Key: {'Set' if config.alpaca.api_key else 'Missing'}")
    print(f"  Secret Key: {'Set' if config.alpaca.secret_key else 'Missing'}")
    print(f"  Paper Trading: {config.alpaca.paper_trading}")
    print(f"  Database: {config.database.database_type}")
    print(f"  Data Provider: {config.data.default_provider}")
    
    # Example 5: Environment Setup Check
    print("\nExample 5: Environment Setup")
    print("-" * 40)
    
    required_env_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("Missing environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nTo fix this:")
        print("   1. Copy .env.example to .env")
        print("   2. Fill in your Alpaca API credentials")
        print("   3. Run the script again")
    else:
        print("All required environment variables are set")
    
    print("\nExample completed!")
    print("\nNext steps:")
    print("  1. Set up your Alpaca API credentials in .env file")
    print("  2. Run backtests with your own strategies")
    print("  3. Explore live trading capabilities (paper trading recommended)")
    print("  4. Check the documentation for advanced features")


if __name__ == "__main__":
    asyncio.run(main())