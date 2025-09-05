"""Data preprocessing module for cleaning and transforming market data."""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple, Any
from datetime import datetime, timedelta
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer

from ..config import get_config
from ..utils import get_logger, timing, validate_or_raise
from .models import BarData

logger = get_logger(__name__)


class DataPreprocessor:
    """Data preprocessing for market data cleaning and transformation."""
    
    def __init__(self):
        """Initialize data preprocessor."""
        self.config = get_config()
        self.scalers = {}
        self.imputers = {}
        
        logger.info("DataPreprocessor initialized")
    
    @timing()
    def clean_bars(
        self,
        bars: List[BarData],
        remove_outliers: bool = True,
        fill_missing: bool = True,
        validate_ohlc: bool = True
    ) -> List[BarData]:
        """Clean bar data by removing outliers and handling missing values.
        
        Args:
            bars: List of BarData objects
            remove_outliers: Whether to remove outliers
            fill_missing: Whether to fill missing values
            validate_ohlc: Whether to validate OHLC relationships
            
        Returns:
            List of cleaned BarData objects
        """
        if not bars:
            return bars
        
        logger.info(
            "Starting data cleaning",
            count=len(bars),
            symbol=bars[0].symbol if bars else None
        )
        
        # Convert to DataFrame for easier processing
        df = self._bars_to_dataframe(bars)
        
        # Validate OHLC relationships
        if validate_ohlc:
            df = self._validate_ohlc(df)
        
        # Remove outliers
        if remove_outliers:
            df = self._remove_outliers(df)
        
        # Fill missing values
        if fill_missing:
            df = self._fill_missing_values(df)
        
        # Convert back to BarData objects
        cleaned_bars = self._dataframe_to_bars(df, bars[0].symbol)
        
        logger.info(
            "Data cleaning completed",
            original_count=len(bars),
            cleaned_count=len(cleaned_bars),
            removed=len(bars) - len(cleaned_bars)
        )
        
        return cleaned_bars
    
    def _bars_to_dataframe(self, bars: List[BarData]) -> pd.DataFrame:
        """Convert BarData objects to DataFrame."""
        data = []
        for bar in bars:
            data.append({
                'timestamp': bar.timestamp,
                'open': bar.open,
                'high': bar.high,
                'low': bar.low,
                'close': bar.close,
                'volume': bar.volume,
                'trade_count': bar.trade_count,
                'vwap': bar.vwap
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        return df
    
    def _dataframe_to_bars(self, df: pd.DataFrame, symbol: str) -> List[BarData]:
        """Convert DataFrame back to BarData objects."""
        bars = []
        for timestamp, row in df.iterrows():
            bar = BarData(
                symbol=symbol,
                timestamp=timestamp,
                open=row['open'],
                high=row['high'],
                low=row['low'],
                close=row['close'],
                volume=int(row['volume']) if pd.notna(row['volume']) else 0,
                trade_count=int(row['trade_count']) if pd.notna(row['trade_count']) else None,
                vwap=row['vwap'] if pd.notna(row['vwap']) else None
            )
            bars.append(bar)
        
        return bars
    
    def _validate_ohlc(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate OHLC relationships and fix inconsistencies."""
        # Check for invalid OHLC relationships
        invalid_mask = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        )
        
        invalid_count = invalid_mask.sum()
        if invalid_count > 0:
            logger.warning(
                "Found invalid OHLC relationships",
                count=invalid_count
            )
            
            # Fix by adjusting high/low to encompass open/close
            df.loc[invalid_mask, 'high'] = df.loc[invalid_mask], [['open', 'close']].max(axis=1)
            df.loc[invalid_mask, 'low'] = df.loc[invalid_mask], [['open', 'close']].min(axis=1)
        
        # Remove rows with zero or negative prices
        price_mask = (
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0)
        )
        
        zero_price_count = price_mask.sum()
        if zero_price_count > 0:
            logger.warning(
                "Removing rows with zero or negative prices",
                count=zero_price_count
            )
            df = df[~price_mask]
        
        return df
    
    def _remove_outliers(
        self,
        df: pd.DataFrame,
        method: str = 'iqr',
        threshold: float = 3.0
    ) -> pd.DataFrame:
        """Remove outliers from price and volume data.
        
        Args:
            df: DataFrame with OHLCV data
            method: Outlier detection method ('iqr', 'zscore', 'modified_zscore')
            threshold: Threshold for outlier detection
        """
        original_count = len(df)
        
        if method == 'iqr':
            df = self._remove_outliers_iqr(df)
        elif method == 'zscore':
            df = self._remove_outliers_zscore(df, threshold)
        elif method == 'modified_zscore':
            df = self._remove_outliers_modified_zscore(df, threshold)
        else:
            logger.warning(f"Unknown outlier detection method: {method}")
            return df
        
        removed_count = original_count - len(df)
        if removed_count > 0:
            logger.info(
                "Outliers removed",
                method=method,
                removed=removed_count,
                remaining=len(df)
            )
        
        return df
    
    def _remove_outliers_iqr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers using Interquartile Range method."""
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            df = df[~outlier_mask]
        
        # Handle volume outliers separately (only upper bound)
        if 'volume' in df.columns:
            Q1_vol = df['volume'].quantile(0.25)
            Q3_vol = df['volume'].quantile(0.75)
            IQR_vol = Q3_vol - Q1_vol
            upper_bound_vol = Q3_vol + 3 * IQR_vol  # More lenient for volume
            
            volume_outlier_mask = df['volume'] > upper_bound_vol
            df = df[~volume_outlier_mask]
        
        return df
    
    def _remove_outliers_zscore(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Remove outliers using Z-score method."""
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            z_scores = np.abs(stats.zscore(df[col]))
            outlier_mask = z_scores > threshold
            df = df[~outlier_mask]
        
        return df
    
    def _remove_outliers_modified_zscore(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Remove outliers using Modified Z-score method."""
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            median = df[col].median()
            mad = np.median(np.abs(df[col] - median))
            modified_z_scores = 0.6745 * (df[col] - median) / mad
            
            outlier_mask = np.abs(modified_z_scores) > threshold
            df = df[~outlier_mask]
        
        return df
    
    def _fill_missing_values(
        self,
        df: pd.DataFrame,
        method: str = 'forward_fill'
    ) -> pd.DataFrame:
        """Fill missing values in the data.
        
        Args:
            df: DataFrame with potential missing values
            method: Method to fill missing values
        """
        missing_count = df.isnull().sum().sum()
        if missing_count == 0:
            return df
        
        logger.info(f"Filling {missing_count} missing values using {method}")
        
        if method == 'forward_fill':
            df = df.fillna(method='ffill')
        elif method == 'backward_fill':
            df = df.fillna(method='bfill')
        elif method == 'interpolate':
            df = df.interpolate(method='linear')
        elif method == 'mean':
            df = df.fillna(df.mean())
        elif method == 'median':
            df = df.fillna(df.median())
        else:
            logger.warning(f"Unknown fill method: {method}, using forward fill")
            df = df.fillna(method='ffill')
        
        # Drop any remaining NaN rows
        df = df.dropna()
        
        return df
    
    @timing()
    def add_technical_indicators(
        self,
        df: pd.DataFrame,
        indicators: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Add technical indicators to the DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            indicators: List of indicators to add
            
        Returns:
            DataFrame with technical indicators
        """
        if indicators is None:
            indicators = ['sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi_14', 'bb_20', 'macd']
        
        logger.info("Adding technical indicators", indicators=indicators)
        
        for indicator in indicators:
            try:
                if indicator.startswith('sma_'):
                    period = int(indicator.split('_')[1])
                    df[indicator] = self._calculate_sma(df['close'], period)
                
                elif indicator.startswith('ema_'):
                    period = int(indicator.split('_')[1])
                    df[indicator] = self._calculate_ema(df['close'], period)
                
                elif indicator.startswith('rsi_'):
                    period = int(indicator.split('_')[1])
                    df[indicator] = self._calculate_rsi(df['close'], period)
                
                elif indicator.startswith('bb_'):
                    period = int(indicator.split('_')[1])
                    bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(df['close'], period)
                    df[f'{indicator}_upper'] = bb_upper
                    df[f'{indicator}_middle'] = bb_middle
                    df[f'{indicator}_lower'] = bb_lower
                
                elif indicator == 'macd':
                    macd_line, signal_line, histogram = self._calculate_macd(df['close'])
                    df['macd_line'] = macd_line
                    df['macd_signal'] = signal_line
                    df['macd_histogram'] = histogram
                
                elif indicator.startswith('atr_'):
                    period = int(indicator.split('_')[1])
                    df[indicator] = self._calculate_atr(df, period)
                
                else:
                    logger.warning(f"Unknown indicator: {indicator}")
            
            except Exception as e:
                logger.error(f"Failed to calculate {indicator}", error=str(e))
        
        return df
    
    def _calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average."""
        return prices.rolling(window=period).mean()
    
    def _calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average."""
        return prices.ewm(span=period).mean()
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(
        self,
        prices: pd.Series,
        period: int = 20,
        std_dev: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band, sma, lower_band
    
    def _calculate_macd(
        self,
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)."""
        ema_fast = prices.ewm(span=fast_period).mean()
        ema_slow = prices.ewm(span=slow_period).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period).mean()
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    @timing()
    def normalize_data(
        self,
        df: pd.DataFrame,
        method: str = 'standard',
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Normalize data using specified method.
        
        Args:
            df: DataFrame to normalize
            method: Normalization method ('standard', 'minmax', 'robust')
            columns: Columns to normalize (default: all numeric columns)
            
        Returns:
            Normalized DataFrame
        """
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        logger.info(f"Normalizing data using {method} method", columns=columns)
        
        df_normalized = df.copy()
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            logger.error(f"Unknown normalization method: {method}")
            return df
        
        # Fit and transform the specified columns
        df_normalized[columns] = scaler.fit_transform(df[columns])
        
        # Store scaler for potential inverse transformation
        scaler_key = f"{method}_{hash(tuple(columns))}"
        self.scalers[scaler_key] = scaler
        
        return df_normalized
    
    def create_features(
        self,
        df: pd.DataFrame,
        lookback_periods: List[int] = [5, 10, 20],
        price_features: bool = True,
        volume_features: bool = True,
        volatility_features: bool = True
    ) -> pd.DataFrame:
        """Create additional features for machine learning.
        
        Args:
            df: DataFrame with OHLCV data
            lookback_periods: Periods for rolling calculations
            price_features: Whether to create price-based features
            volume_features: Whether to create volume-based features
            volatility_features: Whether to create volatility features
            
        Returns:
            DataFrame with additional features
        """
        logger.info("Creating features for machine learning")
        
        df_features = df.copy()
        
        if price_features:
            # Price returns
            df_features['returns'] = df['close'].pct_change()
            
            # Price ratios
            df_features['high_low_ratio'] = df['high'] / df['low']
            df_features['close_open_ratio'] = df['close'] / df['open']
            
            # Price position within the day's range
            df_features['price_position'] = (
                (df['close'] - df['low']) / (df['high'] - df['low'])
            ).fillna(0.5)
            
            # Rolling features
            for period in lookback_periods:
                df_features[f'returns_{period}d'] = df['close'].pct_change(period)
                df_features[f'high_{period}d'] = df['high'].rolling(period).max()
                df_features[f'low_{period}d'] = df['low'].rolling(period).min()
                df_features[f'price_rank_{period}d'] = (
                    df['close'].rolling(period).rank(pct=True)
                )
        
        if volume_features:
            # Volume features
            df_features['volume_sma_ratio'] = (
                df['volume'] / df['volume'].rolling(20).mean()
            )
            
            for period in lookback_periods:
                df_features[f'volume_{period}d'] = df['volume'].rolling(period).mean()
                df_features[f'volume_std_{period}d'] = df['volume'].rolling(period).std()
        
        if volatility_features:
            # Volatility features
            for period in lookback_periods:
                df_features[f'volatility_{period}d'] = (
                    df['returns'].rolling(period).std() * np.sqrt(252)
                )
                df_features[f'price_range_{period}d'] = (
                    (df['high'] - df['low']).rolling(period).mean()
                )
        
        # Remove infinite and NaN values
        df_features = df_features.replace([np.inf, -np.inf], np.nan)
        df_features = df_features.fillna(method='ffill').fillna(0)
        
        logger.info(
            "Feature creation completed",
            original_columns=len(df.columns),
            new_columns=len(df_features.columns),
            added_features=len(df_features.columns) - len(df.columns)
        )
        
        return df_features
    
    def resample_data(
        self,
        df: pd.DataFrame,
        target_frequency: str,
        aggregation_rules: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """Resample data to different frequency.
        
        Args:
            df: DataFrame with datetime index
            target_frequency: Target frequency ('1H', '1D', etc.)
            aggregation_rules: Custom aggregation rules for columns
            
        Returns:
            Resampled DataFrame
        """
        if aggregation_rules is None:
            aggregation_rules = {
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'trade_count': 'sum',
                'vwap': 'mean'
            }
        
        logger.info(
            "Resampling data",
            target_frequency=target_frequency,
            original_shape=df.shape
        )
        
        # Apply aggregation rules only to existing columns
        existing_rules = {
            col: rule for col, rule in aggregation_rules.items()
            if col in df.columns
        }
        
        # For other columns, use mean as default
        for col in df.columns:
            if col not in existing_rules:
                if df[col].dtype in ['float64', 'int64']:
                    existing_rules[col] = 'mean'
                else:
                    existing_rules[col] = 'first'
        
        resampled_df = df.resample(target_frequency).agg(existing_rules)
        
        # Remove rows with NaN values (weekends, holidays)
        resampled_df = resampled_df.dropna()
        
        logger.info(
            "Resampling completed",
            new_shape=resampled_df.shape
        )
        
        return resampled_df
    
    def split_data(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        validation_ratio: float = 0.15,
        test_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets.
        
        Args:
            df: DataFrame to split
            train_ratio: Ratio for training set
            validation_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        # Validate ratios
        total_ratio = train_ratio + validation_ratio + test_ratio
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
        
        n_samples = len(df)
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + validation_ratio))
        
        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        test_df = df.iloc[val_end:]
        
        logger.info(
            "Data split completed",
            total_samples=n_samples,
            train_samples=len(train_df),
            val_samples=len(val_df),
            test_samples=len(test_df)
        )
        
        return train_df, val_df, test_df
    
    def get_data_quality_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality report.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with data quality metrics
        """
        report = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'date_range': {
                'start': df.index.min() if hasattr(df.index, 'min') else None,
                'end': df.index.max() if hasattr(df.index, 'max') else None
            }
        }
        
        # Numeric columns statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            report['numeric_stats'] = df[numeric_cols].describe().to_dict()
        
        # Check for duplicates
        report['duplicate_rows'] = df.duplicated().sum()
        
        # Check for constant columns
        report['constant_columns'] = [
            col for col in df.columns
            if df[col].nunique() <= 1
        ]
        
        logger.info("Data quality report generated", report_keys=list(report.keys()))
        
        return report