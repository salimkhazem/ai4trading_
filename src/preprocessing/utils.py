"""
Preprocessing utilities for AI4Trading system.

This module provides core utilities for financial data preprocessing including
data cleaning, feature engineering, and configuration management.
"""

import logging
import resource
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import gc
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration class for preprocessing parameters ensuring reproducibility."""
    
    # Data paths and days
    base_data_path: str
    train_days: List[str]
    test_days: List[str]
    output_dir: str
    
    # Bar configuration
    bar_type: str = 'time'  # 'time' or 'volume'
    resample_freq: str = '1s'  # for time bars
    nb_bars: int = 10000  # for volume bars
    
    # Window configuration
    window_length: int = 150
    target_window_length: int = 30
    target_column: str = 'wmp_mean'
    
    # Symbol filtering
    symbols_to_exclude: List[str] = None
    symbols_to_keep: List[str] = None
    
    # Processing configuration
    n_jobs: int = -1
    memory_efficient: bool = True
    
    # Feature engineering
    enable_technical_indicators: bool = True
    enable_order_flow_features: bool = True
    enable_time_features: bool = False
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.bar_type not in ['time', 'volume']:
            raise ValueError(f"bar_type must be 'time' or 'volume', got {self.bar_type}")
        
        if self.symbols_to_exclude is None:
            self.symbols_to_exclude = []
        
        if self.symbols_to_keep is None:
            self.symbols_to_keep = []
        
        if self.symbols_to_exclude and self.symbols_to_keep:
            raise ValueError("Cannot specify both symbols_to_exclude and symbols_to_keep")


class MemoryManager:
    """Utility class for memory management and monitoring."""
    
    @staticmethod
    def get_memory_usage_gb() -> float:
        """Get current peak memory usage in gigabytes."""
        try:
            return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)
        except (AttributeError, OSError):
            return -1.0
    
    @staticmethod
    def cleanup() -> None:
        """Force garbage collection."""
        gc.collect()
    
    @classmethod
    def log_memory_usage(cls, step_name: str) -> float:
        """Log current memory usage for a processing step."""
        mem_usage = cls.get_memory_usage_gb()
        if mem_usage > 0:
            logger.info(f"{step_name} - Memory usage: {mem_usage:.2f} GB")
        return mem_usage


class DataCleaner:
    """Handles data cleaning operations with consistent methodology."""
    
    DEFAULT_COLUMNS_TO_DROP = [
        'exch_time', 'exchange', 'first_sequence_number', 'last_sequence_number',
        'first_sym_sequence', 'last_sym_sequence', 'first_time', 'first_exch_time',
        'event_id', 'date'
    ]
    
    @classmethod
    def clean_raw_data(
        cls, 
        df_raw: pd.DataFrame, 
        columns_to_drop: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Clean raw financial data by removing NaNs and unnecessary columns.
        
        Args:
            df_raw: Raw input DataFrame
            columns_to_drop: Custom list of columns to drop, uses default if None
            
        Returns:
            Cleaned DataFrame
            
        Raises:
            ValueError: If DataFrame becomes empty after cleaning
        """
        logger.info("Starting data cleaning...")
        initial_rows = len(df_raw)
        
        # Drop NaN values
        df_clean = df_raw.dropna().copy()
        rows_dropped_nan = initial_rows - len(df_clean)
        percent_dropped_nan = (rows_dropped_nan / initial_rows * 100) if initial_rows > 0 else 0.0
        
        logger.info(
            f"Dropped {rows_dropped_nan:,} rows with NaNs "
            f"({percent_dropped_nan:.2f}%). Shape: {df_clean.shape}"
        )
        
        if df_clean.empty:
            raise ValueError("DataFrame became empty after dropping NaN values")
        
        # Drop unnecessary columns
        if columns_to_drop is None:
            columns_to_drop = cls.DEFAULT_COLUMNS_TO_DROP
            
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
        
        if existing_cols_to_drop:
            df_clean.drop(columns=existing_cols_to_drop, inplace=True)
            logger.info(f"Dropped columns: {existing_cols_to_drop}. Final shape: {df_clean.shape}")
        
        logger.info(f"Data cleaning complete. Final shape: {df_clean.shape}")
        return df_clean


class FeatureEngineer:
    """
    Comprehensive feature engineering for high-frequency trading data.
    
    Computes microstructure features, technical indicators, and order flow metrics
    with proper error handling and logging.
    """
    
    @staticmethod
    def compute_basic_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute basic market microstructure features."""
        logger.debug("Computing basic microstructure features...")
        
        # Basic price features
        df["mid_price"] = (df["L1_bid_price"] + df["L1_ask_price"]) / 2
        df["weighted_mid_price"] = (
            (df["L1_bid_price"] * df["L1_ask_size"] + df["L1_ask_price"] * df["L1_bid_size"]) /
            (df["L1_bid_size"] + df["L1_ask_size"])
        )
        df["spread"] = df["L1_ask_price"] - df["L1_bid_price"]
        
        return df
    
    @staticmethod
    def compute_order_book_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute order book imbalance and depth features."""
        logger.debug("Computing order book features...")
        
        # Order Book Imbalance (OBI) for all levels
        for i in range(1, 11):
            bid_col = f"L{i}_bid_size"
            ask_col = f"L{i}_ask_size"
            if bid_col in df.columns and ask_col in df.columns:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    df[f"obi_L{i}"] = (df[bid_col] - df[ask_col]) / (df[bid_col] + df[ask_col])
        
        # Aggregate depth metrics
        bid_cols = [f"L{i}_bid_size" for i in range(1, 11) if f"L{i}_bid_size" in df.columns]
        ask_cols = [f"L{i}_ask_size" for i in range(1, 11) if f"L{i}_ask_size" in df.columns]
        
        if bid_cols and ask_cols:
            df["cum_bid_vol_10"] = df[bid_cols].sum(axis=1)
            df["cum_ask_vol_10"] = df[ask_cols].sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                df["depth_imbalance_10"] = (
                    (df["cum_bid_vol_10"] - df["cum_ask_vol_10"]) /
                    (df["cum_bid_vol_10"] + df["cum_ask_vol_10"])
                )
        
        return df
    
    @staticmethod
    def compute_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute liquidity impact and price slope features."""
        logger.debug("Computing liquidity features...")
        
        # Liquidity impact
        if all(col in df.columns for col in ["L1_bid_price", "L2_bid_price"]):
            df["liquidity_impact_bid"] = df["L1_bid_price"] - df["L2_bid_price"]
        
        if all(col in df.columns for col in ["L2_ask_price", "L1_ask_price"]):
            df["liquidity_impact_ask"] = df["L2_ask_price"] - df["L1_ask_price"]
        
        # Price slopes
        if all(col in df.columns for col in ["L1_bid_price", "L5_bid_price"]):
            df["bid_price_slope"] = (df["L1_bid_price"] - df["L5_bid_price"]) / 4
        
        if all(col in df.columns for col in ["L5_ask_price", "L1_ask_price"]):
            df["ask_price_slope"] = (df["L5_ask_price"] - df["L1_ask_price"]) / 4
        
        return df
    
    @staticmethod
    def compute_order_flow_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute order flow imbalance and related features."""
        logger.debug("Computing order flow features...")
        
        # Order Flow Imbalance (OFI)
        if all(col in df.columns for col in ["L1_bid_size", "L1_ask_size"]):
            df["ofi"] = df["L1_bid_size"].diff() - df["L1_ask_size"].diff()
        
        # Returns
        if "weighted_mid_price" in df.columns:
            df["return_1"] = df["weighted_mid_price"].pct_change()
            df["return_5"] = df["weighted_mid_price"].pct_change(5)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                df["log_return_1"] = np.log(df["weighted_mid_price"] / df["weighted_mid_price"].shift(1))
                df["log_return_5"] = np.log(df["weighted_mid_price"] / df["weighted_mid_price"].shift(5))
        
        return df
    
    @staticmethod
    def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Compute technical analysis indicators."""
        logger.debug("Computing technical indicators...")
        
        if "weighted_mid_price" not in df.columns:
            logger.warning("weighted_mid_price not found, skipping technical indicators")
            return df
        
        wmp = df["weighted_mid_price"]
        
        # Moving averages
        for window in [10, 20, 100]:
            df[f"sma_{window}"] = wmp.rolling(window=window, min_periods=1).mean()
            df[f"ema_{window}"] = wmp.ewm(span=window, adjust=False, min_periods=1).mean()
            df[f"momentum_{window}"] = wmp - wmp.shift(window)
        
        # RSI calculation
        FeatureEngineer._compute_rsi(df, [10, 20, 100])
        
        # Bollinger Bands
        for window in [10, 20, 100]:
            FeatureEngineer._compute_bollinger_bands(df, window)
        
        return df
    
    @staticmethod
    def _compute_rsi(df: pd.DataFrame, periods: List[int]) -> None:
        """Compute Relative Strength Index for given periods."""
        for period in periods:
            delta = df["weighted_mid_price"].diff()
            gain = delta.clip(lower=0)
            loss = -delta.clip(upper=0)
            
            avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rs = avg_gain / avg_loss
                df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
                df[f"rsi_signal_{period}"] = np.where(
                    df[f"rsi_{period}"] > 70, -1, 
                    np.where(df[f"rsi_{period}"] < 30, 1, 0)
                )
    
    @staticmethod
    def _compute_bollinger_bands(df: pd.DataFrame, window: int) -> None:
        """Compute Bollinger Bands for given window."""
        rolling_mean = df["weighted_mid_price"].rolling(window=window, min_periods=1).mean()
        rolling_std = df["weighted_mid_price"].rolling(window=window, min_periods=1).std()
        
        df[f"bollinger_upper_{window}"] = rolling_mean + 2 * rolling_std
        df[f"bollinger_lower_{window}"] = rolling_mean - 2 * rolling_std
        df[f"band_width_{window}"] = df[f"bollinger_upper_{window}"] - df[f"bollinger_lower_{window}"]
        df[f"bollinger_signal_{window}"] = np.where(
            df["weighted_mid_price"] > df[f"bollinger_upper_{window}"], -1,
            np.where(df["weighted_mid_price"] < df[f"bollinger_lower_{window}"], 1, 0)
        )
    
    @classmethod
    def compute_all_features(
        cls, 
        df: pd.DataFrame, 
        config: PreprocessingConfig
    ) -> pd.DataFrame:
        """
        Compute all features based on configuration.
        
        Args:
            df: Input DataFrame with LOB data
            config: Preprocessing configuration
            
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature engineering...")
        
        # Always compute basic features
        df = cls.compute_basic_features(df)
        df = cls.compute_order_book_features(df)
        df = cls.compute_liquidity_features(df)
        
        if config.enable_order_flow_features:
            df = cls.compute_order_flow_features(df)
        
        if config.enable_technical_indicators:
            df = cls.compute_technical_indicators(df)
        
        # Fill NaN values that may result from calculations
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        return df


class ValidationUtils:
    """Utility functions for data validation and quality checks."""
    
    @staticmethod
    def validate_dataframe(
        df: pd.DataFrame, 
        required_columns: List[str],
        name: str = "DataFrame"
    ) -> None:
        """
        Validate DataFrame has required columns and is not empty.
        
        Args:
            df: DataFrame to validate
            required_columns: List of required column names
            name: Name for logging purposes
            
        Raises:
            ValueError: If validation fails
        """
        if df.empty:
            raise ValueError(f"{name} is empty")
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"{name} missing required columns: {missing_cols}")
    
    @staticmethod
    def validate_feature_array(
        feature_array: np.ndarray, 
        array_name: str = "X_windows"
    ) -> bool:
        """
        Check NumPy array for NaN or Infinity values.
        
        Args:
            feature_array: Array to validate
            array_name: Name for logging purposes
            
        Returns:
            True if array is valid
            
        Raises:
            ValueError: If invalid values are detected
        """
        logger.info(f"Validating array: '{array_name}'...")
        
        has_nan = np.isnan(feature_array).any()
        has_inf = np.isinf(feature_array).any()
        
        if has_nan or has_inf:
            nan_count = np.isnan(feature_array).sum() if has_nan else 0
            inf_count = np.isinf(feature_array).sum() if has_inf else 0
            error_message = (
                f"Invalid values found in array '{array_name}'! "
                f"NaN count: {nan_count}, Inf count: {inf_count}"
            )
            logger.error(error_message)
            raise ValueError(error_message)
        
        logger.info(f"Array '{array_name}' validation passed.")
        return True


def load_day_data(base_data_path: str, day: str) -> Optional[pd.DataFrame]:
    """
    Load raw data for a specific day.
    
    Args:
        base_data_path: Base path to data directory
        day: Day string (e.g., '20250212')
        
    Returns:
        DataFrame with raw data or None if loading fails
    """
    try:
        # This is a placeholder - implement actual data loading logic
        # based on your data format (parquet, csv, etc.)
        data_path = Path(base_data_path) / day
        
        if not data_path.exists():
            logger.warning(f"Data path does not exist: {data_path}")
            return None
        
        # Example for parquet files - adjust based on your data format
        files = list(data_path.glob("*.parquet"))
        if not files:
            logger.warning(f"No data files found in {data_path}")
            return None
        
        logger.info(f"Loading data for day {day} from {len(files)} files...")
        dfs = []
        for file in files:
            df = pd.read_parquet(file)
            dfs.append(df)
        
        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(combined_df)} rows for day {day}")
        return combined_df
        
    except Exception as e:
        logger.error(f"Failed to load data for day {day}: {e}")
        return None 