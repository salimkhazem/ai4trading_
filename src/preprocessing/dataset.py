"""
Dataset processing module : TODO: add multiprocessing and accelerate the pipeline

This module provides classes for constructing different types of bars (time/volume)
and generating sequential windows for machine learning training.
"""

import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import gc
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import PreprocessingConfig, MemoryManager, ValidationUtils

logger = logging.getLogger(__name__)


class BaseBarConstructor(ABC):
    """Abstract base class for bar construction strategies."""
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize bar constructor with configuration.
        
        Args:
            config: Preprocessing configuration object
        """
        self.config = config
    
    @abstractmethod
    def construct_bars(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Construct bars from raw data.
        
        Args:
            df: Input DataFrame with raw data
            symbol: Optional symbol name for logging
            
        Returns:
            DataFrame with constructed bars
        """
        pass
    
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Validate input data for bar construction."""
        if df.empty:
            raise ValueError("Input DataFrame is empty")
        
        if 'time' not in df.columns:
            raise ValueError("Input DataFrame must contain 'time' column")
        
        # Ensure time column is datetime and sorted
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
            df['time'] = pd.to_datetime(df['time'])
        
        if not df['time'].is_monotonic_increasing:
            logger.warning("Data not sorted by time, sorting now...")
            df.sort_values('time', inplace=True)
            df.reset_index(drop=True, inplace=True)


class TimeBarConstructor(BaseBarConstructor):
    """Constructs time-based bars by resampling data at fixed time intervals."""
    
    def construct_bars(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Construct time bars by resampling at fixed intervals.
        
        Args:
            df: Input DataFrame with tick data
            symbol: Optional symbol name for logging
            
        Returns:
            DataFrame with time bars
        """
        self._validate_input(df)
        
        logger.debug(f"Constructing time bars with frequency {self.config.resample_freq}")
        
        # Set time as index for resampling
        df_indexed = df.set_index('time')
        
        # Define aggregation functions for different column types
        agg_funcs = self._get_aggregation_functions(df_indexed)
        
        # Resample and aggregate
        resampled = df_indexed.resample(self.config.resample_freq).agg(agg_funcs)
        
        # Flatten column names if they are MultiIndex
        if isinstance(resampled.columns, pd.MultiIndex):
            resampled.columns = ['_'.join(col).strip() for col in resampled.columns.values]
        
        # Reset index to make time a regular column
        resampled = resampled.reset_index()
        
        # Remove rows with no data (all NaN)
        resampled = resampled.dropna(subset=['weighted_mid_price'])
        
        # Add bar metadata
        resampled['bar_start_time'] = resampled['time']
        resampled['bar_end_time'] = resampled['time']
        resampled['bar_duration_seconds'] = pd.Timedelta(self.config.resample_freq).total_seconds()
        
        logger.info(f"Constructed {len(resampled)} time bars")
        return resampled
    
    def _get_aggregation_functions(self, df: pd.DataFrame) -> Dict[str, Union[str, Dict[str, str]]]:
        """Define aggregation functions for different types of columns."""
        agg_funcs = {}
        
        # Price columns - use OHLC
        price_cols = [col for col in df.columns if 'price' in col.lower()]
        for col in price_cols:
            agg_funcs[col] = ['first', 'last', 'min', 'max', 'mean']
        
        # Size/volume columns - use sum and mean
        size_cols = [col for col in df.columns if 'size' in col.lower()]
        for col in size_cols:
            agg_funcs[col] = ['sum', 'mean', 'min', 'max']
        
        # Count columns - use sum and mean
        count_cols = [col for col in df.columns if '_no' in col.lower()]
        for col in count_cols:
            agg_funcs[col] = ['sum', 'mean']
        
        # Special handling for key columns
        key_columns = {
            'weighted_mid_price': ['first', 'last', 'min', 'max', 'mean', 'std'],
            'mid_price': ['first', 'last', 'min', 'max', 'mean'],
            'spread': ['first', 'last', 'min', 'max', 'mean'],
            'sym': 'first'
        }
        
        for col, func in key_columns.items():
            if col in df.columns:
                agg_funcs[col] = func
        
        return agg_funcs


class VolumeBarConstructor(BaseBarConstructor):
    """Constructs volume-based bars using cumulative volume thresholds."""
    
    def construct_bars(self, df: pd.DataFrame, symbol: Optional[str] = None) -> pd.DataFrame:
        """
        Construct volume bars based on cumulative activity volume.
        
        Args:
            df: Input DataFrame with tick data
            symbol: Optional symbol name for logging
            
        Returns:
            DataFrame with volume bars
        """
        self._validate_input(df)
        
        # Calculate target number of bars for this symbol
        nb_bars = self._calculate_optimal_bars(df)
        
        logger.debug(f"Constructing {nb_bars} volume bars for symbol {symbol or 'unknown'}")
        
        # Calculate activity volume (L1 bid + ask size)
        df = self._calculate_activity_volume(df)
        
        # Calculate volume threshold per bar
        total_volume = df['cumulative_activity_volume'].iloc[-1]
        volume_threshold = total_volume / nb_bars
        
        if volume_threshold <= 0:
            raise ValueError(f"Volume threshold is non-positive: {volume_threshold}")
        
        # Assign bar IDs based on volume thresholds
        df['bar_id'] = (df['cumulative_activity_volume'] // volume_threshold).astype(int)
        
        # Ensure first entries belong to bar 0
        if not df.empty:
            first_bar_id = df.iloc[0]['bar_id']
            df.loc[df['bar_id'] == first_bar_id, 'bar_id'] = 0
        
        # Remove partial last bar
        max_complete_bar_id = int(np.floor(total_volume / volume_threshold)) - 1
        if max_complete_bar_id < 0:
            raise ValueError("Not enough volume to form complete bars")
        
        df_complete = df[df['bar_id'] <= max_complete_bar_id].copy()
        
        # Aggregate data by bar_id
        volume_bars = self._aggregate_volume_bars(df_complete)
        
        logger.info(f"Constructed {len(volume_bars)} volume bars")
        return volume_bars
    
    def _calculate_optimal_bars(self, df: pd.DataFrame) -> int:
        """Calculate optimal number of bars based on data size."""
        min_snapshots_per_bar = 150
        max_bars = 7500
        
        nb_snapshots = len(df)
        optimal_bars = nb_snapshots // min_snapshots_per_bar
        return min(optimal_bars, max_bars, self.config.nb_bars)
    
    def _calculate_activity_volume(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate cumulative activity volume (L1 bid + ask size)."""
        if not all(col in df.columns for col in ['L1_bid_size', 'L1_ask_size']):
            raise ValueError("L1 bid and ask size columns required for volume bars")
        
        df['L1_total_volume'] = df['L1_bid_size'] + df['L1_ask_size']
        df['L1_total_volume'] = df['L1_total_volume'].fillna(0)
        df['cumulative_activity_volume'] = df['L1_total_volume'].cumsum()
        
        return df
    
    def _aggregate_volume_bars(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate tick data into volume bars."""
        # Define columns to exclude from generic aggregation
        exclude_cols = [
            'time', 'weighted_mid_price', 'L1_total_volume', 'bar_id',
            'cumulative_activity_volume', 'sym'
        ]
        
        # Identify feature columns for aggregation
        feature_cols = [
            col for col in df.columns 
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col])
        ]
        
        def aggregate_bar(group: pd.DataFrame) -> pd.Series:
            """Aggregate function for each volume bar."""
            results = {
                'bar_start_time': group['time'].iloc[0],
                'bar_end_time': group['time'].iloc[-1],
                'num_snapshots_in_bar': len(group),
                'actual_volume_in_bar': group['L1_total_volume'].sum(),
                'wmp_mean': group['weighted_mid_price'].mean(),
                'wmp_first': group['weighted_mid_price'].iloc[0],
                'wmp_last': group['weighted_mid_price'].iloc[-1],
                'wmp_min': group['weighted_mid_price'].min(),
                'wmp_max': group['weighted_mid_price'].max(),
                'wmp_median': group['weighted_mid_price'].median(),
                'wmp_std': group['weighted_mid_price'].std(),
            }
            
            # Aggregate other features
            for col in feature_cols:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    results[f"{col}_mean"] = group[col].mean()
                    results[f"{col}_sum"] = group[col].sum()
                    results[f"{col}_min"] = group[col].min()
                    results[f"{col}_max"] = group[col].max()
                    results[f"{col}_std"] = group[col].std()
                    results[f"{col}_median"] = group[col].median()
            
            return pd.Series(results)
        
        # Group by bar_id and aggregate
        volume_bars = df.groupby('bar_id').apply(aggregate_bar, include_groups=False)
        
        # Calculate bar duration
        volume_bars['bar_duration_seconds'] = (
            volume_bars['bar_end_time'] - volume_bars['bar_start_time']
        ).dt.total_seconds()
        
        # Reset index and fill NaN values
        volume_bars = volume_bars.reset_index(drop=True)
        volume_bars = volume_bars.fillna(0)
        
        return volume_bars


class WindowGenerator:
    """Generates sequential sliding windows from bar data for ML training."""
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize window generator with configuration.
        
        Args:
            config: Preprocessing configuration object
        """
        self.config = config
    
    def generate_windows(
        self, 
        df_bars: pd.DataFrame
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[List[str]]]:
        """
        Generate sequential sliding windows from bar data.
        
        Args:
            df_bars: DataFrame with bar data
            
        Returns:
            Tuple of (X_windows, target_windows, window_info, feature_cols)
            Returns (None, None, None, None) if no windows can be generated
        """
        logger.debug(
            f"Generating windows: input_len={self.config.window_length}, "
            f"target_len={self.config.target_window_length}, "
            f"target_col='{self.config.target_column}'"
        )
        
        # Validate input
        required_cols = ['bar_end_time', self.config.target_column]
        ValidationUtils.validate_dataframe(df_bars, required_cols, "Bar DataFrame")
        
        # Identify feature columns
        feature_cols = self._identify_feature_columns(df_bars)
        if not feature_cols:
            logger.error("No feature columns identified for windowing")
            return None, None, None, None
        
        # Prepare feature data
        feature_data = self._prepare_feature_data(df_bars, feature_cols)
        times = df_bars['bar_end_time'].values
        targets = df_bars[self.config.target_column].astype(np.float32).values
        
        # Check if we have enough data
        total_bars = len(df_bars)
        required_bars = self.config.window_length + self.config.target_window_length
        if total_bars < required_bars:
            logger.warning(
                f"Not enough bars ({total_bars}) to create windows "
                f"(required: {required_bars})"
            )
            return None, None, None, None
        
        # Generate windows
        X_windows_list = []
        target_windows_list = []
        window_info_list = []
        
        for i in range(total_bars - required_bars + 1):
            # Input window
            input_end_idx = i + self.config.window_length
            input_window = feature_data[i:input_end_idx]
            
            # Target window
            target_start_idx = input_end_idx
            target_end_idx = target_start_idx + self.config.target_window_length
            target_window = targets[target_start_idx:target_end_idx]
            
            X_windows_list.append(input_window)
            target_windows_list.append(target_window)
            
            # Window metadata
            window_info = {
                'window_end_time': times[input_end_idx - 1],
                'last_target_in_window': targets[input_end_idx - 1]
            }
            window_info_list.append(window_info)
        
        if not X_windows_list:
            logger.warning("No windows generated")
            return self._empty_results(feature_cols)
        
        # Stack into arrays
        X_windows = np.stack(X_windows_list, axis=0).astype(np.float32)
        target_windows = np.stack(target_windows_list, axis=0).astype(np.float32)
        window_info = pd.DataFrame(window_info_list)
        window_info['window_end_time'] = pd.to_datetime(window_info['window_end_time'])
        
        logger.info(
            f"Generated {len(X_windows)} windows - "
            f"X: {X_windows.shape}, Target: {target_windows.shape}"
        )
        
        return X_windows, target_windows, window_info, feature_cols
    
    def _identify_feature_columns(self, df_bars: pd.DataFrame) -> List[str]:
        """Identify columns to use as features for windowing."""
        exclude_cols = [
            'bar_id', 'bar_start_time', 'bar_end_time', 
            'num_snapshots_in_bar', 'actual_volume_in_bar', 'bar_duration_seconds'
        ]
        
        return [col for col in df_bars.columns if col not in exclude_cols]
    
    def _prepare_feature_data(self, df_bars: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Prepare and validate feature data for window generation."""
        try:
            # Convert to float32 and handle NaN/Inf values
            feature_data = df_bars[feature_cols].fillna(0).astype(np.float32).values
            
            if np.isnan(feature_data).any() or np.isinf(feature_data).any():
                logger.warning("NaN or Inf values found after fillna, applying robust handling")
                feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
            
            return feature_data
            
        except Exception as e:
            logger.error(f"Error preparing feature data: {e}")
            raise
    
    def _empty_results(self, feature_cols: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        """Return empty results with correct shapes."""
        empty_x = np.array([]).reshape(0, self.config.window_length, len(feature_cols)).astype(np.float32)
        empty_target = np.array([]).reshape(0, self.config.target_window_length).astype(np.float32)
        empty_info = pd.DataFrame(columns=['window_end_time', 'last_target_in_window'])
        return empty_x, empty_target, empty_info, feature_cols


class SymbolProcessor:
    """Processes individual symbols with bar construction and window generation."""
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize symbol processor with configuration.
        
        Args:
            config: Preprocessing configuration object
        """
        self.config = config
        self.bar_constructor = self._create_bar_constructor()
        self.window_generator = WindowGenerator(config)
    
    def _create_bar_constructor(self) -> BaseBarConstructor:
        """Create appropriate bar constructor based on configuration."""
        if self.config.bar_type == 'time':
            return TimeBarConstructor(self.config)
        elif self.config.bar_type == 'volume':
            return VolumeBarConstructor(self.config)
        else:
            raise ValueError(f"Unknown bar type: {self.config.bar_type}")
    
    def process_symbol(
        self, 
        symbol: str, 
        df_symbol: pd.DataFrame
    ) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]]:
        """
        Process a single symbol: construct bars and generate windows.
        
        Args:
            symbol: Symbol name
            df_symbol: DataFrame with symbol data
            
        Returns:
            Tuple of (X_windows, target_windows, window_info, feature_cols)
            Returns None if processing fails
        """
        try:
            logger.debug(f"Processing symbol {symbol} with {len(df_symbol)} rows")
            
            # Construct bars
            bars_df = self.bar_constructor.construct_bars(df_symbol, symbol)
            
            if bars_df is None or bars_df.empty:
                logger.warning(f"Could not construct bars for symbol {symbol}")
                return None
            
            # Generate windows
            result = self.window_generator.generate_windows(bars_df)
            
            if result[0] is None:  # X_windows is None
                logger.warning(f"Could not generate windows for symbol {symbol}")
                return None
            
            logger.debug(f"Successfully processed symbol {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error processing symbol {symbol}: {e}")
            return None
        finally:
            MemoryManager.cleanup()


def process_symbol_parallel(
    symbol: str,
    df_symbol: pd.DataFrame,
    config: PreprocessingConfig
) -> Optional[Tuple[str, np.ndarray, np.ndarray, pd.DataFrame, List[str]]]:
    """
    Process a single symbol in parallel execution.
    
    This function is designed to be called by joblib.Parallel and includes
    the symbol name in the return tuple for easier result handling.
    
    Args:
        symbol: Symbol name
        df_symbol: DataFrame with symbol data
        config: Preprocessing configuration
        
    Returns:
        Tuple of (symbol, X_windows, target_windows, window_info, feature_cols)
        Returns None if processing fails
    """
    processor = SymbolProcessor(config)
    result = processor.process_symbol(symbol, df_symbol)
    
    if result is None:
        return None
    
    # Add symbol name to result tuple
    return (symbol,) + result 