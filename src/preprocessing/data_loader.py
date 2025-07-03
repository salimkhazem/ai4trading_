"""
Data loading and preprocessing orchestration #TODO: add multiprocessing and accelerate the pipeline

This module provides the main DataProcessor class that orchestrates the complete
preprocessing pipeline from raw data to training-ready windows.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json
import pickle
import hashlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

from .utils import (
    PreprocessingConfig, MemoryManager, DataCleaner, FeatureEngineer, 
    ValidationUtils, load_day_data
)
from .dataset import process_symbol_parallel, SymbolProcessor

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Main preprocessing orchestrator that handles the complete pipeline from
    raw data to training-ready windows with reproducibility guarantees.
    """
    
    def __init__(self, config: PreprocessingConfig):
        """
        Initialize data processor with configuration.
        
        Args:
            config: Preprocessing configuration object
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Setup file logging in addition to console logging."""
        log_file = self.output_dir / "preprocessing.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info("DataProcessor initialized")
        logger.info(f"Config: {self.config}")
    
    def process_all_data(
        self, 
        save_results: bool = True,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        """
        Process all training and test data according to configuration.
        
        Args:
            save_results: Whether to save processed data to disk
            use_cache: Whether to use cached results if available
            
        Returns:
            Tuple of (X_combined, target_combined, metadata_combined, feature_cols)
        """
        start_time = time.time()
        logger.info("Starting complete data processing pipeline")
        MemoryManager.log_memory_usage("Pipeline start")
        
        # Check for cached results
        cache_key = self._generate_cache_key()
        if use_cache:
            cached_result = self._load_cached_result(cache_key)
            if cached_result is not None:
                logger.info("Using cached preprocessing results")
                return cached_result
        
        # Process training data
        logger.info("Processing training data...")
        train_result = self._process_day_list(
            self.config.train_days, 
            data_type="train"
        )
        
        # Process test data
        logger.info("Processing test data...")
        test_result = self._process_day_list(
            self.config.test_days, 
            data_type="test"
        )
        
        # Combine results
        combined_result = self._combine_results([train_result, test_result])
        
        # Validate final results
        self._validate_final_results(combined_result)
        
        # Save results if requested
        if save_results:
            self._save_results(combined_result, cache_key)
        
        # Log completion
        total_time = time.time() - start_time
        logger.info(f"Data processing completed in {total_time:.2f} seconds")
        MemoryManager.log_memory_usage("Pipeline end")
        
        return combined_result
    
    def _process_day_list(
        self, 
        days: List[str], 
        data_type: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        """
        Process a list of days and combine results.
        
        Args:
            days: List of day strings to process
            data_type: Type label for logging ("train" or "test")
            
        Returns:
            Combined results for all days
        """
        logger.info(f"Processing {len(days)} {data_type} days: {days}")
        
        day_results = []
        for day in tqdm(days, desc=f"Processing {data_type} days"):
            try:
                day_result = self._process_single_day(day)
                if day_result is not None:
                    day_results.append(day_result)
                    logger.info(f"Successfully processed {data_type} day {day}")
                else:
                    logger.warning(f"Failed to process {data_type} day {day}")
            except Exception as e:
                logger.error(f"Error processing {data_type} day {day}: {e}")
                continue
        
        if not day_results:
            raise ValueError(f"No {data_type} days were successfully processed")
        
        # Combine all day results
        combined = self._combine_results(day_results)
        logger.info(
            f"Combined {data_type} data: "
            f"X: {combined[0].shape}, Target: {combined[1].shape}, "
            f"Windows: {len(combined[2])}"
        )
        
        return combined
    
    def _process_single_day(
        self, 
        day: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]]:
        """
        Process a single day of data.
        
        Args:
            day: Day string to process
            
        Returns:
            Combined results for the day or None if processing fails
        """
        logger.info(f"Processing day {day}")
        start_time = time.time()
        
        # Load raw data for the day
        df_raw = load_day_data(self.config.base_data_path, day)
        if df_raw is None or df_raw.empty:
            logger.warning(f"No data available for day {day}")
            return None
        
        logger.info(f"Loaded {len(df_raw)} rows for day {day}")
        
        # Clean raw data
        df_clean = DataCleaner.clean_raw_data(df_raw)
        
        # Apply feature engineering
        df_features = FeatureEngineer.compute_all_features(df_clean, self.config)
        
        # Filter symbols if configured
        df_filtered = self._filter_symbols(df_features)
        
        # Group by symbol and process in parallel
        symbol_groups = self._prepare_symbol_groups(df_filtered)
        if not symbol_groups:
            logger.warning(f"No valid symbols found for day {day}")
            return None
        
        # Process symbols in parallel
        symbol_results = self._process_symbols_parallel(symbol_groups)
        
        if not symbol_results:
            logger.warning(f"No symbols were successfully processed for day {day}")
            return None
        
        # Combine symbol results for this day
        day_result = self._combine_symbol_results(symbol_results, day)
        
        # Clean up memory
        MemoryManager.cleanup()
        
        processing_time = time.time() - start_time
        logger.info(f"Day {day} processed in {processing_time:.2f} seconds")
        
        return day_result
    
    def _filter_symbols(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter symbols based on configuration."""
        if 'sym' not in df.columns:
            logger.warning("No 'sym' column found, skipping symbol filtering")
            return df
        
        initial_symbols = df['sym'].nunique()
        
        # Apply exclusion filter
        if self.config.symbols_to_exclude:
            df = df[~df['sym'].isin(self.config.symbols_to_exclude)]
            logger.info(f"Excluded {len(self.config.symbols_to_exclude)} symbols")
        
        # Apply inclusion filter
        if self.config.symbols_to_keep:
            df = df[df['sym'].isin(self.config.symbols_to_keep)]
            logger.info(f"Kept only {len(self.config.symbols_to_keep)} symbols")
        
        final_symbols = df['sym'].nunique()
        logger.info(f"Symbol filtering: {initial_symbols} -> {final_symbols} symbols")
        
        return df
    
    def _prepare_symbol_groups(self, df: pd.DataFrame) -> List[Tuple[str, pd.DataFrame]]:
        """Prepare symbol groups for parallel processing."""
        if 'sym' not in df.columns:
            # If no symbol column, treat entire DataFrame as one group
            return [("ALL", df)]
        
        symbol_groups = []
        for symbol, group_df in df.groupby('sym'):
            if len(group_df) < 1000:  # Minimum rows threshold
                logger.warning(f"Symbol {symbol} has only {len(group_df)} rows, skipping")
                continue
            symbol_groups.append((symbol, group_df.copy()))
        
        logger.info(f"Prepared {len(symbol_groups)} symbol groups for processing")
        return symbol_groups
    
    def _process_symbols_parallel(
        self, 
        symbol_groups: List[Tuple[str, pd.DataFrame]]
    ) -> List[Tuple[str, np.ndarray, np.ndarray, pd.DataFrame, List[str]]]:
        """Process symbols in parallel using joblib."""
        logger.info(f"Processing {len(symbol_groups)} symbols in parallel")
        
        n_jobs = self.config.n_jobs if self.config.n_jobs > 0 else -1
        
        # Process symbols in parallel
        with Parallel(n_jobs=n_jobs, backend='threading') as parallel:
            results = parallel(
                delayed(process_symbol_parallel)(symbol, df_symbol, self.config)
                for symbol, df_symbol in tqdm(
                    symbol_groups, 
                    desc="Processing symbols",
                    leave=False
                )
            )
        
        # Filter out None results
        valid_results = [r for r in results if r is not None]
        
        logger.info(
            f"Successfully processed {len(valid_results)}/{len(symbol_groups)} symbols"
        )
        
        return valid_results
    
    def _combine_symbol_results(
        self, 
        symbol_results: List[Tuple[str, np.ndarray, np.ndarray, pd.DataFrame, List[str]]],
        day: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        """Combine results from multiple symbols for a single day."""
        logger.debug(f"Combining results from {len(symbol_results)} symbols for day {day}")
        
        X_list = []
        target_list = []
        metadata_list = []
        feature_cols = None
        
        for symbol, X_windows, target_windows, window_info, cols in symbol_results:
            X_list.append(X_windows)
            target_list.append(target_windows)
            
            # Add metadata
            metadata_with_symbol = window_info.copy()
            metadata_with_symbol['symbol'] = symbol
            metadata_with_symbol['day'] = day
            metadata_list.append(metadata_with_symbol)
            
            # Use feature columns from first symbol (should be consistent)
            if feature_cols is None:
                feature_cols = cols
        
        if not X_list:
            raise ValueError(f"No valid symbol results to combine for day {day}")
        
        # Combine arrays
        X_combined = np.concatenate(X_list, axis=0)
        target_combined = np.concatenate(target_list, axis=0)
        metadata_combined = pd.concat(metadata_list, ignore_index=True)
        
        logger.debug(
            f"Day {day} combined: X: {X_combined.shape}, "
            f"Target: {target_combined.shape}, Windows: {len(metadata_combined)}"
        )
        
        return X_combined, target_combined, metadata_combined, feature_cols
    
    def _combine_results(
        self, 
        results_list: List[Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]]
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
        """Combine multiple result tuples into a single result."""
        if not results_list:
            raise ValueError("No results to combine")
        
        X_list = []
        target_list = []
        metadata_list = []
        feature_cols = None
        
        for X, target, metadata, cols in results_list:
            X_list.append(X)
            target_list.append(target)
            metadata_list.append(metadata)
            
            if feature_cols is None:
                feature_cols = cols
        
        # Combine arrays
        X_combined = np.concatenate(X_list, axis=0)
        target_combined = np.concatenate(target_list, axis=0)
        metadata_combined = pd.concat(metadata_list, ignore_index=True)
        
        return X_combined, target_combined, metadata_combined, feature_cols
    
    def _validate_final_results(
        self, 
        result: Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]
    ) -> None:
        """Validate the final combined results."""
        X_combined, target_combined, metadata_combined, feature_cols = result
        
        logger.info("Validating final results...")
        
        # Validate shapes
        if X_combined.shape[0] != target_combined.shape[0]:
            raise ValueError("Mismatch between X and target array lengths")
        
        if X_combined.shape[0] != len(metadata_combined):
            raise ValueError("Mismatch between arrays and metadata length")
        
        # Validate data quality
        ValidationUtils.validate_feature_array(X_combined, "X_combined")
        ValidationUtils.validate_feature_array(target_combined, "target_combined")
        
        # Log final statistics
        logger.info(f"Final results validated successfully:")
        logger.info(f"  X shape: {X_combined.shape}")
        logger.info(f"  Target shape: {target_combined.shape}")
        logger.info(f"  Features: {len(feature_cols)}")
        logger.info(f"  Windows: {len(metadata_combined)}")
        logger.info(f"  Days: {metadata_combined['day'].nunique()}")
        logger.info(f"  Symbols: {metadata_combined['symbol'].nunique()}")
    
    def _generate_cache_key(self) -> str:
        """Generate a cache key based on configuration."""
        # Create a hash of the configuration for cache key
        config_dict = {
            'base_data_path': self.config.base_data_path,
            'train_days': sorted(self.config.train_days),
            'test_days': sorted(self.config.test_days),
            'bar_type': self.config.bar_type,
            'resample_freq': self.config.resample_freq,
            'nb_bars': self.config.nb_bars,
            'window_length': self.config.window_length,
            'target_window_length': self.config.target_window_length,
            'target_column': self.config.target_column,
            'symbols_to_exclude': sorted(self.config.symbols_to_exclude),
            'symbols_to_keep': sorted(self.config.symbols_to_keep),
            'enable_technical_indicators': self.config.enable_technical_indicators,
            'enable_order_flow_features': self.config.enable_order_flow_features,
            'enable_time_features': self.config.enable_time_features,
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(config_str.encode()).hexdigest()
        
        logger.debug(f"Generated cache key: {cache_key}")
        return cache_key
    
    def _load_cached_result(
        self, 
        cache_key: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]]:
        """Load cached preprocessing results if available."""
        cache_file = self.output_dir / f"cached_results_{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            logger.info(f"Loading cached results from {cache_file}")
            with open(cache_file, 'rb') as f:
                result = pickle.load(f)
            
            # Validate cached result structure
            if (isinstance(result, tuple) and len(result) == 4 and
                isinstance(result[0], np.ndarray) and isinstance(result[1], np.ndarray) and
                isinstance(result[2], pd.DataFrame) and isinstance(result[3], list)):
                
                logger.info("Successfully loaded cached results")
                return result
            else:
                logger.warning("Cached result has invalid structure, ignoring")
                return None
                
        except Exception as e:
            logger.warning(f"Failed to load cached results: {e}")
            return None
    
    def _save_results(
        self, 
        result: Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]],
        cache_key: str
    ) -> None:
        """Save preprocessing results to disk."""
        X_combined, target_combined, metadata_combined, feature_cols = result
        
        # Save individual components
        np.save(self.output_dir / "X_features.npy", X_combined)
        np.save(self.output_dir / "target_windows.npy", target_combined)
        metadata_combined.to_parquet(self.output_dir / "window_metadata.parquet")
        
        with open(self.output_dir / "feature_columns.json", 'w') as f:
            json.dump(feature_cols, f, indent=2)
        
        # Save complete result as pickle for caching
        cache_file = self.output_dir / f"cached_results_{cache_key}.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        # Save configuration for reproducibility
        config_file = self.output_dir / "preprocessing_config.json"
        config_dict = {
            'base_data_path': self.config.base_data_path,
            'train_days': self.config.train_days,
            'test_days': self.config.test_days,
            'output_dir': self.config.output_dir,
            'bar_type': self.config.bar_type,
            'resample_freq': self.config.resample_freq,
            'nb_bars': self.config.nb_bars,
            'window_length': self.config.window_length,
            'target_window_length': self.config.target_window_length,
            'target_column': self.config.target_column,
            'symbols_to_exclude': self.config.symbols_to_exclude,
            'symbols_to_keep': self.config.symbols_to_keep,
            'n_jobs': self.config.n_jobs,
            'memory_efficient': self.config.memory_efficient,
            'enable_technical_indicators': self.config.enable_technical_indicators,
            'enable_order_flow_features': self.config.enable_order_flow_features,
            'enable_time_features': self.config.enable_time_features,
        }
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir}")


def load_preprocessed_data(
    output_dir: Union[str, Path]
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Load previously saved preprocessing results.
    
    Args:
        output_dir: Directory containing saved preprocessing results
        
    Returns:
        Tuple of (X_features, target_windows, metadata, feature_cols)
        
    Raises:
        FileNotFoundError: If required files are not found
        ValueError: If loaded data is invalid
    """
    output_path = Path(output_dir)
    
    # Load components
    X_features = np.load(output_path / "X_features.npy")
    target_windows = np.load(output_path / "target_windows.npy")
    metadata = pd.read_parquet(output_path / "window_metadata.parquet")
    
    with open(output_path / "feature_columns.json", 'r') as f:
        feature_cols = json.load(f)
    
    # Validate loaded data
    if X_features.shape[0] != target_windows.shape[0]:
        raise ValueError("Mismatch between X and target array lengths")
    
    if X_features.shape[0] != len(metadata):
        raise ValueError("Mismatch between arrays and metadata length")
    
    logger.info(f"Loaded preprocessed data from {output_path}")
    logger.info(f"X shape: {X_features.shape}, Target shape: {target_windows.shape}")
    
    return X_features, target_windows, metadata, feature_cols


def create_preprocessing_config(
    base_data_path: str,
    train_days: List[str],
    test_days: List[str],
    output_dir: str,
    **kwargs
) -> PreprocessingConfig:
    """
    Create preprocessing configuration with sensible defaults.
    
    Args:
        base_data_path: Path to raw data
        train_days: List of training day strings
        test_days: List of test day strings
        output_dir: Output directory for results
        **kwargs: Additional configuration parameters
        
    Returns:
        PreprocessingConfig object
    """
    return PreprocessingConfig(
        base_data_path=base_data_path,
        train_days=train_days,
        test_days=test_days,
        output_dir=output_dir,
        **kwargs
    )


# Example usage and convenience functions
def quick_preprocess(
    base_data_path: str,
    train_days: List[str],
    test_days: List[str],
    output_dir: str,
    bar_type: str = 'time',
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]:
    """
    Quick preprocessing with sensible defaults.
    
    Args:
        base_data_path: Path to raw data
        train_days: List of training day strings
        test_days: List of test day strings
        output_dir: Output directory for results
        bar_type: Type of bars ('time' or 'volume')
        **kwargs: Additional configuration parameters
        
    Returns:
        Tuple of (X_features, target_windows, metadata, feature_cols)
    """
    config = create_preprocessing_config(
        base_data_path=base_data_path,
        train_days=train_days,
        test_days=test_days,
        output_dir=output_dir,
        bar_type=bar_type,
        **kwargs
    )
    
    processor = DataProcessor(config)
    return processor.process_all_data() 