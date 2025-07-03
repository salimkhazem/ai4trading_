"""
AI4Trading Preprocessing Module

This module provides a comprehensive object-oriented preprocessing pipeline
for high-frequency trading data with support for:

- Time-based and volume-based bar construction
- Feature engineering with microstructure indicators
- Sliding window generation for ML training
- Parallel processing and memory management
- Reproducible configurations and caching

Main Components:
    - PreprocessingConfig: Configuration dataclass
    - DataProcessor: Main orchestrator class
    - SymbolProcessor: Individual symbol processing
    - TimeBarConstructor/VolumeBarConstructor: Bar construction strategies
    - FeatureEngineer: Feature computation utilities
    - WindowGenerator: Sliding window creation

Usage:
    from src.preprocessing import DataProcessor, PreprocessingConfig
    
    config = PreprocessingConfig(
        base_data_path="data/raw",
        train_days=["20240101", "20240102"],
        test_days=["20240103"],
        output_dir="output",
        bar_type='time',
        window_length=150
    )
    
    processor = DataProcessor(config)
    X, y, metadata, features = processor.process_all_data()
"""

# Core configuration and processing classes
from .utils import (
    PreprocessingConfig,
    MemoryManager,
    DataCleaner,
    FeatureEngineer,
    ValidationUtils,
    load_day_data
)

from .dataset import (
    BaseBarConstructor,
    TimeBarConstructor, 
    VolumeBarConstructor,
    WindowGenerator,
    SymbolProcessor,
    process_symbol_parallel
)

from .data_loader import (
    DataProcessor,
    load_preprocessed_data,
    create_preprocessing_config,
    quick_preprocess
)

# Convenience imports for common use cases
__all__ = [
    # Main classes
    'DataProcessor',
    'PreprocessingConfig',
    'SymbolProcessor',
    
    # Bar constructors
    'BaseBarConstructor',
    'TimeBarConstructor',
    'VolumeBarConstructor',
    
    # Utility classes
    'FeatureEngineer',
    'WindowGenerator',
    'DataCleaner',
    'MemoryManager',
    'ValidationUtils',
    
    # Functions
    'quick_preprocess',
    'load_preprocessed_data',
    'create_preprocessing_config',
    'load_day_data',
    'process_symbol_parallel',
] 