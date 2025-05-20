import os

# Base directory for processed data
PROCESSED_DATA_DIR_TIME = './processed_data' 
PROCESSED_DATA_DIR_VOLUME = '/mnt/storage_1_10T/citibank/data/processed_data_volume_bars' 

# Specific days to use for training and testing
# First 7 days for train, last 3 for test 
TRAIN_DAYS = ['20250212', '20250213', '20250214', '20250217', '20250218', '20250219', '20250220']
TEST_DAYS = ['20250221', '20250224', '20250225']

# TRAIN_DAYS = ['20250212', '20250213']
# TEST_DAYS = ['20250214']

# Parameters defining the data structure (used in subfolder names)
# -- Time Bars --
RESAMPLE_FREQ = '1s'
# -- Volume Bars --
NB_BARS = 10000 # Default number of bars per symbol per day
# -- Common --
WINDOW_LENGTH = 150 # Input window length 
TARGET_WINDOW_LENGTH = 50 # Target window length 


# --- Model & Training Configuration ---
# Base directory to save models and logs
SAVE_DIR = './saved_models'


def get_data_subdir(
        resample_freq: int = RESAMPLE_FREQ,
        window_length: int = WINDOW_LENGTH,
        target_window_length: int = TARGET_WINDOW_LENGTH
    ) -> str:
    """Constructs the parameter-specific subdirectory name."""
    return f"resample_{resample_freq}_in{window_length}s_tgt{target_window_length}s"

def get_processed_data_path(
        bar_type: str = 'time',
        base_dir_time: str = PROCESSED_DATA_DIR_TIME,
        base_dir_volume: str = PROCESSED_DATA_DIR_VOLUME,
        resample_freq: str = RESAMPLE_FREQ,
        nb_bars: int = NB_BARS,
        window_length: int = WINDOW_LENGTH,
        target_window_length: int = TARGET_WINDOW_LENGTH
    ) -> str:
    """Gets the full path to the specific processed data directory based on bar type."""
    if bar_type == 'time':
        subdir = f"resample_{resample_freq}_in{window_length}s_tgt{target_window_length}s"
        base_dir = base_dir_time
    elif bar_type == 'volume':
        subdir = f"volbars_{nb_bars}_in{window_length}_tgt{target_window_length}"
        base_dir = base_dir_volume
    else:
        raise ValueError(f"Unknown bar_type: {bar_type}. Choose 'time' or 'volume'.")
        
    if not base_dir:
         raise ValueError(f"Base directory for bar_type '{bar_type}' is not set.")
         
    return os.path.join(base_dir, subdir)
