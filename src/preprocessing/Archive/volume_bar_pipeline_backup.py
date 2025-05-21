import logging
import os
import sys
import time
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Tuple, Dict, Optional
import gc
from pathlib import Path

# --- Try importing resource for memory profiling ---
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    # resource module is Unix-specific
    HAS_RESOURCE = False
    logging.warning("Module 'resource' not found. Memory profiling will be skipped.")

# --- Add project root to sys.path ---
current_file_path = Path(__file__).resolve()
project_root_path = current_file_path.parent.parent
project_root_str = str(project_root_path)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
# ------------------------------------

from utils.utils import *

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# --- Profiling Helper ---
def get_memory_usage_gb() -> float:
    """Gets current peak memory usage in GB (Unix only)."""
    if not HAS_RESOURCE:
        return -1.0 # Indicate memory profiling is unavailable
    # ru_maxrss is typically in KB, convert to GB
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

def compute_core_features(df):
    # Midprice
    df["mid_price"] = (df["L1_bid_price"] + df["L1_ask_price"]) / 2
    # Spread
    df["spread"] = df["L1_ask_price"] - df["L1_bid_price"]

    # Weighted MidPrice (WMP)
    df["weighted_mid_price"] = (df["L1_bid_price"] * df["L1_ask_size"] + df["L1_ask_price"] * df["L1_bid_size"]) / (
        df["L1_bid_size"] + df["L1_ask_size"]
    )
    # Order Book Imbalance (OBI) - niveau 1
    df["obi_L1"] = (df["L1_bid_size"] - df["L1_ask_size"]) / (df["L1_bid_size"] + df["L1_ask_size"])
    # Depth imbalance (L1 à L10)
    bid_cols = [f"L{i}_bid_size" for i in range(1, 11)]
    ask_cols = [f"L{i}_ask_size" for i in range(1, 11)]
    df["cum_bid_vol_10"] = df[bid_cols].sum(axis=1)
    df["cum_ask_vol_10"] = df[ask_cols].sum(axis=1)
    df["depth_imbalance_10"] = (df["cum_bid_vol_10"] - df["cum_ask_vol_10"]) / (
        df["cum_bid_vol_10"] + df["cum_ask_vol_10"]
    )
    # Liquidity metrics
    df["liquidity_impact_bid"] = df["L1_bid_price"] - df["L2_bid_price"]
    df["liquidity_impact_ask"] = df["L2_ask_price"] - df["L1_ask_price"]
    return df

def compute_nb_bars(
    symbol_data: pd.DataFrame, 
    min_avg_snapshots_per_bar: int = 150,
    max_nb_bars: int = 7500
) -> int:
    """Create volume bars for all symbols based on the number of snapshots.

    This function calculates the number of volume bars for each symbol in the
    provided data map, ensuring that the number of bars does not exceed the
    specified maximum and is based on a minimum average number of snapshots per bar.

    Args:
        symbols_data_map (Dict[str, pd.DataFrame]): A dictionary mapping symbol names to their corresponding DataFrames.
        min_avg_snapshots_per_bar (int, optional): Minimum average snapshots required per bar. Defaults to 150.
        max_nb_bars (int, optional): Maximum number of bars allowed per symbol. Defaults to 7500.

    Returns:
        Dict[str, int]: A dictionary mapping each symbol to the calculated number of bars.
    """
    nb_snapshots = symbol_data.shape[0]
    nb_bars_for_quality = nb_snapshots // min_avg_snapshots_per_bar
    final_nb_bars = min(nb_bars_for_quality, max_nb_bars)
    return final_nb_bars

def create_volume_bars_with_lob_features(
    df_symbol: pd.DataFrame,    
    nb_bars: int,
    activity_col_name: str = 'L1_total_volume', # Renamed for clarity
    wmp_col_name: str = 'weighted_mid_price',
    time_col_name: str = 'time'
) -> Optional[pd.DataFrame]:
    """
    Create aggregated volume bars for a single symbol based on a target number of bars.

    Aggregates LOB features (mean, sum, min, max, std, median) for each bar.

    Args:
        df_symbol (pd.DataFrame): DataFrame containing LOB data for one symbol, 
                                  sorted by time. Must include time, WMP, and L1 sizes.
        nb_bars (int): Target number of volume bars to create for the given data.
        activity_col_name (str): Name for the calculated L1 volume column.
        wmp_col_name (str): Name of the Weighted Mid-Price column.
        time_col_name (str): Name of the timestamp column.

    Returns:
        Optional[pd.DataFrame]: DataFrame where each row represents one volume bar 
                                with aggregated features, or None if bars cannot be created.
    """
    logging.debug(f"Starting volume bar creation for symbol (shape {df_symbol.shape}) with target {nb_bars} bars.")
    
    # Calculate activity volume (L1 Bid Size + L1 Ask Size)
    if 'L1_bid_size' not in df_symbol.columns or 'L1_ask_size' not in df_symbol.columns:
        logging.error("L1 bid/ask size columns missing. Cannot calculate activity volume.")
        return None
    
    # Compute total volume between bid and ask
    df_symbol[activity_col_name] = df_symbol['L1_bid_size'] + df_symbol['L1_ask_size']
    df_symbol[activity_col_name] = df_symbol[activity_col_name].fillna(0)
    
    # Calculate cumulative activity volume
    df_symbol['cumulative_activity_volume'] = df_symbol[activity_col_name].cumsum()

    required_cols = [time_col_name, wmp_col_name, activity_col_name, 'cumulative_activity_volume']
    if not all(col in df_symbol.columns for col in required_cols):
        logging.error(f"Missing required columns for bar creation: Needs {required_cols}. Found {df_symbol.columns.tolist()}")
        return None
    
    # Ensure data is sorted by time (should be done before calling, but double-check)
    if not df_symbol[time_col_name].is_monotonic_increasing:
         logging.warning("Input data not sorted by time. Sorting now...")
         df_symbol = df_symbol.sort_values(time_col_name).reset_index(drop=True)
    
    data = df_symbol.copy()

    #print(data[[time_col_name, wmp_col_name, activity_col_name, 'cumulative_activity_volume']].head())
    
    # Calculate volume threshold per bar
    total_cumulative_volume = data['cumulative_activity_volume'].max()
    if total_cumulative_volume == 0 or nb_bars == 0:
        logging.warning("Total cumulative volume or number of bars is zero. Cannot create bars.")
        return None
    
    volume_threshold = total_cumulative_volume / nb_bars 
    logging.info(f"Calculated Volume Threshold per bar: {volume_threshold:.2f} (Total Vol: {total_cumulative_volume}, Target Bars: {nb_bars})")

    if volume_threshold <= 0:
         logging.warning(f"Volume threshold is non-positive ({volume_threshold:.2f}). Cannot create bars.")
         return None

    # Assign bar ID based on cumulative volume crossing thresholds
    data['bar_id'] = (data['cumulative_activity_volume'] // volume_threshold).astype(int)

    #print(data[[time_col_name, wmp_col_name, activity_col_name, 'cumulative_activity_volume', 'bar_id']].head())

    # Handle edge case: First few entries might be below the first threshold / Ensure they belong to bar 0
    if data.empty:
        logging.warning("Data is empty before attempting to access first_bar_id. Cannot create bars.")
        return None
    first_bar_id = data.iloc[0]['bar_id'] 
    data.loc[data['bar_id'] == first_bar_id, 'bar_id'] = 0 

    # Exclude snapshots belonging to a potentially partial last bar
    # Ensure we only include bars where the *next* threshold was crossed
    max_possible_bar_id = int(np.floor(total_cumulative_volume / volume_threshold)) -1 
    logging.info(f"max_possible_bar_id: {max_possible_bar_id}")
    if max_possible_bar_id < 0:
        logging.warning("Not enough volume to form even one full bar.")
        return None
        
    data_full_bars = data[data['bar_id'] <= max_possible_bar_id].copy()
    if data_full_bars.empty:
        logging.warning("No full bars could be formed after excluding the partial last bar.")
        return None
        
    logging.debug(f"Data filtered to full bars (Bar IDs 0 to {max_possible_bar_id}). Shape: {data_full_bars.shape}")

    # Identify features to aggregate
    # LOB Level features (L1-L10 price, size, no)
    level_cols = [col for col in data_full_bars.columns if col.startswith('L') and ('_price' in col or '_size' in col or '_no' in col)]
    # Microstructure features (spread, obi, imbalance, etc.)
    lob_features = ['spread', 'obi_L1', 'cum_bid_vol_10','cum_ask_vol_10', 
                    'depth_imbalance_10', 'liquidity_impact_bid',
                    'liquidity_impact_ask']
    # Filter lob_features to only include those present in the dataframe
    lob_features = [f for f in lob_features if f in data_full_bars.columns]
    
    # Combine LOB and microstructure features 
    features_to_aggregate = level_cols + lob_features
    
    # Keep essential info columns
    info_cols_to_keep = [time_col_name, wmp_col_name, activity_col_name]

    # Check if features_to_aggregate is empty
    if not features_to_aggregate:
        logging.warning("No LOB features found to aggregate.")
        return None

    # Aggregation function
    def aggregate_bar(bar_group: pd.DataFrame) -> pd.Series:
        results = {
            'bar_start_time': bar_group[time_col_name].iloc[0],
            'bar_end_time': bar_group[time_col_name].iloc[-1],
            'num_snapshots_in_bar': len(bar_group),
            'actual_volume_in_bar': bar_group[activity_col_name].sum(),
            # WMP aggregations
            'wmp_mean': bar_group[wmp_col_name].mean(),
            'wmp_first': bar_group[wmp_col_name].iloc[0],
            'wmp_last': bar_group[wmp_col_name].iloc[-1],
            'wmp_min': bar_group[wmp_col_name].min(),
            'wmp_max': bar_group[wmp_col_name].max(),
            'wmp_median': bar_group[wmp_col_name].median(),
            'wmp_std': bar_group[wmp_col_name].std(),
        }
        # Aggregate other LOB features
        for col in features_to_aggregate:
            results[f"{col}_mean"] = bar_group[col].mean()
            results[f"{col}_sum"] = bar_group[col].sum()
            results[f"{col}_min"] = bar_group[col].min()
            results[f"{col}_max"] = bar_group[col].max()
            results[f"{col}_std"] = bar_group[col].std()
            results[f"{col}_median"] = bar_group[col].median()
            
        return pd.Series(results)
    
    logging.debug(f"Aggregating {len(data_full_bars['bar_id'].unique())} bars using groupby().apply()...")
    # Group by bar_id and apply the aggregation
    grouped_bars = data_full_bars.groupby('bar_id')
    volume_bars_df = grouped_bars.apply(aggregate_bar)
    logging.debug("Aggregation finished.")

    # Calculate bar duration
    volume_bars_df['bar_duration_seconds'] = (
        volume_bars_df['bar_end_time'] - volume_bars_df['bar_start_time']
    ).dt.total_seconds()

    # Final checks
    volume_bars_df = volume_bars_df.reset_index() 
    # Fill NaNs that might result from std() on single-snapshot bars or empty groups
    volume_bars_df.fillna(0, inplace=True) 
    
    logging.info(f"Successfully created {len(volume_bars_df)} volume bars.")
    return volume_bars_df

def generate_sequential_windows(
    df_bars: pd.DataFrame,
    window_length: int = 100,
    target_window_length: int = 10,
    target_col: str = 'wmp_mean' # Column to use for target prediction
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Generates sequential, sliding windows and corresponding target windows from aggregated bar data.

    Args:
        df_bars (pd.DataFrame): DataFrame where each row is an aggregated bar (time or volume).
                                Must contain 'bar_end_time' and the target_col ('wmp_mean').
        window_length (int): Number of bars in the input window.
        target_window_length (int): Number of bars in the target window.
        target_col (str): The column within df_bars to use for generating the target window values 
                          (e.g., 'wmp_mean', 'wmp_last').

    Returns:
        Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[List[str]]]: 
            - X_windows: NumPy array of input windows (samples, window_length, num_features).
            - target_windows: NumPy array of target values (samples, target_window_length).
            - window_info: DataFrame with metadata for each window (end time, last target value).
            - feature_cols: List of feature names corresponding to the last dimension of X_windows.
            Returns (None, None, None, None) if no windows can be generated.
    """
    logging.info(f"Generating input windows (len={window_length}) and target windows (len={target_window_length}) using target '{target_col}'...")

    required_cols = ['bar_end_time', target_col]
    if not all(col in df_bars.columns for col in required_cols):
        logging.error(f"Input DataFrame for windowing must contain columns: {required_cols}. Found {df_bars.columns.tolist()}")
        return None, None, None, None
        
    # Identify feature columns 
    exclude_cols = ['bar_id', 'bar_start_time', 'bar_end_time', 'num_snapshots_in_bar', 'actual_volume_in_bar', 'bar_duration_seconds']
    feature_cols = [col for col in df_bars.columns if col not in exclude_cols]
    
    if not feature_cols:
        logging.error("No feature columns identified for windowing.")
        return None, None, None, None

    num_features = len(feature_cols)
    logging.info(f"Using {num_features} input features for windows.") 

    # Ensure features are numeric and handle NaNs
    logging.debug("Converting features to float32 and checking for NaNs...")
    try:
        feature_data = df_bars[feature_cols].fillna(0).astype(np.float32).values
        if np.isnan(feature_data).any() or np.isinf(feature_data).any():
             logging.warning("NaN or Inf values found in feature data after fillna(0). Check data pipeline.")
             # Optionally, apply more robust NaN handling like imputation if needed
             feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0) 
    except Exception as e:
        logging.error(f"Error converting features to float32: {e}", exc_info=True)
        return None, None, None, None
    logging.debug("Feature conversion and NaN check complete.")

    # Extract time and target columns
    times = df_bars['bar_end_time'].values
    targets = df_bars[target_col].astype(np.float32).values # Target values

    # Check if enough data exists for at least one window
    total_bars = len(df_bars)
    required_bars = window_length + target_window_length
    if total_bars < required_bars:
        logging.warning(f"Not enough bars ({total_bars}) to create a window of length {window_length} + target {target_window_length}.")
        return None, None, None, None

    X_windows_list, target_windows_list, window_info_list = [], [], []
    
    logging.debug(f"Iterating to create windows (Total bars: {total_bars}, Required per window: {required_bars})...")
    # Iterate through the bars to create sliding windows
    # Stop early enough to allow for a full input window AND a full target window
    for i in range(total_bars - required_bars + 1):
        input_end_idx = i + window_length
        target_start_idx = input_end_idx
        target_end_idx = target_start_idx + target_window_length

        # Slice input features
        input_window_slice = feature_data[i : input_end_idx]
        
        # Slice target variable for the future window
        target_window_slice = targets[target_start_idx : target_end_idx]
        
        X_windows_list.append(input_window_slice)
        target_windows_list.append(target_window_slice)

        # Store metadata: time corresponds to the *end* of the input window
        info = {
            'window_end_time': times[input_end_idx - 1], 
            'last_target_in_window': targets[input_end_idx - 1] # Store last target value of the input window
        }
        window_info_list.append(info)

    logging.debug(f"Finished iterating. Created {len(X_windows_list)} raw windows.")

    if not X_windows_list:
        logging.warning("No windows were generated after iteration.")
        # Return empty structures with correct shapes/columns
        empty_x = np.array([]).reshape(0, window_length, num_features).astype(np.float32)
        empty_tgt = np.array([]).reshape(0, target_window_length).astype(np.float32)
        empty_info = pd.DataFrame(columns=['window_end_time', 'last_target_in_window'])
        return empty_x, empty_tgt, empty_info, feature_cols

    # Stack lists into final NumPy arrays and DataFrame
    logging.debug("Stacking window lists into final arrays...")
    try:
        X_windows = np.stack(X_windows_list, axis=0).astype(np.float32)
        target_windows = np.stack(target_windows_list, axis=0).astype(np.float32)
        window_info = pd.DataFrame(window_info_list)
        # Ensure time column is datetime
        window_info['window_end_time'] = pd.to_datetime(window_info['window_end_time']) 
        # Sort by time just in case (though generation is sequential)
        window_info = window_info.sort_values('window_end_time').reset_index(drop=True) 
        # Reorder X and target arrays based on sorted info
        # This is crucial if parallel processing were used, but good practice anyway
        # Assuming sequential generation, indices should already match sort order
        # X_windows = X_windows[window_info.index] 
        # target_windows = target_windows[window_info.index]
        
    except Exception as e:
         logging.error(f"Error stacking final window arrays: {e}", exc_info=True)
         return None, None, None, None
    logging.debug("Stacking complete.")

    logging.info(f"Window generation complete.")
    logging.info(f"Shape of X_windows: {X_windows.shape}")
    logging.info(f"Shape of target_windows: {target_windows.shape}")
    logging.info(f"Shape of window_info: {window_info.shape}")
    logging.info(f"Data type of X_windows: {X_windows.dtype}")
    logging.info(f"Data type of target_windows: {target_windows.dtype}")

    return X_windows, target_windows, window_info, feature_cols

# --- Helper Function for Parallel Symbol Processing ---
def _process_symbol_bars_windows(
    sym_name: str,
    df_sym: pd.DataFrame,
    nb_bars: int,
    window_length: int,
    target_window_length: int,
    target_col_name: str
) -> Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]]:
    """
    Processes a single symbol: creates volume bars and generates sequential windows.
    Designed to be called by joblib.Parallel.

    Args:
        sym_name (str): The name of the symbol being processed.
        df_sym (pd.DataFrame): DataFrame containing data for only the specified symbol.
        nb_bars (int): Target number of volume bars.
        window_length (int): Length of the input window.
        target_window_length (int): Length of the target window.
        target_col_name (str): Column name for the target variable.

    Returns:
        Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame, List[str]]]: 
            A tuple containing (X_windows, target_windows, window_info, feature_cols) 
            for the symbol, or None if processing fails at any step.
    """
    try:
        # Ensure time column is datetime and sorted (should be done before grouping, but safer here too)
        if 'time' not in df_sym.columns:
             logging.error(f"[{sym_name}] 'time' column missing.")
             return None
        df_sym['time'] = pd.to_datetime(df_sym['time'])
        df_sym = df_sym.sort_values('time').reset_index(drop=True)

        # 1. Create Volume Bars
        logging.debug(f"[{sym_name}] Creating volume bars...")
        bars_df = create_volume_bars_with_lob_features(df_sym, nb_bars)
        del df_sym; gc.collect() 
        if bars_df is None or bars_df.empty:
            logging.warning(f"[{sym_name}] Could not create volume bars. Skipping.")
            return None
        logging.debug(f"[{sym_name}] Created {len(bars_df)} volume bars.")

        # 2. Generate Sequential Windows
        logging.debug(f"[{sym_name}] Generating sequential windows...")
        X_windows_sym, target_windows_sym, window_info_sym, feature_cols = generate_sequential_windows(
            bars_df, window_length, target_window_length, target_col_name
        )
        # bars_df is no longer needed after windows are generated
        del bars_df; gc.collect() 
        if X_windows_sym is None or X_windows_sym.size == 0:
            logging.warning(f"[{sym_name}] No windows generated. Skipping.")
            return None
        
        # Add symbol name to info DataFrame
        window_info_sym['sym'] = sym_name
        logging.debug(f"[{sym_name}] Generated {len(X_windows_sym)} windows.")
        
        return X_windows_sym, target_windows_sym, window_info_sym, feature_cols

    except Exception as e:
        logging.error(f"[{sym_name}] Error during processing: {e}", exc_info=True)
        # Ensure cleanup in case of error during processing
        if 'df_sym' in locals(): del df_sym
        if 'bars_df' in locals(): del bars_df
        gc.collect()
        return None

# --- Main Daily Processing Function ---
def process_day(
    day: str,
    base_data_path: str,
    nb_bars: int,
    window_length: int,
    target_window_length: int,
    target_col_name: str, # Added target column name
    output_dir: str,
    symbols_to_exclude: List[str] # Added symbols to exclude
) -> int:
    """Process a single day: load, feature compute, bar creation, windowing, combine, save."""
    logging.info(f"--- Processing Day: {day} ---")
    day_start_time = time.time()
    day_peak_mem_start = get_memory_usage_gb()
    day_peak_mem_current = day_peak_mem_start

    # --- 1. Load Raw Data ---
    step_start_time = time.time()
    logging.info(f"Step 1: Loading raw data for {day}...")
    df_day_raw = load_and_clean_day_data(base_data_path, day)
    load_time = time.time() - step_start_time
    mem_after_load = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_load)
    logging.info(f"Step 1 (Load) complete: Time={load_time:.2f}s, Peak Mem={mem_after_load:.2f} GB, Shape={df_day_raw.shape if df_day_raw is not None else 'None'}")
    if df_day_raw is None or df_day_raw.empty:
        logging.warning(f"Skipping day {day} due to loading error or no data.")
        return 0

    # --- 2. Clean Data ---
    step_start_time = time.time()
    logging.info("Step 2: Cleaning data...")
    initial_rows = len(df_day_raw)
    df_day_raw.dropna(inplace=True)
    df_clean = df_day_raw.copy()
    rows_dropped = initial_rows - len(df_clean)
    percent_dropped = (rows_dropped / initial_rows * 100) if initial_rows > 0 else 0.0
    logging.info(f"Dropped {rows_dropped:,} rows with NaNs ({percent_dropped:.2f}%). Shape after NaN drop: {df_clean.shape}")
    columns_to_drop = [
        'exch_time', 'exchange', 'first_sequence_number', 'last_sequence_number',
        'first_sym_sequence', 'last_sym_sequence', 'first_time', 'first_exch_time',
        'event_id', 'date'
    ]
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    if existing_cols_to_drop:
        df_clean.drop(columns=existing_cols_to_drop, inplace=True)
        logging.info(f"Dropped unused columns: {existing_cols_to_drop}. Shape now: {df_clean.shape}")
    clean_time = time.time() - step_start_time
    mem_after_clean = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_clean)
    logging.info(f"Step 2 (Clean) complete: Time={clean_time:.2f}s, Peak Mem={mem_after_clean:.2f} GB")
    if df_clean.empty:
        logging.warning(f"Skipping day {day} as it became empty after cleaning.")
        return 0
    del df_day_raw; gc.collect()

    # --- 3. Compute Features ---
    step_start_time = time.time()
    logging.info("Step 3: Computing core features...")
    df_featured = compute_core_features(df_clean)
    feature_time = time.time() - step_start_time
    mem_after_features = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_features)
    logging.info(f"Step 3 (Features) complete: Time={feature_time:.2f}s, Peak Mem={mem_after_features:.2f} GB, Shape={df_featured.shape}")
    del df_clean; gc.collect()

    # --- 4. Filter Symbols ---
    step_start_time = time.time()
    logging.info(f"Step 4: Filtering symbols (excluding {symbols_to_exclude})...")
    initial_syms = df_featured['sym'].nunique()
    df_filtered = df_featured[~df_featured['sym'].isin(symbols_to_exclude)].copy()
    syms_dropped = initial_syms - df_filtered['sym'].nunique()
    filter_time = time.time() - step_start_time
    mem_after_filter = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_filter)
    logging.info(f"Step 4 (Filter Sym) complete: Time={filter_time:.2f}s, Peak Mem={mem_after_filter:.2f} GB")
    logging.info(f"Dropped {syms_dropped} symbols. Shape after symbol filter: {df_filtered.shape}")
    if df_filtered.empty:
        logging.warning(f"Skipping day {day} as it became empty after filtering symbols.")
        return 0
    del df_featured; gc.collect()

    # --- DEBUGGING SINGLE SYMBOL --- For create_volume_bars_with_lob_features
    logging.info("DEBUG: Isolating create_volume_bars_with_lob_features for one symbol.")
    grouped_symbols_debug = df_filtered.groupby('sym')
    #first_sym_name_debug, first_sym_df_debug = next(iter(grouped_symbols_debug))
    first_sym_name_debug = 'FGBLH5'
    first_sym_df_debug = grouped_symbols_debug.get_group(first_sym_name_debug)
    logging.info(f"DEBUG: Processing symbol: {first_sym_name_debug}")
    print(first_sym_df_debug.shape)

    nb_bars_symbol = compute_nb_bars(first_sym_df_debug, max_nb_bars=10000)
    print(nb_bars_symbol)

    bars_df_debug = create_volume_bars_with_lob_features(
        df_symbol=first_sym_df_debug.copy(), 
        nb_bars=nb_bars_symbol,
        activity_col_name='L1_total_volume', 
        wmp_col_name='weighted_mid_price', 
        time_col_name='time' 
    )

    if bars_df_debug is not None:
        logging.info(f"DEBUG: Output bars_df_debug from create_volume_bars_with_lob_features (head):")
        print(bars_df_debug.head())
        logging.info(f"DEBUG: Output bars_df_debug dtypes:")
        print(bars_df_debug.dtypes)
        logging.info(f"DEBUG: Output bars_df_debug shape: {bars_df_debug.shape}")

        # --- DEBUGGING generate_sequential_windows ---
        logging.info(f"DEBUG: Calling generate_sequential_windows for symbol {first_sym_name_debug}...")
        X_windows_debug, target_windows_debug, window_info_debug, feature_cols_debug = generate_sequential_windows(
            df_bars=bars_df_debug,
            window_length=window_length, # Use window_length from process_day args
            target_window_length=target_window_length, # Use target_window_length from process_day args
            target_col=target_col_name # Use target_col_name from process_day args
        )

        if X_windows_debug is not None and window_info_debug is not None:
            logging.info(f"DEBUG: generate_sequential_windows output shapes:")
            logging.info(f"  X_windows_debug: {X_windows_debug.shape}")
            logging.info(f"  target_windows_debug: {target_windows_debug.shape if target_windows_debug is not None else 'None'}")
            logging.info(f"  window_info_debug: {window_info_debug.shape}")
            logging.info(f"  Number of feature_cols_debug: {len(feature_cols_debug)}")

            logging.info(f"DEBUG: window_info_debug head:")
            print(window_info_debug.head())

            if not window_info_debug.empty and X_windows_debug.shape[0] == window_info_debug.shape[0]:
                # Find index of target_col_name in feature_cols_debug to check last_target_in_window
                # This assumes target_col_name (e.g. 'wmp_mean') is also a feature in X_windows
                # If target_col_name might have suffixes like '_mean' added during bar creation, adjust accordingly.
                # For 'wmp_mean', it should be present directly if it was the target_col_name.
                
                target_col_actual_name_in_features = target_col_name # Assuming target_col is a direct feature
                
                # Attempt to find the target_col_name within the feature_cols_debug list
                # It might be that target_col_name itself is not a feature, but rather its components are.
                # For instance, if target_col_name is 'wmp_mean', it should be in feature_cols_debug.

                # Let's find the index of the target_col in feature_cols_debug for verification
                # The target_col specified to generate_sequential_windows is used to *create* target_windows.
                # The last_target_in_window in window_info comes from this original target_col in df_bars.
                # The X_windows contains features, which should include this target_col if it's meant to be an input feature.
                
                if target_col_actual_name_in_features in feature_cols_debug:
                    target_feature_idx = feature_cols_debug.index(target_col_actual_name_in_features)
                    logging.info(f"DEBUG: Verifying alignment for the first window (index 0):")
                    
                    # Info from window_info_debug
                    info_time = window_info_debug.iloc[0]['window_end_time']
                    info_last_target = window_info_debug.iloc[0]['last_target_in_window']
                    
                    # Corresponding value from X_windows_debug
                    # X_windows_debug shape: (num_samples, window_length, num_features)
                    # We need the last bar (-1) of the first window (0), and the target_feature_idx
                    x_last_target_val = X_windows_debug[0, -1, target_feature_idx]

                    # Corresponding bar_end_time from original bars_df_debug
                    # The 'window_end_time' in window_info corresponds to the end time of the (window_length-1)-th bar
                    # in the segment of bars_df_debug that formed this window.
                    # If the first window starts at index 0 of bars_df_debug, its input part ends at index (window_length - 1).
                    original_bar_time = bars_df_debug.iloc[window_length - 1]['bar_end_time']

                    logging.info(f"  window_info_debug[0]: window_end_time='{info_time}', last_target_in_window={info_last_target:.6f}")
                    logging.info(f"  X_windows_debug[0, -1, target_feature_idx ('{target_col_actual_name_in_features}')]: {x_last_target_val:.6f}")
                    logging.info(f"  bars_df_debug.iloc[{window_length - 1}]['bar_end_time']: '{original_bar_time}'")

                    

                    if pd.Timestamp(info_time) == pd.Timestamp(original_bar_time) and np.isclose(info_last_target, x_last_target_val):
                        logging.info("  DEBUG: Alignment for first window looks GOOD.")

                        # --- Comprehensive Check for All Windows ---
                        logging.info("DEBUG: Performing comprehensive alignment check for all windows...")
                        # Extract all last WMP values from X_windows_debug
                        # X_windows_debug[:, -1, target_feature_idx] will give a 1D array of these values
                        all_x_last_targets = X_windows_debug[:, -1, target_feature_idx]
                        
                        # Extract all 'last_target_in_window' values from window_info_debug
                        all_info_last_targets = window_info_debug['last_target_in_window'].values

                        if len(all_x_last_targets) == len(all_info_last_targets):
                            # Compare the two arrays element-wise for approximate equality
                            are_all_aligned = np.all(np.isclose(all_x_last_targets, all_info_last_targets))
                            if are_all_aligned:
                                logging.info("  DEBUG: COMPREHENSIVE ALIGNMENT CHECK PASSED! All last_target_in_window values match X_windows.")
                            else:
                                num_mismatches = np.sum(~np.isclose(all_x_last_targets, all_info_last_targets))
                                logging.error(f"  DEBUG: COMPREHENSIVE ALIGNMENT CHECK FAILED! {num_mismatches} out of {len(all_x_last_targets)} values do not align.")
                                # Optionally print a few mismatches
                                mismatches = np.where(~np.isclose(all_x_last_targets, all_info_last_targets))[0]
                                logging.error(f"    First few mismatch indices: {mismatches[:5]}")
                                for idx_mismatch in mismatches[:3]: # Print details for first 3 mismatches
                                    logging.error(f"      Mismatch at index {idx_mismatch}: X_window_val={all_x_last_targets[idx_mismatch]:.6f}, info_val={all_info_last_targets[idx_mismatch]:.6f}")
                        else:
                            logging.error("  DEBUG: COMPREHENSIVE ALIGNMENT CHECK FAILED! Length mismatch between extracted X_window targets and info targets.")
                        # --- End Comprehensive Check ---

                    else:
                        logging.error("  DEBUG: MISMATCH DETECTED for first window alignment.")
                else:
                    logging.warning(f"DEBUG: Cannot verify last_target_in_window alignment as target_col '{target_col_actual_name_in_features}' not found in feature_cols_debug: {feature_cols_debug[:10]}...")
            else:
                logging.warning("DEBUG: window_info_debug is empty or row count mismatch with X_windows_debug. Cannot verify alignment.")
            
            # --- Save Debug Data to Files ---
            logging.info("DEBUG: Saving X_windows_debug, target_windows_debug, and window_info_debug to files...")
            try:
                np.save("debug_X_windows.npy", X_windows_debug)
                logging.info("  Saved: debug_X_windows.npy")
                if target_windows_debug is not None:
                    np.save("debug_target_windows.npy", target_windows_debug)
                    logging.info("  Saved: debug_target_windows.npy")
                window_info_debug.to_parquet("debug_window_info.parquet", index=False)
                logging.info("  Saved: debug_window_info.parquet")
                logging.info(f"Files saved in current working directory: {os.getcwd()}")
            except Exception as e:
                logging.error(f"  DEBUG: Error saving debug data: {e}")
            # --- End Save Debug Data ---

        else:
            logging.info(f"DEBUG: generate_sequential_windows returned None or empty data for symbol {first_sym_name_debug}.")
        # --- END DEBUGGING generate_sequential_windows ---

    else:
        logging.info("DEBUG: bars_df_debug is None.")

    logging.info("DEBUG: Exiting after single symbol test for create_volume_bars_with_lob_features.")
    exit()
    # --- END DEBUGGING SINGLE SYMBOL ---

    # --- 5. Process Symbols in Parallel --- 
    step_start_time = time.time()
    logging.info(f"Step 5: Processing symbols in parallel for day {day}...")
    # Group data by symbol for parallel processing
    grouped_symbols = df_filtered.groupby('sym')
    num_symbols = len(grouped_symbols)
    logging.info(f"Found {num_symbols} symbols to process in parallel.")
    del df_filtered; gc.collect() # Free memory of the large filtered dataframe

    # Use joblib to parallelize symbol processing
    # n_jobs=-1 uses all available CPU cores
    results = Parallel(n_jobs=-1)(
        delayed(_process_symbol_bars_windows)(
            sym_name=sym_name, 
            df_sym=group.copy(), # Pass a copy to avoid potential issues with shared data
            nb_bars=nb_bars,
            window_length=window_length,
            target_window_length=target_window_length,
            target_col_name=target_col_name
        )
        for sym_name, group in tqdm(grouped_symbols, total=num_symbols, desc=f"Parallel Processing {day}")
    )
    
    logging.info("Parallel processing finished. Collecting results...")

    # Initialize lists to hold results for the *entire day*
    all_X_windows_day = []
    all_target_windows_day = []
    all_window_info_day = []

    # Collect results from parallel jobs, filtering out None results
    successful_symbols = 0
    feature_cols_list = None # Initialize variable to store the feature list
    for result in results:
        if result is not None:
            X_windows_sym, target_windows_sym, window_info_sym, feature_cols = result # Unpack feature_cols
            # Store the feature list from the first successful result
            if feature_cols_list is None:
                feature_cols_list = feature_cols
                logging.info(f"Captured feature list ({len(feature_cols_list)} features) from first successful symbol.")
            
            all_X_windows_day.append(X_windows_sym)
            all_target_windows_day.append(target_windows_sym)
            all_window_info_day.append(window_info_sym)
            successful_symbols += 1
        # No need for else, None results indicate failure already logged in helper
        
    logging.info(f"Successfully collected results from {successful_symbols}/{num_symbols} symbols.")
    del results; gc.collect() # Free the list returned by Parallel

    # --- End of Parallel Symbol Processing ---
    bars_window_time = time.time() - step_start_time
    mem_after_bars_windows = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_bars_windows)
    logging.info(f"Step 5 (Parallel Symbol Processing) complete: Time={bars_window_time:.2f}s, Peak Mem={mem_after_bars_windows:.2f} GB")
    # df_filtered was already deleted before the parallel call

    # --- 6. Combine and Save Daily Results ---
    step_start_time = time.time()
    logging.info(f"Step 6: Combining and saving results for day {day}...")
    day_windows_count = 0
    
    if not all_X_windows_day:
        logging.warning(f"No windows generated for any symbol on day {day}. Skipping saving.")
    elif feature_cols_list is None: # Check if we captured the feature list
        logging.warning(f"No feature list captured for day {day} (likely no symbols succeeded). Skipping saving.")
    else:
        try:
            logging.debug("Concatenating daily X windows...")
            X_windows_day_final = np.concatenate(all_X_windows_day, axis=0)
            del all_X_windows_day; gc.collect()
            
            logging.debug("Concatenating daily target windows...")
            target_windows_day_final = np.concatenate(all_target_windows_day, axis=0)
            del all_target_windows_day; gc.collect()
            
            logging.debug("Concatenating daily window info...")
            window_info_day_final = pd.concat(all_window_info_day, ignore_index=True)
            del all_window_info_day; gc.collect()
            
            logging.debug("Sorting final daily data by window end time...")
            # Sort the info DataFrame first
            window_info_day_final = window_info_day_final.sort_values('window_end_time').reset_index(drop=True)
            # Get the sorted indices - THIS IS THE CRUCIAL STEP TO REORDER ARRAYS
            # However, since we built sequentially and concat *usually* preserves order relative 
            # to the list elements, simply sorting the final info DF *might* be sufficient
            # if the downstream process re-sorts anyway. Let's assume for now sorting info is enough.
            # If issues arise later, explicit reordering of arrays using indices is needed:
            # sorted_indices = window_info_day_final.index
            # X_windows_day_final = X_windows_day_final[sorted_indices] 
            # target_windows_day_final = target_windows_day_final[sorted_indices]

            day_windows_count = len(X_windows_day_final)
            logging.info(f"Combined {day_windows_count} windows for day {day}.")
            logging.info(f"Final Day Shapes: X={X_windows_day_final.shape}, TGT={target_windows_day_final.shape}, INFO={window_info_day_final.shape}")

            # Create the output directory for the day *directly* under OUTPUT_DIR
            day_output_dir = os.path.join(output_dir, day)
            os.makedirs(day_output_dir, exist_ok=True)
            logging.debug(f"Ensured daily output directory exists: {day_output_dir}")

            # Define filenames (no symbol needed)
            x_fname = os.path.join(day_output_dir, f'X_windows_in{window_length}_tgt{target_window_length}.npy')
            tgt_fname = os.path.join(day_output_dir, f'target_windows_in{window_length}_tgt{target_window_length}.npy')
            info_fname = os.path.join(day_output_dir, f'window_info_in{window_length}_tgt{target_window_length}.parquet')

            # Save the combined files
            np.save(x_fname, X_windows_day_final)
            logging.info(f"Saved daily input windows to {x_fname}")
            np.save(tgt_fname, target_windows_day_final)
            logging.info(f"Saved daily target windows to {tgt_fname}")
            window_info_day_final.to_parquet(info_fname, index=False)
            logging.info(f"Saved daily window info to {info_fname}")
            
            # --- Save Feature List --- 
            feature_list_fname = os.path.join(day_output_dir, 'features.txt')
            try:
                with open(feature_list_fname, 'w') as f:
                    for feature_name in feature_cols_list:
                        f.write(f"{feature_name}\n")
                logging.info(f"Saved feature list to {feature_list_fname}")
            except Exception as e:
                logging.error(f"Error saving feature list to {feature_list_fname}: {e}")
            # -------------------------
            
            # Clean up final daily arrays from memory after saving
            del X_windows_day_final, target_windows_day_final, window_info_day_final
            gc.collect()

        except Exception as e:
            logging.error(f"Error combining or saving results for day {day}: {e}", exc_info=True)

    save_time = time.time() - step_start_time
    mem_after_save = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_save)
    logging.info(f"Step 6 (Combine & Save) complete: Time={save_time:.2f}s, Peak Mem={mem_after_save:.2f} GB")

    # --- Day End --- 
    day_end_time = time.time()
    day_duration = day_end_time - day_start_time
    day_peak_mem_overall = day_peak_mem_current
    logging.info(f"--- Finished processing day {day}: Total Time={day_duration:.2f}s, Peak Mem Usage={day_peak_mem_overall:.2f} GB ---")
    gc.collect() # Extra GC call at end of day processing
    return day_windows_count # Return number of windows generated for this day

# --- Main Pipeline Execution ---
def main():
    
    # --- Configuration ---
    logging.info("=== Configuring Volume Bar Pipeline ===")
    BASE_DATA_PATH = '/mnt/user_disk/kfeghoul/storage_1_10T/Citibank/egbs_data_02_25'
    # DAYS_TO_PROCESS = [
    #     '20250212', '20250213', '20250214', '20250217', '20250218',
    #     '20250219', '20250220', '20250221', '20250224', '20250225'
    # ]

    DAYS_TO_PROCESS = [
        '20250212'
    ]

    SYMBOLS_TO_EXCLUDE = ['FGBLM5', 'CONFH5']
    NB_BARS_PER_DAY_SYMBOL = 10000
    WINDOW_LENGTH = 150
    TARGET_WINDOW_LENGTH = 30
    TARGET_COLUMN_NAME = 'wmp_mean'
    BASE_OUTPUT_DIR = '/mnt/user_disk/kfeghoul/storage_1_10T/Citibank/processed_data_volume_bars'
    PARAMS_SUBDIR = f"volbars_{NB_BARS_PER_DAY_SYMBOL}_in{WINDOW_LENGTH}_tgt{TARGET_WINDOW_LENGTH}"
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, PARAMS_SUBDIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")

    # --- Start Processing ---
    total_windows_generated_all_days = 0
    overall_start_time = time.time()
    overall_peak_mem_gb = 0.0
    logging.info("=== Starting Day-by-Day Volume Bar Processing ===")

    for day in DAYS_TO_PROCESS:
        # Call the refactored process_day function
        daily_windows = process_day(
            day=day,
            base_data_path=BASE_DATA_PATH,
            nb_bars=NB_BARS_PER_DAY_SYMBOL,
            window_length=WINDOW_LENGTH,
            target_window_length=TARGET_WINDOW_LENGTH,
            target_col_name=TARGET_COLUMN_NAME,
            output_dir=OUTPUT_DIR,
            symbols_to_exclude=SYMBOLS_TO_EXCLUDE
        )
        total_windows_generated_all_days += daily_windows
        # Memory tracking update (optional, if process_day doesn't capture peak correctly)
        # current_mem = get_memory_usage_gb()
        # overall_peak_mem_gb = max(overall_peak_mem_gb, current_mem)

    # --- End of All Days Loop ---
    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    logging.info("\n=== Finished All Volume Bar Processing ===")
    logging.info(f"Total windows generated across all days: {total_windows_generated_all_days:,}")
    logging.info(f"Total processing time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    # Peak memory reporting relies on process_day correctly finding its peak
    # To get true overall peak, would need checks within main loop or a global tracker
    # logging.info(f"Overall peak memory usage during pipeline: {overall_peak_mem_gb:.2f} GB") 
    logging.info(f"Daily processed files are saved in subdirectories under: {OUTPUT_DIR}")
    logging.info("--- Volume Bar Pipeline Finished ---")

if __name__ == "__main__":
    main() 