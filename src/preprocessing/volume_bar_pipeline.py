import logging
import os
import sys
import time
import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
from typing import List, Tuple, Optional
import gc
from pathlib import Path
import warnings

# --- Add project root to sys.path ---
current_file_path = Path(__file__).resolve()
project_root_path = current_file_path.parent.parent
project_root_str = str(project_root_path)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# --- Import Custom Modules ---
from utils.utils import load_and_clean_day_data
from utils_preprocessing import get_memory_usage_gb, compute_microstructure_features, clean_raw_data

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


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
    activity_col_name: str = 'L1_total_volume', 
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
    
    # Calculate volume threshold per bar
    total_cumulative_volume = data['cumulative_activity_volume'].max()
    volume_threshold = total_cumulative_volume / nb_bars 
    logging.info(f"Calculated Volume Threshold per bar: {volume_threshold:.2f} (Total Vol: {total_cumulative_volume}, Target Bars: {nb_bars})")

    if volume_threshold <= 0:
         logging.warning(f"Volume threshold is non-positive ({volume_threshold:.2f}). Cannot create bars.")
         return None

    # Assign bar ID based on cumulative volume crossing thresholds
    data['bar_id'] = (data['cumulative_activity_volume'] // volume_threshold).astype(int)

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

    # Define columns that are handled specifically or are not features for generic aggregation
    cols_to_exclude_from_generic_aggregation = [
        time_col_name,          
        wmp_col_name,          
        activity_col_name,      
        'bar_id',               
        'cumulative_activity_volume', 
        'sym'                   
    ]
    
    cols_to_exclude_from_generic_aggregation = [
        c for c in cols_to_exclude_from_generic_aggregation if c in data_full_bars.columns
    ]

    features_to_aggregate = [
        col for col in data_full_bars.columns
        if col not in cols_to_exclude_from_generic_aggregation
        and pd.api.types.is_numeric_dtype(data_full_bars[col]) 
    ]
    logging.debug(f"Identified {len(features_to_aggregate)} features for generic aggregation. First 5: {features_to_aggregate[:5] if features_to_aggregate else 'None'}")

    # Aggregation function
    def aggregate_bar(bar_group: pd.DataFrame) -> pd.Series:
        results = {
            'bar_start_time': bar_group[time_col_name].iloc[0],
            'bar_end_time': bar_group[time_col_name].iloc[-1],
            'num_snapshots_in_bar': len(bar_group),
            'actual_volume_in_bar': bar_group[activity_col_name].sum(),
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
            # Catch and log RuntimeWarnings specifically for this column's aggregations
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", RuntimeWarning) # Ensure RuntimeWarnings are caught

                try:
                    mean_val = bar_group[col].mean()
                    sum_val = bar_group[col].sum()
                    min_val = bar_group[col].min()
                    max_val = bar_group[col].max()
                    std_val = bar_group[col].std()
                    median_val = bar_group[col].median()

                    results[f"{col}_mean"] = mean_val
                    results[f"{col}_sum"] = sum_val
                    results[f"{col}_min"] = min_val
                    results[f"{col}_max"] = max_val
                    results[f"{col}_std"] = std_val
                    results[f"{col}_median"] = median_val

                except Exception as e:
                    logging.error(f"Error during generic aggregation for column '{col}': {e}", exc_info=True)
                    # Assign NaNs or 0s if an unexpected error occurs during aggregation
                    results[f"{col}_mean"] = np.nan
                    results[f"{col}_sum"] = np.nan
                    results[f"{col}_min"] = np.nan
                    results[f"{col}_max"] = np.nan
                    results[f"{col}_std"] = np.nan
                    results[f"{col}_median"] = np.nan


                if caught_warnings:
                    for cw in caught_warnings:
                        if issubclass(cw.category, RuntimeWarning) and "empty slice" in str(cw.message).lower():
                            logging.warning(f"Caught RuntimeWarning: '{cw.message}' for column '{col}'. "
                                            f"Bar group shape: {bar_group.shape}. Column '{col}' has {bar_group[col].isnull().sum()} NaNs "
                                            f"out of {len(bar_group[col])} values. First few values of column '{col}': {bar_group[col].head().values}")
            
        return pd.Series(results)

    # Group by bar_id and apply the aggregation
    grouped_bars = data_full_bars.groupby('bar_id')
    volume_bars_df = grouped_bars.apply(aggregate_bar, include_groups=False)

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
    for i in range(total_bars - required_bars + 1):
        input_end_idx = i + window_length
        target_start_idx = input_end_idx
        target_end_idx = target_start_idx + target_window_length

        # Slice input features
        input_window_slice = feature_data[i : input_end_idx]
        
        # Slice target variable for the future window
        target_window_slice = targets[target_start_idx : target_end_idx]

        assert input_window_slice[-1,0] == targets[target_start_idx-1]
        assert input_window_slice[-1,0] == targets[input_end_idx - 1]
        
        X_windows_list.append(input_window_slice)
        target_windows_list.append(target_window_slice)

        # Store metadata: time corresponds to the *end* of the input window
        info = {
            'window_end_time': times[input_end_idx - 1], 
            'last_target_in_window': targets[input_end_idx - 1] # Store last target value of the input window
        }

        # Store the last target value of the input window
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
    X_windows = np.stack(X_windows_list, axis=0).astype(np.float32)
    target_windows = np.stack(target_windows_list, axis=0).astype(np.float32)
    window_info = pd.DataFrame(window_info_list)
    window_info['window_end_time'] = pd.to_datetime(window_info['window_end_time']) 
    window_info = window_info.sort_values('window_end_time').reset_index(drop=True) 
    
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
    
    # Ensure time column is datetime and sorted 
    if 'time' not in df_sym.columns:
            logging.error(f"[{sym_name}] 'time' column missing.")
            return None
    df_sym['time'] = pd.to_datetime(df_sym['time'])
    df_sym = df_sym.sort_values('time').reset_index(drop=True)

    # 1. Create Volume Bars
    logging.debug(f"[{sym_name}] Creating volume bars...")
    bars_df = create_volume_bars_with_lob_features(df_sym, nb_bars)
    logging.debug(f"[{sym_name}] Volume bars created. Shape: {bars_df.shape if bars_df is not None else 'None'}")
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

   


# --- Main Daily Processing Function ---
def process_day(
    day: str,
    base_data_path: str,
    nb_bars: int,
    window_length: int,
    target_window_length: int,
    target_col_name: str,
    output_dir: str,
    symbols_to_exclude: List[str] 
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
  
    # --- 2. Clean Data ---
    step_start_time = time.time()
    logging.info("Step 2: Cleaning data using utility function...")
    df_clean = clean_raw_data(df_day_raw)
    clean_time = time.time() - step_start_time
    mem_after_clean = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_clean)
    logging.info(f"Step 2 (Clean) complete: Time={clean_time:.2f}s, Peak Mem={mem_after_clean:.2f} GB. Shape after clean: {df_clean.shape if df_clean is not None else 'None'}")
    if df_clean is None or df_clean.empty:
        logging.warning(f"Skipping day {day} as it became empty after cleaning.")
        if df_day_raw is not None: del df_day_raw
        gc.collect()
        return 0
    if 'df_day_raw' in locals() and df_day_raw is not None:
         del df_day_raw; gc.collect()

    # --- 3. Filter Symbols ---
    step_start_time = time.time()
    logging.info(f"Step 3: Filtering symbols (excluding {symbols_to_exclude})...")
        
    initial_syms = df_clean['sym'].nunique()
    df_filtered = df_clean[~df_clean['sym'].isin(symbols_to_exclude)].copy()
    syms_dropped = initial_syms - df_filtered['sym'].nunique()

    filter_time = time.time() - step_start_time
    mem_after_filter = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_filter)
    logging.info(f"Step 3 (Filter Sym) complete: Time={filter_time:.2f}s, Peak Mem={mem_after_filter:.2f} GB")
    logging.info(f"Dropped {syms_dropped} symbols. Shape after symbol filter: {df_filtered.shape}")
    del df_clean; gc.collect()

    # --- 4. Compute Features ---
    step_start_time = time.time()
    logging.info("Step 4: Computing core features using utility function on filtered data...")
    sym_dfs = {}
    for sym, sym_df in tqdm(df_filtered.groupby('sym'), total=df_filtered['sym'].nunique(), desc="Computing microstructure features"):
        logging.info(f"    Computed features for {sym}")
        sym_df = compute_microstructure_features(sym_df.copy())
        sym_dfs[sym] = sym_df
        logging.info(f"    Dataset shape: {sym_df.shape}\n")

    #df_featured = compute_microstructure_features(df_filtered) 
    feature_time = time.time() - step_start_time
    mem_after_features = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_features)
    total_rows_in_sym_dfs = sum(df.shape[0] for df in sym_dfs.values())
    logging.info(f"Step 4 (Features) complete: Time={feature_time:.2f}s, Peak Mem={mem_after_features:.2f} GB, Processed {len(sym_dfs)} symbols with a total of {total_rows_in_sym_dfs} rows.")
    del df_filtered; gc.collect() 


    # --- 5. Process Symbols Sequentially with Per-Symbol Checks --- 
    step_start_time = time.time()
    logging.info(f"Step 5: Processing symbols sequentially for day {day} with detailed checks...")
    
    num_symbols = len(sym_dfs)
    logging.info(f"Found {num_symbols} symbols with precomputed features to process sequentially.")
    # del df_featured; gc.collect() # df_featured no longer exists

    all_X_windows_day = []
    all_target_windows_day = []
    all_window_info_day = []
    master_feature_cols_list = None 
    any_symbol_processed_successfully = False

    for sym_name, group_df in tqdm(sym_dfs.items(), total=num_symbols, desc=f"Sequential Processing {day}"):
        logging.info(f"--- Processing symbol: {sym_name} ---")

        # Calculate nb_bars for this specific symbol
        nb_bars_for_sym = compute_nb_bars(group_df.copy(), max_nb_bars=nb_bars) # Use main nb_bars as max_nb_bars for consistency

        symbol_result = _process_symbol_bars_windows(
            sym_name=sym_name,
            df_sym=group_df.copy(), 
            nb_bars=nb_bars_for_sym, 
            window_length=window_length,
            target_window_length=target_window_length,
            target_col_name=target_col_name
        )

        if symbol_result is not None:
            X_windows_sym, target_windows_sym, window_info_sym, features_for_sym = symbol_result
            
            # Feature List Consistency Check
            if master_feature_cols_list is None:
                master_feature_cols_list = features_for_sym
                logging.info(f"  [{sym_name}] Captured master feature list ({len(master_feature_cols_list)} features): {master_feature_cols_list[:5]}...")

            elif master_feature_cols_list != features_for_sym:
                logging.error(f"  [{sym_name}] CRITICAL MISMATCH: Feature list does not match master list. Skipping this symbol.")
                logging.error(f"    Master ({len(master_feature_cols_list)}): {master_feature_cols_list[:5]}...")
                logging.error(f"    Symbol's ({len(features_for_sym)}): {features_for_sym[:5]}...")
                continue 

            # 2. Per-Symbol Alignment Check
            logging.info(f"  [{sym_name}] Performing alignment check for its {len(X_windows_sym)} windows...")
            if not window_info_sym.empty and target_col_name in features_for_sym:
                target_feature_idx_sym = features_for_sym.index(target_col_name)
                all_x_last_targets_sym = X_windows_sym[:, -1, target_feature_idx_sym]
                all_info_last_targets_sym = window_info_sym['last_target_in_window'].values

                if len(all_x_last_targets_sym) == len(all_info_last_targets_sym) and \
                   np.all(np.isclose(all_x_last_targets_sym, all_info_last_targets_sym)):
                    logging.info(f"    [{sym_name}] ALIGNMENT CHECK PASSED for this symbol.")
                else:
                    num_mismatches_sym = np.sum(~np.isclose(all_x_last_targets_sym, all_info_last_targets_sym)) if len(all_x_last_targets_sym) == len(all_info_last_targets_sym) else -1
                    logging.error(f"    [{sym_name}] ALIGNMENT CHECK FAILED for this symbol! Mismatches: {num_mismatches_sym}/{len(all_x_last_targets_sym)}. Skipping this symbol.")
                    continue # Skip to next symbol
            elif window_info_sym.empty:
                 logging.warning(f"  [{sym_name}] window_info_sym is empty. Cannot perform alignment check. Skipping.")
                 continue
            else: # target_col_name not in features_for_sym
                logging.warning(f"  [{sym_name}] Target column '{target_col_name}' not found in its feature list. Cannot perform alignment check. Skipping.")
                continue

            # If all checks passed for this symbol, append its data
            all_X_windows_day.append(X_windows_sym)
            all_target_windows_day.append(target_windows_sym)
            all_window_info_day.append(window_info_sym)
            any_symbol_processed_successfully = True
            logging.info(f"  [{sym_name}] Successfully processed and data appended.")
        else:
            logging.warning(f"  [{sym_name}] _process_symbol_bars_windows returned None. Skipping.")
            
    logging.info("Sequential symbol processing finished.")
    # Free memory of intermediate symbol data if any
    if 'group_df' in locals(): del group_df 
    if 'symbol_result' in locals(): del symbol_result
    if 'X_windows_sym' in locals(): del X_windows_sym
    if 'target_windows_sym' in locals(): del target_windows_sym
    if 'window_info_sym' in locals(): del window_info_sym
    gc.collect()

    # --- End of Sequential Symbol Processing ---
    bars_window_time = time.time() - step_start_time
    mem_after_bars_windows = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_bars_windows)
    logging.info(f"Step 5 (Sequential Symbol Processing) complete: Time={bars_window_time:.2f}s, Peak Mem={mem_after_bars_windows:.2f} GB")

    # --- 6. Combine and Save Daily Results ---
    step_start_time = time.time()
    logging.info(f"Step 6: Combining and saving results for day {day}...")
    day_windows_count = 0
    
    if not all_X_windows_day:
        logging.warning(f"No windows generated for any symbol on day {day} (all_X_windows_day is empty). Skipping saving.")
    elif master_feature_cols_list is None:
        logging.warning(f"No master feature list captured for day {day} (likely no symbols succeeded or all had mismatches). Skipping saving.")
    else:
        try:
            logging.debug("Concatenating daily X windows...")
            X_windows_day_final = np.concatenate(all_X_windows_day, axis=0)
            del all_X_windows_day; gc.collect()
            
            logging.debug("Concatenating daily target windows...")
            target_windows_day_final = np.concatenate(all_target_windows_day, axis=0)
            del all_target_windows_day; gc.collect()
            
            logging.debug("Concatenating daily window info...")
            window_info_day_final = pd.concat(all_window_info_day, ignore_index=False) 
            del all_window_info_day; gc.collect()

            # # --- Robust Sorting and Re-indexing of Combined Data ---
            # logging.info("Applying robust sorting to combined daily data...")
            # if not window_info_day_final.empty:
            #     sorted_indices = np.lexsort((window_info_day_final['window_end_time'].values, window_info_day_final['sym'].values))
                
            #     logging.debug(f"Re-indexing window_info_day_final ({window_info_day_final.shape}) using {len(sorted_indices)} sorted_indices...")
            #     window_info_day_final = window_info_day_final.iloc[sorted_indices].reset_index(drop=True)
                
            #     logging.debug(f"Re-indexing X_windows_day_final ({X_windows_day_final.shape}) using {len(sorted_indices)} sorted_indices...")
            #     X_windows_day_final = X_windows_day_final[sorted_indices]
                
            #     logging.debug(f"Re-indexing target_windows_day_final ({target_windows_day_final.shape}) using {len(sorted_indices)} sorted_indices...")
            #     target_windows_day_final = target_windows_day_final[sorted_indices]
            #     logging.info("Robust sorting and re-indexing of combined data complete.")

            #     # --- Final Alignment Check on Combined Data ---
            #     logging.info("Performing final alignment check on combined and sorted daily data...")
            #     if target_col_name in master_feature_cols_list:
            #         final_target_idx = master_feature_cols_list.index(target_col_name)
            #         final_x_targets = X_windows_day_final[:, -1, final_target_idx]
            #         final_info_targets = window_info_day_final['last_target_in_window'].values
            #         if len(final_x_targets) == len(final_info_targets) and \
            #            np.all(np.isclose(final_x_targets, final_info_targets)):
            #             logging.info("  FINAL COMBINED ALIGNMENT CHECK PASSED!")
            #         else:
            #             num_final_mismatches = np.sum(~np.isclose(final_x_targets, final_info_targets)) if len(final_x_targets) == len(final_info_targets) else -1
            #             logging.error(f"  FINAL COMBINED ALIGNMENT CHECK FAILED! Mismatches: {num_final_mismatches}/{len(final_x_targets)}")
            #     else:
            #         logging.warning(f"  Cannot perform final combined alignment check: target '{target_col_name}' not in master feature list.")
            #     # --- End Final Alignment Check ---
            # else:
            #     logging.warning("window_info_day_final is empty. Skipping sorting and final alignment check.")


            day_windows_count = len(X_windows_day_final) if 'X_windows_day_final' in locals() and X_windows_day_final is not None else 0
            logging.info(f"Combined and sorted {day_windows_count} windows for day {day}.")
            if day_windows_count > 0:
                 logging.info(f"Final Day Shapes: X={X_windows_day_final.shape}, TGT={target_windows_day_final.shape}, INFO={window_info_day_final.shape}")

            # Create the output directory for the day
            day_output_dir = os.path.join(output_dir, day)
            os.makedirs(day_output_dir, exist_ok=True)
            logging.debug(f"Ensured daily output directory exists: {day_output_dir}")

            if day_windows_count > 0:
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
                
                # Save Feature List 
                feature_list_fname = os.path.join(day_output_dir, 'features.txt')
                try:
                    with open(feature_list_fname, 'w') as f:
                        for feature_name in master_feature_cols_list: 
                            f.write(f"{feature_name}\\n")
                    logging.info(f"Saved feature list to {feature_list_fname}")
                except Exception as e:
                    logging.error(f"Error saving feature list to {feature_list_fname}: {e}")
                
                del X_windows_day_final, target_windows_day_final, window_info_day_final
                gc.collect()
            else:
                logging.info(f"No windows to save for day {day}.")

        except Exception as e:
            logging.error(f"Error combining or saving results for day {day}: {e}", exc_info=True)
            if 'all_X_windows_day' in locals(): del all_X_windows_day
            if 'all_target_windows_day' in locals(): del all_target_windows_day
            if 'all_window_info_day' in locals(): del all_window_info_day
            if 'X_windows_day_final' in locals() and X_windows_day_final is not None: del X_windows_day_final
            if 'target_windows_day_final' in locals() and target_windows_day_final is not None: del target_windows_day_final
            if 'window_info_day_final' in locals() and window_info_day_final is not None: del window_info_day_final
            gc.collect()

    save_time = time.time() - step_start_time
    mem_after_save = get_memory_usage_gb()
    day_peak_mem_current = max(day_peak_mem_current, mem_after_save)
    logging.info(f"Step 6 (Combine & Save) complete: Time={save_time:.2f}s, Peak Mem={mem_after_save:.2f} GB")

    # --- Day End --- 
    day_end_time = time.time()
    day_duration = day_end_time - day_start_time
    day_peak_mem_overall = day_peak_mem_current
    logging.info(f"--- Finished processing day {day}: Total Time={day_duration:.2f}s, Peak Mem Usage={day_peak_mem_overall:.2f} GB ---")
    gc.collect() 
    return day_windows_count 

def main():
    
    # --- Configuration ---
    logging.info("=== Configuring Volume Bar Pipeline ===")
    BASE_DATA_PATH = '/mnt//storage_1_10T/citibank/egbs_data_02_25'
    
    DAYS_TO_PROCESS = [
        '20250212', '20250213', '20250214', '20250217', '20250218',
        '20250219', '20250220', '20250221', '20250224', '20250225'
    ]

    # DAYS_TO_PROCESS = [
    #     '20250212'
    # ]

    SYMBOLS_TO_EXCLUDE = ['FGBLM5', 'CONFH5']
    NB_BARS_PER_DAY_SYMBOL = 10000
    WINDOW_LENGTH = 150
    TARGET_WINDOW_LENGTH = 50
    TARGET_COLUMN_NAME = 'wmp_mean'
    BASE_OUTPUT_DIR = '/mnt/storage_1_10T/citibank/data/processed_data_volume_bars'
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
    logging.info(f"Daily processed files are saved in subdirectories under: {OUTPUT_DIR}")
    logging.info("--- Volume Bar Pipeline Finished ---")

if __name__ == "__main__":
    main() 