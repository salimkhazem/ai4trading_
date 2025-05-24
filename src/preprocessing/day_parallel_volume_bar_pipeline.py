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
from joblib import Parallel, delayed # Added for day-level parallelism

# --- Add project root to sys.path ---
current_file_path = Path(__file__).resolve()
project_root_path = current_file_path.parent.parent
project_root_str = str(project_root_path)
if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)

# --- Import Custom Modules ---
from utils.utils import load_day_data
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
    # Logging within this function is per-symbol, so it's fine with parallel days
    # logging.debug(f"Starting volume bar creation for symbol (shape {df_symbol.shape}) with target {nb_bars} bars.")


    # Compute total volume between bid and ask
    df_symbol[activity_col_name] = df_symbol['L1_bid_size'] + df_symbol['L1_ask_size']
    df_symbol[activity_col_name] = df_symbol[activity_col_name].fillna(0)

    # Calculate cumulative activity volume
    df_symbol['cumulative_activity_volume'] = df_symbol[activity_col_name].cumsum()

    required_cols = [time_col_name, wmp_col_name, activity_col_name, 'cumulative_activity_volume']
    if not all(col in df_symbol.columns for col in required_cols):
        logging.error(f"Missing required columns for bar creation: Needs {required_cols}. Found {df_symbol.columns.tolist()}")
        return None

    # Ensure data is sorted by time
    if not df_symbol[time_col_name].is_monotonic_increasing:
         logging.warning("Input data not sorted by time. Sorting now...")
         df_symbol = df_symbol.sort_values(time_col_name).reset_index(drop=True)

    data = df_symbol.copy()

    # Calculate volume threshold per bar
    total_cumulative_volume = data['cumulative_activity_volume'].max()
    if nb_bars <= 0: # Prevent division by zero if nb_bars is somehow 0 or negative
        logging.warning(f"Target number of bars is non-positive ({nb_bars}). Cannot create bars.")
        return None
    volume_threshold = total_cumulative_volume / nb_bars
    # logging.info(f"Calculated Volume Threshold per bar: {volume_threshold:.2f} (Total Vol: {total_cumulative_volume}, Target Bars: {nb_bars})")


    if volume_threshold <= 0:
         logging.warning(f"Volume threshold is non-positive ({volume_threshold:.2f}). Cannot create bars.")
         return None

    # Assign bar ID based on cumulative volume crossing thresholds
    data['bar_id'] = (data['cumulative_activity_volume'] // volume_threshold).astype(int)

    # Handle edge case: First few entries might be below the first threshold / Ensure they belong to bar 0
    if data.empty:
        # logging.warning("Data is empty before attempting to access first_bar_id. Cannot create bars.")
        return None
    first_bar_id = data.iloc[0]['bar_id']
    data.loc[data['bar_id'] == first_bar_id, 'bar_id'] = 0

    # Exclude snapshots belonging to a potentially partial last bar
    # Ensure we only include bars where the *next* threshold was crossed
    max_possible_bar_id = int(np.floor(total_cumulative_volume / volume_threshold)) -1
    # logging.info(f"max_possible_bar_id: {max_possible_bar_id}")
    if max_possible_bar_id < 0:
        # logging.warning("Not enough volume to form even one full bar.")
        return None

    data_full_bars = data[data['bar_id'] <= max_possible_bar_id].copy()
    if data_full_bars.empty:
        # logging.warning("No full bars could be formed after excluding the partial last bar.")
        return None

    # logging.debug(f"Data filtered to full bars (Bar IDs 0 to {max_possible_bar_id}). Shape: {data_full_bars.shape}")

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

    # logging.debug(f"Identified {len(features_to_aggregate)} features for generic aggregation. First 5: {features_to_aggregate[:5] if features_to_aggregate else 'None'}")

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
            with warnings.catch_warnings(record=True) as caught_warnings:
                warnings.simplefilter("always", RuntimeWarning)
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

                except Exception as e: # Broad exception to catch any error during aggregation
                    # logging.error(f"Error during generic aggregation for column '{col}': {e}", exc_info=True) # Consider if too verbose for parallel
                    # Assign NaNs or 0s if an unexpected error occurs
                    results[f"{col}_mean"] = np.nan
                    results[f"{col}_sum"] = np.nan
                    results[f"{col}_min"] = np.nan
                    results[f"{col}_max"] = np.nan
                    results[f"{col}_std"] = np.nan
                    results[f"{col}_median"] = np.nan


                if caught_warnings:
                    for cw in caught_warnings:
                        if issubclass(cw.category, RuntimeWarning) and "empty slice" in str(cw.message).lower():
                            # logging.warning(f"Caught RuntimeWarning: '{cw.message}' for column '{col}'. " # Consider if too verbose
                            #                 f"Bar group shape: {bar_group.shape}. Column '{col}' has {bar_group[col].isnull().sum()} NaNs "
                            #                 f"out of {len(bar_group[col])} values. First few values of column '{col}': {bar_group[col].head().values}")
                            pass # Suppress for now or log to a specific file per process

        return pd.Series(results)

    # Group by bar_id and apply the aggregation
    grouped_bars = data_full_bars.groupby('bar_id')
    volume_bars_df = grouped_bars.apply(aggregate_bar, include_groups=False) # include_groups=False is default in pandas >=2.0

    # Calculate bar duration
    volume_bars_df['bar_duration_seconds'] = (
        volume_bars_df['bar_end_time'] - volume_bars_df['bar_start_time']
    ).dt.total_seconds()

    # Final checks
    volume_bars_df = volume_bars_df.reset_index()
    volume_bars_df.fillna(0, inplace=True)

    # logging.info(f"Successfully created {len(volume_bars_df)} volume bars.")
    return volume_bars_df


def generate_sequential_windows(
    df_bars: pd.DataFrame,
    window_length: int = 100,
    target_window_length: int = 10,
    target_col: str = 'wmp_mean' # Column to use for target prediction
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[pd.DataFrame], Optional[List[str]]]:
    """
    Generates sequential, sliding windows and corresponding target windows from aggregated bar data.
    """
    # logging.info(f"Generating input windows (len={window_length}) and target windows (len={target_window_length}) using target '{target_col}'...")

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
    # logging.info(f"Using {num_features} input features for windows.")

    # Ensure features are numeric and handle NaNs
    # logging.debug("Converting features to float32 and checking for NaNs...")
    try:
        feature_data = df_bars[feature_cols].fillna(0).astype(np.float32).values
        if np.isnan(feature_data).any() or np.isinf(feature_data).any():
             # logging.warning("NaN or Inf values found in feature data after fillna(0). Check data pipeline.")
             feature_data = np.nan_to_num(feature_data, nan=0.0, posinf=0.0, neginf=0.0)
    except Exception as e:
        logging.error(f"Error converting features to float32: {e}", exc_info=True)
        return None, None, None, None
    # logging.debug("Feature conversion and NaN check complete.")

    # Extract time and target columns
    times = df_bars['bar_end_time'].values
    targets = df_bars[target_col].astype(np.float32).values # Target values

    # Check if enough data exists for at least one window
    total_bars = len(df_bars)
    required_bars = window_length + target_window_length
    if total_bars < required_bars:
        # logging.warning(f"Not enough bars ({total_bars}) to create a window of length {window_length} + target {target_window_length}.")
        return None, None, None, None

    X_windows_list, target_windows_list, window_info_list = [], [], []

    # logging.debug(f"Iterating to create windows (Total bars: {total_bars}, Required per window: {required_bars})...")

    # Iterate through the bars to create sliding windows
    for i in range(total_bars - required_bars + 1):
        input_end_idx = i + window_length
        target_start_idx = input_end_idx
        target_end_idx = target_start_idx + target_window_length

        input_window_slice = feature_data[i : input_end_idx]
        target_window_slice = targets[target_start_idx : target_end_idx]
        
        # This assertion logic needs to be robust if target_col is not the first feature.
        # Assuming target_col's processed version is what's in feature_data.
        try:
            target_col_idx_in_features = feature_cols.index(target_col)
            assert np.isclose(input_window_slice[-1, target_col_idx_in_features], targets[input_end_idx - 1], equal_nan=True)
        except ValueError:
            logging.warning(f"Target column '{target_col}' not found in feature_cols for assertion. Skipping assertion.")
        except AssertionError:
            logging.warning(f"Assertion failed: input_window_slice[-1, target_idx] != targets[input_end_idx - 1]. "
                            f"Values: {input_window_slice[-1, target_col_idx_in_features if 'target_col_idx_in_features' in locals() else 'N/A']} vs {targets[input_end_idx - 1]}")
            # Depending on strictness, you might skip this window or log and continue.

        X_windows_list.append(input_window_slice)
        target_windows_list.append(target_window_slice)

        info = {
            'window_end_time': times[input_end_idx - 1],
            'last_target_in_window': targets[input_end_idx - 1]
        }
        window_info_list.append(info)

    # logging.debug(f"Finished iterating. Created {len(X_windows_list)} raw windows.")

    if not X_windows_list:
        # logging.warning("No windows were generated after iteration.")
        empty_x = np.array([]).reshape(0, window_length, num_features).astype(np.float32)
        empty_tgt = np.array([]).reshape(0, target_window_length).astype(np.float32)
        empty_info = pd.DataFrame(columns=['window_end_time', 'last_target_in_window'])
        return empty_x, empty_tgt, empty_info, feature_cols

    # logging.debug("Stacking window lists into final arrays...")
    X_windows = np.stack(X_windows_list, axis=0).astype(np.float32)
    target_windows = np.stack(target_windows_list, axis=0).astype(np.float32)
    window_info = pd.DataFrame(window_info_list)
    window_info['window_end_time'] = pd.to_datetime(window_info['window_end_time'])
    # window_info = window_info.sort_values('window_end_time').reset_index(drop=True) # Sorting done after combining all symbols for the day

    # logging.debug("Stacking complete.")
    # logging.info(f"Window generation complete.")
    # logging.info(f"Shape of X_windows: {X_windows.shape}")
    # logging.info(f"Shape of target_windows: {target_windows.shape}")
    # logging.info(f"Shape of window_info: {window_info.shape}")

    return X_windows, target_windows, window_info, feature_cols


def _process_symbol_bars_windows(
    sym_name: str,
    df_sym: pd.DataFrame,
    nb_bars: int,
    window_length: int,
    target_window_length: int,
    target_col_name: str
) -> Optional[Tuple[str, np.ndarray, np.ndarray, pd.DataFrame, List[str]]]: # Return sym_name
    """
    Processes a single symbol: creates volume bars and generates sequential windows.
    Designed to be called by joblib.Parallel (within a day's processing) or sequentially.
    """
    if 'time' not in df_sym.columns:
        logging.error(f"[{sym_name}] 'time' column missing.")
        return None
    df_sym['time'] = pd.to_datetime(df_sym['time'])
    df_sym = df_sym.sort_values('time').reset_index(drop=True)

    bars_df = create_volume_bars_with_lob_features(df_sym, nb_bars)
    del df_sym; gc.collect()
    if bars_df is None or bars_df.empty:
        logging.warning(f"[{sym_name}] Could not create volume bars for day. Skipping.")
        return None

    X_windows_sym, target_windows_sym, window_info_sym, feature_cols = generate_sequential_windows(
        bars_df, window_length, target_window_length, target_col_name
    )
    del bars_df; gc.collect()
    if X_windows_sym is None or X_windows_sym.size == 0:
        logging.warning(f"[{sym_name}] No windows generated for day. Skipping.")
        return None

    window_info_sym['sym'] = sym_name
    return sym_name, X_windows_sym, target_windows_sym, window_info_sym, feature_cols


def process_day(
    day: str,
    base_data_path: str,
    nb_bars: int,
    window_length: int,
    target_window_length: int,
    target_col_name: str,
    output_dir: str,
    symbols_to_keep: List[str]
) -> int:
    """Process a single day: load, feature compute, bar creation, windowing, combine, save."""
    # This function is now the target for joblib.Parallel in the new script's main.
    # Most logging messages are suitable, some high-frequency ones within loops are reduced.
    logging.info(f"--- [Day {day}] START Processing ---")
    day_start_time = time.time()
    # day_peak_mem_start = get_memory_usage_gb() # Per-process memory might be harder to track globally
    # day_peak_mem_current = day_peak_mem_start

    df_day_raw = load_day_data(base_data_path, day)
    if df_day_raw is None or df_day_raw.empty:
        logging.warning(f"[Day {day}] No raw data loaded. Skipping.")
        return 0
    # logging.info(f"[Day {day}] Raw data loaded. Shape={df_day_raw.shape}")

    df_clean = clean_raw_data(df_day_raw)
    del df_day_raw; gc.collect()
    if df_clean is None or df_clean.empty:
        logging.warning(f"[Day {day}] Data empty after cleaning. Skipping.")
        return 0
    # logging.info(f"[Day {day}] Data cleaned. Shape={df_clean.shape}")

    df_filtered = df_clean[df_clean['sym'].isin(symbols_to_keep)].copy()
    del df_clean; gc.collect()
    if df_filtered.empty:
        logging.warning(f"[Day {day}] No symbols remaining after filtering. Skipping.")
        return 0
    # logging.info(f"[Day {day}] Symbols filtered. Kept {df_filtered['sym'].nunique()}. Shape={df_filtered.shape}")

    sym_dfs = {}
    # tqdm might not render well when days are parallel, consider removing or conditional verbosity
    for sym, sym_df_group in df_filtered.groupby('sym'): # tqdm(df_filtered.groupby('sym'), desc=f"Features Day {day}"):
        sym_dfs[sym] = compute_microstructure_features(sym_df_group.copy())
    del df_filtered; gc.collect()
    # logging.info(f"[Day {day}] Microstructure features computed for {len(sym_dfs)} symbols.")

    all_X_windows_day, all_target_windows_day, all_window_info_day = [], [], []
    master_feature_cols_list = None

    # This loop processes symbols sequentially WITHIN a day.
    # The parallelization is ACROSS days.
    for sym_name, group_df in sym_dfs.items(): # tqdm(sym_dfs.items(), desc=f"Windows Day {day}"):
        if group_df is None or group_df.empty:
            logging.warning(f"[Day {day}, Sym {sym_name}] Empty group_df. Skipping.")
            continue

        nb_bars_for_sym = compute_nb_bars(group_df.copy(), max_nb_bars=nb_bars)
        if nb_bars_for_sym <= 0:
            logging.warning(f"[Day {day}, Sym {sym_name}] Calculated nb_bars_for_sym is {nb_bars_for_sym}. Skipping.")
            del group_df; gc.collect()
            continue
            
        symbol_result = _process_symbol_bars_windows(
            sym_name=sym_name, df_sym=group_df.copy(), nb_bars=nb_bars_for_sym,
            window_length=window_length, target_window_length=target_window_length,
            target_col_name=target_col_name
        )
        del group_df; gc.collect()

        if symbol_result is not None:
            ret_sym_name, X_windows_sym, target_windows_sym, window_info_sym, features_for_sym = symbol_result
            if master_feature_cols_list is None:
                master_feature_cols_list = features_for_sym
            elif master_feature_cols_list != features_for_sym:
                logging.error(f"[Day {day}, Sym {ret_sym_name}] CRITICAL MISMATCH: Feature list. Skipping symbol.")
                continue # Skip this symbol

            # Simplified alignment check for brevity in parallel context
            if not window_info_sym.empty and target_col_name in features_for_sym:
                try:
                    target_feature_idx_sym = features_for_sym.index(target_col_name)
                    all_x_last_targets_sym = X_windows_sym[:, -1, target_feature_idx_sym]
                    all_info_last_targets_sym = window_info_sym['last_target_in_window'].values
                    if not (len(all_x_last_targets_sym) == len(all_info_last_targets_sym) and \
                            np.all(np.isclose(all_x_last_targets_sym, all_info_last_targets_sym, equal_nan=True))):
                        logging.error(f"[Day {day}, Sym {ret_sym_name}] ALIGNMENT CHECK FAILED. Skipping symbol.")
                        continue
                except ValueError: # target_col_name not in features_for_sym
                    logging.error(f"[Day {day}, Sym {ret_sym_name}] Target col for alignment not in features. Skipping symbol.")
                    continue
            elif window_info_sym.empty and (X_windows_sym is not None and X_windows_sym.size > 0):
                logging.warning(f"[Day {day}, Sym {ret_sym_name}] Window info empty but X_windows exist. Skipping symbol.")
                continue


            all_X_windows_day.append(X_windows_sym)
            all_target_windows_day.append(target_windows_sym)
            all_window_info_day.append(window_info_sym)
    
    del sym_dfs; gc.collect()
    if 'symbol_result' in locals(): del symbol_result
    if 'X_windows_sym' in locals(): del X_windows_sym
    if 'target_windows_sym' in locals(): del target_windows_sym
    if 'window_info_sym' in locals(): del window_info_sym
    gc.collect()


    day_windows_count = 0
    if not all_X_windows_day or master_feature_cols_list is None:
        logging.warning(f"[Day {day}] No windows generated or master feature list missing. Skipping save.")
    else:
        try:
            X_windows_day_final = np.concatenate(all_X_windows_day, axis=0)
            del all_X_windows_day; gc.collect()
            target_windows_day_final = np.concatenate(all_target_windows_day, axis=0)
            del all_target_windows_day; gc.collect()
            window_info_day_final = pd.concat(all_window_info_day, ignore_index=True)
            del all_window_info_day; gc.collect()

            # Sort combined data by symbol then by window_end_time before saving
            window_info_day_final = window_info_day_final.sort_values(by=['sym', 'window_end_time']).reset_index(drop=True)
            # Reorder X and target arrays according to the sorted window_info
            # This requires original indices before concat if order is critical or matching to X/target by original symbol blocks.
            # For simplicity here, we assume concatenation order is sufficient if sorting info only.
            # If precise reordering of X/target based on sorted info is needed, store original indices or handle more carefully.

            day_windows_count = len(X_windows_day_final)
            # logging.info(f"[Day {day}] Combined {day_windows_count} windows. Shapes: X={X_windows_day_final.shape}, TGT={target_windows_day_final.shape}, INFO={window_info_day_final.shape}")


            day_output_dir = os.path.join(output_dir, day)
            os.makedirs(day_output_dir, exist_ok=True)

            if day_windows_count > 0:
                x_fname = os.path.join(day_output_dir, f'X_windows_in{window_length}_tgt{target_window_length}.npy')
                tgt_fname = os.path.join(day_output_dir, f'target_windows_in{window_length}_tgt{target_window_length}.npy')
                info_fname = os.path.join(day_output_dir, f'window_info_in{window_length}_tgt{target_window_length}.parquet')

                np.save(x_fname, X_windows_day_final)
                np.save(tgt_fname, target_windows_day_final)
                window_info_day_final.to_parquet(info_fname, index=False)
                # logging.info(f"[Day {day}] Saved data to {day_output_dir}")

                feature_list_fname = os.path.join(day_output_dir, 'features.txt')
                with open(feature_list_fname, 'w') as f:
                    for feature_name in master_feature_cols_list:
                        f.write(f"{feature_name}\\n") # Corrected to \\n for newline
                del X_windows_day_final, target_windows_day_final, window_info_day_final; gc.collect()
        except Exception as e:
            logging.error(f"[Day {day}] Error combining or saving results: {e}", exc_info=True)
            # Clean up large lists if error occurs
            if 'all_X_windows_day' in locals() and all_X_windows_day: del all_X_windows_day
            if 'all_target_windows_day' in locals() and all_target_windows_day: del all_target_windows_day
            if 'all_window_info_day' in locals() and all_window_info_day: del all_window_info_day
            if 'X_windows_day_final' in locals(): del X_windows_day_final
            if 'target_windows_day_final' in locals(): del target_windows_day_final
            if 'window_info_day_final' in locals(): del window_info_day_final
            gc.collect()


    day_duration = time.time() - day_start_time
    logging.info(f"--- [Day {day}] END Processing: Total Time={day_duration:.2f}s, Windows Generated={day_windows_count} ---")
    gc.collect()
    return day_windows_count

def main():
    logging.info("=== Configuring Day-Parallel Volume Bar Pipeline ===")
    BASE_DATA_PATH = '/mnt//storage_1_10T/citibank/egbs_data_02_25'
    DAYS_TO_PROCESS = [
        '20250212', '20250213', '20250214', '20250217', '20250218',
        '20250219', '20250220', '20250221', '20250224', '20250225'
    ]
    # DAYS_TO_PROCESS = ['20250212', '20250213'] # For testing

    SYMBOLS_TO_KEEP = [
        'FBONH5', 'FBTPH5', 'FBTSH5', 'FGBLH5',
        'FGBMH5', 'FGBSH5', 'FGBXH5', 'FOATH5'
    ]
    NB_BARS_PER_DAY_SYMBOL = 10000
    WINDOW_LENGTH = 150
    TARGET_WINDOW_LENGTH = 10
    TARGET_COLUMN_NAME = 'wmp_mean' # Example target column
    BASE_OUTPUT_DIR = '/mnt/storage_1_10T/citibank/data/processed_data_volume_bars_day_parallel' # New output dir
    PARAMS_SUBDIR = f"volbars_{NB_BARS_PER_DAY_SYMBOL}_in{WINDOW_LENGTH}_tgt{TARGET_WINDOW_LENGTH}"
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, PARAMS_SUBDIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    logging.info(f"Output directory: {OUTPUT_DIR}")

    overall_start_time = time.time()
    logging.info(f"=== Starting Day-by-Day Volume Bar Processing (Parallelizing Days) using {len(DAYS_TO_PROCESS)} days ===")

    # Prepare tasks for joblib
    tasks = [
        delayed(process_day)(
            day=day,
            base_data_path=BASE_DATA_PATH,
            nb_bars=NB_BARS_PER_DAY_SYMBOL,
            window_length=WINDOW_LENGTH,
            target_window_length=TARGET_WINDOW_LENGTH,
            target_col_name=TARGET_COLUMN_NAME,
            output_dir=OUTPUT_DIR,
            symbols_to_keep=SYMBOLS_TO_KEEP
        ) for day in DAYS_TO_PROCESS
    ]

    # Execute tasks in parallel for days
    # n_jobs=-1 uses all available cores. Set to 1 for sequential debugging.
    # Consider lower n_jobs if memory per process is very high.
    # joblib's default backend is 'loky', which is process-based.
    logging.info(f"Starting parallel execution for {len(tasks)} days...")
    daily_windows_counts = Parallel(n_jobs=4, verbose=10)(tasks)
    
    total_windows_generated_all_days = sum(filter(None, daily_windows_counts)) # Summing results, filtering Nones if any day failed

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time
    logging.info("\\n=== Finished All Day-Parallel Volume Bar Processing ===")
    logging.info(f"Total windows generated across all days: {total_windows_generated_all_days:,}")
    logging.info(f"Total processing time: {total_duration:.2f} seconds ({total_duration/60:.2f} minutes)")
    logging.info(f"Daily processed files are saved in subdirectories under: {OUTPUT_DIR}")
    logging.info("--- Day-Parallel Volume Bar Pipeline Finished ---")

if __name__ == "__main__":
    main() 