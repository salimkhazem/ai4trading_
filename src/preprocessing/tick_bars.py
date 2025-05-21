import logging
import os
import time
import numpy as np
import pandas as pd
import pickle as pkl
from glob import glob
from tqdm import tqdm
from joblib import Parallel, delayed
from typing import List, Tuple, Dict
import gc

from utils.utils import *

# --- Try importing resource for memory profiling ---
try:
    import resource
    HAS_RESOURCE = True
except ImportError:
    # resource module is Unix-specific
    HAS_RESOURCE = False
    logging.warning("Module 'resource' not found. Memory profiling will be skipped.")

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

# --- Resampling Function ---
def resample_lob_data(df: pd.DataFrame, resample_freq: str = '1s') -> pd.DataFrame:
    """
    Resamples the LOB data to a uniform frequency per symbol using 'mean' aggregation.
    """
    logging.info(f"Resampling data to {resample_freq} frequency using 'mean' aggregation...")
    df_resampled_list = []
    required_cols = ['time', 'sym', 'weighted_mid_price']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame for resampling must contain columns: {required_cols}")

    df['time'] = pd.to_datetime(df['time']) # Ensure time is datetime

    numeric_cols_to_agg = df.select_dtypes(include=np.number).columns.tolist()
    agg_dict = {col: 'mean' for col in numeric_cols_to_agg}

    grouped = df.groupby('sym')
    total_syms = len(grouped)

    for sym, group in tqdm(grouped, total=total_syms, desc="Resampling symbols"):
        if group.empty:
             logging.warning(f"Skipping symbol {sym} due to empty group.")
             continue
        if group['time'].duplicated().any():
            logging.warning(f"Duplicate timestamps found for symbol {sym}. Keeping first occurrence.")
            group = group.drop_duplicates(subset=['time'], keep='first')

        group = group.set_index('time').sort_index()

        current_agg_dict = {k: v for k, v in agg_dict.items() if k in group.columns}
        if not current_agg_dict:
             logging.warning(f"No numeric columns found for symbol {sym} to aggregate.")
             continue

        try:
            group_resampled = group.resample(resample_freq).agg(current_agg_dict)
            group_resampled['sym'] = sym
            df_resampled_list.append(group_resampled)
        except Exception as e:
            logging.error(f"Error resampling symbol {sym}: {e}")

    if not df_resampled_list:
        logging.warning("No data could be resampled.")
        return pd.DataFrame()

    df_final = pd.concat(df_resampled_list).sort_index()
    df_final = df_final.dropna(subset=['weighted_mid_price'])
    df_final = df_final.reset_index()

    cols_order = ['time', 'sym'] + [col for col in df_final.columns if col not in ['time', 'sym']]
    df_final = df_final[cols_order]

    logging.info(f"Resampling complete. Resulting shape: {df_final.shape}")
    return df_final

# --- Window Generation Function ---
def generate_sequential_windows(df_resampled: pd.DataFrame,
                                window_length: int = 100,
                                target_window_length: int = 10, 
                                n_jobs: int = -1
                               ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]: 
    """
    Generates sequential, sliding windows and corresponding target WMP windows.
    """
    logging.info(f"Generating input windows (len={window_length}) and target WMP windows (len={target_window_length})...")

    required_cols = ['time', 'sym', 'weighted_mid_price']
    lob_level_cols = [f"L{i}_{t}_{f}" for i in range(1, 11) for t in ['bid', 'ask'] for f in ['price', 'size', 'no']]
    calculated_features = ['mid_price', 'spread', 'weighted_mid_price', 'obi_L1',
                           'cum_bid_vol_10', 'cum_ask_vol_10', 'depth_imbalance_10',
                           'liquidity_impact_bid', 'liquidity_impact_ask']
    
    potential_feature_cols = lob_level_cols + calculated_features
    feature_cols = [col for col in potential_feature_cols if col in df_resampled.columns]

    if not all(col in df_resampled.columns for col in required_cols): raise ValueError(f"Input DataFrame must contain base columns: {required_cols}")
    if not feature_cols: raise ValueError("No feature columns identified or found in the DataFrame.")
    num_features = len(feature_cols)
    logging.info(f"Using {num_features} input features.") #: {feature_cols}") # Commented out long list

    logging.info("Converting features to float32...")
    for col in feature_cols:
        df_resampled[col] = pd.to_numeric(df_resampled[col], errors='coerce')
        df_resampled[col] = df_resampled[col].astype(np.float32)
    df_resampled = df_resampled.dropna(subset=feature_cols)
    logging.info("Conversion and NaN drop done.")
    if df_resampled.empty:
        logging.warning("DataFrame is empty after converting features to float32 and dropping NaNs.")
        return np.array([]).reshape(0, window_length, num_features).astype(np.float32), pd.DataFrame(columns=['window_end_time', 'sym', 'last_wmp_in_window'])

    grouped = df_resampled.groupby('sym')
    all_sym_names = list(grouped.groups.keys())

    def process_symbol(sym_name: str, group_df: pd.DataFrame) -> Tuple[List[np.ndarray], List[np.ndarray], List[Dict]]: # <-- Added List[np.ndarray]
        group_df = group_df.sort_values('time')
        symbol_X_windows = []
        symbol_target_windows = [] # <-- New list for target WMPs
        symbol_info = []

        # Adjust length check for target window
        if len(group_df) < window_length + target_window_length:
            return [], [], []

        feature_data = group_df[feature_cols].values
        times = group_df['time'].values
        wmps = group_df['weighted_mid_price'].values # Need this for target

        for i in range(len(group_df) - window_length - target_window_length + 1):
            # Input window
            input_window_slice = feature_data[i : i + window_length]
            symbol_X_windows.append(input_window_slice)

            target_start_idx = i + window_length
            target_end_idx = target_start_idx + target_window_length
            target_window_slice = wmps[target_start_idx : target_end_idx] # Extract WMP slice
            symbol_target_windows.append(target_window_slice) # Add WMP sequence

            # Info
            window_end_index = i + window_length - 1
            info = {'window_end_time': times[window_end_index], 'sym': sym_name, 'last_wmp_in_window': wmps[window_end_index]}
            symbol_info.append(info)

        return symbol_X_windows, symbol_target_windows, symbol_info # <-- Return target list

    logging.info("Starting parallel processing of symbols...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_symbol)(sym_name, group.copy())
        for sym_name, group in tqdm(grouped, total=len(all_sym_names), desc="Processing symbols")
    )
    logging.info("Parallel processing finished. Stacking results...")

    all_X_windows_list = []
    all_target_windows_list = [] 
    all_window_info_list = []
    for x_wins, target_wins, info in results: 
        if x_wins: 
            all_X_windows_list.extend(x_wins)
            all_target_windows_list.extend(target_wins) 
            all_window_info_list.extend(info)

    if not all_X_windows_list:
        logging.warning("No windows were generated.")
        # Return empty arrays/DataFrame with correct dimensions/columns
        empty_x = np.array([]).reshape(0, window_length, num_features).astype(np.float32)
        empty_tgt = np.array([]).reshape(0, target_window_length).astype(np.float32)
        empty_info = pd.DataFrame(columns=['window_end_time', 'sym', 'last_wmp_in_window'])
        return empty_x, empty_tgt, empty_info

    # Stack input and target windows
    X_windows = np.stack(all_X_windows_list, axis=0).astype(np.float32)
    target_windows = np.stack(all_target_windows_list, axis=0).astype(np.float32) # <-- Stack targets
    window_info = pd.DataFrame(all_window_info_list)
    window_info = window_info.sort_values('window_end_time').reset_index(drop=True)
    logging.info("Stacking complete.")

    logging.info(f"Window generation complete.")
    logging.info(f"Shape of X_windows: {X_windows.shape}")
    logging.info(f"Shape of target_windows: {target_windows.shape}") 
    logging.info(f"Shape of window_info: {window_info.shape}")
    logging.info(f"Data type of X_windows: {X_windows.dtype}")
    logging.info(f"Data type of target_windows: {target_windows.dtype}")

    return X_windows, target_windows, window_info 


if __name__ == "__main__":

    # --- Configuration ---
    BASE_DATA_PATH = '/mnt/user_disk/kfeghoul/storage_1_10T/Citibank/egbs_data_02_25'
    # Use specific days or get all valid ones
    # valid_days = get_valid_day_folders(BASE_DATA_PATH)
    DAYS_TO_PROCESS = [
        '20250212',
        '20250213',
        '20250214',
        '20250217',
        '20250218',
        '20250219',
        '20250220',
        '20250221',
        '20250224',
        '20250225'
    ]   
    SYMBOLS_TO_EXCLUDE = ['FGBLM5', 'CONFH5'] # Low frequency symbols
    RESAMPLE_FREQ = '1s'
    WINDOW_LENGTH = 100 # Input window length (seconds)
    TARGET_WINDOW_LENGTH = 10 # Target window length (seconds)
    N_JOBS = -1
    BASE_OUTPUT_DIR = './processed_data'
    PARAMS_SUBDIR = f"resample_{RESAMPLE_FREQ}_in{WINDOW_LENGTH}s_tgt{TARGET_WINDOW_LENGTH}s" 
    OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, PARAMS_SUBDIR)
    #INTERMEDIATE_DIR = os.path.join(OUTPUT_DIR, 'intermediate_daily') # Daily files go here
    CLEANUP_INTERMEDIATE = False # Keep daily files

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    #os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

    # --- Process Data Day-by-Day loop ---
    intermediate_window_files = []
    intermediate_info_files = []
    total_rows_processed = 0
    overall_start_time = time.time()
    overall_peak_mem_mb = 0.0

    logging.info("=== Starting Day-by-Day Processing ===")

    for day in DAYS_TO_PROCESS:
        logging.info(f"--- Processing Day: {day} ---")
        day_start_time = time.time()
        day_peak_mem_start = get_memory_usage_gb()
        day_peak_mem_current = day_peak_mem_start

        # 1. Load Data for the day
        step_start_time = time.time()
        df_day_raw = load_and_clean_day_data(BASE_DATA_PATH, day)
        load_time = time.time() - step_start_time
        mem_after_load = get_memory_usage_gb()
        day_peak_mem_current = max(day_peak_mem_current, mem_after_load)
        logging.info(f"Step 1 (Load) complete: Time={load_time:.2f}s, Peak Mem={mem_after_load:.2f} GB")
        if df_day_raw.empty:
            logging.warning(f"Skipping day {day} due to loading error or no data.")
            continue

        # 2. Clean Data (Drop NaNs)
        step_start_time = time.time()
        initial_rows = len(df_day_raw)
        df_day_raw.dropna(inplace=True) # Dropna in-place
        df_clean = df_day_raw.copy() # Now copy the cleaned data
        rows_dropped = initial_rows - len(df_clean)
        percent_dropped = (rows_dropped / initial_rows * 100) if initial_rows > 0 else 0.0
        clean_time = time.time() - step_start_time
        mem_after_clean = get_memory_usage_gb()
        day_peak_mem_current = max(day_peak_mem_current, mem_after_clean)
        logging.info(f"Step 2 (Clean NaNs) complete: Time={clean_time:.2f}s, Peak Mem={mem_after_clean:.2f} GB")
        logging.info(f"Dropped {rows_dropped:,} rows with NaNs ({percent_dropped:.2f}%).")
        logging.info(f"Shape after NaN drop: {df_clean.shape}")

        if df_clean.empty:
            logging.warning(f"Skipping day {day} as it became empty after dropping NaNs.")
            continue

        # 2b. Drop Unused Columns
        step_start_time = time.time()
        columns_to_drop = [
            'exch_time', 'exchange', 'first_sequence_number',
            'last_sequence_number', 'first_sym_sequence', 'last_sym_sequence',
            'first_time', 'first_exch_time', 'event_id', 'date'
        ]
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
        if existing_cols_to_drop:
            df_clean.drop(columns=existing_cols_to_drop, inplace=True)
            logging.info(f"Dropped: {existing_cols_to_drop}")
            logging.info(f"Shape after dropping unused columns: {df_clean.shape}")

        # 3. Compute Core Features (In-Place)
        step_start_time = time.time()
        df_featured = compute_core_features(df_clean) # Modifies df_clean
        feature_time = time.time() - step_start_time
        mem_after_features = get_memory_usage_gb()
        day_peak_mem_current = max(day_peak_mem_current, mem_after_features)
        logging.info(f"Step 3 (Features) complete: Time={feature_time:.2f}s, Peak Mem={mem_after_features:.2f} GB")
        logging.info("Computed high frequency trading features.")

        # 4. Filter Symbols
        step_start_time = time.time()
        logging.info(f"\nFiltering out symbols: {SYMBOLS_TO_EXCLUDE}")
        df_filtered = df_featured[~df_featured['sym'].isin(SYMBOLS_TO_EXCLUDE)].copy()
        filter_time = time.time() - step_start_time
        mem_after_filter = get_memory_usage_gb()
        day_peak_mem_current = max(day_peak_mem_current, mem_after_filter)
        logging.info(f"Step 4 (Filter Sym) complete: Time={filter_time:.2f}s, Peak Mem={mem_after_filter:.2f} GB")
        logging.info(f"Shape after symbol filter: {df_filtered.shape}")

        if df_filtered.empty:
            logging.warning(f"Skipping day {day} as it became empty after filtering symbols.")
            continue

        # 5. Resample Data
        step_start_time = time.time()
        df_resampled = resample_lob_data(df_filtered, resample_freq=RESAMPLE_FREQ)
        resample_time = time.time() - step_start_time
        mem_after_resample = get_memory_usage_gb()
        day_peak_mem_current = max(day_peak_mem_current, mem_after_resample)
        logging.info(f"Step 5 (Resample) complete: Time={resample_time:.2f}s, Peak Mem={mem_after_resample:.2f} GB")
        if df_resampled.empty:
            logging.warning(f"Skipping day {day} as resampling resulted in an empty DataFrame.")
            continue

        # 6. Generate Sequential Windows for the day
        step_start_time = time.time()
        logging.info("Generating sequential windows...")
        X_windows_day, target_windows_day, window_info_day = generate_sequential_windows(
            df_resampled,
            window_length=WINDOW_LENGTH,
            target_window_length=TARGET_WINDOW_LENGTH,
            n_jobs=N_JOBS
        )
        del df_resampled; gc.collect() # Free resampled df memory
        window_time = time.time() - step_start_time
        mem_after_window = get_memory_usage_gb()
        day_peak_mem_current = max(day_peak_mem_current, mem_after_window)
        logging.info(f"Step 6 (Windows) complete: Time={window_time:.2f}s, Peak Mem={mem_after_window:.2f} GB")

        # 7. Save Intermediate Results for the day
        step_start_time = time.time()
        # Check X_windows_day as before, assume target_windows exists if X exists
        if X_windows_day.size > 0:
            logging.info("Saving intermediate data for the day...")
            try:
                # Create day-specific subdirectory
                day_output_dir = os.path.join(OUTPUT_DIR, day)
                os.makedirs(day_output_dir, exist_ok=True)

                # Save input windows
                windows_filename = os.path.join(day_output_dir, f'X_windows_in{WINDOW_LENGTH}_tgt{TARGET_WINDOW_LENGTH}.npy')
                np.save(windows_filename, X_windows_day)
                logging.info(f"Saved daily input windows to {windows_filename}")

                # Save target windows <-- New save step
                target_filename = os.path.join(day_output_dir, f'target_windows_in{WINDOW_LENGTH}_tgt{TARGET_WINDOW_LENGTH}.npy')
                np.save(target_filename, target_windows_day)
                logging.info(f"Saved daily target windows to {target_filename}")

                # Save window info
                info_filename = os.path.join(day_output_dir, f'window_info_in{WINDOW_LENGTH}_tgt{TARGET_WINDOW_LENGTH}.parquet')
                window_info_day.to_parquet(info_filename, index=False)
                logging.info(f"Saved daily window info to {info_filename}")

                total_rows_processed += len(X_windows_day) # Use X length for count
                save_time = time.time() - step_start_time
                mem_after_save = get_memory_usage_gb()
                day_peak_mem_current = max(day_peak_mem_current, mem_after_save)
                logging.info(f"Step 7 (Save) complete: Time={save_time:.2f}s, Peak Mem={mem_after_save:.2f} GB")

            except Exception as e:
                logging.error(f"Error saving intermediate data for day {day}: {e}")
                save_time = time.time() - step_start_time
                logging.info(f"Step 7 (Save) FAILED: Time={save_time:.2f}s")
        else:
            logging.warning("No windows generated for this day, skipping saving.")
            logging.info(f"Step 7 (Save) skipped.")

        day_end_time = time.time()
        day_duration = day_end_time - day_start_time
        day_peak_mem_overall = day_peak_mem_current
        overall_peak_mem_mb = max(overall_peak_mem_mb, day_peak_mem_overall)

        logging.info(f"--- Finished processing day {day}: Total Time={day_duration:.2f}s, Peak Mem Usage={day_peak_mem_overall:.2f} GB ---")

        # Explicitly delete objects for the day
        # Explicitly delete target windows array too
        del X_windows_day, target_windows_day, window_info_day
        gc.collect()

    overall_end_time = time.time()
    total_duration = overall_end_time - overall_start_time

    logging.info("\n=== Finished Day-by-Day Processing ===")
    logging.info(f"Total windows generated: {total_rows_processed:,}")
    logging.info(f"Total processing time: {total_duration:.2f} seconds")
    logging.info(f"Overall peak memory usage: {overall_peak_mem_mb:.2f} GB")
    logging.info(f"Daily processed files are saved in subdirectories under: {OUTPUT_DIR}")
    logging.info("--- Pipeline Finished ---")
    

    # --- 8. Combine Intermediate Results --- (REMOVED)
    # if not intermediate_window_files or not intermediate_info_files:
    #     print("Error: No intermediate files were generated. Cannot combine results. Exiting.")
    #     exit()
    #
    # print(f"\nCombining results from {len(intermediate_window_files)} days...")
    # try:
    #     # Load and concatenate window arrays
    #     print("Loading and concatenating window arrays...")
    #     all_X_windows = [np.load(f) for f in tqdm(intermediate_window_files, desc="Loading npy files")]
    #     X_windows_final = np.concatenate(all_X_windows, axis=0)
    #     print(f"Final X_windows shape: {X_windows_final.shape}")
    #     del all_X_windows # Free memory
    #     gc.collect()
    #
    #     # Load and concatenate info dataframes
    #     print("Loading and concatenating info dataframes...")
    #     all_window_info = [pd.read_parquet(f) for f in tqdm(intermediate_info_files, desc="Loading parquet files")]
    #     window_info_final = pd.concat(all_window_info, ignore_index=True)
    #     # Ensure time is datetime and sort
    #     window_info_final['window_end_time'] = pd.to_datetime(window_info_final['window_end_time'])
    #     window_info_final = window_info_final.sort_values('window_end_time').reset_index(drop=True)
    #     print(f"Final window_info shape: {window_info_final.shape}")
    #     del all_window_info # Free memory
    #     gc.collect()
    #
    # except Exception as e:
    #     print(f"Error combining intermediate files: {e}")
    #     exit()
    #
    # # --- 9. Save Final Combined Results --- (REMOVED)
    # # if X_windows_final.size > 0:
    # #     print("\nSaving final combined data...")
    # #     try:
    # #         final_windows_filename = os.path.join(OUTPUT_DIR, f'X_windows_final_{WINDOW_LENGTH}s.npy')
    # #         np.save(final_windows_filename, X_windows_final)
    # #         print(f"Saved final X_windows to {final_windows_filename}")
    # #
    # #         final_info_filename = os.path.join(OUTPUT_DIR, f'window_info_final_{WINDOW_LENGTH}s.parquet')
    # #         window_info_final.to_parquet(final_info_filename, index=False)
    # #         print(f"Saved final window_info to {final_info_filename}")
    # #
    # #         print("Final saving complete.")
    # #     except Exception as e:
    # #         print(f"Error saving final combined data: {e}")
    # # else:
    # #     print("\nCombined data is empty, skipping final saving.")
    #
    # # --- 10. Cleanup Intermediate Files (Optional) --- (REMOVED)
    # # if CLEANUP_INTERMEDIATE:
    # #     print("\nCleaning up intermediate daily files...")
    # #     cleaned_count = 0
    # #     try:
    # #         for f in intermediate_window_files + intermediate_info_files:
    # #             if os.path.exists(f):
    # #                 os.remove(f)
    # #                 cleaned_count += 1
    # #         # Remove the intermediate directory if empty
    # #         if not os.listdir(INTERMEDIATE_DIR):
    # #             os.rmdir(INTERMEDIATE_DIR)
    # #         print(f"Removed {cleaned_count} intermediate files and directory.")
    # #     except Exception as e:
    # #         print(f"Error during cleanup: {e}")

    print(f"\n--- Pipeline Finished ---")
    print(f"Daily processed files are saved in: {OUTPUT_DIR}") 