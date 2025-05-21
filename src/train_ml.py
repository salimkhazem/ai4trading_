import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging
from scipy.stats import skew, kurtosis # Optional, for more stats

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def load_and_combine_daily_data(processed_data_dir: str, days: list, window_length: int, target_window_length: int) -> tuple:
    '''Loads and combines data for specified days.'''
    
    all_X_windows_list = []
    all_target_windows_list = []
    all_window_info_list = []
    feature_names_list = None # To store feature names from the first day

    logger.info(f"Starting data loading for days: {days}")
    for day in tqdm(days, total=len(days), desc="Loading daily data"):
        logger.debug(f"Processing data for day: {day}")
        day_data_path = os.path.join(processed_data_dir, day)
        if not os.path.exists(day_data_path):
            logger.warning(f"Data path not found for day: {day}. Skipping.")
            continue
        
        x_windows_path = os.path.join(day_data_path, f'X_windows_in{window_length}_tgt{target_window_length}.npy')
        target_windows_path = os.path.join(day_data_path, f'target_windows_in{window_length}_tgt{target_window_length}.npy')
        window_info_path = os.path.join(day_data_path, f'window_info_in{window_length}_tgt{target_window_length}.parquet')
        features_path = os.path.join(day_data_path, 'features.txt')

        try:
            X_day = np.load(x_windows_path)
            target_day = np.load(target_windows_path)
            info_day = pd.read_parquet(window_info_path)
            
            all_X_windows_list.append(X_day)
            all_target_windows_list.append(target_day)
            all_window_info_list.append(info_day)

            if feature_names_list is None and os.path.exists(features_path):
                with open(features_path, 'r') as f:
                    feature_names_list = [line.strip() for line in f if line.strip()]
                logger.info(f"Loaded feature names ({len(feature_names_list)}) from day: {day}")

            logger.debug(f"Day {day}: X_shape={X_day.shape}, TGT_shape={target_day.shape}, INFO_shape={info_day.shape}")
        except FileNotFoundError as e:
            logger.warning(f"File not found for day {day} ({e}). Skipping.")
        except Exception as e:
            logger.error(f"Error loading data for day {day}: {e}", exc_info=True)
            continue
            
    if not all_X_windows_list:
        logger.error("No data was loaded. Exiting.")
        return None, None, None, None

    logger.info("Concatenating all loaded daily data...")
    X_combined = np.concatenate(all_X_windows_list, axis=0)
    target_combined = np.concatenate(all_target_windows_list, axis=0)
    window_info_combined = pd.concat(all_window_info_list, ignore_index=True)
    
    logger.info(f"Concatenated shapes: X={X_combined.shape}, Target={target_combined.shape}, Info={window_info_combined.shape}")
    
    # Robust sorting (as in volume_bar_pipeline.py)
    # This ensures alignment if data wasn't perfectly sorted per day or if concat changed order.
    # logger.info("Applying robust sorting to combined data (by sym, then window_end_time)...")
    # if not window_info_combined.empty:
    #     # Ensure 'window_end_time' is datetime for proper sorting
    #     window_info_combined['window_end_time'] = pd.to_datetime(window_info_combined['window_end_time'])
    #     
    #     sorted_indices = np.lexsort((window_info_combined['window_end_time'].astype(np.int64), window_info_combined['sym'].astype(str)))
    #     
    #     X_combined = X_combined[sorted_indices]
    #     target_combined = target_combined[sorted_indices]
    #     window_info_combined = window_info_combined.iloc[sorted_indices].reset_index(drop=True)
    #     logger.info("Robust sorting applied.")
    # else:
    #     logger.warning("Window_info_combined is empty, skipping robust sorting.")

    return X_combined, target_combined, window_info_combined, feature_names_list

def create_tabular_features(X_windows: np.ndarray, feature_names: list = None, stats_to_compute=None) -> pd.DataFrame:
    """Creates tabular features from windowed data by calculating statistics over the time dimension."""
    if stats_to_compute is None:
        stats_to_compute = ['mean', 'std', 'median', 'min', 'max']
        
    num_samples, window_len, num_original_features = X_windows.shape
    
    if feature_names is None:
        feature_names = [f"orig_feat_{i}" for i in range(num_original_features)]
    elif len(feature_names) != num_original_features:
        logger.warning(f"Length of provided feature_names ({len(feature_names)}) does not match "
                       f"number of features in X_windows ({num_original_features}). Using generic names.")
        feature_names = [f"orig_feat_{i}" for i in range(num_original_features)]

    logger.info(f"Starting tabular feature engineering for {num_original_features} original features with stats: {stats_to_compute}...")
    
    all_new_features_data = []
    all_new_feature_names = []

    for i in tqdm(range(num_original_features), desc="  Extracting stats per original feature"):
        feature_over_window = X_windows[:, :, i]  # Shape: (num_samples, window_len)
        original_name = feature_names[i]

        if 'mean' in stats_to_compute:
            all_new_features_data.append(np.mean(feature_over_window, axis=1))
            all_new_feature_names.append(f"{original_name}_mean_win{window_len}")
        if 'std' in stats_to_compute:
            all_new_features_data.append(np.std(feature_over_window, axis=1))
            all_new_feature_names.append(f"{original_name}_std_win{window_len}")
        if 'median' in stats_to_compute:
            all_new_features_data.append(np.median(feature_over_window, axis=1))
            all_new_feature_names.append(f"{original_name}_median_win{window_len}")
        if 'min' in stats_to_compute:
            all_new_features_data.append(np.min(feature_over_window, axis=1))
            all_new_feature_names.append(f"{original_name}_min_win{window_len}")
        if 'max' in stats_to_compute:
            all_new_features_data.append(np.max(feature_over_window, axis=1))
            all_new_feature_names.append(f"{original_name}_max_win{window_len}")
        # Add more stats here if needed (e.g., skew, kurtosis, last value)

    X_tabular = np.column_stack(all_new_features_data)
    tabular_df = pd.DataFrame(X_tabular, columns=all_new_feature_names)
    logger.info(f"Tabular features created. Shape: {tabular_df.shape}")
    return tabular_df

def generate_tercile_labels_from_relative_change(relative_change_values: np.ndarray) -> np.ndarray:
    '''Generates tercile labels (0, 1, 2) directly from relative change values.'''
    # Handle NaNs and Infs in relative_change_values before percentile calculation
    # by creating a temporary array with NaNs/Infs removed for percentile calculation.
    # Labels for these will effectively be NaN if not handled, or can be assigned a default.
    
    finite_relative_change = relative_change_values[np.isfinite(relative_change_values)]
    if len(finite_relative_change) < 3: # Need at least 3 points to define 2 terciles
        logger.warning(f"Not enough finite relative change values ({len(finite_relative_change)}) to compute terciles. Returning all labels as 1 (Stationary).")
        return np.full(len(relative_change_values), 1, dtype=np.int64)

    lower_threshold = np.percentile(finite_relative_change, 100/3)
    upper_threshold = np.percentile(finite_relative_change, 200/3)
    
    logger.info(f"Calculated Tercile Thresholds (direct): Lower={lower_threshold:.6f}, Upper={upper_threshold:.6f}")

    # Initialize labels to 1 (Stationary). NaNs in relative_change_values will remain 1 or need specific handling.
    labels = np.full(len(relative_change_values), 1, dtype=np.int64)
    
    # Apply thresholds. This comparison will be False for NaNs in relative_change_values.
    labels[relative_change_values <= lower_threshold] = 0  # Down
    labels[relative_change_values > upper_threshold] = 2   # Up
    
    # For samples that were NaN/Inf in relative_change_values, their labels will be 1.
    # You might want to explicitly mark them, e.g., with -1, if they should be excluded later.
    # num_invalid_rc = np.sum(~np.isfinite(relative_change_values))
    # if num_invalid_rc > 0:
    #     logger.info(f"{num_invalid_rc} samples had non-finite relative_change; their labels are 1 (Stationary) by default or need explicit handling.")

    label_counts = pd.Series(labels).value_counts(normalize=True).sort_index()
    logger.info(f"Generated Tercile Labels distribution:\n{label_counts}")
    return labels

def process_days_for_ml(
    days_to_process: list, 
    processed_data_dir: str, 
    window_length: int, 
    target_window_length: int,
    nb_bars_per_day_symbol: int # Added for filename consistency
    ) -> tuple[pd.DataFrame | None, np.ndarray | None, pd.DataFrame | None]:
    """
    Processes a list of days to generate tabular features and labels for ML.

    Args:
        days_to_process (list): List of day strings to process.
        processed_data_dir (str): Directory containing the processed daily data.
        window_length (int): Length of the input window.
        target_window_length (int): Length of the target window.
        nb_bars_per_day_symbol (int): Number of bars per day per symbol (used for constructing filenames).


    Returns:
        tuple[pd.DataFrame | None, np.ndarray | None, pd.DataFrame | None]: 
            - X_tabular_df_final: DataFrame of tabular features.
            - Y_labels_final: NumPy array of labels.
            - window_info_final: DataFrame containing window information (optional, can be None).
            Returns (None, None, None) if processing fails.
    """
    logger.info(f"--- Starting ML Data Processing for Days: {days_to_process} ---")

    # --- 1. Load and Combine Data ---
    X_windows, target_windows, window_info, feature_names = load_and_combine_daily_data(
        processed_data_dir, days_to_process, window_length, target_window_length
    )

    if X_windows is None:
        logger.error(f"Data loading failed for days: {days_to_process}. Cannot proceed.")
        return None, None, None

    # --- 2. Create Tabular Features ---
    if feature_names is None:
        logger.warning("Feature names not loaded from features.txt. Tabular features will have generic names.")
        if days_to_process: # Check if days_to_process is not empty
            common_features_path = os.path.join(processed_data_dir, days_to_process[0], 'features.txt')
            if os.path.exists(common_features_path):
                try:
                    with open(common_features_path, 'r') as f:
                        feature_names = [line.strip() for line in f if line.strip()]
                    logger.info(f"Loaded feature names ({len(feature_names)}) from common path: {common_features_path}")
                except Exception as e:
                    logger.error(f"Could not load feature names from {common_features_path}: {e}")
            else:
                logger.warning(f"Common features.txt not found at {common_features_path}")
        else:
            logger.warning("No days specified, cannot infer common features path.")


    X_tabular_df = create_tabular_features(X_windows, feature_names, stats_to_compute=['mean', 'std', 'median', 'min', 'max']) # Pass window_length

    # --- 3. Calculate Relative Mean Target ---
    logger.info("Calculating relative mean target...")
    last_val_input = window_info['last_target_in_window'].values.astype(np.float64)
    mean_target_vals = np.mean(target_windows, axis=1).astype(np.float64)
    
    relative_mean_target = np.full_like(last_val_input, np.nan, dtype=np.float64)
    valid_denom_mask = np.isfinite(last_val_input) & (last_val_input != 0)
    
    relative_mean_target[valid_denom_mask] = \
        (mean_target_vals[valid_denom_mask] - last_val_input[valid_denom_mask]) / last_val_input[valid_denom_mask]
    relative_mean_target[np.isinf(relative_mean_target)] = np.nan
    
    num_invalid_rmt = np.sum(~np.isfinite(relative_mean_target))
    if num_invalid_rmt > 0:
        logger.warning(f"{num_invalid_rmt} samples have non-finite (NaN/Inf) relative_mean_target.")

    # --- 4. Generate Tercile Labels ---
    Y_labels = generate_tercile_labels_from_relative_change(relative_mean_target)

    # --- 5. Filter based on finite relative_mean_target ---
    # finite_rmt_indices = np.isfinite(relative_mean_target)
    # window_info_final = None # Initialize
    #
    # if np.sum(~finite_rmt_indices) > 0:
    #     logger.info(f"Filtering out {np.sum(~finite_rmt_indices)} samples with non-finite relative_mean_target.")
    #     X_tabular_df_final = X_tabular_df[finite_rmt_indices].reset_index(drop=True)
    #     Y_labels_final = Y_labels[finite_rmt_indices]
    #     if window_info is not None: # Ensure window_info was loaded
    #         window_info_final = window_info[finite_rmt_indices].reset_index(drop=True)
    # else:
    #     X_tabular_df_final = X_tabular_df
    #     Y_labels_final = Y_labels
    #     window_info_final = window_info # Assign original if no filtering

    X_tabular_df_final = X_tabular_df
    Y_labels_final = Y_labels
    window_info_final = window_info

    logger.info(f"Final shapes for days {days_to_process}: X_tabular={X_tabular_df_final.shape}, Y_labels={Y_labels_final.shape}")
    if window_info_final is not None:
        logger.info(f"Final window_info shape: {window_info_final.shape}")
    
    logger.info(f"--- Finished ML Data Processing for Days: {days_to_process} ---")
    return X_tabular_df_final, Y_labels_final, window_info_final

def main():
    logger.info("--- Starting Tabular Data Generation for ML ---")

    # --- Configuration (mirrors ML.ipynb for consistency) ---
    NB_BARS_PER_DAY_SYMBOL = 10000 # This is part of the directory name, not directly used in nb_bars calc here
    WINDOW_LENGTH = 150
    TARGET_WINDOW_LENGTH = 30
    BASE_OUTPUT_DIR = '/mnt/user_disk/kfeghoul/storage_1_10T/Citibank/processed_data_volume_bars'
    PARAMS_SUBDIR = f"volbars_{NB_BARS_PER_DAY_SYMBOL}_in{WINDOW_LENGTH}_tgt{TARGET_WINDOW_LENGTH}"
    PROCESSED_DATA_DIR = os.path.join(BASE_OUTPUT_DIR, PARAMS_SUBDIR)

    # For demonstration, using TRAIN_DAYS. Adapt as needed (e.g., combine TRAIN_DAYS and TEST_DAYS or use specific days)
    # TRAIN_DAYS = ['20250212', '20250213', '20250214', '20250217', '20250218', '20250219', '20250220']
    # TEST_DAYS = ['20250221', '20250224', '20250225']

    TRAIN_DAYS = ['20250212', '20250213']
    TEST_DAYS = ['20250214']

    # --- Process Training Data ---
    logger.info("=== PROCESSING TRAINING DATA ===")
    X_train_tabular, Y_train_labels, _ = process_days_for_ml(
        days_to_process=TRAIN_DAYS,
        processed_data_dir=PROCESSED_DATA_DIR,
        window_length=WINDOW_LENGTH,
        target_window_length=TARGET_WINDOW_LENGTH,
        nb_bars_per_day_symbol=NB_BARS_PER_DAY_SYMBOL
    )

    if X_train_tabular is not None and Y_train_labels is not None:
        train_output_base = f"train_tabular_data_vol_{NB_BARS_PER_DAY_SYMBOL}_in{WINDOW_LENGTH}_tgt{TARGET_WINDOW_LENGTH}"
        X_train_tabular.to_parquet(f"{train_output_base}_features.parquet", index=False)
        logger.info(f"Saved TRAINING tabular features to {train_output_base}_features.parquet")
        np.save(f"{train_output_base}_labels.npy", Y_train_labels)
        logger.info(f"Saved TRAINING labels to {train_output_base}_labels.npy")
    else:
        logger.error("Training data processing failed. Skipping saving training data.")

    # --- Process Test Data ---
    logger.info("\n=== PROCESSING TEST DATA ===")
    X_test_tabular, Y_test_labels, _ = process_days_for_ml(
        days_to_process=TEST_DAYS,
        processed_data_dir=PROCESSED_DATA_DIR,
        window_length=WINDOW_LENGTH,
        target_window_length=TARGET_WINDOW_LENGTH,
        nb_bars_per_day_symbol=NB_BARS_PER_DAY_SYMBOL
    )

    if X_test_tabular is not None and Y_test_labels is not None:
        test_output_base = f"test_tabular_data_vol_{NB_BARS_PER_DAY_SYMBOL}_in{WINDOW_LENGTH}_tgt{TARGET_WINDOW_LENGTH}"
        X_test_tabular.to_parquet(f"{test_output_base}_features.parquet", index=False)
        logger.info(f"Saved TEST tabular features to {test_output_base}_features.parquet")
        np.save(f"{test_output_base}_labels.npy", Y_test_labels)
        logger.info(f"Saved TEST labels to {test_output_base}_labels.npy")
    else:
        logger.error("Test data processing failed. Skipping saving test data.")

    logger.info("--- Tabular Data Generation for ML Finished ---")

if __name__ == "__main__":
    # Example of how you might adjust for specific days or load feature names
    # This main function provides a basic structure.
    
    # TODO: Add argparse for command-line configuration if needed
    # For example, to specify days, window_length, target_window_length, output paths etc.
    
    main() 