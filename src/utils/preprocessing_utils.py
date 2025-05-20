import logging
import numpy as np
import gc
import pickle
import sys
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import List, Optional

# Configure logging for this module
logger = logging.getLogger(__name__)

def fit_save_scaler_incrementally(
    train_days: List[str],
    base_data_dir: Path,
    input_window_length: int,
    target_window_length: int,
    output_dir: Path,
    bar_type: str = 'time',
    resample_freq: str = '1s',
    nb_bars: int = 10000
) -> Optional[StandardScaler]:
    """
    Fits a StandardScaler incrementally using data from specified training days
    and saves the fitted scaler to a file.

    Args:
        train_days (List[str]): List of training day strings (e.g., ['20250212']).
        base_data_dir (Path): Path object for the base directory containing day subfolders.
        input_window_length (int): Length of the input window used in filenames.
        target_window_length (int): Length of the target window used in filenames.
        output_dir (Path): Directory where the fitted scaler ('fitted_scaler.pkl') will be saved.
        bar_type (str): Type of bar data being processed ('time' or 'volume').
        resample_freq (str): Resampling frequency for volume bars.
        nb_bars (int): Number of bars for volume bars.

    Returns:
        Optional[StandardScaler]: The fitted StandardScaler object, or None if fitting fails.
    """
    scaler = None # Initialize scaler
    try:
        logger.info("Initializing StandardScaler...")
        scaler = StandardScaler()
        processed_days = 0
        logger.info(f"Incrementally fitting scaler using training days: {train_days}")

        # Loop through training days for partial fitting
        for day in tqdm(train_days, desc="Fitting Scaler Incrementally"):
            day_data_dir = Path(base_data_dir) / day # Use Path
            
            if bar_type == 'time':
                x_fname_stem = f'X_windows_in{input_window_length}_tgt{target_window_length}.npy'
            elif bar_type == 'volume':
                x_fname_stem = f'X_windows_in{input_window_length}_tgt{target_window_length}.npy'
            else:
                logger.error(f"Unsupported bar_type '{bar_type}' in fit_save_scaler_incrementally. Skipping day {day}.")
                continue
            
            x_fname = day_data_dir / x_fname_stem

            if x_fname.exists():
                try:
                    logging.debug(f"Loading X for day {day}...")
                    X_day = np.load(str(x_fname)) 

                    if X_day is None or X_day.size == 0:
                        logging.warning(f"X_windows for day {day} is empty, skipping partial_fit.")
                        continue

                    # Reshape for scaler: (n_samples, window_len, n_features) -> (n_samples * window_len, n_features)
                    n_samples, window_len, n_features = X_day.shape
                    X_day_reshaped = X_day.reshape(-1, n_features)

                    # Handle potential all-NaN columns within this chunk
                    nan_cols = np.all(np.isnan(X_day_reshaped), axis=0)
                    if np.any(nan_cols):
                        logging.warning(f"Found {np.sum(nan_cols)} feature column(s) with all NaN values in day {day}. Scaling will ignore these.")
                        # X_day_reshaped = np.nan_to_num(X_day_reshaped) # Optional: handle NaNs if needed

                    # Partially fit the scaler
                    logging.debug(f"Partial fitting scaler with {X_day_reshaped.shape[0]} steps from day {day}...")
                    scaler.partial_fit(X_day_reshaped)
                    processed_days += 1

                    # Clear memory for this day's data
                    del X_day, X_day_reshaped
                    gc.collect()

                except Exception as e:
                    logging.warning(f"Could not load or process day {day} for scaler fitting: {e}")
            else:
                logging.warning(f"X_windows file not found for day {day} ({x_fname}). Skipping for scaler fitting.")

        if processed_days == 0:
            logger.error("No training data days were successfully processed for scaler fitting.")
            return None # Return None on failure

        logger.info(f"Scaler fitted incrementally using {processed_days} days.")
        logger.info(f"Scaler Mean (first 5 features): {scaler.mean_[:5] if scaler.mean_ is not None else 'N/A'}")
        logger.info(f"Scaler Scale (std dev) (first 5 features): {scaler.scale_[:5] if scaler.scale_ is not None else 'N/A'}")

        # Ensure output directory exists (although it should have been created by results_handler)
        output_dir.mkdir(parents=True, exist_ok=True)
        scaler_save_path = output_dir / 'fitted_scaler.pkl'
        with open(scaler_save_path, 'wb') as f:
            pickle.dump(scaler, f)
        logger.info(f"Scaler saved to {scaler_save_path}")

        return scaler 

    except Exception as e:
        logger.error(f"Error during incremental scaler fitting: {e}", exc_info=True)
        return None 

def validate_feature_array(feature_array: np.ndarray, array_name: str = "X_windows") -> bool:
    """
    Checks a NumPy array for the presence of NaN or Infinity values.
    Logs errors and raises ValueError if invalid values are found.

    Args:
        feature_array (np.ndarray): The array to validate.
        array_name (str): Name of the array for logging purposes (e.g., "Training X_windows").

    Returns:
        bool: True if the array is valid (no NaN/Inf found).

    Raises:
        ValueError: If NaN or Inf values are detected in the array.
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
    else:
        logger.info(f"No NaN/Inf found in array '{array_name}'.")
        return True 