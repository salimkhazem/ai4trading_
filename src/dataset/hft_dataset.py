import gc
import logging
import numpy as np
import os
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from tqdm import tqdm
from typing import List, Tuple, Optional


class HFTDataset(Dataset):
    """
    PyTorch Dataset for loading HFT processed windows, optionally applying scaling,
    and generating labels using different strategies.

    Args:
        data_dir (str): Base directory containing the parameter-specific folder
                        (e.g., './processed_data/resample_1s_in100s_tgt10s').
        days (List[str]): List of day strings (e.g., ['20250212', '20250213'])
                          to load data for.
        input_window_length (int): Length of the input window (must match saved data).
        target_window_length (int): Length of the target window (must match saved data).
        scaler (Optional[StandardScaler]): A pre-fitted StandardScaler instance.
                                           If provided, scaling is applied to X_windows.
                                           If None, no scaling is applied.
        labeling_strategy (str): Strategy for label generation.
                                   Choices: 'median', 'directional', 'tercile'. Defaults to 'median'.
        bar_type (str): Type of bar used for resampling.
        resample_freq (str): Frequency of resampling.
        nb_bars (int): Number of bars to load.
    """
    def __init__(self,
                 data_dir: str,
                 days: List[str],
                 input_window_length: int,
                 target_window_length: int,
                 scaler: Optional[StandardScaler] = None,
                 labeling_strategy: str = "median",
                 bar_type: str = 'time',
                 resample_freq: str = '1s',
                 nb_bars: int = 10000
                 ):
        super().__init__()
        self.base_data_dir = data_dir 
        self.days = days
        self.input_window_length = input_window_length
        self.target_window_length = target_window_length
        self.scaler = scaler # Store the scaler
        self.labeling_strategy = labeling_strategy # Store strategy
        if self.labeling_strategy not in ["median", "directional", "tercile"]:
            raise ValueError(f"Unsupported labeling_strategy: {self.labeling_strategy}. Choose 'median', 'directional', or 'tercile'.")
        # Store bar parameters
        self.bar_type = bar_type
        self.resample_freq = resample_freq
        self.nb_bars = nb_bars
        self.intermediate_dir = self.base_data_dir

        self.X_windows = None
        self.target_windows = None
        self.window_info = None
        self.labels = None 
        self.relative_change = None 
        self.lower_tercile_threshold = None 
        self.upper_tercile_threshold = None 
        
        # Load data first
        self._load_and_combine_data()

        # Generate labels (which calculates relative_change from ORIGINAL X_windows)
        self._generate_labels() # This will call _calculate_relative_change internally

        # Apply scaling (to X_windows that the model will see)
        #    Scaling is applied AFTER relative_change and labels are determined from original data
        if self.scaler:
            self._apply_scaling()

        # Filter out samples where relative_change ended up as NaN
        #    This is done after all transformations to X_windows for practical array manipulation.
        #    The decision to filter is based on self.relative_change, which was computed from unscaled data.
        if self.relative_change is not None:
            nan_rc_indices = np.isnan(self.relative_change)
            num_nan_rc = np.sum(nan_rc_indices)

            if num_nan_rc > 0:
                logging.info(f"Excluding {num_nan_rc} samples due to NaN relative_change (potentially from zero denominators in original data or other issues).")
                
                valid_indices = ~nan_rc_indices
                
                # Ensure arrays/DataFrames are not None before attempting to slice
                if self.X_windows is not None:
                    self.X_windows = self.X_windows[valid_indices]
                
                if self.target_windows is not None:
                    self.target_windows = self.target_windows[valid_indices]
                
                if isinstance(self.window_info, pd.DataFrame):
                    self.window_info = self.window_info.iloc[valid_indices].reset_index(drop=True)
                elif self.window_info is not None: 
                     logging.warning("window_info is not None but not a DataFrame, cannot filter by iloc. Check data loading.")

                if self.labels is not None:
                    self.labels = self.labels[valid_indices]
                
                # Filter relative_change itself, so its stats (if re-calculated/logged) are based on the final dataset
                self.relative_change = self.relative_change[valid_indices]
                
                logging.info(f"Dataset size after excluding NaN relative_change samples: {len(self.labels) if self.labels is not None else 'N/A'}")
                if self.labels is not None and len(self.labels) == 0:
                    logging.error("All samples were excluded due to NaN relative_change. Check data pipeline and definition of relative_change.")
                    # Consider raising an error if no valid data remains
                    # raise ValueError("No valid samples remaining in dataset after NaN relative_change exclusion.")

    def _load_and_combine_data(self):
        """Loads and combines data from specified daily directories."""
        all_X_windows_list = []
        all_target_windows_list = []
        all_window_info_list = []
        days_loaded_count = 0

        logging.info(f"Attempting to load data for days: {self.days} from base directory structure {self.base_data_dir}")

        for day in tqdm(self.days, desc=f"Loading Dataset Days {self.days}"):
            day_data_dir = os.path.join(self.base_data_dir, day)
            
            # Filename construction should be consistent regardless of bar type based on previous steps
            x_fname_stem = f'X_windows_in{self.input_window_length}_tgt{self.target_window_length}.npy'
            tgt_fname_stem = f'target_windows_in{self.input_window_length}_tgt{self.target_window_length}.npy'
            info_fname_stem = f'window_info_in{self.input_window_length}_tgt{self.target_window_length}.parquet'
            
            x_fname = os.path.join(day_data_dir, x_fname_stem)
            tgt_fname = os.path.join(day_data_dir, tgt_fname_stem)
            info_fname = os.path.join(day_data_dir, info_fname_stem)

            if os.path.exists(x_fname) and os.path.exists(tgt_fname) and os.path.exists(info_fname):
                try:
                    logging.debug(f"Loading X for {day}...")
                    all_X_windows_list.append(np.load(x_fname))
                    logging.debug(f"Loading TGT for {day}...")
                    all_target_windows_list.append(np.load(tgt_fname))
                    logging.debug(f"Loading INFO for {day}...")
                    all_window_info_list.append(pd.read_parquet(info_fname))
                    days_loaded_count += 1
                    logging.debug(f"Successfully loaded data for day {day}")
                except Exception as e:
                    logging.warning(f"Could not load data for day {day}: {e}")
            else:
                logging.warning(f"Data files missing for day {day} in {day_data_dir}. Files expected: {os.path.basename(x_fname)}, {os.path.basename(tgt_fname)}, {os.path.basename(info_fname)}. Skipping.")


        if days_loaded_count == 0:
            raise FileNotFoundError(f"No data loaded. Check path and days: {self.base_data_dir}, {self.days}")

        logging.info(f"Combining data from {days_loaded_count} days...")
        self.X_windows = np.concatenate(all_X_windows_list, axis=0)
        del all_X_windows_list; gc.collect()
        self.target_windows = np.concatenate(all_target_windows_list, axis=0)
        del all_target_windows_list; gc.collect()
        self.window_info = pd.concat(all_window_info_list, ignore_index=True)
        del all_window_info_list; gc.collect()

        # # --- Corrected Sorting Logic ---
        # self.window_info['window_end_time'] = pd.to_datetime(self.window_info['window_end_time'])
        
        # # Get indices that would sort the current self.window_info by time.
        # # This ensures that even if concatenation resulted in an unsorted DataFrame,
        # # we get the correct order to apply to all related arrays.
        # # Using .values is important if sort_indices is used to index numpy arrays.
        # true_sort_indices = self.window_info['window_end_time'].argsort().values
        
        # # Apply these robust sort indices to all three structures
        # self.X_windows = self.X_windows[true_sort_indices]
        # self.target_windows = self.target_windows[true_sort_indices]
        # self.window_info = self.window_info.iloc[true_sort_indices].reset_index(drop=True)
        # # --- End Corrected Sorting Logic ---


        if not (len(self.X_windows) == len(self.target_windows) == len(self.window_info)):
             raise ValueError("Mismatch in loaded data lengths after concatenation and sorting!")

        logging.info(f"Dataset loaded and sorted: X={self.X_windows.shape}, TGT={self.target_windows.shape}, INFO={self.window_info.shape}")

    def _apply_scaling(self):
        """Applies the pre-fitted scaler to the X_windows data."""
        if self.X_windows is None or self.X_windows.size == 0:
            logging.warning("X_windows is empty, skipping scaling.")
            return

        if self.scaler is None:
            logging.warning("Scaler is None, skipping scaling.")
            return

        # Scaler expects 2D array: (n_samples * window_len, n_features)
        n_samples, window_len, n_features = self.X_windows.shape
        # Reshape to 2D for scaler
        X_reshaped = self.X_windows.reshape(-1, n_features)

        logging.info(f"Applying StandardScaler transform to X_windows (shape: {X_reshaped.shape})...")
        try:
            X_scaled_reshaped = self.scaler.transform(X_reshaped)
            # Reshape back to original 3D shape
            self.X_windows = X_scaled_reshaped.reshape(n_samples, window_len, n_features)
            logging.info("Scaling applied successfully.")
        except Exception as e:
            logging.error(f"Error applying scaler transform: {e}", exc_info=True)
            # Decide how to handle error: raise, log, or proceed with unscaled data?
            # For now, just log the error and proceed (X_windows remains unscaled).
            logging.warning("Proceeding with unscaled data due to scaler error.")

    def _calculate_relative_change(self) -> np.ndarray:
        """
        Calculates the relative change between the last WMP of the input window
        and a summary statistic of the target window's WMP.
        Handles NaNs and division by zero. Stores the result in self.relative_change.
        Returns the calculated relative change array.
        """
        wmp_mean_feature_idx = 0 
        logging.info(f"HFTDataset: Using wmp_mean_feature_idx = {wmp_mean_feature_idx} for X_windows alignment checks and potential direct use.")
        
        info_column_for_last_val = 'last_target_in_window'

        # --- ALIGNMENT CHECK between X_windows and window_info ---
        if (self.X_windows is not None and
            self.window_info is not None and
            info_column_for_last_val in self.window_info.columns and
            self.X_windows.shape[0] == self.window_info.shape[0] and
            self.X_windows.ndim == 3 and
            self.X_windows.shape[2] > wmp_mean_feature_idx):
            
            logging.info(f"HFTDataset: Performing alignment check between X_windows[:, -1, {wmp_mean_feature_idx}] and window_info['{info_column_for_last_val}']...")
            
            x_check_values = self.X_windows[:, -1, wmp_mean_feature_idx].astype(np.float64)
            info_check_values = self.window_info[info_column_for_last_val].values.astype(np.float64)

            if np.all(np.isclose(x_check_values, info_check_values)):
                logging.info(f"  HFTDataset: ALIGNMENT CHECK PASSED. X_windows[:, -1, {wmp_mean_feature_idx}] and window_info['{info_column_for_last_val}'] are aligned.")
            else:
                num_mismatches = np.sum(~np.isclose(x_check_values, info_check_values))
                logging.error(
                    f"  HFTDataset: CRITICAL ALIGNMENT CHECK FAILED! "
                    f"{num_mismatches}/{len(x_check_values)} mismatches detected between "
                    f"X_windows[:, -1, {wmp_mean_feature_idx}] and window_info['{info_column_for_last_val}']."
                )
                mismatch_indices = np.where(~np.isclose(x_check_values, info_check_values))[0]
                for i in mismatch_indices[:3]: # Log first 3 mismatches
                    logging.error(f"    Mismatch Example - Index {i}: X_val={x_check_values[i]:.8f}, Info_val={info_check_values[i]:.8f}, Diff={(x_check_values[i] - info_check_values[i]):.8e}")
                raise AssertionError(
                    "Critical data misalignment detected in HFTDataset between X_windows and window_info. Check logs."
                )
        else:
            details = (
                f"X_windows shape: {self.X_windows.shape if self.X_windows is not None else 'None'}, "
                f"window_info shape: {self.window_info.shape if self.window_info is not None else 'None'}, "
                f"'{info_column_for_last_val}' in columns: {info_column_for_last_val in self.window_info.columns if self.window_info is not None else 'N/A'}, "
                f"X_windows.ndim==3: {self.X_windows.ndim == 3 if self.X_windows is not None else 'N/A'}, "
                f"X_windows.shape[2] > {wmp_mean_feature_idx}: {self.X_windows.shape[2] > wmp_mean_feature_idx if self.X_windows is not None and self.X_windows.ndim == 3 else 'N/A'}"
            )
            logging.warning(
                f"HFTDataset: Alignment check skipped. Conditions not met. Details: {details}"
            )
        # --- END ALIGNMENT CHECK ---

        logging.info(f"Calculating relative change using window_info['{info_column_for_last_val}'] for initial value...")
        if self.X_windows is None or self.target_windows is None or self.window_info is None:
            raise ValueError("X_windows, target_windows, or window_info must be loaded before calculating relative change.")
        if info_column_for_last_val not in self.window_info.columns:
            raise ValueError(f"Required column '{info_column_for_last_val}' not found in window_info. Available columns: {self.window_info.columns.tolist()}")

        # Use self.window_info for the last value of the input window's target variable.
        # last_val_in_input_window = self.X_windows[:, -1, wmp_mean_feature_idx].astype(np.float64) # Original line, using wmp_mean_feature_idx
        last_val_in_input_window = self.window_info[info_column_for_last_val].values.astype(np.float64)

        # Choose summary statistic based on strategy (mean of target window WMPs)
        summary_future_wmp = np.mean(self.target_windows, axis=1).astype(np.float64)

        print(last_val_in_input_window[:100])
        print(summary_future_wmp[:100])

        # Calculate relative change, handle potential division by zero or NaNs
        relative_change = np.full_like(last_val_in_input_window, fill_value=np.nan, dtype=np.float64)
        # Ensure valid_mask considers only finite values to prevent issues with np.nan comparison
        valid_mask = (np.isfinite(last_val_in_input_window)) & (last_val_in_input_window != 0) & (np.isfinite(summary_future_wmp))

        relative_change[valid_mask] = (summary_future_wmp[valid_mask] - last_val_in_input_window[valid_mask]) / last_val_in_input_window[valid_mask]

        # Replace potential infinities resulting from tiny denominators or other operations
        relative_change[np.isinf(relative_change)] = np.nan

        self.relative_change = relative_change # Store for potential analysis
        logging.info(f"Relative change calculated. Shape: {self.relative_change.shape}, NaNs: {np.isnan(self.relative_change).sum()}")
        return self.relative_change

    def _generate_labels(self):
        """Dispatches label generation to the appropriate strategy method."""
        if self.relative_change is None:
            # Calculate relative change if not already done (should be done by constructor)
             self._calculate_relative_change()
             if self.relative_change is None: # Check again
                 logging.error("Failed to calculate relative change.")
                 # Handle error appropriately, maybe raise exception or set default labels
                 self.labels = np.zeros(len(self.window_info), dtype=np.int64) # Default to 0
                 return

        if self.labeling_strategy == "median":
            self._generate_labels_median()
        elif self.labeling_strategy == "directional":
            self._generate_labels_directional()
        elif self.labeling_strategy == "tercile":
            self._generate_labels_tercile()
        else:
            # This case should be caught by __init__, but belts and suspenders
            raise ValueError(f"Invalid labeling strategy '{self.labeling_strategy}' encountered during generation.")

        logging.info(f"Label generation complete using '{self.labeling_strategy}' strategy.")
        label_counts = pd.Series(self.labels).value_counts(normalize=True).sort_index()
        logging.info(f"Final Label distribution:\n{label_counts}")

    def _generate_labels_median(self):
        """
        Generates labels based on the median of the pre-calculated relative change.
        Label is 1 if change > median, 0 otherwise.
        """
        logging.info("Generating labels using MEDIAN relative change threshold...")
        if self.relative_change is None:
             logging.error("Relative change not calculated. Cannot generate median labels.")
             self.labels = np.zeros(len(self.window_info), dtype=np.int64) # Default
             return

        # Calculate the median relative change, ignoring NaNs
        median_change = np.nanmedian(self.relative_change)
        logging.info(f"Calculated Median Relative Change Threshold: {median_change:.6f}")

        # Apply threshold: 1 if > median_change, 0 otherwise
        # NaNs in relative_change will result in 0
        self.labels = np.where(self.relative_change > median_change, 1, 0).astype(np.int64)

    def _generate_labels_directional(self):
        """
        Generates labels based on the sign of the pre-calculated relative change.
        Label is 1 if change > 0, 0 otherwise.
        """
        logging.info("Generating labels using DIRECTIONAL (relative change > 0) threshold...")
        if self.relative_change is None:
             logging.error("Relative change not calculated. Cannot generate directional labels.")
             self.labels = np.zeros(len(self.window_info), dtype=np.int64) # Default
             return

        # Apply threshold: 1 if > 0, 0 otherwise
        # NaNs in relative_change will result in 0
        self.labels = np.where(self.relative_change > 0, 1, 0).astype(np.int64)

    def _generate_labels_tercile(self):
        """
        Generates three labels (0: Down, 1: Stationary, 2: Up) based on terciles
        of the pre-calculated relative change.
        Aims for roughly equal distribution across the three classes.
        """
        logging.info("Generating labels using TERCILE relative change thresholds...")
        if self.relative_change is None:
            logging.error("Relative change not calculated. Cannot generate tercile labels.")
            self.labels = np.zeros(len(self.window_info) if self.window_info is not None else 0, dtype=np.int64) 
            return

        # Filter out NaNs before calculating percentiles
        # This ensures thresholds are based on valid, computable relative changes
        valid_relative_change = self.relative_change[~np.isnan(self.relative_change)]
        if len(valid_relative_change) == 0:
            logging.error("No valid (non-NaN) relative changes available to calculate terciles.")
            # If all relative_change values are NaN, all samples will be excluded later by __init__.
            # For now, set labels to a default; they'll be filtered.
            self.labels = np.full(len(self.relative_change), 1, dtype=np.int64) # Default to stationary (1)
            self.lower_tercile_threshold = np.nan # Mark thresholds as NaN
            self.upper_tercile_threshold = np.nan
            logging.warning("Tercile thresholds are NaN as no valid relative changes were found.")
            return

        # Calculate the 33.33rd and 66.67th percentiles as thresholds
        lower_threshold = np.percentile(valid_relative_change, 100/3)
        upper_threshold = np.percentile(valid_relative_change, 200/3)

        # Store thresholds
        self.lower_tercile_threshold = lower_threshold
        self.upper_tercile_threshold = upper_threshold

        logging.info(f"Calculated Tercile Thresholds: Lower={self.lower_tercile_threshold:.6f}, Upper={self.upper_tercile_threshold:.6f}")

        # Initialize labels array. Samples with NaN relative_change will be filtered out later
        # by the logic in __init__. Their label value here is temporary if they are NaN.
        self.labels = np.full(len(self.relative_change), 1, dtype=np.int64) # Default to stationary (1)

        # Assign labels based on thresholds. NaNs in self.relative_change will not satisfy these conditions.
        # This means NaN relative_change samples will keep the default label 1 for now,
        # but they will be filtered out by the __init__ method's exclusion logic later.
        self.labels[self.relative_change <= self.lower_tercile_threshold] = 0  # Down
        self.labels[self.relative_change > self.upper_tercile_threshold] = 2   # Up
        
        # Explicitly set the middle band for non-NaN values
        # This overwrites the default '1' for values that fall into the middle tercile
        # and are not NaN.
        if not (np.isnan(self.lower_tercile_threshold) or np.isnan(self.upper_tercile_threshold)):
            self.labels[
                (~np.isnan(self.relative_change)) &  # Ensure not NaN
                (self.relative_change > self.lower_tercile_threshold) & 
                (self.relative_change <= self.upper_tercile_threshold)
            ] = 1 # Stationary
        
        # The specific logging about "NaN relative_change labeled as Stationary (1)"
        # has been removed. Samples with NaN relative_change are now handled by the
        # exclusion logic in the __init__ method, preventing them from affecting training.

    def __len__(self) -> int:
        """Returns the total number of samples (windows)."""
        return len(self.window_info)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieves a sample (input window features and calculated label).

        Args:
            idx (int): Index of the sample.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple containing the input window features
                                              and the binary classification label (0 or 1).
        """
        input_features = self.X_windows[idx]
        label = self.labels[idx]

        # Convert to tensors
        # Label needs to be LongTensor for CrossEntropyLoss if model outputs 2 classes
        # Or FloatTensor if model outputs 1 logit and uses BCEWithLogitsLoss
        return torch.tensor(input_features, dtype=torch.float32), torch.tensor(label, dtype=torch.long) 
