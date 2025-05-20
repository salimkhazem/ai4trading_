import os
import time
import pandas as pd
import pickle as pkl
from glob import glob
from typing import Any, List, Tuple, Dict


def load_pickle(filename:str) -> Any:
    '''Load data from a pickle (pkl) file.'''
    with open(filename, 'rb') as handle:
        data = pkl.load(handle)
    return data

def save_pickle(filename: str, data: Any) -> None:
    '''Save data in the pickle (pkl) format.'''
    with open(filename, 'wb') as handle:
        pkl.dump(data, handle, protocol=pkl.HIGHEST_PROTOCOL)

def is_valid_date_folder(name: str) -> bool:
    '''Check if a folder name is an 8-digit date.'''
    return name.isdigit() and len(name) == 8

def get_valid_day_folders(base_path: str) -> List[str]:
    '''Get sorted list of valid date folders in the base path.'''
    try:
        return sorted([f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f)) and is_valid_date_folder(f)])
    except FileNotFoundError:
        print(f"Error: Base path not found: {base_path}")
        return []

def load_and_clean_day_data(base_path: str, day: str) -> pd.DataFrame:
    '''Loads, concatenates, and cleans parquet files for a specific day.'''
    print(f"--- Loading {day} ---")
    day_path = os.path.join(base_path, day, 'chunks_parquet_levels')
    parquet_files = sorted(glob(os.path.join(day_path, '*.parquet')))

    if not parquet_files:
        print(f"[!] No parquet files found for {day} in {day_path}")
        return pd.DataFrame()

    start_time = time.time()
    try:
        dfs = [pd.read_parquet(f) for f in parquet_files]
        if not dfs:
            print(f"[!] Failed to read any parquet files for {day}")
            return pd.DataFrame()
        df_day = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        print(f"Error reading or concatenating parquet files for {day}: {e}")
        return pd.DataFrame()

    # Clean column names
    df_day.columns = [str(col).strip() for col in df_day.columns]

    # Parse 'time' to datetime
    if 'time' in df_day.columns:
        df_day['time'] = pd.to_datetime(df_day['time'])
    else:
        print(f"[!] 'time' column not found for {day}")
        return pd.DataFrame()

    # Add trading day as separate column
    try:
        df_day['date'] = pd.to_datetime(day, format='%Y%m%d')
    except ValueError:
        print(f"[!] Invalid date format for folder: {day}")
        return pd.DataFrame()

    print(f"Loaded {len(df_day):,} rows for {day} in {time.time() - start_time:.2f}s")
    return df_day