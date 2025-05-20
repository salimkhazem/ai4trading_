import logging
import pandas as pd
import numpy as np 

import resource
HAS_RESOURCE = True


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def get_memory_usage_gb() -> float:
    """Gets current peak memory usage in gigabytes.

    Returns:
        float: Peak memory usage in GB, or -1.0 if the 'resource' module is not available.
    """
    if not HAS_RESOURCE:
        return -1.0
    
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0) 


def clean_raw_data(df_raw: pd.DataFrame, columns_to_drop_custom: list[str] = None) -> pd.DataFrame:
    """Cleans raw financial data by dropping NaNs and specified columns.

    Args:
        df_raw (pd.DataFrame): The raw input DataFrame.
        columns_to_drop_custom (list[str], optional): A list of specific columns to drop. 
                                                      Defaults to a predefined list if None.

    Returns:
        pd.DataFrame: The cleaned DataFrame.
    """
    logging.info("Starting data cleaning...")
    # --- Drop NaNs ---
    initial_rows = len(df_raw)
    df_clean = df_raw.dropna().copy() 
    rows_dropped_nan = initial_rows - len(df_clean)
    percent_dropped_nan = (rows_dropped_nan / initial_rows * 100) if initial_rows > 0 else 0.0
    logging.info(f"Dropped {rows_dropped_nan:,} rows with NaNs ({percent_dropped_nan:.2f}%). Shape after NaN drop: {df_clean.shape}")

    if columns_to_drop_custom is None:
        columns_to_drop = [
            'exch_time', 'exchange', 'first_sequence_number', 'last_sequence_number',
            'first_sym_sequence', 'last_sym_sequence', 'first_time', 'first_exch_time',
            'event_id', 'date'
        ]
    else:
        columns_to_drop = columns_to_drop_custom
        
    existing_cols_to_drop = [col for col in columns_to_drop if col in df_clean.columns]
    
    if existing_cols_to_drop:
        df_clean.drop(columns=existing_cols_to_drop, inplace=True)
        logging.info(f"Dropped specified columns: {existing_cols_to_drop}. Shape now: {df_clean.shape}")
    else:
        logging.info("No specified columns found to drop, or 'columns_to_drop_custom' was empty.")
        
    logging.info(f"Cleaning complete. Final shape: {df_clean.shape}")
    return df_clean


def compute_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Computes microstructure features from the given DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame containing microstructure data.

    Returns:
        pd.DataFrame: The DataFrame with microstructure features added.
    """
    # Midprice
    df["mid_price"] = (df["L1_bid_price"] + df["L1_ask_price"]) / 2

    # Weighted MidPrice (WMP)
    df["weighted_mid_price"] = (df["L1_bid_price"] * df["L1_ask_size"] + df["L1_ask_price"] * df["L1_bid_size"]) / (
        df["L1_bid_size"] + df["L1_ask_size"]
    )

    # Spread
    df["spread"] = df["L1_ask_price"] - df["L1_bid_price"]

    # Order Book Imbalance (OBI) 
    for i in range(1, 11):
        bid = f"L{i}_bid_size"
        ask = f"L{i}_ask_size"
        df[f"obi_L{i}"] = (df[bid] - df[ask]) / (df[bid] + df[ask])

    # Depth imbalance (L1 to L10)
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
    df["bid_price_slope"] = (df["L1_bid_price"] - df["L5_bid_price"]) / 4
    df["ask_price_slope"] = (df["L5_ask_price"] - df["L1_ask_price"]) / 4

    # Order Flow Imbalance (OFI)
    df["ofi"] = df["L1_bid_size"].diff() - df["L1_ask_size"].diff()

    # Simple Moving Average (SMA)
    df["sma_10"] = df["weighted_mid_price"].rolling(window=10).mean()
    df["sma_20"] = df["weighted_mid_price"].rolling(window=20).mean()
    df["sma_100"] = df["weighted_mid_price"].rolling(window=100).mean()
    #df["sma_200"] = df["weighted_mid_price"].rolling(window=200).mean()

    # Exponential Moving Average (EMA)
    df["ema_10"] = df["weighted_mid_price"].ewm(span=10, adjust=False).mean()
    df["ema_20"] = df["weighted_mid_price"].ewm(span=20, adjust=False).mean()
    df["ema_100"] = df["weighted_mid_price"].ewm(span=100, adjust=False).mean()
    #df["ema_200"] = df["weighted_mid_price"].ewm(span=200, adjust=False).mean()

    # Momentum
    df["momentum_10"] = df["weighted_mid_price"] - df["weighted_mid_price"].shift(10)
    df["momentum_20"] = df["weighted_mid_price"] - df["weighted_mid_price"].shift(20)
    df["momentum_100"] = df["weighted_mid_price"] - df["weighted_mid_price"].shift(100)
    #df["momentum_200"] = df["weighted_mid_price"] - df["weighted_mid_price"].shift(200)

    # Order Flow Imbalance (OFI)
    df["ofi"] = df["L1_bid_size"].diff() - df["L1_ask_size"].diff()

    # Returns
    df["return_1"] = df["weighted_mid_price"].pct_change()
    df["return_5"] = df["weighted_mid_price"].pct_change(5)
    df["log_return_1"] = np.log(df["weighted_mid_price"] / df["weighted_mid_price"].shift(1))
    df["log_return_5"] = np.log(df["weighted_mid_price"] / df["weighted_mid_price"].shift(5))
    
    # Relative Strength Index (RSI)
    def compute_rsi(df, period=14):
        delta = df["weighted_mid_price"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Wilder's method using EMA
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()

        rs = avg_gain / avg_loss

        df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
        df[f"rsi_signal_{period}"] = np.where(df[f"rsi_{period}"] > 70, -1, np.where(df[f"rsi_{period}"] < 30, 1, 0))

        return df

    df = compute_rsi(df, period=10)
    df = compute_rsi(df, period=20)
    df = compute_rsi(df, period=100)
    #df = compute_rsi(df, period=200)

    # Bollinger Bands
    def compute_bollinger_bands(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Compute Bollinger Bands for the given DataFrame.

        Args:
            df (pd.DataFrame): DataFrame containing the 'mid_price' column.
            window (int): The rolling window size for the Bollinger Bands.

        Returns:
            pd.DataFrame: DataFrame with Bollinger Bands and signals added.
        """
        rolling_mean = df["weighted_mid_price"].rolling(window=window).mean()
        rolling_std = df["weighted_mid_price"].rolling(window=window).std()

        df[f"bollinger_upper_{window}"] = rolling_mean + 2 * rolling_std
        df[f"bollinger_lower_{window}"] = rolling_mean - 2 * rolling_std
        df[f"band_width_{window}"] = df[f"bollinger_upper_{window}"] - df[f"bollinger_lower_{window}"]
        df[f"bollinger_signal_{window}"] = np.where(df["weighted_mid_price"] > df[f"bollinger_upper_{window}"], -1, 
                                           np.where(df["weighted_mid_price"] < df[f"bollinger_lower_{window}"], 1, 0))
        return df

    df = compute_bollinger_bands(df, window=10)
    df = compute_bollinger_bands(df, window=20)
    df = compute_bollinger_bands(df, window=100)
    #df = compute_bollinger_bands(df, window=200)
    
    # Time encoding
    # if pd.api.types.is_datetime64_any_dtype(df["time"]):
    #     df["seconds_since_midnight"] = (
    #         df["time"].dt.hour * 3600
    #         + df["time"].dt.minute * 60
    #         + df["time"].dt.second
    #         + df["time"].dt.microsecond / 1e6
    #     )
    
    return df

