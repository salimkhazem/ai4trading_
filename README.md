# AI4Trading: High-Frequency Trading with Deep Learning

## Description

This project aims to develop and evaluate deep learning models for high-frequency trading (HFT) tasks. It includes a comprehensive pipeline for data preprocessing, feature engineering, model training, and results analysis, with a focus on using volume bars as input for market movement prediction.

## Project Status

[comment]: <> (Indicate current status, e.g., Actively Developed, Maintenance Mode, Proof of Concept)
*Development in Progress*

## Directory Structure

```
AI4Trading/
├── notebooks/             # Jupyter notebooks for experimentation and analysis
├── src/
│   ├── preprocessing/     # Scripts for data cleaning, feature engineering, bar creation
│   │   ├── volume_bar_pipeline.py
│   │   ├── volume_bar_pipeline_parallel.py
│   │   ├── day_parallel_volume_bar_pipeline.py
│   │   └── utils_preprocessing.py # (Actual location may be src/utils)
│   ├── dataset/           # PyTorch Dataset class (e.g., HFTDataset)
│   ├── models/            # Model definitions (Transformer, RNN, TCN, etc.)
│   ├── trainers/          # Training and evaluation loop logic
│   ├── utils/             # Helper utilities (config, logging, plotting, etc.)
│   ├── train.py           # Main script for training deep learning models
│   ├── train_deepspeed.py # Main script for distributed training with DeepSpeed
│   ├── train_ml.py        # Script for training classical machine learning models
│   └── ...
├── .gitignore
├── README.md
└── ds_config.json         # DeepSpeed configuration file
```

## Core Components

### 1. Data Preprocessing (`src/preprocessing/`)

This directory contains scripts responsible for transforming raw financial data into a format suitable for model training. The general workflow involves cleaning, feature calculation, creating volume bars, and then generating sequential input and target windows.

Key scripts include:

*   **`utils_preprocessing.py`** (located in `src/utils/` or `src/preprocessing/`):
    *   Provides utility functions for common preprocessing tasks such as:
        *   `clean_raw_data()`: Cleans the initial raw dataset.
        *   `compute_microstructure_features()`: Calculates various microstructure features from the data.
        *   `get_memory_usage_gb()`: Helper to monitor memory usage.

*   **`volume_bar_pipeline.py`**:
    *   Processes raw data sequentially, day by day, and symbol by symbol within each day.
    *   Performs data loading, cleaning, microstructure feature computation, volume bar creation (using `create_volume_bars_with_lob_features`), and sequential window generation (`generate_sequential_windows`).
    *   Outputs daily processed data (X_windows.npy, target_windows.npy, window_info.parquet, features.txt) into structured directories.

*   **`volume_bar_pipeline_parallel.py`**:
    *   Similar to `volume_bar_pipeline.py` but parallelizes the processing of **symbols within each day** using `joblib`.
    *   Days are still processed sequentially.
    *   This can significantly speed up preprocessing if a single day contains many symbols.

*   **`day_parallel_volume_bar_pipeline.py`**:
    *   Processes **days in parallel** using `joblib`.
    *   Within each day, symbols are processed sequentially (as in `volume_bar_pipeline.py`).
    *   This is beneficial when processing a large number of independent daily data files.

**Output of Preprocessing:**
The scripts in this folder typically generate files for each processed day, usually including:
*   `X_windows_in<INPUT_LEN>_tgt<TARGET_LEN>.npy`: NumPy array of input feature windows.
*   `target_windows_in<INPUT_LEN>_tgt<TARGET_LEN>.npy`: NumPy array of corresponding target windows.
*   `window_info_in<INPUT_LEN>_tgt<TARGET_LEN>.parquet`: Parquet file containing metadata for each window (e.g., timestamps, symbol).
*   `features.txt`: A list of feature names used.
These are stored in subdirectories (typically within a main `processed_data` directory, which is not versioned), structured by parameters (e.g., bar type, window lengths) and then by day.


### 2. Model Training (`src/train.py`)

`train.py` is the primary script for training deep learning models for HFT binary or multi-class prediction tasks.

**Key Functionalities:**

*   **Argument Parsing**: Uses `argparse` to accept a wide range of command-line arguments for controlling the training process, including:
    *   General settings: `task_type`, `labeling_strategy`, `bar_type`, `nb_bars`.
    *   Model parameters: `model_name` (transformer, lstm, gru, tcn), dimensions (`d_model`, `nhead`, `dim_ff`, `hidden_dim`), `num_layers`, `dropout`, `output_dim`.
    *   Training hyperparameters: `epochs`, `batch_size`, `lr`, `weight_decay`, `early_stopping_patience`, `num_workers`, `seed`.
*   **Environment Setup**: Initializes random seeds for reproducibility and selects the appropriate device (CPU/GPU).
*   **Data Loading**: 
    *   Uses `utils.config.py` to determine data paths and parameters (e.g., `TRAIN_DAYS`, `TEST_DAYS`, `WINDOW_LENGTH`, `TARGET_WINDOW_LENGTH`).
    *   Fits a `StandardScaler` on the training days' data using `utils.preprocessing_utils.fit_save_scaler_incrementally`.
    *   Instantiates `dataset.hft_dataset.HFTDataset` for training and validation/testing, passing the fitted scaler.
*   **Model Initialization**: 
    *   Initializes the chosen model architecture (e.g., `TransformerEncoder`, `RNN`, `TCN` from `src/models/`) with specified dimensions and feature size derived from the data.
*   **Training Orchestration**:
    *   Sets up the optimizer (e.g., `AdamW`) and loss function (e.g., `CrossEntropyLoss`).
    *   Calls the `train_model` function (likely from `src/trainers/trainer.py`) which contains the main training and validation loop, including early stopping logic.
    *   Saves the best performing model weights.
*   **Evaluation**: 
    *   After training, loads the best model and calls `evaluate_model` (likely from `src/trainers/trainer.py`) to get final metrics on the test set.
*   **Results Handling & Logging**:
    *   Uses `utils.results_handler.py` to set up output directories for the run (typically within a main `results` directory, which is not versioned).
    *   Saves training arguments, training history (epoch-wise and batch-wise), evaluation metrics, confusion matrix, and various plots (e.g., loss, accuracy, relative change histograms) to the run-specific directory.
    *   Appends results to an aggregate CSV file for comparison across runs.
    *   Manages logging throughout the process using Python's `logging` module.

**Related Training Scripts:**
*   `src/train_deepspeed.py`: Adapts the training process for distributed training using Microsoft DeepSpeed.
*   `src/train_ml.py`: A separate pipeline for training classical machine learning models (e.g., scikit-learn models).

## Usage

### 1. Data Preprocessing

Navigate to the `src/preprocessing/` directory. Choose one of the pipelines:

*   **Sequential Processing:**
    ```bash
    python volume_bar_pipeline.py
    ```
*   **Symbol-Parallel Processing (within each day):**
    ```bash
    python volume_bar_pipeline_parallel.py
    ```
*   **Day-Parallel Processing:**
    ```bash
    python day_parallel_volume_bar_pipeline.py
    ```
    Modify parameters (paths, days, symbols, window lengths, etc.) directly within the chosen script's `main()` function or adapt them to use command-line arguments if preferred.

### 2. Model Training

Navigate to the `src/` directory.

*   **Train a Deep Learning Model:**
    ```bash
    python train.py --model_name transformer --epochs 50 --batch_size 128 --lr 1e-4 --labeling_strategy tercile --bar_type volume --d_model 128 --nhead 8 ... (other arguments)
    ```
    Refer to the `parse_arguments()` function in `train.py` for all available command-line options.

*   **Train with DeepSpeed (example):**
    ```bash
    deepspeed train_deepspeed.py --deepspeed_config ds_config.json --model_name transformer ... (other arguments)
    ```
    Ensure `ds_config.json` is configured correctly.

## Configuration

Key configurations can be found in:

*   **`src/utils/config.py`**: Contains paths, default data parameters (`TRAIN_DAYS`, `TEST_DAYS`), window sizes, etc.
*   **Preprocessing Scripts (`src/preprocessing/*.py`)**: `main()` functions within these scripts define data paths, days to process, symbols, and bar/window parameters.
*   **`train.py` (and related training scripts)**: Default values for arguments are set in `parse_arguments()`. Many can be overridden via command line.
*   **`ds_config.json`**: Configuration for DeepSpeed if used.


[comment]: <> (## Roadmap / Future Work)

[comment]: <> (## Contributing)
[comment]: <> (Guidelines for contributing to the project, if applicable.)

[comment]: <> (## License)
[comment]: <> (Specify the project license, e.g., MIT, Apache 2.0)
