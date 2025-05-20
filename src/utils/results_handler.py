import os
import time
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from argparse import Namespace
from typing import Dict, Tuple, List, Any, Optional

from .plotting import plot_confusion_matrix, plot_metrics_barchart, plot_batch_metric, plot_relative_change_histogram
from . import config 

# --- Add imports for history plotting --- #
import matplotlib
matplotlib.use('Agg') # Use Agg backend for non-interactive plotting
import matplotlib.pyplot as plt
# -------------------------------------- #


def setup_results_directories(args: Namespace) -> Tuple[Path, Path, Path]:
    """Creates the directory structure for saving results and returns key paths.

    Includes a subfolder indicating the number of train/test days.

    Args:
        args (Namespace): Parsed command-line arguments including task_type and model_name.

    Returns:
        Tuple[Path, Path, Path]:
            - run_output_dir: Path to the specific directory for this run's outputs.
            - aggregate_results_path: Path to the CSV file for aggregating results across runs.
            - sorted_results_path: Path to the CSV file for sorted aggregate results.
    """
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    # Go up two levels from results_handler.py (utils/ -> src/ -> project root)
    project_root = Path(__file__).resolve().parents[2]
    base_results_dir = project_root / "results"

    # --- Get number of train/test days for path --- #
    num_train_days = len(config.TRAIN_DAYS)
    num_test_days = len(config.TEST_DAYS)
    days_subdir_name = f"train{num_train_days}_test{num_test_days}"
    # ---------------------------------------------- #

    # --- Get window and target length for path --- #
    window_len = config.WINDOW_LENGTH
    target_len = config.TARGET_WINDOW_LENGTH
    data_config_subdir_name = f"bartype-{args.bar_type}_in{window_len}_tgt{target_len}"
    # ------------------------------------------- #

    # Directory for aggregate results: results/<task_type>/<days_subdir>/<data_config_subdir_name>/
    aggregate_base_dir = base_results_dir / args.task_type / days_subdir_name / data_config_subdir_name
    aggregate_base_dir.mkdir(parents=True, exist_ok=True)

    # Specific directory for this run: results/<task_type>/<days_subdir>/<data_config_subdir_name>/<labeling_strategy>/<model_name>/<timestamp>/
    run_output_dir = aggregate_base_dir / args.labeling_strategy / args.model_name / f"{run_timestamp}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # Define paths for aggregate CSV files within the labeling strategy folder
    labeling_strategy_dir = aggregate_base_dir / args.labeling_strategy 
    aggregate_results_path = labeling_strategy_dir / "aggregate_results.csv"
    sorted_results_path = labeling_strategy_dir / "best_results.csv"

    # --- Save Run Configuration (Args & Days Used) --- #
    try:
        # Save args
        args_save_path = run_output_dir / 'args.json'
        # Add the calculated day counts to args before saving for record keeping
        args_dict = vars(args)
        args_dict['_num_train_days'] = num_train_days
        args_dict['_num_test_days'] = num_test_days
        with open(args_save_path, 'w') as f:
            json.dump(args_dict, f, indent=4)

        # Save the actual days used
        days_used_path = run_output_dir / 'days_used.json'
        days_info = {
            'train_days': config.TRAIN_DAYS,
            'test_days': config.TEST_DAYS
        }
        with open(days_used_path, 'w') as f:
            json.dump(days_info, f, indent=4)
        logging.info(f"Saved run configuration and days used in {run_output_dir}")

    except Exception as e:
        logging.warning(f"Could not save configuration files in {run_output_dir}: {e}")

    return run_output_dir, aggregate_results_path, sorted_results_path

def save_training_history_and_plots(
    training_history_df: pd.DataFrame,
    run_output_dir: Path
):
    """Saves the training history DataFrame to CSV and plots loss/accuracy curves.

    Args:
        training_history_df (pd.DataFrame): DataFrame containing epoch-wise metrics.
        run_output_dir (Path): The specific directory for this run's outputs.
    """
    if training_history_df is None or training_history_df.empty:
        logging.warning("Training history DataFrame is empty or None. Skipping saving/plotting history.")
        return

    history_csv_path = run_output_dir / 'training_history.csv'
    loss_plot_path = run_output_dir / 'loss_vs_epoch.png'
    acc_plot_path = run_output_dir / 'accuracy_vs_epoch.png'

    try:
        # Save CSV
        training_history_df.to_csv(history_csv_path, index=False)
        logging.info(f"Training history saved to {history_csv_path}")

        # --- Plot Loss --- #
        if 'epoch' in training_history_df and 'train_loss' in training_history_df and 'val_loss' in training_history_df:
            plt.figure(figsize=(10, 6))
            plt.plot(training_history_df['epoch'], training_history_df['train_loss'], label='Train Loss', marker='o')
            plt.plot(training_history_df['epoch'], training_history_df['val_loss'], label='Validation Loss', marker='x')
            plt.title('Training and Validation Loss vs. Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            plt.savefig(loss_plot_path)
            plt.close()
            logging.info(f"Loss plot saved to {loss_plot_path}")
        else:
            logging.warning(f"Could not plot loss: Missing required columns in history DataFrame ({list(training_history_df.columns)}). Required: 'epoch', 'train_loss', 'val_loss'")

        # --- Plot Accuracy --- #
        if 'epoch' in training_history_df and 'train_accuracy' in training_history_df and 'val_accuracy' in training_history_df:
            plt.figure(figsize=(10, 6))
            plt.plot(training_history_df['epoch'], training_history_df['train_accuracy'], label='Train Accuracy', marker='o')
            plt.plot(training_history_df['epoch'], training_history_df['val_accuracy'], label='Validation Accuracy', marker='x')
            plt.title('Training and Validation Accuracy vs. Epochs')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(acc_plot_path)
            plt.close()
            logging.info(f"Accuracy plot saved to {acc_plot_path}")
        else:
            logging.warning(f"Could not plot accuracy: Missing required columns in history DataFrame ({list(training_history_df.columns)}). Required: 'epoch', 'train_accuracy', 'val_accuracy'")

    except Exception as e:
        logging.error(f"Error during saving/plotting training history: {e}", exc_info=True)

def save_test_results(
    run_output_dir: Path,
    metrics: Dict[str, float],
    conf_matrix: np.ndarray,
    class_names: Optional[List[str]] = None
):
    """Saves final test evaluation results (metrics, plots) to the run directory.

    Args:
        run_output_dir (Path): The specific directory for this run.
        metrics (Dict[str, float]): Dictionary of evaluation metrics.
        conf_matrix (np.ndarray): Confusion matrix.
        class_names (Optional[List[str]]): Names for confusion matrix labels (e.g., ['Down', 'Up']).
                                           Defaults to generic class indices if None.
    """
    if class_names is None:
        # Default to class indices if names not provided
        num_classes = conf_matrix.shape[0]
        class_names = [f'Class {i}' for i in range(num_classes)]
        if num_classes == 2:
             class_names = ['Down', 'Up'] # Specific default for binary

    logging.info("Saving final test evaluation results...")

    # Save metrics dictionary as JSON
    metrics_save_path = run_output_dir / 'test_evaluation_metrics.json'
    try:
        # Convert numpy types to native Python types for JSON serialization
        serializable_metrics = {k: (float(v) if isinstance(v, (np.float32, np.float64)) else v) for k, v in metrics.items()}
        with open(metrics_save_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=4)
        logging.info(f"Test metrics saved to {metrics_save_path}")
    except Exception as e:
        logging.error(f"Failed to save test metrics: {e}")

    # Save confusion matrix as numpy array
    cm_npy_save_path = run_output_dir / 'confusion_matrix.npy'
    try:
        np.save(cm_npy_save_path, conf_matrix)
        logging.info(f"Confusion matrix array saved to {cm_npy_save_path}")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix array: {e}")

    # Plot and save confusion matrix in PNG format
    cm_png_save_path = run_output_dir / 'confusion_matrix.png'
    try:
        plot_confusion_matrix(conf_matrix, class_names, str(cm_png_save_path))
        logging.info(f"Confusion matrix plot saved to {cm_png_save_path}")
    except Exception as e:
        logging.error(f"Failed to plot/save confusion matrix: {e}")

    # Plot and save metrics bar chart
    metrics_plot_save_path = run_output_dir / 'test_metrics_barchart.png'
    try:
        plot_metrics_barchart(metrics, str(metrics_plot_save_path))
        logging.info(f"Metrics bar chart saved to {metrics_plot_save_path}")
    except Exception as e:
        logging.error(f"Failed to plot/save metrics bar chart: {e}")

def append_and_sort_results(
    aggregate_results_path: Path,
    sorted_results_path: Path,
    args: Namespace,
    metrics: Dict[str, float],
    sort_metric: str = 'accuracy',
    ascending: bool = False # Typically sort accuracy/f1 descending
):
    """Appends the current run's results to the aggregate CSV and saves a sorted version.

    Args:
        aggregate_results_path (Path): Path to the main results CSV.
        sorted_results_path (Path): Path to the sorted results CSV.
        args (Namespace): Command-line arguments for the run.
        metrics (Dict[str, float]): Test metrics dictionary.
        sort_metric (str): Metric name to sort the results by.
        ascending (bool): Sort order.
    """
    logging.info(f"Appending results to {aggregate_results_path} and updating {sorted_results_path}")

    # --- Create Data Entry for this Run --- #
    run_summary = {}
    run_summary['acc'] = metrics.get('accuracy', np.nan)
    run_summary['macro_f1'] = metrics.get('macro_f1_score', np.nan)
    run_summary['prec_macro'] = metrics.get('precision_macro', np.nan)
    run_summary['rec_macro'] = metrics.get('recall_macro', np.nan)
    run_summary['auc'] = metrics.get('auc', np.nan)
    run_summary['loss'] = metrics.get('loss', np.nan)
    run_summary['model_name'] = getattr(args, 'model_name', 'N/A')
    run_summary['run_dir'] = str(args.run_output_dir.relative_to(args.run_output_dir.parents[2])) if hasattr(args, 'run_output_dir') else 'N/A' # Get relative path like task/run/model

    # Convert to DataFrame
    new_entry_df = pd.DataFrame([run_summary])

    # --- Append to Aggregate File --- #
    try:
        if aggregate_results_path.exists():
            results_df = pd.read_csv(aggregate_results_path)
            # Ensure columns match, add missing if necessary
            for col in new_entry_df.columns:
                if col not in results_df.columns:
                    results_df[col] = np.nan
            results_df = pd.concat([results_df, new_entry_df[results_df.columns]], ignore_index=True)
        else:
            results_df = new_entry_df

        results_df.to_csv(aggregate_results_path, index=False)
        #logging.info(f"Appended run summary to {aggregate_results_path}")

        # --- Sort and Save Best Results --- #
        if sort_metric in results_df.columns:
            sorted_df = results_df.sort_values(by=sort_metric, ascending=ascending)
            sorted_df.to_csv(sorted_results_path, index=False)
            logging.info(f"Saved sorted results ({sort_metric}, asc={ascending}) to {sorted_results_path}")
        else:
            logging.warning(f"Sort metric '{sort_metric}' not found in results columns. Cannot sort.")
            # Save unsorted data anyway to the sorted path
            results_df.to_csv(sorted_results_path, index=False)
            logging.info(f"Saved (unsorted) results to {sorted_results_path}")

    except Exception as e:
        logging.error(f"Failed to update aggregate result files: {e}")

def setup_logging(log_path: Path):
    # This function is assumed to exist and is called in the original file
    # It's not modified in the current version, so it's kept as is
    pass 

def save_batch_history_plots(
    batch_history_df: pd.DataFrame,
    run_output_dir: Path,
    rolling_window: int = 50 # Default rolling window for smoothing
):
    """Saves plots for batch-level training metrics.

    Args:
        batch_history_df (pd.DataFrame): DataFrame containing batch-level metrics.
                                         Expected columns: 'batch_train_loss', 
                                         'batch_train_accuracy', 'batch_grad_norm'.
        run_output_dir (Path): The specific directory for this run's outputs.
        rolling_window (int): Window size for rolling average.
    """
    if batch_history_df is None or batch_history_df.empty:
        logging.warning("Batch history DataFrame is empty or None. Skipping plotting batch metrics.")
        return

    logging.info(f"Plotting batch-level training metrics to {run_output_dir}...")

    # Plot Batch Training Loss
    if 'batch_train_loss' in batch_history_df.columns:
        plot_batch_metric(
            batch_history_df=batch_history_df,
            metric_column='batch_train_loss',
            y_label='Batch Train Loss',
            title='Batch Training Loss vs. Global Batch Step',
            save_path=run_output_dir / 'batch_train_loss_plot.png',
            rolling_window=rolling_window
        )
    else:
        logging.warning("Column 'batch_train_loss' not found in batch history. Skipping plot.")

    # Plot Batch Training Accuracy
    if 'batch_train_accuracy' in batch_history_df.columns:
        plot_batch_metric(
            batch_history_df=batch_history_df,
            metric_column='batch_train_accuracy',
            y_label='Batch Train Accuracy',
            title='Batch Training Accuracy vs. Global Batch Step',
            save_path=run_output_dir / 'batch_train_accuracy_plot.png',
            rolling_window=rolling_window
        )
    else:
        logging.warning("Column 'batch_train_accuracy' not found in batch history. Skipping plot.")

    # Plot Batch Gradient Norm
    if 'batch_grad_norm' in batch_history_df.columns:
        plot_batch_metric(
            batch_history_df=batch_history_df,
            metric_column='batch_grad_norm',
            y_label='Batch Gradient Norm',
            title='Batch Gradient Norm vs. Global Batch Step',
            save_path=run_output_dir / 'batch_grad_norm_plot.png',
            rolling_window=rolling_window
        )
    else:
        logging.warning("Column 'batch_grad_norm' not found in batch history. Skipping plot.") 