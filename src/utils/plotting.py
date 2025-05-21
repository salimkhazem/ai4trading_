import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import pandas as pd
from typing import Dict, Optional
import logging
from pathlib import Path

# Configure logging for this module
logger = logging.getLogger(__name__)

def plot_confusion_matrix(cm: np.ndarray, class_names: list, output_path: str, title: str = 'Confusion Matrix'):
    """Plots and saves a confusion matrix heatmap.

    Args:
        cm (np.ndarray): The confusion matrix array.
        class_names (list): List of class names for labels.
        output_path (str): Path to save the plot image (e.g., .../confusion_matrix.png).
        title (str): Title for the plot.
    """
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    

def plot_metrics_barchart(metrics: Dict[str, float], output_path: str, title: str = 'Test Set Metrics'):
    """Plots and saves a bar chart of key classification metrics.

    Args:
        metrics (Dict[str, float]): Dictionary containing metric names and values
                                     (e.g., {'accuracy': 0.9, 'f1_score': 0.85, ...}).
                                     Excludes 'loss'.
        output_path (str): Path to save the plot image (e.g., .../test_metrics.png).
        title (str): Title for the plot.
    """
    # Select key metrics for plotting (exclude loss)
    plot_metrics = {k: v for k, v in metrics.items() if k != 'loss'}
    if not plot_metrics:
        print("No metrics (excluding loss) provided for plotting.")
        return

    try:
        df = pd.DataFrame([plot_metrics])
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(data=df)
        plt.title(title)
        plt.ylabel('Score')
        plt.ylim(0, 1.05) # Set y-axis limits for typical metric range

        # Add text labels above bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.4f')

        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
    except Exception as e:
        print(f"Error plotting metrics bar chart: {e}")


def plot_relative_change_histogram(
    relative_change: np.ndarray,
    labeling_strategy: str,
    save_path: Path,
    lower_tercile_threshold: Optional[float] = None,
    upper_tercile_threshold: Optional[float] = None
):
    """Plots and saves a histogram of relative price changes.

    Args:
        relative_change (np.ndarray): Array of calculated relative changes.
        labeling_strategy (str): The strategy used ('median', 'directional', or 'tercile')
                                   to determine the threshold line(s).
        save_path (Path): Path object where the plot should be saved.
        lower_tercile_threshold (Optional[float]): The lower threshold for tercile strategy.
        upper_tercile_threshold (Optional[float]): The upper threshold for tercile strategy.
    """
    logger.info(f"Plotting histogram of relative changes (Strategy: {labeling_strategy})...")
    if relative_change is None:
        logger.warning("Received None for relative_change array. Cannot plot histogram.")
        return

    try:
        plt.figure(figsize=(10, 6))
        # Filter out NaNs before plotting histogram
        valid_changes = relative_change[~np.isnan(relative_change)]

        if len(valid_changes) > 0:
            plt.hist(valid_changes, bins=100, color='skyblue', edgecolor='black')
            plt.title(f'Histogram of Relative Change (Strategy: {labeling_strategy})')
            plt.xlabel('Relative Change [(Mean Future WMP - Last Input WMP) / Last Input WMP]') 
            plt.ylabel('Frequency')
            plt.grid(axis='y', alpha=0.75)

            # Add a vertical line for the threshold
            if labeling_strategy == 'median':
                median_val = np.nanmedian(valid_changes)
                plt.axvline(median_val, color='red', linestyle='dashed', linewidth=1,
                            label=f'Median Threshold = {median_val:.6f}')
                plt.legend()
            elif labeling_strategy == 'directional':
                 plt.axvline(0, color='green', linestyle='dashed', linewidth=1, label='Threshold = 0')
                 plt.legend()
            elif labeling_strategy == 'tercile':
                if lower_tercile_threshold is not None and upper_tercile_threshold is not None:
                    plt.axvline(lower_tercile_threshold, color='purple', linestyle='dashed', linewidth=1,
                                label=f'Lower Tercile = {lower_tercile_threshold:.6f}')
                    plt.axvline(upper_tercile_threshold, color='orange', linestyle='dashed', linewidth=1,
                                label=f'Upper Tercile = {upper_tercile_threshold:.6f}')
                    plt.legend()
                else:
                    logger.warning("Tercile strategy selected, but thresholds not provided for plotting.")
            else:
                logger.warning(f"Unknown labeling strategy '{labeling_strategy}' for histogram threshold line.")

            plt.savefig(save_path)
            plt.close() # Close the figure to free memory
            logger.info(f"Relative change histogram saved to {save_path}")
        else:
             logger.warning("No valid relative change values found to plot histogram.")

    except Exception as e:
        logger.error(f"Error plotting histogram: {e}", exc_info=True)


def plot_batch_metric(
    batch_history_df: pd.DataFrame,
    metric_column: str,
    y_label: str,
    title: str,
    save_path: Path,
    rolling_window: Optional[int] = None
):
    """Plots a specific metric from the batch history DataFrame against global batch steps.

    Args:
        batch_history_df (pd.DataFrame): DataFrame containing batch-level metrics.
                                         Expected to have 'epoch' and 'batch_idx' columns,
                                         and the specified 'metric_column'.
        metric_column (str): The name of the column in batch_history_df to plot.
        y_label (str): Label for the Y-axis.
        title (str): Title for the plot.
        save_path (Path): Path object where the plot should be saved.
        rolling_window (Optional[int]): If provided, plots a rolling mean of the metric.
    """
    if batch_history_df.empty or metric_column not in batch_history_df.columns:
        logger.warning(f"Batch history is empty or metric '{metric_column}' not found. Skipping plot: {title}")
        return

    try:
        # Create a global step. If using just index, ensure df is sorted by epoch then batch_idx
        # For simplicity, we'll use the DataFrame index as a proxy for global step assuming it's sequential.
        # A more robust global step could be created if needed:
        # batch_history_df['global_step'] = batch_history_df.groupby('epoch').cumcount() + 
        #                                     (batch_history_df['epoch'] - 1) * max_batches_per_epoch
        global_step = batch_history_df.index

        plt.figure(figsize=(12, 6))
        plt.plot(global_step, batch_history_df[metric_column], label=y_label, alpha=0.6)

        if rolling_window and rolling_window > 0:
            rolling_mean = batch_history_df[metric_column].rolling(window=rolling_window, min_periods=1).mean()
            plt.plot(global_step, rolling_mean, label=f'{y_label} (Roll Avg {rolling_window})', color='red')
        
        plt.title(title)
        plt.xlabel('Global Batch Step')
        plt.ylabel(y_label)
        plt.legend()
        plt.grid(True, alpha=0.5)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        logger.info(f"Batch metric plot saved to {save_path}")
    except Exception as e:
        logger.error(f"Error plotting batch metric '{title}': {e}", exc_info=True)


# Example usage (if needed for testing):
# if __name__ == '__main__':
#     # Example CM
#     cm_example = np.array([[100, 10], [5, 85]])
#     plot_confusion_matrix(cm_example, ['Class 0', 'Class 1'], './cm_example.png')
#
#     # Example Metrics
#     metrics_example = {'accuracy': 0.925, 'precision': 0.894, 'recall': 0.944, 'f1_score': 0.918, 'auc': 0.975, 'loss': 0.123}
#     plot_metrics_barchart(metrics_example, './metrics_example.png') 