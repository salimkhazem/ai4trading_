import argparse
import logging
import numpy as np
import os
import sys
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# Project utilities 
import utils.config as config
from dataset.hft_dataset import HFTDataset 
from models import TransformerEncoder, RNN, TCN 
from trainers.trainer import train_model, evaluate_model 
from utils.helpers import setup_logging, get_device, seed_it_all 
from utils.results_handler import setup_results_directories, save_test_results, append_and_sort_results, save_training_history_and_plots, save_batch_history_plots
from utils.preprocessing_utils import fit_save_scaler_incrementally, validate_feature_array
from utils.plotting import plot_relative_change_histogram

import warnings
warnings.filterwarnings("ignore")


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the HFT model training script."""
    parser = argparse.ArgumentParser(description='Train a deep learning model for HFT binary prediction.')

    # General settings
    parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'], help='Type of task')
    parser.add_argument('--labeling_strategy', type=str, default='median', choices=['median', 'directional', 'tercile'], help='Strategy for label generation in HFTDataset')
    parser.add_argument('--bar_type', type=str, default='volume', choices=['time', 'volume'], help='Type of input bars to use (time or volume)')
    parser.add_argument('--nb_bars', type=int, default=config.NB_BARS, help='Number of volume bars per day/symbol (used if bar_type=volume)')

    # Model parameters
    model_params = parser.add_argument_group('Model Parameters') 
    model_params.add_argument('--model_name', choices=['transformer', 'lstm', 'gru', 'tcn'], default='transformer', help="Select the deep learning model architecture.")
    # --- Transformer Specific --- #
    model_params.add_argument('--d_model', type=int, default=128, help='Dimension of the transformer model embeddings (d_model)')
    model_params.add_argument('--nhead', type=int, default=8, help='Number of attention heads (for Transformer)')
    model_params.add_argument('--dim_ff', type=int, default=1024, help='Dimension of the feedforward network (for Transformer)')
    # --- RNN/TCN Specific --- # 
    model_params.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size (for LSTM, GRU, TCN)')
    # --- Common --- #
    model_params.add_argument('--num_layers', type=int, default=4, help='Number of layers in the model')
    model_params.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    model_params.add_argument('--output_dim', type=int, default=2, help='Output dimension (typically 2 for binary classification)')

    # Training parameters
    train_params = parser.add_argument_group('Training Hyperparameters') 
    train_params.add_argument('--epochs', type=int, default=50, help='Maximum number of training epochs')
    train_params.add_argument('--batch_size', type=int, default=128, help='Batch size')
    train_params.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_params.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
    train_params.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping based on validation loss')
    train_params.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    train_params.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    args = parser.parse_args()
    logger.info(f"Arguments parsed: {vars(args)}")
    return args


def main(args: argparse.Namespace):
    
    # --- Enable anomaly detection for debugging NaN gradients --- #
    torch.autograd.set_detect_anomaly(True)
    
    # --- Setup Environment --- #
    seed_it_all(args.seed)
    device = get_device()

    # --- Setup output directories and paths for saving results --- #
    logger.info("Setting up directories and loading arguments.")
    run_output_dir, aggregate_results_path, sorted_results_path = setup_results_directories(args)

    # --- Setup Logging --- #
    setup_logging(log_path=run_output_dir / 'training.log')

    # Initial log messages
    log_message = (
        f"Starting training run: {run_output_dir.parent.name}\n"
        f"  Task Type: {args.task_type}\n"
        f"  Model Name: {args.model_name}\n"
        f"  Config Loaded: utils/config.py\n"
        f"  Args: {vars(args)}\n" # Use vars(args) for cleaner dict display
        f"  Results Dir: {run_output_dir}"
    )
    logging.info(log_message)

    # --- Load Data --- # 
    base_data_dir = config.get_processed_data_path(
        bar_type=args.bar_type,
        resample_freq=config.RESAMPLE_FREQ,
        nb_bars=args.nb_bars,
        window_length=config.WINDOW_LENGTH,
        target_window_length=config.TARGET_WINDOW_LENGTH
    )
    logging.info(f"Selected Bar Type: {args.bar_type}")
    logging.info(f"Loading data from base directory: {base_data_dir}")
    logging.info(f"Using Train Days: {config.TRAIN_DAYS}")
    logging.info(f"Using Test Days: {config.TEST_DAYS}")

    # --- Fit Scaler using Utility Function --- #
    logging.info("Attempting to fit and save scaler...")
    scaler = fit_save_scaler_incrementally(
        train_days=config.TRAIN_DAYS,
        base_data_dir=Path(base_data_dir), # Ensure it's a Path object
        input_window_length=config.WINDOW_LENGTH,
        target_window_length=config.TARGET_WINDOW_LENGTH,
        output_dir=run_output_dir, # Pass the specific run output directory
        bar_type=args.bar_type,
        resample_freq=config.RESAMPLE_FREQ,
        nb_bars=args.nb_bars
    )

    if scaler is None:
        logging.error("Scaler fitting failed. Cannot proceed with training.")
        sys.exit(1)
    logging.info("Scaler fitted and saved successfully.")

    # --- Create Datasets with Fitted Scaler --- #
    logging.info(f"Loading final datasets with fitted scaler using '{args.labeling_strategy}' strategy...")
    logging.info(f"Train Days: {config.TRAIN_DAYS}")
    logging.info(f"Test/Validation Days: {config.TEST_DAYS}")

    train_dataset = HFTDataset(
        data_dir=base_data_dir,
        days=config.TRAIN_DAYS,
        input_window_length=config.WINDOW_LENGTH,
        target_window_length=config.TARGET_WINDOW_LENGTH,
        scaler=scaler, 
        labeling_strategy=args.labeling_strategy,
        bar_type=args.bar_type,
        resample_freq=config.RESAMPLE_FREQ,
        nb_bars=args.nb_bars
    )

    test_dataset = HFTDataset(
        data_dir=base_data_dir,
        days=config.TEST_DAYS,
        input_window_length=config.WINDOW_LENGTH,
        target_window_length=config.TARGET_WINDOW_LENGTH,
        scaler=scaler, 
        labeling_strategy=args.labeling_strategy,
        bar_type=args.bar_type,
        resample_freq=config.RESAMPLE_FREQ,
        nb_bars=args.nb_bars
    )
    
    # --- Plot Histogram of Relative Change (from Training Data) --- #
    hist_save_path = run_output_dir / 'relative_change_histogram.png'
    
    # Prepare arguments for histogram plotting
    plot_args = {
        'relative_change': train_dataset.relative_change,
        'labeling_strategy': args.labeling_strategy,
        'save_path': hist_save_path
    }
    
    # Add tercile thresholds if applicable
    if args.labeling_strategy == "tercile":
        if hasattr(train_dataset, 'lower_tercile_threshold') and train_dataset.lower_tercile_threshold is not None:
            plot_args['lower_tercile_threshold'] = train_dataset.lower_tercile_threshold
        if hasattr(train_dataset, 'upper_tercile_threshold') and train_dataset.upper_tercile_threshold is not None:
            plot_args['upper_tercile_threshold'] = train_dataset.upper_tercile_threshold
    
    # --- Add Debug Prints for Relative Change ---    
    if hasattr(train_dataset, 'relative_change') and train_dataset.relative_change is not None:
        try:
            print(f"Relative change stats: Min={np.nanmin(train_dataset.relative_change):.6f}, Max={np.nanmax(train_dataset.relative_change):.6f}, Mean={np.nanmean(train_dataset.relative_change):.6f}, Median={np.nanmedian(train_dataset.relative_change):.6f}")
            print(f"Number of NaN relative changes: {np.isnan(train_dataset.relative_change).sum()}")
            print(f"Number of Inf relative changes after NaN replacement: {np.isinf(train_dataset.relative_change).sum()}") # Should be 0 if handled in dataset
        except Exception as e:
            print(f"Error printing basic relative_change stats: {e}")
        
        wmp_mean_feature_idx = 0 
        if hasattr(train_dataset, 'X_windows') and train_dataset.X_windows is not None and train_dataset.X_windows.ndim == 3 and train_dataset.X_windows.shape[1] > 0 and train_dataset.X_windows.shape[2] > wmp_mean_feature_idx:
            try:
                last_bar_wmp_values = train_dataset.X_windows[:, -1, wmp_mean_feature_idx]
                num_abs_tiny_denominators = np.count_nonzero(np.abs(last_bar_wmp_values) < 1e-5) 
                num_zero_denominators = np.count_nonzero(last_bar_wmp_values == 0)
                print(f"Number of X_windows last bar WMP == 0: {num_zero_denominators}")
                print(f"Number of X_windows last bar WMP with abs value < 1e-5: {num_abs_tiny_denominators}")
            except Exception as e:
                print(f"Error printing denominator stats: {e}")
        else:
            print("Could not perform denominator check: X_windows not in expected state (None, not 3D, or too small).")
            
        if args.labeling_strategy == "tercile":
            if hasattr(train_dataset, 'lower_tercile_threshold') and train_dataset.lower_tercile_threshold is not None:
                 print(f"Lower Tercile Threshold: {train_dataset.lower_tercile_threshold:.6f}")
            if hasattr(train_dataset, 'upper_tercile_threshold') and train_dataset.upper_tercile_threshold is not None:
                 print(f"Upper Tercile Threshold: {train_dataset.upper_tercile_threshold:.6f}")
    else:
        print("train_dataset.relative_change is None or not present. Cannot print stats.")
    # --- End Debug Prints ---
            
    plot_relative_change_histogram(**plot_args)

    #exit()

    # exit() # You can comment this out to proceed with training if stats look good

    # --- Validate Training Data Features --- #
    validate_feature_array(train_dataset.X_windows, "Training Features (X_windows)")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )
    
    val_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device == 'cuda' else False
    )

    # --- Model Initialization --- # 
    if train_dataset.X_windows is None or train_dataset.X_windows.shape[-1] == 0:
         raise ValueError("Training data is empty or has zero features after loading.")
    feature_dim = train_dataset.X_windows.shape[-1] 

    logging.info(f"Initializing {args.model_name} with input_dim={feature_dim}, num_classes={args.output_dim}")
    
    if args.model_name == 'transformer': 
        model = TransformerEncoder(
            input_dim=feature_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            dim_feedforward=args.dim_ff,
            num_layers=args.num_layers,
            num_classes=args.output_dim,
            dropout=args.dropout
        )
    
    elif args.model_name == 'tcn':
        model = TCN(
            input_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            output_dim=args.output_dim,
            num_layers=args.num_layers,
            dropout=args.dropout
        )

    elif args.model_name in ['lstm', 'gru', 'rnn']:
        model = RNN(
            args.model_name, 
            input_dim=feature_dim, 
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers, 
            output_dim=args.output_dim, 
            dropout=args.dropout
            )

    # Move model to device
    model.to(device)

    # --- Optimizer and Loss Function --- #
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()

    # --- Training --- #
    logging.info("Starting model training with early stopping...")
    best_model_save_path = os.path.join(run_output_dir, 'best_model.pt')
    
    # Modified to expect two DataFrames from train_model
    training_epoch_history_df, training_batch_history_df = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader, # Use test set loader for validation
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=args.epochs,
        device=device,
        early_stopping_patience=args.early_stopping_patience,
        save_path=best_model_save_path
    )
    logging.info(f"Model training completed. Best model saved to {best_model_save_path}")

    # --- Save Training History & Plot Metrics --- #
    save_training_history_and_plots(training_epoch_history_df, run_output_dir)
    save_batch_history_plots(training_batch_history_df, run_output_dir)

    # --- Final Evaluation on Test Set --- #
    logging.info("--- Starting Final Evaluation on Test Set ---")
    final_metrics = None # Initialize variable to store metrics if evaluation happens
    
    best_model_path = run_output_dir / 'best_model.pt'

    if best_model_path.exists(): # Check if a best model was actually saved
        logging.info(f"Loading best model weights from {best_model_path} for final evaluation...")
        model.load_state_dict(torch.load(str(best_model_path), map_location=device))
        logging.info("Best model state loaded successfully into model.")

        # --- Evaluate --- #
        test_metrics, test_conf_matrix = evaluate_model(
            model=model, 
            data_loader=val_loader,
            criterion=criterion,
            device=device
        )
        final_metrics = test_metrics 

        # --- Save Individual Run Results --- #
        if args.labeling_strategy == "median" or args.labeling_strategy == "directional":
            class_names = ['Down', 'Up']
        elif args.labeling_strategy == "tercile":
            class_names = ['Down', 'Neutral', 'Up']
            
        
        save_test_results(
            run_output_dir=run_output_dir,
            metrics=test_metrics,
            conf_matrix=test_conf_matrix,
            class_names=class_names 
        )

    # --- Append and Sort Results --- #
    args.run_output_dir = run_output_dir
    append_and_sort_results(
        aggregate_results_path=aggregate_results_path, 
        sorted_results_path=sorted_results_path,     
        args=args,
        metrics=final_metrics,
        sort_metric='macro_f1', 
        ascending=False 
    )

    # --- Delete Saved Model Weights ---
    try:
        # Ensure best_model_path is defined (it should be from earlier)
        if best_model_path.exists():
            best_model_path.unlink() # Delete the file
            logging.info(f"Deleted saved model weights: {best_model_path}")
        else:
            logging.warning(f"Could not delete model weights: File not found at {best_model_path}")
    except Exception as e:
        logging.error(f"Error deleting model weights at {best_model_path}: {e}", exc_info=True)
    # ------------------------------------------ #

    # --- Final Log Messages --- #
    logging.info(f"Training run finished successfully.")
    logging.info(f"Run results saved in: {run_output_dir}")


if __name__ == "__main__":
    cli_args = parse_arguments()
    main(cli_args)