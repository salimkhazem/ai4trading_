# train_deepspeed.py
import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import torch
import deepspeed # Import deepspeed
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler # Import DistributedSampler

# Project utilities
import utils.config as config
from dataset.hft_dataset import HFTDataset
from models import TransformerEncoder, RNN, TCN
from trainers.trainer import train_model_deepspeed, evaluate_model
from utils.helpers import setup_logging, seed_it_all 
from utils.results_handler import setup_results_directories, save_test_results, append_and_sort_results, save_training_history_and_plots
from utils.preprocessing_utils import fit_save_scaler_incrementally, validate_feature_array
from utils.plotting import plot_relative_change_histogram

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments for the DeepSpeed HFT model training script."""
    parser = argparse.ArgumentParser(description='Train a deep learning model for HFT binary prediction using DeepSpeed.')

    # --- Keep most arguments, but remove ones handled by DeepSpeed config --- #
    # General settings
    parser.add_argument('--task_type', type=str, default='classification', choices=['classification', 'regression'], help='Type of task')
    parser.add_argument('--labeling_strategy', type=str, default='median', choices=['median', 'directional'], help='Strategy for label generation in HFTDataset')

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
    # train_params.add_argument('--batch_size', type=int, default=128, help='Global batch size (handled by DeepSpeed config: train_batch_size)') # Commented out - Use DeepSpeed config
    train_params.add_argument('--lr', type=float, default=1e-4, help='Learning rate (handled by DeepSpeed config)') # Commented out
    # train_params.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay (handled by DeepSpeed config)') # Commented out
    train_params.add_argument('--early_stopping_patience', type=int, default=10, help='Patience for early stopping based on validation loss')
    train_params.add_argument('--num_workers', type=int, default=4, help='Number of data loader workers')
    train_params.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')

    # --- Add DeepSpeed arguments --- #
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank passed from distributed launcher")
    # Include DeepSpeed configuration options
    parser = deepspeed.add_config_arguments(parser)
    # ---------------------------- #

    args = parser.parse_args()
    logger.info(f"Arguments parsed: {vars(args)}")
    return args

def main(args: argparse.Namespace):

    # --- Must initialize distributed backend --- #
    deepspeed.init_distributed(dist_backend='nccl')
    # ----------------------------------------- #

    # --- Setup Environment --- #
    seed_it_all(args.seed)

    # --- Determine output directories and paths (all ranks) --- #
    # Call setup_results_directories on all ranks to get the path,
    # but only rank 0 will actually create/log within setup_results_directories.
    logger.info(f"Rank {args.local_rank}: Determining output directories...")
    run_output_dir, aggregate_results_path, sorted_results_path = setup_results_directories(args)
    logger.info(f"Rank {args.local_rank}: Determined run_output_dir: {run_output_dir}")

    # --- Setup Logging & Create Dirs (only on rank 0) --- #
    if args.local_rank == 0:
        logger.info("Rank 0: Setting up logging...")
        # setup_results_directories already created the dir if needed,
        # but it's safe to ensure it exists again.
        run_output_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(log_path=run_output_dir / 'training.log')
        # Log initial messages (only rank 0)
        log_message = (
            f"Starting DeepSpeed training run: {run_output_dir.parent.name}\n"
            f"  Task Type: {args.task_type}\n"
            f"  Model Name: {args.model_name}\n"
            f"  Config Loaded: utils/config.py\n"
            f"  DeepSpeed Config: {args.deepspeed_config}\n"
            f"  Args: {vars(args)}\n"
            f"  Results Dir: {run_output_dir}"
        )
        logging.info(log_message)
    else:
        pass

    # --- Load Data Config --- #
    base_data_dir = config.get_processed_data_path()
    if args.local_rank == 0:
        logging.info(f"Rank 0: Loading data from base directory: {base_data_dir}")
        logging.info(f"Rank 0: Using Train Days: {config.TRAIN_DAYS}")
        logging.info(f"Rank 0: Using Test Days: {config.TEST_DAYS}")

    # --- Fit Scaler (Run only on Rank 0, then broadcast/load) --- #
    scaler = None
    # Construct scaler path using the determined run_output_dir (all ranks know this)
    scaler_save_path = run_output_dir / 'fitted_scaler.pkl'

    if args.local_rank == 0:
        logging.info("Rank 0: Attempting to fit and save scaler...")
        scaler = fit_save_scaler_incrementally(
            train_days=config.TRAIN_DAYS,
            base_data_dir=Path(base_data_dir),
            input_window_length=config.WINDOW_LENGTH,
            target_window_length=config.TARGET_WINDOW_LENGTH,
            output_dir=run_output_dir # Pass determined output dir
        )
        if scaler is None:
            logging.error("Scaler fitting failed on Rank 0. Aborting.")
            # Need a way to signal other ranks to exit cleanly
            # Using sys.exit(1) might work if caught by launcher
            sys.exit(1)
        logging.info("Scaler fitted and saved successfully by Rank 0.")

    # Barrier: Wait for Rank 0 to finish fitting/saving the scaler
    # Use torch barrier AFTER the main work of rank 0 is done.
    if torch.distributed.is_initialized():
        logger.info(f"Rank {args.local_rank}: Waiting at barrier after scaler fit/save.")
        torch.distributed.barrier()
        logger.info(f"Rank {args.local_rank}: Passed barrier after scaler fit/save.")

    # Load scaler on non-rank 0 processes
    if args.local_rank != 0:
        logger.info(f"Rank {args.local_rank}: Attempting to load scaler from {scaler_save_path}")
        if scaler_save_path and scaler_save_path.exists():
            try:
                with open(scaler_save_path, 'rb') as f:
                    scaler = pickle.load(f)
                logging.info(f"Scaler loaded successfully on Rank {args.local_rank}.")
            except Exception as e:
                 logging.error(f"Rank {args.local_rank} failed to load scaler: {e}", exc_info=True)
                 sys.exit(1)
        else:
            logging.error(f"Scaler file not found at {scaler_save_path} on Rank {args.local_rank}. Aborting.")
            sys.exit(1)
    elif scaler is None: # Extra check for rank 0 if fitting failed but didn't exit
        logging.error("Rank 0: Scaler is None after fitting attempt and barrier. Aborting.")
        sys.exit(1)

    # --- Create Datasets (all ranks) --- #
    if args.local_rank == 0:
        logging.info(f"Rank 0: Loading final datasets with fitted scaler using '{args.labeling_strategy}' strategy...")

    try:
        train_dataset = HFTDataset(
            data_dir=base_data_dir,
            days=config.TRAIN_DAYS,
            input_window_length=config.WINDOW_LENGTH,
            target_window_length=config.TARGET_WINDOW_LENGTH,
            scaler=scaler,
            labeling_strategy=args.labeling_strategy
        )
        test_dataset = HFTDataset(
            data_dir=base_data_dir,
            days=config.TEST_DAYS,
            input_window_length=config.WINDOW_LENGTH,
            target_window_length=config.TARGET_WINDOW_LENGTH,
            scaler=scaler,
            labeling_strategy=args.labeling_strategy
        )
    except Exception as e:
        # Use local_rank in error message
        logging.error(f"Error creating datasets on Rank {args.local_rank}: {e}", exc_info=True)
        sys.exit(1)

    # --- Plot Histogram (only on Rank 0) --- #
    if args.local_rank == 0:
        hist_save_path = run_output_dir / 'relative_change_histogram.png'
        plot_relative_change_histogram(
            relative_change=train_dataset.relative_change,
            labeling_strategy=args.labeling_strategy,
            save_path=hist_save_path
        )

    # --- Validate Training Data Features (on all ranks after scaling) --- #
    try:
        validate_feature_array(train_dataset.X_windows, f"Training Features Rank {args.local_rank}")
    except ValueError as e:
        sys.exit(1)

    # --- Model Initialization (before DeepSpeed init) --- #
    feature_dim = train_dataset.X_windows.shape[-1]
    if args.local_rank == 0:
        logging.info(f"Rank 0: Initializing {args.model_name} with input_dim={feature_dim}, num_classes={args.output_dim}")

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
    else:
        raise ValueError(f"Unknown model type: {args.model_name}")

    # --- DeepSpeed Initialization (creates model_engine) --- #
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters()
    )
    criterion = torch.nn.CrossEntropyLoss()
    device = model_engine.local_rank

    # --- Create DataLoaders with Distributed Sampler --- #
    # Access micro batch size *after* deepspeed.initialize() from model_engine
    effective_micro_batch_size = model_engine.train_micro_batch_size_per_gpu()
    logger.info(f"Rank {args.local_rank}: Using effective micro batch size per GPU: {effective_micro_batch_size}")

    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=effective_micro_batch_size, # Use batch size from model_engine
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        test_dataset,
        batch_size=effective_micro_batch_size, # Use the same for validation
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --- Training --- #
    if args.local_rank == 0:
        logging.info("Rank 0: Starting model training with DeepSpeed...")

    training_history_df = train_model_deepspeed(
        model_engine=model_engine,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        num_epochs=args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=run_output_dir / "deepspeed_checkpoints"
    )

    # --- Save History & Plot Metrics (only on Rank 0) --- #
    if args.local_rank == 0:
         logging.info("Rank 0: Training finished. Saving history and plots...")
         save_training_history_and_plots(training_history_df, run_output_dir)

    # --- Final Evaluation on Test Set (only on Rank 0) --- #
    final_metrics = None
    test_conf_matrix = None
    if args.local_rank == 0:
        logging.info("--- Rank 0: Starting Final Evaluation on Test Set ---")

        # Determine best checkpoint tag
        best_tag = None
        best_tag_file = run_output_dir / "deepspeed_checkpoints" / "best_checkpoint_tag.txt"
        if best_tag_file.exists():
            with open(best_tag_file, 'r') as f:
                best_tag = f.read().strip()
            logging.info(f"Rank 0: Found best checkpoint tag: {best_tag}")
        else:
            logging.warning(f"Rank 0: Best checkpoint tag file not found at {best_tag_file}. Cannot load best model for evaluation.")

        if best_tag:
            best_checkpoint_path = run_output_dir / "deepspeed_checkpoints" / best_tag
            try:
                logger.info(f"Rank 0: Loading checkpoint '{best_tag}' from {run_output_dir / 'deepspeed_checkpoints'}")
                load_path, client_state = model_engine.load_checkpoint(run_output_dir / "deepspeed_checkpoints", best_tag)
                if load_path is None:
                    logging.error(f"Rank 0: Could not load best checkpoint using tag '{best_tag}'")
                else:
                    logging.info(f"Rank 0: Best DeepSpeed checkpoint loaded from {load_path}")
                    test_metrics, test_conf_matrix = evaluate_model(
                        model=model_engine,
                        data_loader=val_loader,
                        criterion=criterion,
                        device=device
                    )
                    final_metrics = test_metrics
            except Exception as e:
                 logging.error(f"Rank 0: Error loading or evaluating best DeepSpeed checkpoint: {e}", exc_info=True)

        # --- Save Individual Run Results (only on Rank 0) --- #
        if final_metrics is not None and test_conf_matrix is not None:
            save_test_results(
                run_output_dir=run_output_dir,
                metrics=final_metrics,
                conf_matrix=test_conf_matrix,
                class_names=['Down', 'Up']
            )
        else:
             logging.warning("Rank 0: Final metrics or confusion matrix not available. Skipping saving test results.")

        # --- Append and Sort Results (only on Rank 0) --- #
        if final_metrics is not None and aggregate_results_path is not None:
            args.run_output_dir = run_output_dir # Ensure path is set for handler
            append_and_sort_results(
                aggregate_results_path=aggregate_results_path,
                sorted_results_path=sorted_results_path,
                args=args,
                metrics=final_metrics,
                sort_metric='test_macro_f1_score', # Ensure this key exists in final_metrics
                ascending=False
            )
        else:
             logging.warning("Rank 0: Final metrics or aggregate path not available. Skipping appending results.")

        # Optional: Delete checkpoints

        logging.info(f"Rank 0: DeepSpeed training run finished successfully.")
        logging.info(f"Rank 0: Run results saved in: {run_output_dir}")

    # Barrier at the end to ensure all processes finish cleanly
    if torch.distributed.is_initialized():
        torch.distributed.barrier()

if __name__ == "__main__":
    cli_args = parse_arguments()
    main(cli_args) 