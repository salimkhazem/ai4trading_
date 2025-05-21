import deepspeed
import gc
import logging
import numpy as np
import pandas as pd  # Added for history DataFrame
import time
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LogisticRegression


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module, # Pass criterion explicitly
    num_epochs: int,
    device: str,
    save_path: str, # Path to save the BEST model
    val_loader: torch.utils.data.DataLoader, # Required again for early stopping
    early_stopping_patience: int, # Re-added
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]: # MODIFIED return type
    """
    Train the classification model with early stopping based on validation loss.
    Logs epoch-level and batch-level metrics.

    Args:
        model: The model to train.
        train_loader: DataLoader for training data.
        optimizer: Optimizer for training.
        criterion: Loss function.
        num_epochs: Maximum number of epochs.
        device: Device ('cuda' or 'cpu').
        save_path: Path to save the BEST model based on validation loss.
        val_loader: DataLoader for validation data (used for early stopping).
        early_stopping_patience: Patience for early stopping.
        scheduler: Optional learning rate scheduler.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - DataFrame with epoch-level training history.
            - DataFrame with batch-level training history.
    """
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    best_train_loss_at_best_val = float('inf')

    epoch_history_data = [] 
    batch_history_data = [] 
    run_output_dir = Path(save_path).parent

    logging.info(f"Starting training for max {num_epochs} epochs with early stopping patience {early_stopping_patience}...")
    logging.info(f"Best model will be saved to: {save_path}")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        batch_train_losses_epoch = [] 
        batch_train_accuracies_epoch = [] 
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()


            # # --- Get the first feature and target --- #

            # inputs_cpu = inputs[:,:,0].cpu().numpy()
            # targets_cpu = targets[:,0].cpu().numpy()

            # # --- Train Logistic Regression --- #
            # lr = LogisticRegression()
            # lr.fit(inputs_cpu, targets_cpu)
            # predictions = lr.predict(inputs_cpu)  # Calculate predictions using logistic regression
            # accuracy = accuracy_score(targets_cpu, predictions)  # Print accuracy
            # print(f"Logistic Regression Accuracy: {accuracy:.4f}")  # Display accuracy


            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            preds = torch.argmax(outputs, dim=1)
            batch_acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            current_batch_loss = loss.item()
            batch_train_losses_epoch.append(current_batch_loss)
            batch_train_accuracies_epoch.append(batch_acc)
            batch_history_data.append({
                'epoch': epoch + 1,
                'batch_idx': batch_idx + 1,
                'batch_train_loss': current_batch_loss,
                'batch_train_accuracy': batch_acc,
                'batch_grad_norm': total_norm
            })
            progress_bar.set_postfix({'train_loss': current_batch_loss, 'train_acc': batch_acc, 'grad_norm': total_norm})
        
        epoch_train_loss = np.mean(batch_train_losses_epoch) if batch_train_losses_epoch else 0.0
        epoch_train_acc = np.mean(batch_train_accuracies_epoch) if batch_train_accuracies_epoch else 0.0
        gc.collect()

        # --- Validation phase --- #
        model.eval()

        batch_val_losses = []
        all_probs = []
        all_preds = []
        all_targets_val = []

        progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for batch_val in progress_bar_val: 
                inputs_val, targets_val = batch_val 
                inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
                outputs_val = model(inputs_val)
                loss_val = criterion(outputs_val, targets_val)
                batch_val_losses.append(loss_val.item())
                probs_val = torch.softmax(outputs_val, dim=1)
                preds_val_batch = torch.argmax(outputs_val, dim=1)
                all_probs.extend(probs_val.cpu().numpy())
                all_preds.extend(preds_val_batch.cpu().numpy())
                all_targets_val.extend(targets_val.cpu().numpy())
                progress_bar_val.set_postfix({'val_loss': loss_val.item()})
        
        epoch_val_loss = np.mean(batch_val_losses) if batch_val_losses else float('inf')
        gc.collect()
        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets_val)
        all_probs_np = np.array(all_probs)
        epoch_val_acc = accuracy_score(all_targets_np, all_preds_np)
        precision, recall, epoch_val_macro_f1, _ = precision_recall_fscore_support(
            all_targets_np, all_preds_np, average='macro', zero_division=0
        )
        try:
             if len(np.unique(all_targets_np)) > 1:
                 if all_probs_np.shape[1] > 2:
                     epoch_val_auc = roc_auc_score(all_targets_np, all_probs_np, multi_class='ovr', average='macro')
                 else:
                     epoch_val_auc = roc_auc_score(all_targets_np, all_probs_np[:, 1])
             else:
                 epoch_val_auc = -1
                 logging.warning(f"Epoch {epoch+1} Val AUC not calculated: Only one class present.")
        except ValueError as e:
             epoch_val_auc = -1
             logging.warning(f"Epoch {epoch+1} Val AUC calculation error: {e}")

        epoch_end_time = time.time()
        log_message = (
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss: {epoch_train_loss:.4f} | "
            f"Train Acc: {epoch_train_acc:.4f} | "
            f"Val Loss: {epoch_val_loss:.4f} | "
            f"Val Acc: {epoch_val_acc:.4f} | "
            f"Val Macro F1: {epoch_val_macro_f1:.4f} | "
            f"Val AUC: {epoch_val_auc:.4f} | "
            f"Time: {epoch_end_time - epoch_start_time:.2f}s"
        )
        logging.info(log_message)
        epoch_history_data.append({
            'epoch': epoch + 1,
            'train_loss': epoch_train_loss,
            'train_accuracy': epoch_train_acc, 
            'val_loss': epoch_val_loss,
            'val_accuracy': epoch_val_acc,
            'val_macro_f1': epoch_val_macro_f1,
            'val_precision_macro': precision,
            'val_recall_macro': recall,
            'val_auc': epoch_val_auc,
            'epoch_time_s': epoch_end_time - epoch_start_time
        })

        if scheduler:
             if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                 scheduler.step(epoch_val_loss)
             else:
                 scheduler.step()

        if epoch_val_loss < best_val_loss:
            logging.info(f"Validation loss improved ({best_val_loss:.4f} -> {epoch_val_loss:.4f}). Saving model to {save_path}")
            best_val_loss = epoch_val_loss
            best_epoch = epoch + 1
            best_train_loss_at_best_val = epoch_train_loss
            patience_counter = 0
            try:
                torch.save(model.state_dict(), save_path)
            except Exception as e:
                logging.error(f"Error saving model: {e}")
        else:
            patience_counter += 1
            logging.info(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")

        if patience_counter >= early_stopping_patience:
            logging.info(f"Early stopping triggered at epoch {epoch + 1} due to no improvement in validation loss for {early_stopping_patience} epochs.")
            break

    logging.info(f"\nTraining finished.")
    if best_epoch > 0:
         logging.info(f"Best model saved at epoch {best_epoch} with validation loss: {best_val_loss:.4f} (Train loss: {best_train_loss_at_best_val:.4f})")
    else:
         logging.warning("Training completed without saving a best model (validation loss never improved or validation failed).")

    training_epoch_history_df = pd.DataFrame(epoch_history_data)
    
    batch_history_df = pd.DataFrame() # Initialize as empty
    if batch_history_data: 
        batch_history_df = pd.DataFrame(batch_history_data)
        batch_history_csv_path = run_output_dir / 'training_batch_history.csv'
        try:
            batch_history_df.to_csv(batch_history_csv_path, index=False)
            logging.info(f"Batch-level training history saved to {batch_history_csv_path}")
        except Exception as e:
            logging.error(f"Failed to save batch-level training history: {e}")

    return training_epoch_history_df, batch_history_df


def evaluate_model(
    model: nn.Module,
    data_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = 'cuda',
) -> Tuple[Dict[str, float], np.ndarray]: # Return metrics and confusion matrix
    """
    Evaluate the classification model on a dataset.

    Args:
        model: The trained model (state dict should be loaded).
        data_loader: DataLoader for the evaluation data.
        criterion: The loss function.
        device: Device ('cuda' or 'cpu').

    Returns:
        Tuple containing:
         - Dictionary of metrics (loss, accuracy, precision, recall, f1, auc).
         - Confusion matrix (numpy array).
    """
    model.eval()
    model.to(device)
    all_probs = []
    all_preds = []
    all_targets = []
    total_loss = 0.0

    logging.info("Starting evaluation...")
    progress_bar = tqdm(data_loader, desc="Evaluating", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset) if len(data_loader.dataset) > 0 else 0
    all_probs_np = np.array(all_probs)
    all_preds_np = np.array(all_preds)
    all_targets_np = np.array(all_targets)

    all_targets_np = np.array(all_targets_np)
    all_preds_np = np.array(all_preds_np)

    accuracy = accuracy_score(all_targets_np, all_preds_np)
    # Use macro average for balanced classes
    precision, recall, macro_f1, _ = precision_recall_fscore_support(
        all_targets_np, all_preds_np, average='macro', zero_division=0
    )
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(all_targets_np, all_preds_np)

    try:
        if len(np.unique(all_targets_np)) > 1:
            if all_probs_np.shape[1] > 2:
                auc = roc_auc_score(all_targets_np, all_probs_np, multi_class='ovr', average='macro')
            else:
                auc = roc_auc_score(all_targets_np, all_probs_np[:, 1])
        else:
            auc = -1
            logging.warning("Evaluation AUC not calculated: Only one class present.")
    except ValueError as e:
        auc = -1
        logging.warning(f"Evaluation AUC calculation error: {e}")

    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision_macro': precision,
        'recall_macro': recall,
        'macro_f1_score': macro_f1,
        'auc': auc
    }
    # Log metrics and confusion matrix
    logging.info(f"Evaluation Metrics: {metrics}")
    logging.info(f"Confusion Matrix:\n{conf_matrix}")

    gc.collect()
    # Return both metrics dict and confusion matrix array
    return metrics, conf_matrix


def train_model_deepspeed(
    model_engine: deepspeed.runtime.engine.DeepSpeedEngine,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    num_epochs: int,
    early_stopping_patience: int, # Keep arg for compatibility, but ignore
    checkpoint_dir: Path,
) -> pd.DataFrame:
    
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

    history = []

    if rank == 0:
        logging.info(f"Starting DeepSpeed training for {num_epochs} epochs (simple checkpointing)...")
        logging.info(f"Checkpoints will be saved in: {checkpoint_dir}")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # --- Training phase --- #
        model_engine.train()
        batch_train_losses = []
        batch_train_accuracies = []
        # Set epoch for DistributedSampler (important for shuffling)
        train_loader.sampler.set_epoch(epoch)

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False, disable=(rank != 0))
        for batch in progress_bar:
            inputs, targets = batch
            # Data loading handled by DeepSpeed engine's move_to_device if needed, or manual
            inputs = inputs.to(model_engine.local_rank)
            targets = targets.to(model_engine.local_rank)

            outputs = model_engine(inputs)
            loss = criterion(outputs, targets)

            # DeepSpeed backward pass
            model_engine.backward(loss)

            # DeepSpeed step (handles optimizer step, gradient clipping, scheduler)
            model_engine.step()

            # Calculate batch metrics (on each rank for potential logging/debugging)
            # Note: These are batch metrics on this rank's micro-batch
            batch_loss_cpu = loss.item()
            batch_train_losses.append(batch_loss_cpu)
            with torch.no_grad(): # Accuracy calculation
                preds = torch.argmax(outputs, dim=1)
                batch_acc = accuracy_score(targets.cpu().numpy(), preds.cpu().numpy())
                batch_train_accuracies.append(batch_acc)

            if rank == 0:
                # Grad norm is harder to get easily with DeepSpeed ZeRO > 0
                progress_bar.set_postfix({'train_loss': batch_loss_cpu, 'train_acc': batch_acc})

        # Average training metrics across batches FOR THIS RANK
        epoch_rank_train_loss = np.mean(batch_train_losses) if batch_train_losses else 0.0
        epoch_rank_train_acc = np.mean(batch_train_accuracies) if batch_train_accuracies else 0.0
        gc.collect() # Optional: collect garbage

        # --- Validation phase (Rank 0) --- #
        epoch_val_loss = float('inf')
        epoch_val_acc = 0.0
        epoch_val_macro_f1 = 0.0
        epoch_val_auc = -1.0
        precision = 0.0
        recall = 0.0
        if rank == 0:
            batch_val_losses = []
            all_probs = []
            all_preds = []
            all_targets_val = []
            progress_bar_val = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
            with torch.no_grad():
                for batch in progress_bar_val:
                    inputs, targets = batch
                    # Move validation data to rank 0's device
                    inputs = inputs.to(model_engine.local_rank)
                    targets = targets.to(model_engine.local_rank)

                    outputs = model_engine(inputs)
                    loss = criterion(outputs, targets)
                    batch_val_losses.append(loss.item())

                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    all_probs.extend(probs.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_targets_val.extend(targets.cpu().numpy())
                    progress_bar_val.set_postfix({'val_loss': loss.item()})

            epoch_val_loss = np.mean(batch_val_losses) if batch_val_losses else float('inf')
            gc.collect()

            # Calculate validation metrics on rank 0
            if all_targets_val:
                all_preds_np = np.array(all_preds)
                all_targets_np = np.array(all_targets_val)
                all_probs_np = np.array(all_probs)

                epoch_val_acc = accuracy_score(all_targets_np, all_preds_np)
                precision, recall, epoch_val_macro_f1, _ = precision_recall_fscore_support(
                    all_targets_np, all_preds_np, average='macro', zero_division=0
                )
                try:
                     if len(np.unique(all_targets_np)) > 1:
                         if all_probs_np.shape[1] > 2:
                             epoch_val_auc = roc_auc_score(all_targets_np, all_probs_np, multi_class='ovr', average='macro')
                         else:
                             epoch_val_auc = roc_auc_score(all_targets_np, all_probs_np[:, 1])
                     else: epoch_val_auc = -1
                except ValueError: epoch_val_auc = -1
            else:
                 logging.warning(f"Epoch {epoch+1} Rank 0: No validation targets collected.")

        # --- Synchronize Validation Loss (still useful for logging) --- #
        if world_size > 1:
            val_loss_tensor = torch.tensor(epoch_val_loss, device=model_engine.local_rank)
            torch.distributed.broadcast(val_loss_tensor, src=0)
            epoch_val_loss_synced = val_loss_tensor.item()
        else:
            epoch_val_loss_synced = epoch_val_loss

        # --- Logging and History (Rank 0) --- #
        epoch_end_time = time.time()
        if rank == 0:
            log_message = (
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss (Rank 0): {epoch_rank_train_loss:.4f} | " # Note: This is Rank 0's avg batch loss
                f"Train Acc (Rank 0): {epoch_rank_train_acc:.4f} | "
                f"Val Loss (Sync): {epoch_val_loss_synced:.4f} | "
                f"Val Acc: {epoch_val_acc:.4f} | "
                f"Val Macro F1: {epoch_val_macro_f1:.4f} | "
                f"Val AUC: {epoch_val_auc:.4f} | "
                f"Time: {epoch_end_time - epoch_start_time:.2f}s"
            )
            logging.info(log_message)
            history.append({
                'epoch': epoch + 1,
                'train_loss': epoch_rank_train_loss, # Store rank 0 train loss
                'train_accuracy': epoch_rank_train_acc, # Store rank 0 train acc
                'val_loss': epoch_val_loss_synced,
                'val_accuracy': epoch_val_acc,
                'val_macro_f1': epoch_val_macro_f1,
                'val_precision_macro': precision,
                'val_recall_macro': recall,
                'val_auc': epoch_val_auc,
                'epoch_time_s': epoch_end_time - epoch_start_time
            })

        # --- Unconditional Checkpoint Saving --- #
        save_tag = f"epoch_{epoch+1}" # Simple tag

        # Barrier before saving
        if world_size > 1:
            logging.info(f"Rank {rank}: Reaching barrier just before save_checkpoint with tag {save_tag}.")
            torch.distributed.barrier()
            logging.info(f"Rank {rank}: Passed barrier just before save_checkpoint.")

        # All ranks call save_checkpoint with the simple, identical tag
        try:
            logging.info(f"Rank {rank}: Calling save_checkpoint with tag: {save_tag}")
            model_engine.save_checkpoint(checkpoint_dir, save_tag)
            logging.info(f"Rank {rank}: Finished save_checkpoint call.")
        except Exception as e:
            logging.error(f"Rank {rank}: Error during save_checkpoint: {e}", exc_info=True)
            break # Stop training on save error

        # --- Barrier before next epoch --- #
        if world_size > 1:
            logging.info(f"Rank {rank}: Reaching end-of-epoch barrier.")
            torch.distributed.barrier()
            logging.info(f"Rank {rank}: Passed end-of-epoch barrier.")

    # --- End of Training --- #
    if rank == 0:
        logging.info(f"\nDeepSpeed training finished ({num_epochs} epochs completed).")
        training_history_df = pd.DataFrame(history)
        return training_history_df
    else:
        return pd.DataFrame()
