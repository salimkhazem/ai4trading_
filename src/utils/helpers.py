import logging
import os
import random
import numpy as np
import torch

def setup_logging(log_path: str, level=logging.INFO):
    """Configures logging to file and console.

    Args:
        log_path (str): Path to the log file.
        level: The logging level (e.g., logging.INFO, logging.DEBUG).
    """
    log_dir = os.path.dirname(log_path)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler() # Log to console as well
        ]
    )
    logging.info(f"Logging configured. Log file: {log_path}")

def get_device() -> str:
    """Determines the appropriate device (CUDA or CPU) for PyTorch.

    Returns:
        str: 'cuda' if CUDA is available, otherwise 'cpu'.
    """
    if torch.cuda.is_available():
        device = 'cuda'
        logging.info(f"CUDA available. Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = 'cpu'
        logging.info("CUDA not available. Using CPU.")
    return device

def seed_it_all(seed: int):
    """Sets random seeds for reproducibility.

    Args:
        seed (int): The seed value.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logging.info(f"Set random seed to {seed}") 