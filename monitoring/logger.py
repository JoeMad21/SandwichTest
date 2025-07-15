"""Centralized logging configuration for the project."""

import logging
import sys
import warnings
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from transformers import logging as transformers_logging


def setup_clean_environment():
    """Suppress verbose warnings and configure clean output."""
    # Set environment variables to reduce verbosity
    os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
    
    # Suppress transformers verbosity
    transformers_logging.set_verbosity_error()
    
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    
    # Reduce other library verbosity
    logging.getLogger("accelerate").setLevel(logging.ERROR)
    logging.getLogger("torch.distributed").setLevel(logging.ERROR)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
    logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)


def setup_logger(
    name: str = "rlhf_training",
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    console: bool = True,
    file: bool = True,
    redirect_stdout: bool = True
) -> tuple:
    """Setup a logger with console and file handlers."""
    # Setup clean environment first
    setup_clean_environment()
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Setup log directory
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Console handler - clean format
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    # File handler - detailed format
    if file and log_dir:
        file_handler = logging.FileHandler(log_dir / f'training_{timestamp}.log')
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # Redirect stdout/stderr to files if requested
    original_stdout, original_stderr = None, None
    if redirect_stdout and log_dir:
        original_stdout, original_stderr = setup_output_redirection(log_dir, timestamp)
    
    return logger, original_stdout, original_stderr


def setup_output_redirection(log_dir: Path, timestamp: str):
    """Redirect stdout/stderr to log files while preserving console for logger."""
    
    # Create separate files for stdout and stderr
    stdout_file = open(log_dir / f'console_output_{timestamp}.log', 'w', buffering=1)
    stderr_file = open(log_dir / f'console_errors_{timestamp}.log', 'w', buffering=1)
    
    # Store original streams
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # Create filtered output that only shows logger messages and progress bars on console
    class FilteredOutput:
        def __init__(self, file_handle, console_handle):
            self.file = file_handle
            self.console = console_handle
            
        def write(self, text):
            # Always write to file
            self.file.write(text)
            self.file.flush()
            
            # Only show on console if it's our logger or progress bars
            show_patterns = ['INFO:', 'ERROR:', 'WARNING:', '%|', 'it/s', 'Loading checkpoint']
            if any(pattern in text for pattern in show_patterns):
                self.console.write(text)
                self.console.flush()
                
        def flush(self):
            self.file.flush()
            self.console.flush()
    
    # Replace stdout and stderr
    sys.stdout = FilteredOutput(stdout_file, original_stdout)
    sys.stderr = FilteredOutput(stderr_file, original_stderr)
    
    return original_stdout, original_stderr


def restore_output_redirection(original_stdout, original_stderr):
    """Restore original stdout/stderr."""
    if hasattr(sys.stdout, 'file'):
        sys.stdout.file.close()
    if hasattr(sys.stderr, 'file'):
        sys.stderr.file.close()
        
    sys.stdout = original_stdout
    sys.stderr = original_stderr


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


# Convenience functions
def log_model_info(logger: logging.Logger, model_name: str, num_params: int):
    """Log model information."""
    logger.info(f"🤖 Model: {model_name}")
    logger.info(f"📊 Parameters: {num_params:,}")


def log_training_start(logger: logging.Logger, config: dict):
    """Log training configuration at start."""
    logger.info("🚀 " + "=" * 48)
    logger.info("🎯 Starting PPO training")
    logger.info("⚙️  Configuration:")
    for key, value in config.items():
        logger.info(f"   {key}: {value}")
    logger.info("=" * 50)


def log_training_complete(logger: logging.Logger, total_time: float = None):
    """Log training completion."""
    time_str = f" (Total time: {total_time:.1f}s)" if total_time else ""
    logger.info("🎉 " + "=" * 48)
    logger.info(f"🏁 Training completed{time_str}")
    logger.info("=" * 50)
