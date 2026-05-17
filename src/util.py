import torch
import sys
import signal
import logging
import psutil

def signal_handler(sig, frame):
    """Handle Ctrl-C gracefully."""
    logging.info("\n\nShutting down gracefully...")
    sys.exit(0)

def setup_signal_handlers():
    signal.signal(signal.SIGINT, signal_handler)

def get_device() -> str:
    if torch.backends.mps.is_available():
        logging.debug("Using MPS (Metal Performance Shaders)")
        return "mps"
    elif torch.cuda.is_available():
        logging.debug("Using CUDA")
        return "cuda"
    else:
        logging.debug("Using CPU")
        return "cpu"

def print_memory_usage():
    memory = psutil.virtual_memory()
    logging.info(f"System RAM: {memory.percent}% used ({memory.used / 1024**3:.1f}GB / {memory.total / 1024**3:.1f}GB)")
    if torch.backends.mps.is_available():
        logging.info("MPS is being used for acceleration")
