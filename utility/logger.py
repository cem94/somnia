"""
Logger management for the Somnia model.
This module initializes and configures a logger for the Somnia model,
allowing for logging of events and errors during model training and evaluation.

The logger writes to both console and daily log files in the logs directory.
Log files are named by date (YYYY-MM-DD.log) for easy organization.
"""

import os
import logging
import datetime
from utility import paths

# Global logger instance
LOGGER = None


def setup_logger(level=logging.INFO) -> logging.Logger:
    """
    Set up and configure the logger for the Somnia model.
    
    Creates a logger that outputs to both console and a daily log file.
    The log file is automatically named based on the current date.
    This function is called only once to initialize the global LOGGER.
    
    Args:
        level (int): The logging level to set. Default is logging.INFO.
        
    Returns:
        logging.Logger: The configured logger instance.
    """
    global LOGGER
    if LOGGER is None:
        # Ensure the logs directory exists
        os.makedirs(paths.LOGS_DIR, exist_ok=True)
        
        # Create logger
        LOGGER = logging.getLogger('Somnia')
        LOGGER.setLevel(level)
        
        # Create formatters
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(log_format)
        
        # File handler - daily log files
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        log_file_path = os.path.join(paths.LOGS_DIR, f"{current_date}.log")
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        
        # Console handler for immediate feedback
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(level)
        
        # Add handlers to logger
        LOGGER.addHandler(file_handler)
        LOGGER.addHandler(console_handler)
        
    return LOGGER


# Initialize the global logger instance
LOGGER = setup_logger()