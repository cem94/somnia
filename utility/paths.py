"""
Path management module for the Somnia project.
This module provides centralized path management and ensures all directory paths
are properly initialized and accessible throughout the project.

All paths are defined as absolute paths relative to the project root directory,
which is automatically detected as two levels up from this file's location.
"""
import os

# Global project root variable
PROJECT_ROOT = None


def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    
    The project root is determined as two directory levels up from this file,
    which should correspond to the main project directory.
    This function is called only once to initialize the global PROJECT_ROOT.
    
    Returns:
        str: The absolute path to the project root directory.
    """
    global PROJECT_ROOT
    if PROJECT_ROOT is None:
        try:
            current_file = os.path.abspath(__file__)
            PROJECT_ROOT = os.path.dirname(os.path.dirname(current_file))
        except NameError:
            # Fix the issue for colab
            PROJECT_ROOT = os.getcwd()

    return PROJECT_ROOT

# Initialize project root directory
PROJECT_ROOT = get_project_root()

# Core directory paths
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")

# Data subdirectories
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
ANALYSIS_DIR = os.path.join(DATA_DIR, "analysis")
PLOTS_DIR = os.path.join(ANALYSIS_DIR, "plots")

# Model subdirectories
TOKENIZER_DIR = os.path.join(MODEL_DIR, "tokenizer")
OUTPUT_DIR = os.path.join(MODEL_DIR, "output")
PLOT_OUTPUT_DIR = os.path.join(MODEL_DIR, "plots")
HYPERPARAMETERS_DIR = os.path.join(MODEL_DIR, "hyperparameters")

# Specific file paths
PROCESSED_OUTPUT_FILE = os.path.join(PROCESSED_DATA_DIR, "training_data.jsonl")

# External configuration paths
KAGGLE_DIR = os.path.expanduser("~/.kaggle")
KAGGLE_CONFIG_FILE = os.path.join(KAGGLE_DIR, "kaggle.json")

# Export model paths
EXPORT_MODEL_DIR = os.path.join(PROJECT_ROOT, "export_device")
EXPORT_OUTPUT_DIR = os.path.join(EXPORT_MODEL_DIR, "android_model")