"""
Download datasets from Kaggle and organize them into a structured directory.

This module handles downloading and organizing datasets for the Somnia model:
- children_stories: https://www.kaggle.com/datasets/edenbd/children-stories-text-corpus
- fairy_tales: https://www.kaggle.com/datasets/annbengardt/fairy-tales-from-around-the-world

Requirements:
- Kaggle API token configured at ~/.kaggle/kaggle.json
- kaggle package installed: pip install kaggle
"""

import os
import sys
from kaggle import api
from utility.logger import LOGGER
from utility.paths import RAW_DATA_DIR, KAGGLE_CONFIG_FILE

# Dataset configurations
DATASETS = {    
    "children_stories": "edenbd/children-stories-text-corpus",
    "fairy_tales": "annbengardt/fairy-tales-from-around-the-world" 
}


def verify_kaggle_credentials():
    """ 
    Verify that Kaggle API credentials are properly configured.
    
    Checks for the existence of the Kaggle configuration file and validates
    that the API can authenticate successfully.

    Raises:
        FileNotFoundError: If the Kaggle API token file is not found.
        Exception: If API authentication fails.
    """
    if not os.path.exists(KAGGLE_CONFIG_FILE):
        error_msg = (f"Kaggle API token not found at {KAGGLE_CONFIG_FILE}. "
                    "Please download your API token from kaggle.com/account")
        LOGGER.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    LOGGER.info("Kaggle API token file found, verifying authentication...")
    
    try:
        # Test API authentication
        api.authenticate()
        LOGGER.info("Kaggle API authentication successful")
    except Exception as e:
        LOGGER.error(f"Kaggle API authentication failed: {str(e)}")
        raise


def is_dataset_downloaded(dataset_name: str) -> bool:
    """
    Check if a dataset has already been downloaded and contains files.
    
    Args:
        dataset_name (str): Name of the dataset to check.
    
    Returns:
        bool: True if dataset exists and contains files, False otherwise.
    """
    dataset_path = os.path.join(RAW_DATA_DIR, dataset_name)
    
    if not os.path.exists(dataset_path) or not os.path.isdir(dataset_path):
        LOGGER.debug(f"Dataset directory '{dataset_path}' does not exist")
        return False
    
    if not os.listdir(dataset_path):
        LOGGER.debug(f"Dataset directory '{dataset_path}' is empty")
        return False
    
    LOGGER.info(f"Dataset '{dataset_name}' already exists with files")
    return True


def organize_dataset_files(dataset_path: str):
    """
    Move files from subdirectories to the main dataset directory.
    
    This function flattens the directory structure by moving all files
    from subdirectories to the root dataset directory, then removes
    the empty subdirectories.
    
    Args:
        dataset_path (str): Path to the dataset directory to organize.
    """
    LOGGER.debug(f"Organizing files in {dataset_path}")
    
    for item in os.listdir(dataset_path):
        item_path = os.path.join(dataset_path, item)
        
        if os.path.isdir(item_path):
            # Move all files from subdirectory to main directory
            for file_name in os.listdir(item_path):
                source_file = os.path.join(item_path, file_name)
                destination_file = os.path.join(dataset_path, file_name)
                
                if os.path.isfile(source_file):
                    os.rename(source_file, destination_file)
                    LOGGER.debug(f"Moved: {file_name}")
            
            # Remove empty subdirectory
            os.rmdir(item_path)
            LOGGER.debug(f"Removed empty directory: {item}")


def download_dataset(dataset_name: str, dataset_id: str):
    """
    Download and extract a Kaggle dataset if not already present.
    
    Args:
        dataset_name (str): Name for the local dataset directory.
        dataset_id (str): Kaggle dataset identifier (owner/dataset-name).
        
    Raises:
        Exception: If download or extraction fails.
    """
    if is_dataset_downloaded(dataset_name):
        LOGGER.info(f"Dataset '{dataset_name}' already exists, skipping download")
        return

    LOGGER.info(f"Downloading dataset: {dataset_name} ({dataset_id})")
    
    dataset_path = os.path.join(RAW_DATA_DIR, dataset_name)
    os.makedirs(dataset_path, exist_ok=True)
    
    try:
        # Download and unzip dataset
        api.dataset_download_files(dataset_id, path=dataset_path, unzip=True)
        
        # Organize files (flatten directory structure)
        organize_dataset_files(dataset_path)
        
        LOGGER.info(f"Successfully downloaded and organized: {dataset_name}")
        
    except Exception as e:
        LOGGER.error(f"Failed to download dataset '{dataset_name}': {str(e)}")
        raise


def log_dataset_contents(dataset_name: str):
    """
    Log information about the contents of a downloaded dataset.
    
    Args:
        dataset_name (str): Name of the dataset to inspect.
    """
    dataset_path = os.path.join(RAW_DATA_DIR, dataset_name)
    
    try:
        if not os.path.exists(dataset_path):
            LOGGER.warning(f"Dataset directory '{dataset_path}' not found")
            return
            
        files = [f for f in os.listdir(dataset_path) 
                if os.path.isfile(os.path.join(dataset_path, f))]
        
        LOGGER.info(f"Dataset '{dataset_name}' contains {len(files)} files")
        
        if files:
            LOGGER.debug(f"Files in '{dataset_name}': {', '.join(files[:5])}"
                        f"{'...' if len(files) > 5 else ''}")
        
    except Exception as e:
        LOGGER.error(f"Error inspecting dataset '{dataset_name}': {str(e)}")


def main():
    """
    Main function to download all configured datasets from Kaggle.
    
    This function orchestrates the complete download process:
    1. Verify Kaggle API credentials
    2. Download each configured dataset
    3. Log dataset contents for verification
    """
    LOGGER.info("Starting Kaggle dataset download process")
    
    try:
        # Verify API access before starting downloads
        verify_kaggle_credentials()
        
        # Process each dataset
        for dataset_name, dataset_id in DATASETS.items():
            LOGGER.info(f"Processing dataset: {dataset_name}")
            download_dataset(dataset_name, dataset_id)
            log_dataset_contents(dataset_name)

        LOGGER.info("All datasets processed successfully")
        
    except Exception as e:
        LOGGER.error(f"Dataset download process failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()