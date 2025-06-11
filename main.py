"""
Entry point for the Somnia Model pipeline.
This script orchestrates the data processing, tokenizer training, and model training for the Somnia Model.

Pipeline stages:
1. Data Processing: Fetch raw data, prepare datasets, and perform analysis.
2. Tokenizer Training: Train custom tokenizer on processed data.
3. Model Training: Train the main Somnia model using prepared data and tokenizer.
4. Model Export: Export the trained model for deployment on target devices.
"""

import time
from data.fetch import fetch_data
from data.analysis import analysis
from data.processed import prepare_dataset
from model.train_tokenizer import main as train_tokenizer_main
from model.hyperparameter import main as train_model_main
from export_device.export_model import main as export_model_main
from utility.logger import LOGGER


def data_processing():
    """
    Execute the data processing pipeline.
    
    This function performs three sequential operations:
    1. Fetch raw data from configured sources.
    2. Prepare and clean the dataset for training.
    3. Perform exploratory data analysis.
    """
    start_time = time.time()
    LOGGER.info("Pipeline Stage 1: Starting data processing.")
    
    LOGGER.info("Step 1/3: Fetching raw data.")
    fetch_data.main()
    LOGGER.info("Raw data fetched successfully.")
    
    LOGGER.info("Step 2/3: Preparing the dataset.")
    prepare_dataset.main()
    LOGGER.info("Dataset preparation completed successfully.")
    
    LOGGER.info("Step 3/3: Performing data analysis.")
    analysis.main()
    LOGGER.info("Data analysis completed successfully.")
    
    duration = time.time() - start_time
    LOGGER.info(f"Pipeline Stage 1 completed in {duration:.2f} seconds.")


def train_tokenizer():
    """
    Train the custom tokenizer for the Somnia model.
    
    The tokenizer is trained on the processed dataset and creates
    the vocabulary needed for model training.
    """
    start_time = time.time()
    LOGGER.info("Pipeline Stage 2: Starting tokenizer training.")
    
    train_tokenizer_main()
    LOGGER.info("Tokenizer training completed successfully.")
    
    duration = time.time() - start_time
    LOGGER.info(f"Pipeline Stage 2 completed in {duration:.2f} seconds.")


def train_model():
    """
    Train the main Somnia model.
    
    This function trains the neural network model using the processed
    data and the previously trained tokenizer.
    """
    start_time = time.time()
    LOGGER.info("Pipeline Stage 3: Starting model training.")
    
    train_model_main(skip_hyperparameter_search=False)
    LOGGER.info("Model training completed successfully.")
    
    duration = time.time() - start_time
    LOGGER.info(f"Pipeline Stage 3 completed in {duration:.2f} seconds.")


def export_model():
    """
    Export the trained Somnia model.
    
    This function handles the model export process, preparing it
    for deployment or inference on target devices.
    """
    start_time = time.time()
    LOGGER.info("Pipeline Stage 4: Starting model export.")
    
    export_model_main()
    LOGGER.info("Model export completed successfully.")
    
    duration = time.time() - start_time
    LOGGER.info(f"Pipeline Stage 4 completed in {duration:.2f} seconds.")


def main():
    """
    Main execution function for the Somnia Model pipeline.
    
    Orchestrates the complete training pipeline by running data processing,
    tokenizer training, and model training in sequence.
    """
    pipeline_start = time.time()
    LOGGER.info("Starting Somnia Model pipeline execution.")
    
    #data_processing()
    #train_tokenizer()
    train_model()
    #export_model()
    
    total_duration = time.time() - pipeline_start
    LOGGER.info(f"Somnia Model pipeline execution completed successfully in {total_duration:.2f} seconds.")

if __name__ == "__main__":
    main()