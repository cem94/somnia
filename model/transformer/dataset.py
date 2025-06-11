"""
Module for fairy tales dataset processing.

This module provides a PyTorch Dataset class for loading and tokenizing fairy tale stories
for autoregressive language model training. It handles JSONL file loading, tokenization,
and proper error handling for ML pipeline integration.

Usage:
    Initialize FairyTaleDataset() to load stories and tokenizer.
    Use with PyTorch DataLoader for training pipeline integration.
"""

import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utility.logger import LOGGER
from utility.paths import PROCESSED_OUTPUT_FILE, TOKENIZER_DIR
from model.transformer.tokenizer_config import TokenizerConfig


class FairyTaleDataset(Dataset):
    """
    Custom dataset class for loading and tokenizing fairy tale stories.
    
    This dataset handles the loading and preprocessing of fairy tale text data
    for training an autoregressive language model. It uses a provided tokenizer
    and supports proper error handling for ML pipeline integration.
    
    Attributes:
        data (list): List of story texts loaded from the JSONL file.
        tokenizer (AutoTokenizer): Tokenizer for text encoding.
    """
    
    def __init__(self, tokenizer, filepath: str = PROCESSED_OUTPUT_FILE):
        """
        Initialize the FairyTaleDataset.

        Args:
            tokenizer: Tokenizer instance for text encoding.
            filepath: Path to the JSONL file containing the stories.

        Raises:
            FileNotFoundError: If the stories file cannot be found.
            json.JSONDecodeError: If the file contains invalid JSON.
            ValueError: If no stories are loaded from the file.
        """
        self.data = []
        self.tokenizer = tokenizer
        
        self._load_data(filepath)
        
        LOGGER.info(f"Pipeline Stage 3: FairyTaleDataset initialized with {len(self.data)} stories.")

    def _load_data(self, filepath: str) -> None:
        """
        Load story data from JSONL file.
        
        Args:
            filepath: Path to the JSONL file.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If no valid stories are found.
        """
        try:
            with open(filepath, "r", encoding="utf-8") as file:
                for line_num, line in enumerate(file, 1):
                    self.data.append(json.loads(line)["text"])
        except FileNotFoundError:
            LOGGER.error(f"Stories file not found: {filepath}.")
            raise
        except Exception as error:
            LOGGER.error(f"Unexpected error loading data from {filepath}: {error}.")
            raise
            
        if not self.data:
            raise ValueError(f"No valid stories found in {filepath}.")
            
        LOGGER.debug(f"Successfully loaded {len(self.data)} stories from {filepath}.")

    def __len__(self) -> int:
        """
        Return the number of stories in the dataset.
        
        Returns:
            Number of stories available.
        """
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Retrieve and tokenize a story from the dataset.

        Args:
            idx: Index of the story to retrieve.

        Returns:
            Dictionary containing tokenized data:
                - input_ids: Tensor of token IDs (shape: [max_length]).
                - attention_mask: Tensor indicating valid tokens (1) vs padding (0).
                - labels: Tensor of target token IDs for language modeling.

        Raises:
            IndexError: If idx is out of range.
        """
        if idx >= len(self.data) or idx < 0:
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.data)}.")
            
        story_text = self.data[idx]

        try:
            # Tokenize the story with padding and truncation
            encoding = self.tokenizer(
                story_text,
                truncation=True,
                padding="max_length",
                max_length=TokenizerConfig.MAX_SEQ_LEN,
                return_tensors="pt"
            )

            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)
            
            # For causal language modeling, labels are the same as input_ids
            labels = input_ids.clone()

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels
            }
            
        except Exception as error:
            LOGGER.error(f"Error tokenizing story at index {idx}: {error}.")
            raise


def main():
    """
    Main function to demonstrate dataset loading and tokenization.
    """

    LOGGER.info("Pipeline Stage 3: Starting FairyTaleDataset demonstration.")
    
    try:
        # Load tokenizer first
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        
        # Initialize dataset with tokenizer
        dataset = FairyTaleDataset(tokenizer, PROCESSED_OUTPUT_FILE)
        
        LOGGER.info(f"Dataset successfully loaded with {len(dataset)} stories.")
        
        # Test first item tokenization
        if len(dataset) > 0:
            first_item = dataset[0]
            LOGGER.debug("First story tokenization successful.")
            LOGGER.debug(f"Input shape: {first_item['input_ids'].shape}.")
            LOGGER.debug(f"Attention mask shape: {first_item['attention_mask'].shape}.")
            LOGGER.debug(f"Labels shape: {first_item['labels'].shape}.")
            
            # Additional information
            vocab_size = len(dataset.tokenizer)
            max_token_id = first_item['input_ids'].max().item()
            LOGGER.debug(f"Tokenizer vocab size: {vocab_size}, max token ID: {max_token_id}.")
            
        else:
            LOGGER.warning("Dataset is empty - no stories to demonstrate.")
            
        LOGGER.info("Pipeline Stage 3: FairyTaleDataset demonstration completed successfully.")
            
    except Exception as error:
        LOGGER.error(f"Dataset loading failed: {error}.")
        raise


if __name__ == "__main__":
    main()