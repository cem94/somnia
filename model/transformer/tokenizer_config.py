"""
Configuration module for the tokenizer used in the LLaMA-based language model.

This module provides the configuration class for the tokenizer,
including vocabulary settings, special tokens, and tokenizer parameters.
"""

import os
from typing import Dict, Any
from utility.logger import LOGGER
from utility.paths import TOKENIZER_DIR, PROCESSED_OUTPUT_FILE


class TokenizerConfig:
    """
    Configuration class for the tokenizer used in the LLaMA-based language model.

    This class stores the tokenizer's vocabulary size, special token IDs,
    and provides methods to estimate the tokenizer size and print a summary
    of the configuration.

    Attributes:
        SEED (int): Random seed for reproducibility.
        VOCAB_SIZE (int): Size of the vocabulary.
        MAX_SEQ_LEN (int): Maximum sequence length.
        
        # Special token IDs and values
        BOS_TOKEN_ID (int): Beginning of sequence token ID.
        BOS_TOKEN_VALUE (str): Beginning of sequence token value.
        EOS_TOKEN_ID (int): End of sequence token ID.
        EOS_TOKEN_VALUE (str): End of sequence token value.
        UNK_TOKEN_ID (int): Unknown token ID.
        UNK_TOKEN_VALUE (str): Unknown token value.
        PAD_TOKEN_ID (int): Padding token ID.
        PAD_TOKEN_VALUE (str): Padding token value.
        
        dataset_path (str): Path to the dataset file.
        output_dir (str): Output directory for tokenizer files.
        evaluate (bool): Whether to evaluate tokenizer performance.
        config (Dict[str, Any]): Complete tokenizer configuration dictionary.
    """

    # Global constants - centralized configuration used across the project
    SEED = 42
    VOCAB_SIZE = 20000
    MAX_SEQ_LEN = 2048

    # Special token IDs and values - standardized across all components
    BOS_TOKEN_ID = 0
    BOS_TOKEN_VALUE = "<s>"
    EOS_TOKEN_ID = 1
    EOS_TOKEN_VALUE = "</s>"
    UNK_TOKEN_ID = 2
    UNK_TOKEN_VALUE = "<unk>"
    PAD_TOKEN_ID = 3
    PAD_TOKEN_VALUE = "<pad>"

    def __init__(self):
        """Initialize the tokenizer configuration with predefined values."""
        # Set configuration paths and parameters
        self.dataset_path = PROCESSED_OUTPUT_FILE
        self.output_dir = TOKENIZER_DIR
        self.evaluate = True
        
        # Validate configuration
        self._validate_configuration()
        
        # Build tokenizer configuration dictionary
        self.config = self._build_tokenizer_config()

    def _validate_configuration(self) -> None:
        """Validate the configuration parameters."""
        LOGGER.debug("Starting tokenizer configuration validation.")
        
        # Validate constants
        assert self.VOCAB_SIZE > 0, f"Vocabulary size must be positive, got {self.VOCAB_SIZE}."
        assert self.MAX_SEQ_LEN > 0, f"Max sequence length must be positive, got {self.MAX_SEQ_LEN}."
        assert self.SEED >= 0, f"Seed must be non-negative, got {self.SEED}."
        
        # Validate special token IDs are unique and non-negative
        token_ids = [self.BOS_TOKEN_ID, self.EOS_TOKEN_ID, self.UNK_TOKEN_ID, self.PAD_TOKEN_ID]
        assert all(tid >= 0 for tid in token_ids), "All token IDs must be non-negative."
        assert len(set(token_ids)) == len(token_ids), "All token IDs must be unique."
        assert all(tid < self.VOCAB_SIZE for tid in token_ids), "All token IDs must be within vocabulary size."
        
        LOGGER.debug("Token configuration validated successfully.")
        
        # Check if dataset file exists
        if not os.path.exists(self.dataset_path):
            LOGGER.warning(f"Dataset path does not exist: {self.dataset_path}.")
        else:
            LOGGER.debug(f"Dataset path validated: {self.dataset_path}.")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        LOGGER.debug(f"Output directory validated/created: {self.output_dir}.")
        
        LOGGER.info("Tokenizer configuration validation completed successfully.")

    def _build_tokenizer_config(self) -> Dict[str, Any]:
        """
        Build the tokenizer configuration dictionary.
        
        Returns:
            Dict[str, Any]: Complete tokenizer configuration.
        """
        config = {
            "add_bos_token": False,
            "add_eos_token": False,
            "add_prefix_space": False,
            "bos_token": self.BOS_TOKEN_VALUE,
            "eos_token": self.EOS_TOKEN_VALUE,
            "unk_token": self.UNK_TOKEN_VALUE,
            "pad_token": self.PAD_TOKEN_VALUE,
            "model_max_length": self.MAX_SEQ_LEN,
            "tokenizer_class": "PreTrainedTokenizerFast",
            "clean_up_tokenization_spaces": False,
            "additional_special_tokens": [],
            "spaces_between_special_tokens": False,
            "sp_model_kwargs": {},
            "added_tokens_decoder": self._build_special_tokens_decoder()
        }
        
        LOGGER.debug("Tokenizer configuration built successfully.")
        return config

    def _build_special_tokens_decoder(self) -> Dict[int, Dict[str, Any]]:
        """
        Build the special tokens decoder configuration.
        
        Returns:
            Dict[int, Dict[str, Any]]: Special tokens decoder mapping.
        """
        special_tokens = [
            (self.BOS_TOKEN_ID, self.BOS_TOKEN_VALUE),
            (self.EOS_TOKEN_ID, self.EOS_TOKEN_VALUE),
            (self.UNK_TOKEN_ID, self.UNK_TOKEN_VALUE),
            (self.PAD_TOKEN_ID, self.PAD_TOKEN_VALUE)
        ]
        
        decoder = {}
        for token_id, token_value in special_tokens:
            decoder[token_id] = {
                "content": token_value,
                "lstrip": False,
                "normalized": False,
                "rstrip": False,
                "single_word": False,
                "special": True
            }
        
        LOGGER.debug(f"Special tokens decoder built for {len(decoder)} tokens.")
        return decoder

    def estimate_model_size_mb(self) -> float:
        """
        Estimate the size of the tokenizer in megabytes (MB).
        
        This method calculates the size based on the vocabulary size
        and configuration data using more accurate estimations.
        
        Returns:
            float: Estimated tokenizer size in MB.
        """
        # Vocabulary mapping with more realistic token length estimation
        # BPE tokens typically range from 1-15 characters, avg ~6 characters
        avg_token_length = 6
        vocab_size_bytes = self.VOCAB_SIZE * (4 + avg_token_length)  # 4 bytes for ID + avg token length
        
        # Configuration data
        config_size_bytes = len(str(self.config).encode('utf-8'))
        
        # Special tokens decoder
        decoder_size_bytes = len(str(self.config.get('added_tokens_decoder', {})).encode('utf-8'))
        
        # Additional overhead for tokenizer files (merges, etc.)
        overhead_bytes = self.VOCAB_SIZE * 2  # Approximate overhead for BPE merges
        
        total_size_bytes = vocab_size_bytes + config_size_bytes + decoder_size_bytes + overhead_bytes
        size_mb = total_size_bytes / (1024 * 1024)
        
        LOGGER.debug(f"Tokenizer size estimation: {total_size_bytes:,} bytes, {size_mb:.2f} MB.")
        return size_mb

    def get_special_tokens(self) -> Dict[str, int]:
        """
        Get mapping of special token names to their IDs.
        
        Returns:
            Dict[str, int]: Special token name to ID mapping.
        """
        return {
            "bos": self.BOS_TOKEN_ID,
            "eos": self.EOS_TOKEN_ID,
            "unk": self.UNK_TOKEN_ID,
            "pad": self.PAD_TOKEN_ID
        }

    def print_config_summary(self) -> None:
        """Log a summary of the tokenizer configuration using the logging module."""
        estimated_size = self.estimate_model_size_mb()
        
        summary_lines = [
            "Tokenizer Configuration Summary",
            "=" * 50,
            "",
            "Paths and Files:",
            f"  - Dataset Path: {self.dataset_path}",
            f"  - Output Directory: {self.output_dir}",
            f"  - Dataset Exists: {os.path.exists(self.dataset_path)}",
            "",
            "Core Configuration:",
            f"  - Vocabulary Size: {self.VOCAB_SIZE:,}",
            f"  - Max Sequence Length: {self.MAX_SEQ_LEN}",
            f"  - Random Seed: {self.SEED}",
            f"  - Evaluate Tokenizer: {self.evaluate}",
            "",
            "Special Tokens (ID -> Value):",
            f"  - BOS Token ({self.BOS_TOKEN_ID}): {self.BOS_TOKEN_VALUE}",
            f"  - EOS Token ({self.EOS_TOKEN_ID}): {self.EOS_TOKEN_VALUE}",
            f"  - UNK Token ({self.UNK_TOKEN_ID}): {self.UNK_TOKEN_VALUE}",
            f"  - PAD Token ({self.PAD_TOKEN_ID}): {self.PAD_TOKEN_VALUE}",
            "",
            f"Estimated Tokenizer Size: {estimated_size:.2f} MB"
        ]
        
        # Log each line of the summary
        for line in summary_lines:
            LOGGER.info(line)