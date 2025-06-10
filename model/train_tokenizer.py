"""
Custom tokenizer module for story generation.

This module trains a Byte-Pair Encoding (BPE) tokenizer specifically optimized
for story generation tasks. It processes JSONL text data and creates a vocabulary
tailored for narrative content.

Usage:
    Run main() to train and evaluate the tokenizer, or use individual functions
    for specific tasks like train_custom_tokenizer() or evaluate_tokenizer().
"""

import os
import json
import random
from typing import Generator
from transformers import AutoTokenizer
from model.transformer.tokenizer_config import TokenizerConfig
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
from utility.logger import LOGGER

# Set fixed random seed for reproducibility
random.seed(TokenizerConfig.SEED)


def _load_texts_from_jsonl(file_path: str) -> Generator[str, None, None]:
    """
    Generator function to read and yield text data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file containing text data.
        
    Yields:
        str: Text content from each line of the JSONL file.
        
    Raises:
        FileNotFoundError: If the specified file doesn't exist.
        json.JSONDecodeError: If JSON parsing fails.
    """
    if not os.path.exists(file_path):
        LOGGER.error(f"Dataset file not found: {file_path}.")
        raise FileNotFoundError(f"Dataset file not found: {file_path}.")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                try:
                    data = json.loads(line.strip())
                    yield data['text']
                except json.JSONDecodeError as error:
                    LOGGER.debug(f"Skipping invalid JSON line: {error}.")
                    continue
                    
        LOGGER.debug(f"Successfully processed file {file_path}.")
        
    except Exception as error:
        LOGGER.error(f"Error reading dataset file: {error}.")
        raise


def train_custom_tokenizer(tokenizer_config: TokenizerConfig) -> bool:
    """
    Train a Byte-Pair Encoding (BPE) tokenizer for story generation.
    
    Args:
        tokenizer_config: Configuration object containing tokenizer parameters.
        
    Returns:
        bool: True if training successful, False otherwise.
    """
    LOGGER.info("Pipeline Stage 2: Starting BPE tokenizer training.")
    
    try:
        # Initialize BPE tokenizer
        tokenizer = Tokenizer(models.BPE())
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Define special tokens from configuration
        special_tokens = [
            tokenizer_config.BOS_TOKEN_VALUE,
            tokenizer_config.EOS_TOKEN_VALUE,
            tokenizer_config.UNK_TOKEN_VALUE,
            tokenizer_config.PAD_TOKEN_VALUE,
        ]

        LOGGER.info(f"Configuring BPE trainer with vocabulary size: {tokenizer_config.VOCAB_SIZE}.")
        trainer = trainers.BpeTrainer(
            vocab_size=tokenizer_config.VOCAB_SIZE,
            special_tokens=special_tokens,
            show_progress=True,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
        )

        # Load training data
        LOGGER.info(f"Loading training data from: {tokenizer_config.dataset_path}.")
        texts = _load_texts_from_jsonl(tokenizer_config.dataset_path)

        # Train tokenizer
        LOGGER.info("Training tokenizer on text data.")
        tokenizer.train_from_iterator(texts, trainer=trainer)
        tokenizer.decoder = decoders.ByteLevel()

        # Create output directory
        os.makedirs(tokenizer_config.output_dir, exist_ok=True)
        
        # Save tokenizer configuration first
        config_file_path = os.path.join(tokenizer_config.output_dir, "tokenizer_config.json")
        with open(config_file_path, "w", encoding="utf-8") as config_file:
            json.dump(tokenizer_config.config, config_file, ensure_ascii=False, indent=4)
        
        # Save tokenizer file
        tokenizer_file_path = os.path.join(tokenizer_config.output_dir, "tokenizer.json")
        tokenizer.save(tokenizer_file_path)
        
        # Load as Transformers tokenizer and set special tokens
        LOGGER.info("Converting to Transformers-compatible format.")
        transformers_tokenizer = AutoTokenizer.from_pretrained(tokenizer_config.output_dir, local_files_only=True)
        
        # Set special tokens explicitly
        transformers_tokenizer.bos_token = tokenizer_config.BOS_TOKEN_VALUE
        transformers_tokenizer.eos_token = tokenizer_config.EOS_TOKEN_VALUE
        transformers_tokenizer.unk_token = tokenizer_config.UNK_TOKEN_VALUE
        transformers_tokenizer.pad_token = tokenizer_config.PAD_TOKEN_VALUE
        
        # Save final tokenizer (overwrites with correct Transformers format)
        transformers_tokenizer.save_pretrained(tokenizer_config.output_dir)
        
        LOGGER.info(f"Tokenizer training completed and saved to {tokenizer_config.output_dir}.")
        return True
        
    except Exception as error:
        LOGGER.error(f"Tokenizer training failed: {error}.")
        return False


def evaluate_tokenizer(tokenizer_path: str) -> bool:
    """
    Load and evaluate the trained tokenizer on sample text.
    
    Args:
        tokenizer_path: Path to the directory containing the trained tokenizer.
        
    Returns:
        bool: True if evaluation successful, False otherwise.
    """
    LOGGER.info("Starting tokenizer evaluation.")
    
    if not os.path.exists(tokenizer_path):
        LOGGER.error(f"Tokenizer path not found: {tokenizer_path}.")
        return False
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Test samples for evaluation
        test_samples = [
            "Once upon a time, there was a little rabbit who loved to hop through the forest.",
            "The quick brown fox jumps over the lazy dog.",
            "In a galaxy far, far away, adventures await those brave enough to seek them."
        ]

        LOGGER.info(f"Evaluating tokenizer on {len(test_samples)} test samples.")
        
        for i, sample_text in enumerate(test_samples, 1):
            LOGGER.debug(f"Evaluating sample {i}: {sample_text[:50]}...")
            
            # Encode the text
            encoded_inputs = tokenizer(sample_text, return_tensors=None)
            token_count = len(encoded_inputs['input_ids'])
            
            # Decode back to text
            decoded_text = tokenizer.decode(encoded_inputs['input_ids'], skip_special_tokens=True)
            
            # Check round-trip encoding/decoding
            text_preserved = decoded_text.strip() == sample_text.strip()
            
            LOGGER.debug(f"Sample {i} results: {token_count} tokens, text preserved: {text_preserved}.")
            
            if not text_preserved:
                LOGGER.warning(f"Text preservation failed for sample {i}.")
                LOGGER.debug(f"Original: '{sample_text}'")
                LOGGER.debug(f"Decoded:  '{decoded_text}'")

        # Log vocabulary statistics
        LOGGER.info(f"Tokenizer evaluation completed successfully. Vocabulary size: {tokenizer.vocab_size}.")
        return True
        
    except Exception as error:
        LOGGER.error(f"Tokenizer evaluation failed: {error}.")
        return False


def main() -> bool:
    """
    Main function to train and evaluate the custom tokenizer.
    
    Returns:
        bool: True if entire process successful, False otherwise.
    """
    LOGGER.info("Pipeline Stage 2: Starting tokenizer training pipeline.")
    
    try:
        # Initialize configuration
        config = TokenizerConfig()
        config.print_config_summary()
        
        # Train the tokenizer
        training_success = train_custom_tokenizer(config)
        if not training_success:
            LOGGER.error("Tokenizer training failed.")
            return False
        
        # Evaluate if requested
        if config.evaluate:
            evaluation_success = evaluate_tokenizer(config.output_dir)
            if not evaluation_success:
                LOGGER.warning("Tokenizer evaluation failed, but training was successful.")
                return True  # Training succeeded even if evaluation failed
        
        LOGGER.info("Pipeline Stage 2 completed successfully.")
        return True
        
    except Exception as error:
        LOGGER.error(f"Pipeline Stage 2 failed: {error}.")
        raise


if __name__ == '__main__':
    success = main()