"""
Somnia Transformer Model Export Utility for Android Deployment.

This module exports a trained SomniaTransformer model to TorchScript format
for direct Android deployment without API dependencies.

Usage:
    python export_device/export_model.py
"""

import os
import json
import torch
from model.transformer.model import SomniaTransformer
from transformers import PreTrainedTokenizerFast
from utility.logger import LOGGER
from utility.paths import OUTPUT_DIR, TOKENIZER_DIR, EXPORT_OUTPUT_DIR

class ModelWrapper(torch.nn.Module):
    """
    Wrapper for SomniaTransformer optimized for Android deployment.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.test_prompts = [
            "Once upon a time",
            "In a kingdom far away",
            "There lived a brave princess",
            ""
        ]

    def forward(self, input_ids: torch.Tensor):
        """Forward pass for model inference - returns only logits for TorchScript compatibility."""
        output = self.model(input_ids)
        # Return only logits to avoid CausalLMOutputWithPast serialization issues
        return output.logits
    
    def get_random_prompt(self) -> str:
        """Get a random test prompt."""
        import random
        return random.choice(self.test_prompts)


def _prepare_model_for_export(model: torch.nn.Module) -> torch.nn.Module:
    """
    Prepare model for ExecuTorch export.
    
    Args:
        model: The trained transformer model
        
    Returns:
        Wrapped model ready for export
    """
    model.eval()
    
    # Wrap model for export
    wrapped_model = ModelWrapper(model)
    return wrapped_model


def _export_to_torchscript(model: torch.nn.Module, example_input: torch.Tensor, output_dir: str) -> None:
    """
    Export model to TorchScript format for Android deployment.
    
    Args:
        model: Prepared model for export
        example_input: Example input tensor for tracing
        output_dir: Directory to save the exported model
    """
    LOGGER.info("Tracing model with TorchScript...")
    
    # Trace the model for mobile deployment
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    LOGGER.info("Optimizing model for mobile deployment...")
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    model_path = os.path.join(output_dir, "model.pt")
    torch.jit.save(optimized_model, model_path)
    
    LOGGER.info(f"TorchScript model saved to: {model_path}")


def _save_android_metadata(model_info: dict, tokenizer: PreTrainedTokenizerFast, output_dir: str) -> None:
    """
    Save essential metadata for Android app integration.
    
    Args:
        model_info: Model training information from checkpoint
        tokenizer: Trained tokenizer
        output_dir: Directory to save metadata
    """
    metadata = {
        "model_info": {
            "epoch": model_info.get("epoch", 0),
            "step": model_info.get("step", 0),
            "architecture": "SomniaTransformer"
        },
        "tokenizer_info": {
            "vocab_size": tokenizer.vocab_size,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            "unk_token_id": tokenizer.unk_token_id
        },
        "generation_config": {
            "max_length": 512,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
    }
    
    metadata_path = os.path.join(output_dir, "android_config.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    LOGGER.info(f"Android metadata saved to: {metadata_path}")


def _export_model(model: torch.nn.Module, model_info: dict, tokenizer: PreTrainedTokenizerFast) -> None:
    """
    Export model for Android deployment using TorchScript.
    
    Args:
        model: Trained transformer model
        model_info: Model checkpoint information
        tokenizer: Trained tokenizer
        output_dir: Output directory for Android-ready files
    """
    os.makedirs(EXPORT_OUTPUT_DIR, exist_ok=True)
    LOGGER.info(f"Starting Android model export to: {EXPORT_OUTPUT_DIR}")
    
    # Prepare model for mobile deployment
    export_model_wrapper = _prepare_model_for_export(model)
    
    # Create example input for tracing using random prompt
    prompt = export_model_wrapper.get_random_prompt()
    LOGGER.info(f"Using prompt for tracing: '{prompt}'")
    example_input = tokenizer.encode(prompt, return_tensors="pt")
    
    # Export to TorchScript format
    _export_to_torchscript(export_model_wrapper, example_input, EXPORT_OUTPUT_DIR)
    
    # Save Android-specific metadata
    _save_android_metadata(model_info, tokenizer, EXPORT_OUTPUT_DIR)
    
    LOGGER.info("Android model export completed successfully!")


def main() -> None:
    """
    Load model and tokenizer from checkpoints and export for Android deployment.
    """
    # Load model checkpoint
    model_path = os.path.join(OUTPUT_DIR, "llama_model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model llama not found: {model_path}")
    
    LOGGER.info("Loading model from saving direcotry...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_info = SomniaTransformer.load_checkpoint(model_path, map_location=device)
    LOGGER.info(f"Model loaded - Epoch: {model_info['epoch']}, Step: {model_info['step']}")
    
    # Load tokenizer
    if not os.path.exists(TOKENIZER_DIR):
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_DIR}")
    
    LOGGER.info("Loading tokenizer...")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)
    
    # Export for Android
    _export_model(model, model_info, tokenizer)


if __name__ == "__main__":
    main()