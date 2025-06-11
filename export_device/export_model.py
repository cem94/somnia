"""
Somnia Transformer Model Export Utility for Android Deployment.

This module exports a trained SomniaTransformer model to TorchScript format
for direct Android deployment. Tokenization is handled separately on Android side.

Usage:
    python export_device/export_model.py
"""

import os
import torch
from transformers import PreTrainedTokenizerFast
from model.transformer.model import SomniaTransformer
from utility.logger import LOGGER
from utility.paths import OUTPUT_DIR, EXPORT_OUTPUT_DIR, TOKENIZER_DIR

class ModelWrapper(torch.nn.Module):
    """
    Simple wrapper for SomniaTransformer for Android deployment.
    """
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor):
        """Forward pass - input_ids to logits."""
        output = self.model(input_ids)
        return output.logits


def _export_to_torchscript(model: torch.nn.Module, example_input: torch.Tensor, output_dir: str) -> None:
    """Export model to TorchScript format."""
    LOGGER.info("Exporting to TorchScript...")
    
    traced_model = torch.jit.trace(model, example_input)
    optimized_model = torch.jit.optimize_for_inference(traced_model)
    
    model_path = os.path.join(output_dir, "model.pt")
    torch.jit.save(optimized_model, model_path)
    
    LOGGER.info(f"Model saved to: {model_path}.")


def main() -> None:
    """Load and export model for Android."""
    LOGGER.info("Pipeline Stage 4: Starting model export for Android deployment.")
    
    # Load model
    model_path = os.path.join(OUTPUT_DIR, "llama_model.pt")
    if not os.path.exists(model_path):
        LOGGER.error(f"Model not found: {model_path}.")
        raise FileNotFoundError(f"Model not found: {model_path}.")
    
    LOGGER.info("Loading trained model.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_info = SomniaTransformer.load_checkpoint(model_path, map_location=device)
    
    # Prepare for export
    model.eval()
    wrapped_model = ModelWrapper(model)
    
    # Load tokenizer
    if not os.path.exists(TOKENIZER_DIR):
        LOGGER.error(f"Tokenizer not found: {TOKENIZER_DIR}.")
        raise FileNotFoundError(f"Tokenizer not found: {TOKENIZER_DIR}.")
    
    LOGGER.info("Loading tokenizer for export validation.")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(TOKENIZER_DIR)

    # Generate example inputs for tracing
    LOGGER.info("Generating example inputs for model tracing.")
    input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
    
    # Export model
    LOGGER.info("Creating output directory for Android export.")
    os.makedirs(EXPORT_OUTPUT_DIR, exist_ok=True)
    _export_to_torchscript(wrapped_model, input_ids, EXPORT_OUTPUT_DIR)
    
    LOGGER.info("Pipeline Stage 4 completed successfully. Model export finished.")


if __name__ == "__main__":
    main()