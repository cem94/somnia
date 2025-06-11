""" 
Training module for the Transformer model.

This module implements the core training loop and utilities for the Somnia
Transformer model. It handles model initialization, training data management,
checkpointing, and inference evaluation.

Key Features:
    - Cosine learning rate scheduling with warmup
    - Mixed precision training support
    - Gradient accumulation and clipping
    - Checkpoint saving and loading
    - Training metrics tracking and visualization
    - Sample text generation for evaluation

Usage:
    Run main() with optional hyperparameters to train the model,
    or use individual functions like train_model() for specific tasks.
"""

import os
import time
import math
import torch
import warnings
from torch import optim, nn
from contextlib import nullcontext
from torch.utils.data import DataLoader
from typing import Tuple, Dict, Any, Optional
from model.transformer.model import SomniaTransformer
from model.transformer.dataset import FairyTaleDataset
from model.transformer.llama_config import LLamaConfig
from model.transformer.tokenizer_config import TokenizerConfig
from model.metrics import MetricsTracker
from utility.logger import LOGGER
from utility.paths import TOKENIZER_DIR

# Ensure LLamaConfig global variable is safe for serialization
torch.serialization.add_safe_globals([LLamaConfig])

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def get_cosine_schedule_with_warmup(current_step: int, total_steps: int, 
                                  peak_lr: float, warmup_ratio: float = 0.1) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.
    
    Args:
        current_step: Current training step.
        total_steps: Total number of training steps.
        peak_lr: Peak learning rate after warmup.
        warmup_ratio: Fraction of total steps for warmup phase.
    
    Returns:
        Current learning rate.
    """
    warmup_steps = int(total_steps * warmup_ratio)
    
    if current_step < warmup_steps:
        # Linear warmup
        return peak_lr * current_step / warmup_steps
    else:
        # Cosine decay
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * peak_lr * (1 + math.cos(math.pi * progress))


def load_checkpoint_if_exists(model: SomniaTransformer, optimizer: optim.Optimizer, 
                             config: LLamaConfig) -> Tuple[int, int, float]:
    """
    Load the latest checkpoint if it exists.
    
    Returns:
        Tuple of (start_epoch, global_step, best_loss).
    """
    checkpoint_path = os.path.join(config.out_dir, config.CHECKPOINT_NAME)
    
    if not os.path.exists(checkpoint_path):
        return 0, 0, float('inf')
    
    LOGGER.info(f"Loading checkpoint from {checkpoint_path}.")

    model_loaded, checkpoint_info = SomniaTransformer.load_checkpoint(
        checkpoint_path, map_location=config.device
    )
    model.load_state_dict(model_loaded.state_dict())
    
    if checkpoint_info['optimizer_state_dict'] is not None:
        optimizer.load_state_dict(checkpoint_info['optimizer_state_dict'])
    
    return (
        checkpoint_info['epoch'],
        checkpoint_info['step'],
        checkpoint_info.get('loss', float('inf'))
    )


def save_checkpoint_and_plots(model: SomniaTransformer, optimizer: optim.Optimizer, 
                            metrics: MetricsTracker, config: LLamaConfig, 
                            epoch: int, step: int, loss: float, 
                            checkpoint_name: str = LLamaConfig.CHECKPOINT_NAME) -> None:
    """
    Save model checkpoint and training plots.
    
    Args:
        model: The transformer model.
        optimizer: The optimizer.
        metrics: Training metrics object.
        config: Training configuration.
        epoch: Current epoch.
        step: Current step.
        loss: Current loss.
        checkpoint_name: Name for the checkpoint file.
    """
    checkpoint_path = os.path.join(config.out_dir, checkpoint_name)
    
    model.save_checkpoint(
        checkpoint_path=checkpoint_path,
        epoch=epoch,
        step=step,
        optimizer_state=optimizer.state_dict(),
        scheduler_state=None,
        loss=loss,
        metadata={
            'total_steps': config.epochs * config.batch_size,
            'config': config.__dict__
        }
    )
    
    metrics.save_plots_and_metrics(config.plot_out_dir)
    LOGGER.debug(f"Checkpoint and plots saved to {checkpoint_path}.")


def _estimate_time_remaining(start_time: float, current_step: int, total_steps: int) -> str:
    """
    Estimate remaining training time.
    
    Args:
        start_time: Training start timestamp
        current_step: Current training step
        total_steps: Total training steps
        
    Returns:
        Formatted time string (HH:MM format)
    """
    elapsed = time.time() - start_time
    if current_step == 0:
        return "Unknown"
    
    steps_per_second = current_step / elapsed
    remaining_steps = total_steps - current_step
    remaining_seconds = remaining_steps / steps_per_second
    
    hours = int(remaining_seconds // 3600)
    minutes = int((remaining_seconds % 3600) // 60)
    return f"{hours:02d}h:{minutes:02d}m"


def _load_tokenizer(tokenizer_path: str = TOKENIZER_DIR) -> Any:
    """
    Load the tokenizer from the specified path.
    
    Args:
        tokenizer_path: Path to the tokenizer directory.
        
    Returns:
        Loaded tokenizer object.
    """
    from transformers import AutoTokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        LOGGER.info(f"Tokenizer loaded successfully from {tokenizer_path}.")
        return tokenizer
    except Exception as error:
        LOGGER.error(f"Failed to load tokenizer from {tokenizer_path}: {error}.")
        raise


def train_model(model: SomniaTransformer, train_loader: DataLoader, tokenizer, config: LLamaConfig) -> MetricsTracker:
    """
    Train the transformer model on the given dataset and configuration.
    
    Args:
        model: The transformer model to train.
        train_loader: DataLoader for training data.
        tokenizer: Tokenizer for text processing.
        config: Training configuration.
    
    Returns:
        MetricsTracker object containing training history.
    """
    LOGGER.info("Pipeline Stage 3: Starting model training.")
    
    # Initialize training components
    loss_function = nn.CrossEntropyLoss(reduction='none')
    autocast_context = nullcontext() if config.device == "cpu" else torch.cuda.amp.autocast()
    gradient_scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype in ['float16', 'bfloat16']))

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )
        
    metrics_tracker = MetricsTracker(config.plot_out_dir)
    
    # Load checkpoint if exists
    start_epoch, global_step, best_loss = load_checkpoint_if_exists(model, optimizer, config)    
    steps_per_epoch = len(train_loader)
    total_training_steps = config.epochs * steps_per_epoch
    
    LOGGER.info(f"Training configuration: {config.epochs} epochs, {steps_per_epoch} steps per epoch.")
    LOGGER.info(f"Total training steps: {total_training_steps}, starting from step: {global_step}.")
    LOGGER.debug(f"Gradient accumulation steps: {config.accumulation_steps}.")
    
    # Start training loop
    training_start_time = time.time()
    
    try:
        for current_epoch in range(start_epoch, config.epochs):
            model.train()
            epoch_start_time = time.time()
            epoch_total_loss = 0.0
            accumulate_loss = 0.0
            accumulate_learning_rate = 0.0
            
            LOGGER.info(f"Starting epoch {current_epoch + 1}/{config.epochs}.")
            
            for batch_step, training_batch in enumerate(train_loader):
                # Skip steps if resuming from checkpoint
                current_step_in_epoch = current_epoch * steps_per_epoch + batch_step
                if current_step_in_epoch < global_step:
                    continue
                    
                # Move data to device
                input_token_ids = training_batch["input_ids"].to(config.device, non_blocking=True)
                target_labels = training_batch["labels"].to(config.device, non_blocking=True)
                attention_mask = training_batch["attention_mask"].to(config.device, non_blocking=True)
                
                # Update learning rate with cosine schedule
                current_learning_rate = get_cosine_schedule_with_warmup(global_step, total_training_steps, config.learning_rate)
                accumulate_learning_rate += current_learning_rate
                optimizer.param_groups[0]['lr'] = current_learning_rate

                # Forward pass with mixed precision
                with autocast_context:
                    model_outputs = model(input_ids=input_token_ids)
                    output_logits = model_outputs.logits
                    
                    # Calculate loss with proper masking for next token prediction
                    shifted_logits = output_logits[..., :-1, :].contiguous()
                    shifted_labels = target_labels[..., 1:].contiguous()
                    shifted_attention_mask = attention_mask[..., 1:].contiguous()
                    
                    token_losses = loss_function(
                        shifted_logits.view(-1, shifted_logits.size(-1)), 
                        shifted_labels.view(-1)
                    ).view(shifted_labels.size())
                    
                    # Apply attention mask and normalize
                    masked_loss = (token_losses * shifted_attention_mask).sum() / shifted_attention_mask.sum()
                    scaled_loss_for_accumulation = masked_loss / config.accumulation_steps
                
                # Backward pass
                gradient_scaler.scale(scaled_loss_for_accumulation).backward()
                epoch_total_loss += masked_loss.item()
                
                # Update best loss tracking BEFORE accumulation check
                current_loss_value = masked_loss.item()
                accumulate_loss += current_loss_value
                if current_loss_value < best_loss:
                    best_loss = current_loss_value
                    LOGGER.debug(f"New best loss: {best_loss:.4f} at step {global_step}.")
                
                # Optimization step with gradient accumulation
                if (batch_step + 1) % config.accumulation_steps == 0:
                    # Unscale gradients for clipping
                    gradient_scaler.unscale_(optimizer)
                    
                    # Calculate gradient norm before clipping
                    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    
                    # Optimizer step
                    gradient_scaler.step(optimizer)
                    gradient_scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Record metrics
                    metrics_tracker.add_metrics(
                        step=global_step,
                        epoch=current_epoch,
                        loss=accumulate_loss / config.accumulation_steps,
                        lr=accumulate_learning_rate / config.accumulation_steps,
                        grad_norm=gradient_norm.item() if torch.is_tensor(gradient_norm) else gradient_norm
                    )

                    # Reset accumulators
                    accumulate_loss = 0.0
                    accumulate_learning_rate = 0.0
                
                global_step += 1
                
                # Progress logging
                if batch_step % config.log_interval == 0:
                    elapsed_training_time = time.time() - training_start_time
                    estimated_time_remaining = _estimate_time_remaining(training_start_time, global_step, total_training_steps)
                    current_perplexity = math.exp(min(current_loss_value, 10))
                    
                    LOGGER.info(
                        f"Epoch {current_epoch+1:2d}/{config.epochs} | "
                        f"Step {batch_step:4d}/{steps_per_epoch} | "
                        f"Loss: {current_loss_value:.4f} | "
                        f"Perplexity: {current_perplexity:.2f} | "
                        f"LR: {current_learning_rate:.2e} | "
                        f"Time: {elapsed_training_time/60:.1f}m | "
                        f"ETA: {estimated_time_remaining}"
                    )
                
                # Periodic checkpoint saving
                if (global_step) % config.save_interval == 0:
                    save_checkpoint_and_plots(
                        model=model,
                        optimizer=optimizer,
                        metrics=metrics_tracker,
                        config=config,
                        epoch=current_epoch,
                        step=global_step,
                        loss=current_loss_value
                    )
            
            # End of epoch summary
            epoch_duration = time.time() - epoch_start_time
            average_epoch_loss = epoch_total_loss / len(train_loader)
            epoch_perplexity = math.exp(min(average_epoch_loss, 10))

            # Generate sample text after each epoch
            generate_sample_text(model, tokenizer, config.device)

            # Clear CUDA cache to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                LOGGER.debug("Cleared CUDA cache after epoch completion.")
            
            LOGGER.info(f"Epoch {current_epoch+1} completed: avg loss {average_epoch_loss:.4f}, "
                       f"perplexity {epoch_perplexity:.2f}, "
                       f"time {epoch_duration/60:.1f}m.")
    
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted by user.")
        raise
    except Exception as training_error:
        LOGGER.error(f"Training error: {training_error}.")
        raise
    finally:
        # Save final checkpoint and plots
        save_checkpoint_and_plots(
            model=model,
            optimizer=optimizer,
            metrics=metrics_tracker,
            config=config,
            epoch=current_epoch if 'current_epoch' in locals() else 0,
            step=global_step,
            loss=best_loss
        )
        
        total_training_time = time.time() - training_start_time
        LOGGER.info(f"Training completed in {total_training_time/3600:.2f} hours.")
    
    return metrics_tracker


def generate_sample_text(model: SomniaTransformer, tokenizer, device: str) -> None:
    """
    Generate sample text to evaluate model performance.
    
    Args:
        model: The trained transformer model.
        tokenizer: Tokenizer for text processing.
        device: Device to run inference on.
    """
    test_prompts = [
        "Once upon a time",
        "In a kingdom far away", 
        "There lived a brave princess",
        ""  # Empty prompt test
    ]
    
    model.eval()
    LOGGER.info("Generating sample text outputs.")
    
    with torch.no_grad():
        for prompt_index, test_prompt in enumerate(test_prompts, 1):
            LOGGER.debug(f"Sample {prompt_index} with prompt: '{test_prompt}'" if test_prompt else "Sample with empty prompt.")
            
            try:
                if not test_prompt:  # BOS token
                    input_token_ids = torch.tensor([[TokenizerConfig.BOS_TOKEN_ID]], device=device)
                else:
                    input_token_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to(device)
                
                generated_token_ids = model.generate(
                    input_ids=input_token_ids,
                    max_new_tokens=min(200, TokenizerConfig.MAX_SEQ_LEN - input_token_ids.size(1)),
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    use_cache=True
                )
                
                generated_text = tokenizer.decode(generated_token_ids[0], skip_special_tokens=True)
                LOGGER.info(f"Generated: {generated_text}")
                
            except Exception as generation_error:
                LOGGER.error(f"Error generating text for prompt '{test_prompt}': {generation_error}.")
                continue
    

def main(hyperparameters: Optional[Dict[str, Any]] = None) -> float:
    """
    Main training function that accepts hyperparameters and returns best score.
    
    Args:
        hyperparameters: Dictionary of hyperparameters to override default config.
                        If None, uses default configuration.
    
    Returns:
        Best loss achieved during training.
    """
    LOGGER.info("Pipeline Stage 3: Starting Somnia Transformer training.")
    
    # Set random seeds for reproducibility
    torch.manual_seed(TokenizerConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TokenizerConfig.SEED)
        torch.cuda.manual_seed_all(TokenizerConfig.SEED)
    
    # Initialize configuration
    training_config = LLamaConfig()
    
    # Override with provided hyperparameters
    if hyperparameters:
        for parameter_name, parameter_value in hyperparameters.items():
            if hasattr(training_config, parameter_name):
                setattr(training_config, parameter_name, parameter_value)
                LOGGER.debug(f"Set {parameter_name} = {parameter_value}.")
            else:
                LOGGER.warning(f"Unknown hyperparameter: {parameter_name}.")
    
    # Create output directory
    os.makedirs(training_config.out_dir, exist_ok=True)
    
    # Initialize model
    LOGGER.info("Initializing transformer model.")
    transformer_model = SomniaTransformer(training_config).to(training_config.device)
    
    # Log model information
    total_parameters = transformer_model.get_num_parameters(only_trainable=True)
    memory_usage_info = transformer_model.get_memory_usage()
    
    LOGGER.info(f"Model initialized: {total_parameters/1e6:.3f}M parameters, "
               f"{memory_usage_info['total_mb']:.2f} MB memory usage.")
    LOGGER.debug(f"Training device: {training_config.device}, mixed precision: {training_config.dtype}.")
    

    # Load tokenizer
    text_tokenizer = _load_tokenizer()

    # Initialize dataset and dataloader with tokenizer
    LOGGER.info("Loading training dataset.")
    training_dataset = FairyTaleDataset(text_tokenizer)
    training_dataloader = DataLoader(
        training_dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    LOGGER.info(f"Dataset loaded: {len(training_dataset)} samples, "
               f"batch size {training_config.batch_size}, "
               f"{len(training_dataloader)} batches per epoch.")
    
    # Train the model
    training_metrics = train_model(transformer_model, training_dataloader, text_tokenizer, training_config)
    
    # Calculate best loss from metrics instead of tracking variable
    if training_metrics.losses:
        best_training_loss = min(training_metrics.losses)
        LOGGER.info(f"Best loss from metrics: {best_training_loss:.4f}.")
    else:
        best_training_loss = float('inf')
        LOGGER.warning("No metrics recorded during training.")
    
    LOGGER.info(f"Pipeline Stage 3 completed successfully. Best loss: {best_training_loss:.4f}.")
    return best_training_loss


if __name__ == "__main__":
    main()