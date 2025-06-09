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

# Ensure LLamaConfig global variable is safe for serialization
torch.serialization.add_safe_globals([LLamaConfig])

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


def get_cosine_schedule_with_warmup(current_step: int, total_steps: int, 
                                  peak_lr: float, warmup_ratio: float = 0.1) -> float:
    """
    Cosine annealing learning rate schedule with linear warmup.
    
    Args:
        current_step: Current training step
        total_steps: Total number of training steps
        peak_lr: Peak learning rate after warmup
        warmup_ratio: Fraction of total steps for warmup phase
    
    Returns:
        Current learning rate
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
        Tuple of (start_epoch, global_step, best_loss)
    """
    checkpoint_dir = config.out_dir
    checkpoint_path = os.path.join(checkpoint_dir, 'model_checkpoint.pt')
    
    if not os.path.exists(checkpoint_path):
        return 0, 0, float('inf')
    
    LOGGER.info(f"Loading checkpoint from {checkpoint_path}")

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
                            checkpoint_name: str = "llama_model.pt") -> None:
    """
    Save model checkpoint and training plots.
    
    Args:
        model: The transformer model
        optimizer: The optimizer
        metrics: Training metrics object
        config: Training configuration
        epoch: Current epoch
        step: Current step
        loss: Current loss
        checkpoint_name: Name for the checkpoint file
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
    LOGGER.debug(f"Checkpoint and plots saved to {checkpoint_path}")


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


def train_model(model: SomniaTransformer, train_loader: DataLoader, config: LLamaConfig) -> MetricsTracker:
    """
    Train the transformer model on the given dataset and configuration.
    
    Args:
        model: The transformer model to train
        train_loader: DataLoader for training data
        config: Training configuration
    
    Returns:
        MetricsTracker object containing training history
    """
    LOGGER.info("Starting model training")
    
    # Initialize training components
    model.train()
    device = config.device
    loss_fn = nn.CrossEntropyLoss(reduction='none')
    ctx = nullcontext() if device == "cpu" else torch.cuda.amp.autocast()
    scaler = torch.cuda.amp.GradScaler(enabled=(config.dtype in ['float16', 'bfloat16']))

    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-8
    )
        
    metrics = MetricsTracker(config.plot_out_dir)
    
    # Load checkpoint if exists
    start_epoch, global_step, best_loss = load_checkpoint_if_exists(model, optimizer, config)    
    steps_per_epoch = len(train_loader)
    total_steps = config.epochs * steps_per_epoch
    remaining_steps = total_steps - global_step
    
    LOGGER.info(f"Training configuration: {config.epochs} epochs, {steps_per_epoch} steps per epoch")
    LOGGER.info(f"Total training steps: {total_steps}, starting from step: {global_step}")
    LOGGER.debug(f"Gradient accumulation steps: {config.accumulation_steps}")
    
    start_time = time.time()
    
    try:
        for epoch in range(start_epoch, config.epochs):
            epoch_start_time = time.time()
            epoch_loss = 0.0
            
            LOGGER.info(f"Starting epoch {epoch + 1}/{config.epochs}")
            
            for step, batch in enumerate(train_loader):
                # Skip steps if resuming from checkpoint
                current_step_in_epoch = epoch * steps_per_epoch + step
                if current_step_in_epoch < global_step:
                    continue
                    
                # Move data to device
                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                
                # Update learning rate with cosine schedule
                current_lr = get_cosine_schedule_with_warmup(global_step, total_steps, config.learning_rate)
                optimizer.param_groups[0]['lr'] = current_lr

                # Forward pass with mixed precision
                with ctx:
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits
                    
                    # Calculate loss with proper masking
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    shift_mask = attention_mask[..., 1:].contiguous()
                    
                    loss = loss_fn(
                        shift_logits.view(-1, shift_logits.size(-1)), 
                        shift_labels.view(-1)
                    ).view(shift_labels.size())
                    
                    # Apply attention mask and normalize
                    masked_loss = (loss * shift_mask).sum() / shift_mask.sum()
                    scaled_loss = masked_loss / config.accumulation_steps
                
                # Backward pass
                scaler.scale(scaled_loss).backward()
                epoch_loss += masked_loss.item()
                
                # Optimization step with gradient accumulation
                if (step + 1) % config.accumulation_steps == 0:
                    # Unscale gradients for clipping
                    scaler.unscale_(optimizer)
                    
                    # Calculate gradient norm before clipping
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    
                    # Optimizer step
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Record metrics
                    metrics.add_metrics(
                        step=global_step,
                        epoch=epoch,
                        loss=masked_loss.item(),
                        lr=current_lr,
                        grad_norm=grad_norm.item() if torch.is_tensor(grad_norm) else grad_norm
                    )
                else:
                    # Record metrics even without optimizer step for tracking
                    metrics.add_metrics(
                        step=global_step,
                        epoch=epoch,
                        loss=masked_loss.item(),
                        lr=current_lr,
                        grad_norm=float('nan')
                    )
                
                global_step += 1
                
                if masked_loss.item() < best_loss:
                    best_loss = masked_loss.item()
                
                # Progress logging
                if step % config.log_interval == 0:
                    elapsed = time.time() - start_time
                    time_remaining = _estimate_time_remaining(start_time, global_step, total_steps)
                    
                    LOGGER.info(
                        f"Epoch {epoch+1:2d}/{config.epochs} | "
                        f"Step {step:4d}/{steps_per_epoch} | "
                        f"Loss: {masked_loss.item():.4f} | "
                        f"Perplexity: {math.exp(min(masked_loss.item(), 10)):.2f} | "
                        f"LR: {current_lr:.2e} | "
                        f"Time: {elapsed/60:.1f}m | "
                        f"ETA: {time_remaining}"
                    )
                
                # Checkpoint saving
                if (global_step) % config.save_interval == 0 and global_step > 0:
                    save_checkpoint_and_plots(
                        model=model,
                        optimizer=optimizer,
                        metrics=metrics,
                        config=config,
                        epoch=epoch,
                        step=global_step,
                        loss=masked_loss.item()
                    )
            
            # End of epoch summary
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / len(train_loader)
            
            LOGGER.info(f"Epoch {epoch+1} completed: avg loss {avg_epoch_loss:.4f}, "
                       f"perplexity {math.exp(min(avg_epoch_loss, 10)):.2f}, "
                       f"time {epoch_time/60:.1f}m")
    
    except KeyboardInterrupt:
        LOGGER.info("Training interrupted by user")
        raise
    except Exception as e:
        LOGGER.error(f"Training error: {e}")
        raise
    finally:
        # Save final checkpoint and plots
        save_checkpoint_and_plots(
            model=model,
            optimizer=optimizer,
            metrics=metrics,
            config=config,
            epoch=epoch,
            step=global_step,
            loss=best_loss
        )
        
        total_time = time.time() - start_time
        LOGGER.info(f"Training completed in {total_time/3600:.2f} hours")
    
    return metrics


def generate_sample_text(model: SomniaTransformer, tokenizer, device: str) -> None:
    """
    Generate sample text to evaluate model performance.
    
    Args:
        model: The trained transformer model
        tokenizer: Tokenizer for text processing
        device: Device to run inference on
    """
    test_prompts = [
        "Once upon a time",
        "In a kingdom far away",
        "There lived a brave princess",
        ""  # Empty prompt test
    ]
    
    model.eval()
    LOGGER.info("Generating sample text outputs")
    
    with torch.no_grad():
        for i, prompt in enumerate(test_prompts, 1):
            LOGGER.debug(f"Sample {i} with prompt: '{prompt}'" if prompt else "Sample with empty prompt")
            
            try:
                if not prompt:  # BOS token
                    input_ids = torch.tensor([[TokenizerConfig.BOS_TOKEN_ID]], device=device)
                else:
                    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=min(200, TokenizerConfig.MAX_SEQ_LEN - input_ids.size(1)),
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    use_cache=True
                )
                
                generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                LOGGER.info(f"Generated: {generated_text}")
                
            except Exception as e:
                LOGGER.error(f"Error generating text for prompt '{prompt}': {e}")
                raise
    

def main(hyperparams: Optional[Dict[str, Any]] = None) -> float:
    """
    Main training function that accepts hyperparameters and returns best score.
    
    Args:
        hyperparams: Dictionary of hyperparameters to override default config.
                    If None, uses default configuration.
    
    Returns:
        Best loss achieved during training
    """
    LOGGER.info("Starting Somnia Transformer training")
    
    # Set random seeds for reproducibility
    torch.manual_seed(TokenizerConfig.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TokenizerConfig.SEED)
        torch.cuda.manual_seed_all(TokenizerConfig.SEED)
    
    # Initialize configuration
    config = LLamaConfig()
    
    # Override with provided hyperparameters
    if hyperparams:
        for key, value in hyperparams.items():
            if hasattr(config, key):
                setattr(config, key, value)
                LOGGER.debug(f"Set {key} = {value}")
            else:
                LOGGER.warning(f"Unknown hyperparameter: {key}")
    
    # Create output directory
    os.makedirs(config.out_dir, exist_ok=True)
    
    # Initialize model
    LOGGER.info("Initializing transformer model")
    model = SomniaTransformer(config).to(config.device)
    
    # Log model information
    total_params = model.get_num_parameters(only_trainable=True)
    memory_usage = model.get_memory_usage()
    
    LOGGER.info(f"Model initialized: {total_params/1e6:.3f}M parameters, "
               f"{memory_usage['total_mb']:.2f} MB memory usage")
    LOGGER.debug(f"Training device: {config.device}, mixed precision: {config.dtype}")
    
    # Initialize dataset and dataloader
    LOGGER.info("Loading training dataset")
    train_dataset = FairyTaleDataset()
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=False,
        num_workers=0  # Avoid multiprocessing issues
    )
    
    LOGGER.info(f"Dataset loaded: {len(train_dataset)} samples, "
               f"batch size {config.batch_size}, "
               f"{len(train_loader)} batches per epoch")
    
    # Train the model
    training_metrics = train_model(model, train_loader, config)
    
    # Calculate best loss
    best_loss = min(training_metrics.losses) if training_metrics.losses else float('inf')
    
    # Generate sample outputs for full training
    if not hyperparams or config.epochs > 5:
        generate_sample_text(model, train_dataset.tokenizer, config.device)
        
        # Save training summary
        summary_path = os.path.join(config.out_dir, "training_summary.txt")
        with open(summary_path, 'w') as f:
            f.write("SOMNIA TRANSFORMER TRAINING SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Model Parameters: {total_params/1e6:.3f}M\n")
            f.write(f"Training Samples: {len(train_dataset)}\n")
            f.write(f"Epochs: {config.epochs}\n")
            f.write(f"Batch Size: {config.batch_size}\n")
            f.write(f"Learning Rate: {config.learning_rate}\n")
            f.write(f"Best Loss: {best_loss:.4f}\n")
            
            if training_metrics.losses:
                f.write(f"Final Loss: {training_metrics.losses[-1]:.4f}\n")
                f.write(f"Final Perplexity: {training_metrics.perplexities[-1]:.2f}\n")
        
        LOGGER.info(f"Training summary saved to {summary_path}")
    
    LOGGER.info(f"Training completed successfully, best loss: {best_loss:.4f}")
    return best_loss


if __name__ == "__main__":
    main()