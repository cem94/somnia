"""
Configuration module for the LLaMA-based language model.

This module provides the configuration class for the transformer model,
including all hyperparameters, training settings, and model architecture
parameters.
"""

import torch
from transformers import PretrainedConfig
from model.transformer.tokenizer_config import TokenizerConfig
from utility.paths import TOKENIZER_DIR, OUTPUT_DIR, PLOT_OUTPUT_DIR, PROCESSED_OUTPUT_FILE
from utility.logger import LOGGER


class LLamaConfig(PretrainedConfig):
    """
    Configuration class for the LLaMA-based language model.
    
    This configuration class stores all the hyperparameters and settings
    of the model, including architecture parameters, training settings,
    and Mixture of Experts configuration.

    Attributes:
        model_type (str): Model type identifier
        
        # Architecture parameters
        dim (int): Embedding dimension
        n_layers (int): Number of transformer layers
        n_heads (int): Number of attention heads
        n_kv_heads (int): Number of key-value attention heads (for multi-query attention)
        vocab_size (int): Size of the vocabulary
        hidden_dim (int, optional): Hidden dimension of the feedforward network
        multiple_of (int): Used to calculate hidden_dim if hidden_dim is None
        norm_eps (float): Epsilon value for layer normalization
        max_seq_len (int): Maximum sequence length
        rope_theta (int): Theta value for rotary positional embeddings
        dropout (float): Dropout probability
        flash_attn (bool): Whether to use Flash Attention (if available)
        
        # Training parameters
        epochs (int): Number of training epochs
        batch_size (int): Training batch size
        learning_rate (float): Learning rate for optimizer
        accumulation_steps (int): Gradient accumulation steps
        grad_clip (float): Gradient clipping value
        warmup_iters (int): Number of warmup iterations
        log_interval (int): Logging interval in steps
        save_interval (int): Model saving interval in epochs
        
        # System parameters
        device (str): Device to use for training
        dtype (str): Data type for model parameters
        out_dir (str): Output directory for model checkpoints
        data_path (str): Path to training data
        tokenizer_path (str): Path to tokenizer directory
    """

    model_type = "transformerlm"

    def __init__(self, 
                 # Architecture parameters
                 dim: int = 512,
                 n_layers: int = 8,
                 n_heads: int = 8,
                 n_kv_heads: int = 2,
                 hidden_dim: int = None,
                 multiple_of: int = 64,
                 norm_eps: float = 1e-5,
                 rope_theta: int = 1e6,
                 dropout: float = 0.0,
                 flash_attn: bool = True,
                 
                 # Training parameters
                 epochs: int = 5,
                 batch_size: int = 8,
                 learning_rate: float = 5e-4,
                 accumulation_steps: int = 8,
                 grad_clip: float = 1.0,
                 warmup_iters: int = 0,
                 log_interval: int = 10,
                 save_interval: int = 10):
        """
        Initialize the LLaMA configuration.

        Args:
            # Architecture parameters
            dim (int): Embedding dimension (default: 512)
            n_layers (int): Number of transformer layers (default: 8)
            n_heads (int): Number of attention heads (default: 8)
            n_kv_heads (int): Number of key-value attention heads (default: 2)
            hidden_dim (int, optional): Hidden dimension of FFN. If None, calculated automatically
            multiple_of (int): Rounding factor for hidden_dim calculation (default: 64)
            norm_eps (float): Layer normalization epsilon (default: 1e-5)
            rope_theta (int): RoPE theta parameter (default: 1e6)
            dropout (float): Dropout probability (default: 0.0)
            flash_attn (bool): Whether to use Flash Attention (default: True)
            
            # Training parameters
            epochs (int): Number of training epochs (default: 5)
            batch_size (int): Training batch size (default: 8)
            learning_rate (float): Learning rate (default: 5e-4)
            accumulation_steps (int): Gradient accumulation steps (default: 8)
            grad_clip (float): Gradient clipping value (default: 1.0)
            warmup_iters (int): Warmup iterations (default: 0)
            log_interval (int): Logging interval (default: 10)
            save_interval (int): Save interval (default: 10)
        """
        # Set architecture parameters
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = TokenizerConfig.VOCAB_SIZE
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = TokenizerConfig.MAX_SEQ_LEN
        self.rope_theta = rope_theta
        self.dropout = dropout
        self.flash_attn = flash_attn
        
        # Set training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.accumulation_steps = accumulation_steps
        self.grad_clip = grad_clip
        self.warmup_iters = warmup_iters
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Set system parameters
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dtype = "bfloat16"
        self.out_dir = OUTPUT_DIR
        self.plot_out_dir = PLOT_OUTPUT_DIR
        self.data_path = PROCESSED_OUTPUT_FILE
        self.tokenizer_path = TOKENIZER_DIR
        
        # Validate configuration
        self._validate_configuration()
        
        # Initialize parent class
        super().__init__()

    def _validate_configuration(self) -> None:
        """Validate the configuration parameters."""
        LOGGER.debug("Starting configuration validation...")
        
        # Validate architecture parameters
        assert self.dim > 0, f"Embedding dimension must be positive, got {self.dim}"
        assert self.n_layers > 0, f"Number of layers must be positive, got {self.n_layers}"
        assert self.n_heads > 0, f"Number of heads must be positive, got {self.n_heads}"
        assert self.dim % self.n_heads == 0, f"Embedding dimension ({self.dim}) must be divisible by number of heads ({self.n_heads})"
        
        # Fixed validation for n_kv_heads
        assert self.n_kv_heads > 0 and self.n_kv_heads <= self.n_heads, \
            f"Number of KV heads ({self.n_kv_heads}) must be positive and <= number of heads ({self.n_heads})"
        
        assert self.vocab_size > 0, f"Vocabulary size must be positive, got {self.vocab_size}"
        assert self.max_seq_len > 0, f"Max sequence length must be positive, got {self.max_seq_len}"
        assert self.norm_eps > 0, f"Norm epsilon must be positive, got {self.norm_eps}"
        assert 0.0 <= self.dropout <= 1.0, f"Dropout must be between 0 and 1, got {self.dropout}"
        
        LOGGER.debug("Architecture parameters validated successfully")
        
        # Validate training parameters
        assert self.learning_rate > 0, f"Learning rate must be positive, got {self.learning_rate}"
        assert self.batch_size > 0, f"Batch size must be positive, got {self.batch_size}"
        assert self.epochs > 0, f"Number of epochs must be positive, got {self.epochs}"
        assert self.accumulation_steps > 0, f"Accumulation steps must be positive, got {self.accumulation_steps}"
        assert self.grad_clip > 0, f"Gradient clipping must be positive, got {self.grad_clip}"
        assert self.log_interval > 0, f"Log interval must be positive, got {self.log_interval}"
        assert self.save_interval > 0, f"Save interval must be positive, got {self.save_interval}"
        
        LOGGER.debug("Training parameters validated successfully")
        
        LOGGER.info("Configuration validation completed successfully")
            

    def estimate_model_size_mb(self) -> float:
        """
        Estimate the size of the model in megabytes (MB).
        
        This method calculates the total number of parameters in the model
        and estimates the size based on 4 bytes per parameter (float32).
        
        Returns:
            float: Estimated model size in MB
        """
        # Embedding parameters
        embedding_params = self.vocab_size * self.dim
        
        # Calculate FFN hidden dimension if not specified
        if self.hidden_dim is None:
            ffn_hidden = (4 * self.dim * 2) // 3
            ffn_hidden = self.multiple_of * ((ffn_hidden + self.multiple_of - 1) // self.multiple_of)
        else:
            ffn_hidden = self.hidden_dim
        
        # Calculation for multi-query attention
        head_dim = self.dim // self.n_heads
        attention_params = (
            self.dim * self.dim +  # wq (query projection)
            self.n_kv_heads * head_dim * self.dim +  # wk (key projection)
            self.n_kv_heads * head_dim * self.dim +  # wv (value projection)
            self.dim * self.dim    # wo (output projection)
        )
        
        # Standard FFN parameters
        ffn_params = 3 * self.dim * ffn_hidden  # w1, w2, w3
        
        # Layer normalization parameters
        layer_norm_params = 2 * self.dim  # attention and ffn layer norms
        
        # Total layer parameters
        layer_params = attention_params + ffn_params + layer_norm_params
        
        # Total parameters
        total_params = (
            embedding_params +  # input embeddings
            (self.n_layers * layer_params) +  # all transformer layers
            self.dim +  # final layer norm
            self.vocab_size * self.dim  # output projection (lm_head)
        )
        
        # Estimate size in MB (4 bytes per parameter for float32)
        size_mb = (total_params * 4) / (1024 * 1024)
        
        LOGGER.debug(f"Model size estimation: {total_params:,} parameters, {size_mb:.1f} MB")
        return size_mb
       
    
    def print_config_summary(self) -> None:
        """Log a summary of the model configuration using the logging module."""
        estimated_size = self.estimate_model_size_mb()
        
        summary_lines = [
            "LLaMA Model Configuration Summary",
            "=" * 50,
            "",
            "Architecture Parameters:",
            f"  - Embedding Dimension: {self.dim}",
            f"  - Number of Layers: {self.n_layers}",
            f"  - Attention Heads: {self.n_heads}",
            f"  - Key-Value Heads: {self.n_kv_heads}",
            f"  - Vocabulary Size: {self.vocab_size:,}",
            f"  - Max Sequence Length: {self.max_seq_len}",
            f"  - Hidden Dimension: {self.hidden_dim or 'Auto-calculated'}",
            f"  - Dropout: {self.dropout}",
            f"  - Flash Attention: {self.flash_attn}",
            f"  - RoPE Theta: {self.rope_theta}",
            "",
            "Training Parameters:",
            f"  - Epochs: {self.epochs}",
            f"  - Batch Size: {self.batch_size}",
            f"  - Learning Rate: {self.learning_rate}",
            f"  - Gradient Accumulation Steps: {self.accumulation_steps}",
            f"  - Gradient Clipping: {self.grad_clip}",
            f"  - Warmup Iterations: {self.warmup_iters}",
            f"  - Log Interval: {self.log_interval}",
            f"  - Save Interval: {self.save_interval}",
            "",
            "System Configuration:",
            f"  - Device: {self.device}",
            f"  - Data Type: {self.dtype}",
            f"  - Output Directory: {self.out_dir}",
            f"  - Data Path: {self.data_path}",
            f"  - Tokenizer Path: {self.tokenizer_path}",
            "",
            f"Estimated Model Size: {estimated_size:.1f} MB"
        ]
        
        # Log each line of the summary
        for line in summary_lines:
            LOGGER.info(line)