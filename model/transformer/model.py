"""
Somnia Transformer Language Model Implementation.

This module implements a complete transformer-based language model with:
- Multi-head attention with rotary positional embeddings
- Feed-forward networks with SiLU activation
- RMS normalization
- KV caching for efficient inference
- Checkpoint saving/loading functionality
- Text generation with various sampling strategies
"""

import math
import os
import torch
from torch import nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from typing import Optional, Tuple, List, Dict, Any
from model.transformer.llama_config import LLamaConfig
from model.transformer.tokenizer_config import TokenizerConfig
from transformers.modeling_outputs import CausalLMOutputWithPast
from utility.logger import LOGGER


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    
    More stable alternative to LayerNorm for transformer models.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization to input tensor."""
        norm = x.float() * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * norm.type_as(x)


def precompute_rotary_embeddings(dim: int, max_seq_len: int = 32768, theta: float = 1e6) -> torch.Tensor:
    """
    Precompute rotary positional embeddings (RoPE).

    Args:
        dim: Embedding dimension (must be even)
        max_seq_len: Maximum sequence length to precompute
        theta: Base frequency for rotary embeddings

    Returns:
        torch.Tensor: Precomputed complex exponentials for rotary embeddings
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    positions = torch.arange(max_seq_len, device=freqs.device)
    freqs = torch.outer(positions, freqs).float()
    rotary_emb = torch.polar(torch.ones_like(freqs), freqs)
    return rotary_emb


def apply_rotary_embeddings(query: torch.Tensor, key: torch.Tensor, 
                          rotary_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary positional embeddings to query and key tensors.

    Args:
        query: Query tensor [batch, seq_len, n_heads, head_dim]
        key: Key tensor [batch, seq_len, n_kv_heads, head_dim]
        rotary_emb: Precomputed rotary embeddings

    Returns:
        Tuple of query and key tensors with rotary embeddings applied
    """
    def reshape_for_broadcast(rotary_emb: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Reshape rotary embeddings to match tensor dimensions."""
        ndim = x.ndim
        assert 0 <= 1 < ndim
        assert rotary_emb.shape == (x.shape[1], x.shape[-1])
        shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
        return rotary_emb.view(*shape)

    # Convert to complex representation for efficient rotation
    query_complex = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
    key_complex = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
    rotary_emb = reshape_for_broadcast(rotary_emb, query_complex)

    # Apply rotation and convert back to real representation
    query_rotated = torch.view_as_real(query_complex * rotary_emb).flatten(3)
    key_rotated = torch.view_as_real(key_complex * rotary_emb).flatten(3)
    
    return query_rotated.type_as(query), key_rotated.type_as(key)


def repeat_key_value_heads(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat key-value heads for grouped-query attention.

    Args:
        x: Key or value tensor [batch, seq_len, n_kv_heads, head_dim]
        n_rep: Number of repetitions per head

    Returns:
        torch.Tensor: Repeated tensor [batch, seq_len, n_kv_heads * n_rep, head_dim]
    """
    if n_rep == 1:
        return x
    
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    return (
        x[:, :, :, None, :]
        .expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
        .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim)
    )


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention with rotary positional embeddings and optional KV caching.
    """

    def __init__(self, config: LLamaConfig):
        super().__init__()
        
        # Attention configuration
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads if config.n_kv_heads is not None else config.n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = config.dim // config.n_heads
        
        assert config.n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        
        # Linear projections
        self.query_proj = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.key_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.value_proj = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)
        
        # Dropout layers
        self.attention_dropout = nn.Dropout(config.dropout)
        self.residual_dropout = nn.Dropout(config.dropout)
        self.dropout_prob = config.dropout
        
        # Flash attention support
        self.use_flash_attention = (
            hasattr(torch.nn.functional, 'scaled_dot_product_attention') and 
            config.flash_attn
        )
        
        # Causal mask for standard attention
        if not self.use_flash_attention:
            causal_mask = torch.full(
                (config.max_seq_len, config.max_seq_len), 
                float("-inf"), 
                dtype=torch.float32
            )
            causal_mask = torch.triu(causal_mask, diagonal=1)
            self.register_buffer("causal_mask", causal_mask, persistent=False)
    
    def forward(self,
                hidden_states: torch.Tensor,
                rotary_emb: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for multi-head attention.

        Args:
            hidden_states: Input tensor [batch, seq_len, dim]
            rotary_emb: Rotary positional embeddings
            past_key_value: Cached key-value pairs from previous steps
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (attention_output, updated_cache)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Apply linear projections
        query = self.query_proj(hidden_states)
        key = self.key_proj(hidden_states)
        value = self.value_proj(hidden_states)
        
        # Reshape for multi-head attention
        query = query.view(batch_size, seq_len, self.n_heads, self.head_dim)
        key = key.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        value = value.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        
        # Apply rotary positional embeddings
        query, key = apply_rotary_embeddings(query, key, rotary_emb)
        
        # Handle KV cache
        if past_key_value is not None:
            past_key, past_value = past_key_value
            key = torch.cat([past_key, key], dim=1)
            value = torch.cat([past_value, value], dim=1)
        
        past_kv = (key, value) if use_cache else None
        
        # Repeat key-value heads and transpose for attention computation
        query = query.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        key = repeat_key_value_heads(key, self.n_rep).transpose(1, 2)
        value = repeat_key_value_heads(value, self.n_rep).transpose(1, 2)
        
        # Compute attention
        if self.use_flash_attention and seq_len > 1:
            # Use Flash Attention for efficiency
            dropout_p = self.dropout_prob if self.training else 0.0
            attention_output = F.scaled_dot_product_attention(
                query, key, value,
                attn_mask=None,
                dropout_p=dropout_p,
                is_causal=True
            )
        else:
            # Standard attention implementation
            attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply causal mask
            if hasattr(self, 'causal_mask'):
                current_seq_len = attention_scores.size(-1)
                causal_mask = self.causal_mask[:current_seq_len, :current_seq_len]
                attention_scores = attention_scores + causal_mask
            
            attention_probs = F.softmax(attention_scores.float(), dim=-1).type_as(query)
            attention_probs = self.attention_dropout(attention_probs)
            attention_output = torch.matmul(attention_probs, value)
        
        # Reshape and apply output projection
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, -1)
        attention_output = self.residual_dropout(self.output_proj(attention_output))
        
        return attention_output, past_kv


class FeedForwardNetwork(nn.Module):
    """
    Position-wise feed-forward network with SiLU activation.
    """

    def __init__(self, config: LLamaConfig):
        super().__init__()
        
        # Calculate hidden dimension
        if config.hidden_dim is None:
            hidden_dim = 4 * config.dim
            hidden_dim = int(2 * hidden_dim / 3)
            # Round to nearest multiple
            hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
            config.hidden_dim = hidden_dim
        
        # Linear layers
        self.gate_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.down_proj = nn.Linear(config.hidden_dim, config.dim, bias=False)
        self.up_proj = nn.Linear(config.dim, config.hidden_dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply feed-forward transformation: SiLU(xW1) * (xW3) -> (result)W2"""
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        down = self.down_proj(gate * up)
        return self.dropout(down)


class TransformerBlock(nn.Module):
    """
    Single transformer block with attention and feed-forward layers.
    """

    def __init__(self, layer_id: int, config: LLamaConfig):
        super().__init__()
        self.layer_id = layer_id
        
        # Core components
        self.self_attention = MultiHeadAttention(config)
        self.feed_forward = FeedForwardNetwork(config)
        
        # Layer normalization
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self,
                hidden_states: torch.Tensor,
                rotary_emb: torch.Tensor,
                past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                use_cache: bool = False) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass through transformer block.

        Args:
            hidden_states: Input tensor
            rotary_emb: Rotary positional embeddings
            past_key_value: Cached key-value pairs
            use_cache: Whether to return updated cache

        Returns:
            Tuple of (output_states, updated_cache)
        """
        # Self-attention with residual connection
        normed_hidden_states = self.attention_norm(hidden_states)
        attention_output, past_kv = self.self_attention(
            normed_hidden_states,
            rotary_emb,
            past_key_value=past_key_value,
            use_cache=use_cache
        )
        hidden_states = hidden_states + attention_output
        
        # Feed-forward with residual connection  
        normed_hidden_states = self.ffn_norm(hidden_states)
        ffn_output = self.feed_forward(normed_hidden_states)
        output_states = hidden_states + ffn_output
        
        return output_states, past_kv


class SomniaTransformer(PreTrainedModel):
    """
    Main Somnia Transformer Language Model with checkpoint functionality.
    """
    
    def __init__(self, config: LLamaConfig):
        super().__init__(config)
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layers = config.n_layers
        
        # Core model components
        self.token_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([
            TransformerBlock(layer_id, config) for layer_id in range(self.n_layers)
        ])
        self.final_norm = RMSNorm(config.dim, config.norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        
        # Tie token embeddings and output weights for parameter efficiency
        self.token_embeddings.weight = self.lm_head.weight
        
        # Precompute rotary positional embeddings
        self.register_buffer(
            "rotary_embeddings",
            precompute_rotary_embeddings(
                dim=config.dim // config.n_heads,
                max_seq_len=config.max_seq_len,
                theta=config.rope_theta
            ),
            persistent=False
        )
        
        # Initialize output container
        self.causal_lm_output = CausalLMOutputWithPast()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
                use_cache: bool = False,
                **kwargs) -> CausalLMOutputWithPast:
        """
        Forward pass through the transformer model.

        Args:
            input_ids: Input token IDs [batch, seq_len]
            past_key_values: Cached key-value pairs from previous forward passes
            use_cache: Whether to return key-value cache for next forward pass

        Returns:
            CausalLMOutputWithPast containing logits and optionally past_key_values
        """
        # Initialize cache if not provided
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = kwargs.get('start_pos', 0)
        
        # Get token embeddings
        hidden_states = self.dropout(self.token_embeddings(input_ids))
        
        # Get relevant rotary embeddings for current sequence
        seq_len = input_ids.size(1)
        rotary_emb = self.rotary_embeddings[start_pos:start_pos + seq_len]
        
        # Pass through transformer layers
        new_past_key_values = []
        for layer_idx, layer in enumerate(self.layers):
            hidden_states, past_kv = layer(
                hidden_states,
                rotary_emb,
                past_key_value=past_key_values[layer_idx],
                use_cache=use_cache
            )
            new_past_key_values.append(past_kv)
        
        # Final normalization and projection to vocabulary
        hidden_states = self.final_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        # Prepare output
        self.causal_lm_output.logits = logits
        self.causal_lm_output.past_key_values = new_past_key_values if use_cache else None
        
        return self.causal_lm_output

    def save_checkpoint(self, 
                        checkpoint_path: str, 
                        epoch: int, 
                        step: int, 
                        optimizer_state: Optional[Dict] = None,
                        scheduler_state: Optional[Dict] = None,
                        loss: Optional[float] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Save model checkpoint with training state.

        Args:
            checkpoint_path: Path to save checkpoint.
            epoch: Current training epoch.
            step: Current training step.
            optimizer_state: Optimizer state dict.
            scheduler_state: Learning rate scheduler state dict.
            loss: Current training loss.
            metadata: Additional metadata to save.
        """
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'loss': loss,
            'metadata': metadata or {}
        }
        
        if optimizer_state is not None:
            checkpoint['optimizer_state_dict'] = optimizer_state
            
        if scheduler_state is not None:
            checkpoint['scheduler_state_dict'] = scheduler_state
        
        torch.save(checkpoint, checkpoint_path)
        LOGGER.info(f"Checkpoint saved to {checkpoint_path} at epoch {epoch}, step {step}.")

    @classmethod
    def load_checkpoint(cls, 
                       checkpoint_path: str, 
                       map_location: Optional[str] = None) -> Tuple['SomniaTransformer', Dict[str, Any]]:
        """
        Load model from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file.
            map_location: Device to load checkpoint on.

        Returns:
            Tuple of (model, checkpoint_info) where checkpoint_info contains
            training state information.
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        
        # Create model from saved config
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Extract training state info
        checkpoint_info = {
            'epoch': checkpoint.get('epoch', 0),
            'step': checkpoint.get('step', 0),
            'loss': checkpoint.get('loss', None),
            'optimizer_state_dict': checkpoint.get('optimizer_state_dict', None),
            'scheduler_state_dict': checkpoint.get('scheduler_state_dict', None),
            'metadata': checkpoint.get('metadata', {})
        }
        
        LOGGER.info(f"Model loaded from checkpoint {checkpoint_path}.")
        LOGGER.debug(f"Checkpoint info: epoch {checkpoint_info['epoch']}, step {checkpoint_info['step']}.")
        
        return model, checkpoint_info

    @torch.inference_mode()
    def generate(self,
                input_ids: torch.Tensor,
                max_new_tokens: int = 256,
                temperature: float = 0.8,
                top_p: float = 0.9,
                repetition_penalty: float = 1.1,
                use_cache: bool = True,
                **kwargs) -> torch.Tensor:
        """
        Generate text using various sampling strategies.

        Args:
            input_ids: Input prompt tokens [batch, seq_len].
            max_new_tokens: Maximum number of new tokens to generate.
            temperature: Sampling temperature (higher = more random).
            top_p: Nucleus sampling probability threshold.
            repetition_penalty: Penalty for repeating tokens.
            use_cache: Whether to use KV caching for efficiency.

        Returns:
            torch.Tensor: Generated token sequence [batch, seq_len + new_tokens].
        """
        # Validate parameters
        assert temperature > 0.0, "Temperature must be positive."
        assert 0.0 < top_p <= 1.0, "Top-p must be between 0 and 1."
        assert repetition_penalty >= 1.0, "Repetition penalty must be >= 1.0."
        
        # Initialize generation variables
        generated_ids = input_ids.clone()
        past_key_values = None
        
        LOGGER.debug(f"Starting text generation: max_tokens={max_new_tokens}, temperature={temperature}.")
        
        # Generation loop
        for step in range(max_new_tokens):
            # Prepare input for current step
            if past_key_values is None or not use_cache:
                model_input = generated_ids
            else:
                model_input = generated_ids[:, -1:] 
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.config.dtype in ['float16', 'bfloat16']):
                outputs = self(
                    input_ids=model_input,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    start_pos=generated_ids.size(1) - model_input.size(1) if use_cache else 0
                )
            
            logits = outputs.logits[:, -1, :]  # Get logits for last position
            past_key_values = outputs.past_key_values if use_cache else None
            
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                self._apply_repetition_penalty(logits, generated_ids, repetition_penalty)
            
            # Apply temperature scaling
            logits = logits / temperature
            
            # Apply top-p filtering
            if top_p < 1.0:
                logits = self._apply_top_p_filtering(logits, top_p)
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Check for EOS token
            if (next_token == TokenizerConfig.EOS_TOKEN_ID).any():
                LOGGER.debug(f"Generation stopped at step {step} due to EOS token.")
                break
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            # Clear cache periodically to manage memory
            if use_cache and step % 500 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        LOGGER.debug(f"Text generation completed: {generated_ids.size(1)} total tokens.")
        return generated_ids

    def _apply_repetition_penalty(self, 
                                 logits: torch.Tensor, 
                                 generated_ids: torch.Tensor, 
                                 penalty: float) -> None:
        """Apply repetition penalty to logits in-place."""
        for batch_idx in range(logits.size(0)):
            prev_tokens = generated_ids[batch_idx].unique()
            for token_id in prev_tokens:
                if logits[batch_idx, token_id] > 0:
                    logits[batch_idx, token_id] /= penalty
                else:
                    logits[batch_idx, token_id] *= penalty

    def _apply_top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) sampling to logits."""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Remove tokens with cumulative probability above threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Keep at least one token
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        
        # Scatter back to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        return logits

    def get_num_parameters(self, only_trainable: bool = True) -> int:
        """
        Get total number of model parameters.

        Args:
            only_trainable: If True, count only trainable parameters

        Returns:
            int: Number of parameters
        """
        if only_trainable:
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
        else:
            return sum(p.numel() for p in self.parameters())

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get model memory usage information.

        Returns:
            Dict containing memory usage statistics in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        
        return {
            'parameters_mb': param_size / (1024 ** 2),
            'buffers_mb': buffer_size / (1024 ** 2),
            'total_mb': (param_size + buffer_size) / (1024 ** 2)
        }