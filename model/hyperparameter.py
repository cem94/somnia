"""
Hyperparameter optimization module for Somnia Transformer.

This module provides systematic hyperparameter search capabilities for the Somnia Transformer:
  - Targeted search optimization for ~80MB model size
  - Persistent result storage and recovery
  - Comprehensive trial tracking and validation
  - Final training with best found configuration

Usage:
    Run main() to execute hyperparameter optimization with default settings.
    Or run main(skip_hyperparameter_search=False) for full optimization.
"""

import os
import json
import time
import shutil
import random
import warnings
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field

from model.transformer.tokenizer_config import TokenizerConfig
from model.train_model import main as train_model

from utility.paths import HYPERPARAMETERS_DIR, OUTPUT_DIR, PLOT_OUTPUT_DIR
from utility.logger import LOGGER

# Suppress warnings for cleaner output during optimization
warnings.filterwarnings('ignore')


@dataclass
class HyperparameterSearchSpace:
    """
    Hyperparameter search space definition optimized for ~80MB model size.
    
    Defines targeted ranges for ~80MB model size:
    - Architecture parameters optimized for memory efficiency
    - Training parameters tuned for Colab GPU constraints
    """
    
    # Architecture parameters - optimized for ~80MB model size
    # These combinations are pre-validated for architectural consistency
    model_configs: Any = field(default_factory=lambda: [
    # Small configs (~60MB)
    {'dim': 384, 'n_layers': 8, 'n_heads': 6, 'n_kv_heads': 4},
    {'dim': 448, 'n_layers': 8, 'n_heads': 8, 'n_kv_heads': 4},
    {'dim': 384, 'n_layers': 10, 'n_heads': 8, 'n_kv_heads': 4},
    {'dim': 512, 'n_layers': 6, 'n_heads': 8, 'n_kv_heads': 4},
    # Medium configs (~70-90MB)
    {'dim': 448, 'n_layers': 10, 'n_heads': 8, 'n_kv_heads': 4},
    {'dim': 512, 'n_layers': 8, 'n_heads': 8, 'n_kv_heads': 2},
    {'dim': 576, 'n_layers': 6, 'n_heads': 12, 'n_kv_heads': 4},
    {'dim': 512, 'n_layers': 10, 'n_heads': 8, 'n_kv_heads': 4},
    {'dim': 384, 'n_layers': 12, 'n_heads': 12, 'n_kv_heads': 4},
    {'dim': 640, 'n_layers': 6, 'n_heads': 10, 'n_kv_heads': 5},
    {'dim': 512, 'n_layers': 12, 'n_heads': 8, 'n_kv_heads': 4},
    {'dim': 576, 'n_layers': 10, 'n_heads': 12, 'n_kv_heads': 4},
    # Large config (~90-100MB)
    {'dim': 640, 'n_layers': 8, 'n_heads': 10, 'n_kv_heads': 5},
    {'dim': 512, 'n_layers': 14, 'n_heads': 8, 'n_kv_heads': 4},
    {'dim': 576, 'n_layers': 12, 'n_heads': 12, 'n_kv_heads': 6},
    {'dim': 640, 'n_layers': 10, 'n_heads': 10, 'n_kv_heads': 5},
    ])
    
    dropout: Any = (0.05, 0.2)
    learning_rate: Any = (5e-5, 8e-4) 
    batch_size: Any = field(default_factory=lambda: [8, 16]) 
    accumulation_steps: Any = field(default_factory=lambda: [4, 8])
    grad_clip: Any = 1.0
    
    # Training duration
    epochs: Any = 3
    
    def calculate_model_size_mb(self, config: Dict[str, Any]) -> float:
        """
        Calculate approximate model size in MB for given configuration.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            Estimated model size in megabytes
        """
        dim = config['dim']
        n_layers = config['n_layers']
        n_heads = config['n_heads']
        n_kv_heads = config['n_kv_heads']
        vocab_size = TokenizerConfig.VOCAB_SIZE
        
        # Embedding parameters (tied weights, so counted once)
        embedding_params = vocab_size * dim
        
        # Attention parameters per layer
        head_dim = dim // n_heads
        attention_params_per_layer = (
            dim * (n_heads * head_dim) +  # Query projection
            dim * (n_kv_heads * head_dim) +  # Key projection  
            dim * (n_kv_heads * head_dim) +  # Value projection
            (n_heads * head_dim) * dim 
        )
        
        # Feed-forward parameters per layer (SwiGLU variant)
        hidden_dim = int(2 * (4 * dim) / 3)
        hidden_dim = ((hidden_dim + 63) // 64) * 64  # Round to multiple of 64
        ffn_params_per_layer = (
            dim * hidden_dim * 2 +  # Gate and up projections
            hidden_dim * dim  # Down projection
        )
        
        # Layer norm parameters per layer
        norm_params_per_layer = dim * 2  # Attention norm + FFN norm
        
        # Final layer norm
        final_norm_params = dim
        
        # Total parameters
        total_params = (
            embedding_params +
            n_layers * (attention_params_per_layer + ffn_params_per_layer + norm_params_per_layer) +
            final_norm_params
        )
        
        # Convert to MB (assuming float16, so 2 bytes per parameter)
        size_mb = (total_params * 2) / (1024 * 1024)
        
        return size_mb
    
    def validate_hyperparameter_combination(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate hyperparameter combinations - now using pre-validated configs.
        
        Args:
            config: Dictionary of hyperparameters to validate
            
        Returns:
            Validated configuration
        """
        # Basic sanity checks
        assert config['dim'] % config['n_heads'] == 0, f"dim ({config['dim']}) must be divisible by n_heads ({config['n_heads']})"
        assert config['n_heads'] % config['n_kv_heads'] == 0, f"n_heads ({config['n_heads']}) must be divisible by n_kv_heads ({config['n_kv_heads']})"
        
        # Log model size
        model_size = self.calculate_model_size_mb(config)
        LOGGER.debug(f"Model configuration size: {model_size:.1f} MB")
        
        return config


@dataclass
class HyperparameterTrialResult:
    """
    Results container for a single hyperparameter optimization trial.
    
    Stores complete trial information including configuration, performance metrics,
    timing data, and error information for failed trials.
    """
    trial_identifier: str
    hyperparameter_config: Dict[str, Any]
    best_validation_loss: float
    total_training_time: float
    trial_successful: bool
    error_description: Optional[str] = None


class HyperparameterOptimizer:
    """
    Hyperparameter optimizer using targeted search strategy for ~80MB models.
    
    Performs systematic exploration of optimized hyperparameter space with:
    - Persistent result storage and recovery
    - Trial validation and error handling
    - Best configuration tracking
    """
    
    def __init__(self, maximum_trials: int = 8):
        """
        Initialize the hyperparameter optimizer with search configuration.
        
        Args:
            maximum_trials: Maximum number of optimization trials to execute
        """
        self.search_space = HyperparameterSearchSpace()
        self.maximum_trials = maximum_trials
        self.trial_results: List[HyperparameterTrialResult] = []
        self.best_trial_result: Optional[HyperparameterTrialResult] = None
        
        # Initialize output directory structure
        os.makedirs(HYPERPARAMETERS_DIR, exist_ok=True)
        
        # Attempt to load any previous optimization results
        self._load_existing_results()
        
        LOGGER.info("Hyperparameter optimizer initialized successfully.")
        LOGGER.info(f"Maximum trials configured: {maximum_trials}.")
        LOGGER.info(f"Results will be saved to: {HYPERPARAMETERS_DIR}.")
    
    def _load_existing_results(self) -> None:
        """
        Load previously saved optimization results for continuation.
        
        Attempts to restore optimization state from persistent storage,
        allowing continuation of interrupted optimization runs.
        """
        results_file_path = os.path.join(HYPERPARAMETERS_DIR, "optimization_results.json")
        
        if not os.path.exists(results_file_path):
            LOGGER.debug("No previous optimization results found.")
            return
            
        try:
            with open(results_file_path, 'r', encoding='utf-8') as results_file:
                stored_data = json.load(results_file)
                
                # Restore trial results
                if 'results' in stored_data:
                    self.trial_results = [
                        HyperparameterTrialResult(**result_data) 
                        for result_data in stored_data['results']
                    ]
                
                # Restore best result
                if stored_data.get('best_result'):
                    self.best_trial_result = HyperparameterTrialResult(**stored_data['best_result'])
                
                LOGGER.info(f"Loaded {len(self.trial_results)} previous optimization results.")
                
        except Exception as load_error:
            LOGGER.error(f"Failed to load previous optimization results: {load_error}.")
            LOGGER.warning("Starting optimization from scratch.")
    
    def _save_optimization_results(self) -> None:
        """
        Save current optimization results to persistent storage.
        
        Serializes all trial results, best configuration, and metadata
        to enable recovery and analysis of optimization progress.
        """
        results_file_path = os.path.join(HYPERPARAMETERS_DIR, "optimization_results.json")
        
        optimization_data = {
            'results': [asdict(result) for result in self.trial_results],
            'best_result': asdict(self.best_trial_result) if self.best_trial_result else None,
            'search_space_config': asdict(self.search_space),
            'optimization_metadata': {
                'maximum_trials': self.maximum_trials,
                'save_timestamp': time.time()
            }
        }
        
        try:
            with open(results_file_path, 'w', encoding='utf-8') as results_file:
                json.dump(optimization_data, results_file, indent=2)
            
            LOGGER.debug(f"Optimization results saved to {results_file_path}.")
            
        except Exception as save_error:
            LOGGER.error(f"Failed to save optimization results: {save_error}.")
    
    def _sample_targeted_hyperparameters(self) -> Dict[str, Any]:
        """
        Sample hyperparameters from pre-validated model configurations.
        
        Uses the pre-calculated model configurations to ensure consistent
        ~80MB model size while varying training hyperparameters.
        
        Returns:
            Dictionary of sampled hyperparameters with model size info.
        """
        # Select a model configuration
        model_config = random.choice(self.search_space.model_configs).copy()
        
        # Add training hyperparameters
        sampled_config = model_config.copy()
        
        # Sample training parameters
        sampled_config['dropout'] = random.uniform(*self.search_space.dropout)
        sampled_config['learning_rate'] = random.uniform(*self.search_space.learning_rate)
        sampled_config['batch_size'] = random.choice(self.search_space.batch_size)
        sampled_config['accumulation_steps'] = random.choice(self.search_space.accumulation_steps)
        sampled_config['grad_clip'] = self.search_space.grad_clip
        sampled_config['epochs'] = self.search_space.epochs
        
        # Calculate and log model size
        model_size = self.search_space.calculate_model_size_mb(sampled_config)
        
        # Validate configuration
        validated_config = self.search_space.validate_hyperparameter_combination(sampled_config)
        
        LOGGER.debug(f"Sampled configuration: {validated_config}.")
        LOGGER.debug(f"Estimated model size: {model_size:.1f} MB.")
        
        return validated_config
    
    def _execute_single_trial(self, trial_hyperparameters: Dict[str, Any], trial_number: int) -> HyperparameterTrialResult:
        """
        Execute a single hyperparameter optimization trial.
        
        Runs complete training with specified hyperparameters and captures
        performance metrics, timing data, and error information.
        
        Args:
            trial_hyperparameters: Hyperparameter configuration to evaluate
            trial_number: Sequential trial number for identification
            
        Returns:
            HyperparameterTrialResult containing complete trial information
        """
        trial_id = f"trial_{trial_number:03d}"
        trial_start_time = time.time()
        
        try:
            LOGGER.info(f"Starting optimization trial {trial_number}/{self.maximum_trials}: {trial_id}.")
            LOGGER.debug(f"Trial hyperparameters: {trial_hyperparameters}.")
            
            # Configure trial-specific output directory
            trial_output_dir = os.path.join(HYPERPARAMETERS_DIR, trial_id)
            
            # Clean up any existing trial directory
            if os.path.exists(trial_output_dir):
                shutil.rmtree(trial_output_dir)
                LOGGER.debug(f"Cleaned up existing trial directory: {trial_output_dir}")
            
            # Set trial-specific configuration
            trial_config = trial_hyperparameters.copy()
            trial_config['out_dir'] = trial_output_dir
            trial_config['plot_out_dir'] = trial_output_dir
            
            # Optimize trial configuration for hyperparameter search
            trial_config['log_interval'] = 50
            trial_config['save_interval'] = 5000
            
            # Execute model training
            best_loss = train_model(trial_config)
            
            # Calculate trial metrics
            trial_duration = time.time() - trial_start_time
            trial_successful = best_loss != float('inf')
            
            # Create trial result
            trial_result = HyperparameterTrialResult(
                trial_identifier=trial_id,
                hyperparameter_config=trial_hyperparameters.copy(),
                best_validation_loss=best_loss,
                total_training_time=trial_duration,
                trial_successful=trial_successful
            )
            
            if trial_successful:
                LOGGER.info(f"Trial {trial_id} completed successfully.")
                LOGGER.info(f"Best validation loss: {best_loss:.4f}.")
                LOGGER.info(f"Training duration: {trial_duration/60:.1f} minutes.")
            else:
                LOGGER.warning(f"Trial {trial_id} failed - infinite loss returned.")
            
            return trial_result
            
        except Exception as trial_error:
            trial_duration = time.time() - trial_start_time
            error_message = str(trial_error)
            
            LOGGER.error(f"Trial {trial_id} failed with error: {error_message}.")
            
            return HyperparameterTrialResult(
                trial_identifier=trial_id,
                hyperparameter_config=trial_hyperparameters.copy(),
                best_validation_loss=float('inf'),
                total_training_time=trial_duration,
                trial_successful=False,
                error_description=error_message
            )
            
        except KeyboardInterrupt:
            LOGGER.warning("Optimization interrupted by user.")
            self._save_optimization_results()
            raise
    
    def execute_optimization(self) -> Optional[HyperparameterTrialResult]:
        """
        Execute the complete hyperparameter optimization process.
        
        Performs systematic search across the pre-validated hyperparameter space,
        tracking progress and maintaining persistent state for recovery.
        
        Returns:
            Best trial result found during optimization, or None if no trials succeeded
        """
        LOGGER.info("Pipeline Stage 3: Starting targeted hyperparameter optimization for ~80MB model.")
        LOGGER.info(f"Optimization strategy: Targeted search with {self.maximum_trials} trials.")
        
        optimization_start_time = time.time()

        # Determine starting trial number (for continuation of interrupted runs)
        starting_trial = len(self.trial_results) + 1
        successful_trial_count = starting_trial - 1
        
        if starting_trial > 1:
            LOGGER.info(f"Continuing optimization from trial {starting_trial}.")
        
        # Execute optimization trials
        for trial_number in range(starting_trial, self.maximum_trials + 1):
            # Generate trial configuration
            trial_hyperparameters = self._sample_targeted_hyperparameters()
            
            # Execute trial
            trial_result = self._execute_single_trial(trial_hyperparameters, trial_number)
            self.trial_results.append(trial_result)
            
            # Update best result tracking
            if trial_result.trial_successful:
                successful_trial_count += 1
                
                if (self.best_trial_result is None or 
                    trial_result.best_validation_loss < self.best_trial_result.best_validation_loss):
                    self.best_trial_result = trial_result
                    LOGGER.info("New best configuration discovered!")
                    LOGGER.info(f"Best validation loss: {trial_result.best_validation_loss:.4f}.")
            
            # Save intermediate results for recovery
            self._save_optimization_results()
            
            # Progress reporting
            elapsed_time = time.time() - optimization_start_time
            LOGGER.info(f"Optimization progress: {trial_number}/{self.maximum_trials} trials completed.")
            LOGGER.info(f"Successful trials: {successful_trial_count}, Elapsed time: {elapsed_time/60:.1f} minutes.")
        
        # Generate optimization summary
        total_optimization_time = time.time() - optimization_start_time
        self._log_optimization_summary(successful_trial_count, total_optimization_time)
        
        return self.best_trial_result
    
    def _log_optimization_summary(self, successful_trials: int, total_time: float) -> None:
        """
        Log comprehensive optimization summary statistics.
        
        Args:
            successful_trials: Number of trials that completed successfully
            total_time: Total optimization time in seconds
        """
        LOGGER.info("=" * 50)
        LOGGER.info("HYPERPARAMETER OPTIMIZATION SUMMARY")
        LOGGER.info(f"Total trials executed: {len(self.trial_results)}")
        LOGGER.info(f"Successful trials: {successful_trials}")
        LOGGER.info(f"Success rate: {successful_trials/len(self.trial_results)*100:.1f}%")
        LOGGER.info(f"Total optimization time: {total_time/60:.1f} minutes")
        
        if self.best_trial_result:
            LOGGER.info(f"Best validation loss achieved: {self.best_trial_result.best_validation_loss:.4f}.")
            LOGGER.info(f"Best configuration: {self.best_trial_result.hyperparameter_config}.")
        else:
            LOGGER.warning("No successful trials found during optimization.")
    
    def execute_final_training(self, best_hyperparameters: Dict[str, Any]) -> float:
        """
        Execute final training with the best hyperparameter configuration.
        
        Performs extended training with the optimal hyperparameters found during
        the search process, using full training features and extended epochs.
        
        Args:
            best_hyperparameters: Best hyperparameter configuration from optimization
            
        Returns:
            Final best validation loss achieved during extended training
        """
        LOGGER.info("=" * 50)
        LOGGER.info("EXECUTING FINAL TRAINING WITH OPTIMAL CONFIGURATION")
        
        # Configure final training parameters
        extended_epochs = 100
        final_training_config = best_hyperparameters.copy()
        final_training_config['epochs'] = extended_epochs
        final_training_config['out_dir'] = OUTPUT_DIR
        final_training_config['plot_out_dir'] = PLOT_OUTPUT_DIR
        
        # Enable full training features
        final_training_config['log_interval'] = 20
        final_training_config['save_interval'] = 500
        
        LOGGER.info(f"Final training configuration: {final_training_config}.")
        LOGGER.info(f"Extended training epochs: {extended_epochs}.")
        
        # Execute final training
        final_training_start = time.time()
        
        try:
            final_best_loss = train_model(final_training_config)
            final_training_duration = time.time() - final_training_start
            
            # Log final training results
            LOGGER.info("=" * 50)
            LOGGER.info("FINAL TRAINING COMPLETED SUCCESSFULLY")
            LOGGER.info(f"Final best validation loss: {final_best_loss:.4f}.")
            LOGGER.info(f"Final training duration: {final_training_duration/3600:.2f} hours.")
            
            # Save complete optimization results
            self._save_final_results(best_hyperparameters, final_best_loss, 
                                   extended_epochs, final_training_duration)
            
            return final_best_loss
            
        except Exception as final_training_error:
            LOGGER.error(f"Final training failed: {final_training_error}.")
            raise
    
    def _save_final_results(self, best_config: Dict[str, Any], final_loss: float, 
                           epochs: int, training_time: float) -> None:
        """
        Save comprehensive final optimization and training results.
        
        Args:
            best_config: Best hyperparameter configuration
            final_loss: Final validation loss achieved
            epochs: Number of epochs in final training
            training_time: Final training duration in seconds
        """
        final_results_data = {
            'optimization_summary': {
                'best_hyperparameters': best_config,
                'optimization_best_loss': (self.best_trial_result.best_validation_loss 
                                          if self.best_trial_result else float('inf')),
                'final_training_best_loss': final_loss,
                'final_training_epochs': epochs,
                'final_training_duration_hours': training_time / 3600
            }
        }
        
        final_results_path = os.path.join(OUTPUT_DIR, "final_optimization_results.json")
        
        try:
            with open(final_results_path, 'w', encoding='utf-8') as results_file:
                json.dump(final_results_data, results_file, indent=2)
            
            LOGGER.info(f"Final results saved to {final_results_path}.")
            
        except Exception as save_error:
            LOGGER.error(f"Failed to save final results: {save_error}.")


def main(skip_hyperparameter_search: bool = True) -> float:
    """
    Main hyperparameter optimization execution function.
    
    Orchestrates the complete optimization process including hyperparameter search
    and final training with optimal configuration.
    
    Args:
        skip_hyperparameter_search: If True, use recommended config instead of searching
        
    Returns:
        Final best validation loss achieved
    """
    # Set random seed for reproducible optimization
    random.seed(TokenizerConfig.SEED)
    LOGGER.info(f"Random seed set to {TokenizerConfig.SEED} for reproducible optimization.")

    try:
        if skip_hyperparameter_search:
            LOGGER.info("Hyperparameter search skipped - using default configuration.")
            return train_model()

        # Initialize and execute hyperparameter optimization
        LOGGER.info("Initializing hyperparameter optimization system.")
        hyperparameter_optimizer = HyperparameterOptimizer()
        
        # Execute optimization process
        best_optimization_result = hyperparameter_optimizer.execute_optimization()

        # Execute final training with best configuration
        if best_optimization_result is not None:
            LOGGER.info("Proceeding with final training using optimal hyperparameters.")
            LOGGER.info(f"Optimal configuration: {best_optimization_result.hyperparameter_config}.")
            
            final_loss = hyperparameter_optimizer.execute_final_training(
                best_optimization_result.hyperparameter_config
            )
            
            return final_loss
        else:
            LOGGER.error("No successful optimization trials found - cannot proceed with final training.")
            raise RuntimeError("Hyperparameter optimization failed to find viable configuration.")
    except Exception as error:
        LOGGER.error(f"Hyperparameter optimization failed: {error}.")


if __name__ == "__main__":
    LOGGER.info("Starting hyperparameter optimization system")
    
    optimization_result = main(skip_hyperparameter_search=False)
    LOGGER.info(f"Hyperparameter optimization completed successfully")
    LOGGER.info(f"Final validation loss: {optimization_result:.4f}")