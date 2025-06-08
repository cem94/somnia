"""
Module for training metrics tracking and visualization.

This module provides comprehensive tracking and visualization of training metrics
including loss, learning rate, perplexity, and gradient norms. It supports
checkpoint recovery and generates detailed training progress plots.

Usage:
    Initialize MetricsTracker() and use add_metrics() during training.
    Call save_plots_and_metrics() to persist data and generate visualizations.
"""

import os
import json
import math
import matplotlib.pyplot as plt
from typing import List, Optional
from utility.logger import LOGGER


class MetricsTracker:
    """
    Class to track and visualize training metrics with checkpoint support.
    
    This class handles the collection, storage, and visualization of training metrics
    throughout the model training process. It supports checkpoint recovery for
    interrupted training sessions and generates comprehensive progress plots.
    
    Attributes:
        losses: List of training loss values
        learning_rates: List of learning rate values
        steps: List of training step numbers
        epochs: List of epoch numbers
        grad_norms: List of gradient norm values
        perplexities: List of computed perplexity values
        metric_dir: Directory for storing metrics files
        metrics_file: Full path to the metrics JSON file
    """
    
    def __init__(self, metric_dir: str, metrics_file: str = "training_metrics.json"):
        """
        Initialize the MetricsTracker.
        
        Args:
            metric_dir: Directory path for storing metrics files
            metrics_file: Name of the metrics JSON file
        """
        self.losses: List[float] = []
        self.learning_rates: List[float] = []
        self.steps: List[int] = []
        self.epochs: List[int] = []
        self.grad_norms: List[float] = []
        self.perplexities: List[float] = []
        self.metric_dir = metric_dir
        self.metrics_file = os.path.join(self.metric_dir, metrics_file)

        # Ensure directory exists
        os.makedirs(self.metric_dir, exist_ok=True)
        
        # Load existing metrics if file exists
        if os.path.exists(self.metrics_file):
            self._load_metrics()
            
        LOGGER.info(f"MetricsTracker initialized with storage at: {self.metric_dir}")
    
    def add_metrics(self, step: int, epoch: int, loss: float, lr: float, grad_norm: Optional[float] = None) -> None:
        """
        Add training metrics for current step.
        
        Args:
            step: Current training step
            epoch: Current epoch number
            loss: Training loss value
            lr: Current learning rate
            grad_norm: Gradient norm value (optional)
        """
        self.steps.append(step)
        self.epochs.append(epoch)
        self.losses.append(loss)
        self.learning_rates.append(lr) 
        
        # Calculate perplexity with overflow protection
        perplexity = math.exp(min(loss, 10.0))  # Cap to avoid overflow
        self.perplexities.append(perplexity)
        
        if grad_norm is not None:
            self.grad_norms.append(grad_norm)
            
        LOGGER.debug(f"Step {step}: Loss={loss:.4f}, LR={lr:.2e}, Perplexity={perplexity:.2f}")
    
    def save_metrics(self) -> None:
        """
        Save metrics to file for checkpoint recovery.
        
        Persists all collected metrics to a JSON file for recovery
        in case of training interruption.
        """
        if not self.metrics_file:
            LOGGER.warning("No metrics file path specified, skipping save")
            return
                    
        metrics_data = {
            'steps': self.steps,
            'epochs': self.epochs,
            'losses': self.losses,
            'learning_rates': self.learning_rates,
            'perplexities': self.perplexities,
            'grad_norms': self.grad_norms
        }
        
        try:
            with open(self.metrics_file, 'w', encoding='utf-8') as f:
                json.dump(metrics_data, f, indent=2)
            LOGGER.debug(f"Metrics saved to {self.metrics_file}")
        except Exception as e:
            LOGGER.error(f"Failed to save metrics to {self.metrics_file}: {e}")
    
    def _load_metrics(self) -> None:
        """
        Load metrics from checkpoint file.
        
        Attempts to load previously saved metrics for training recovery.
        If loading fails, initializes with empty metrics.
        """
        if not self.metrics_file or not os.path.exists(self.metrics_file):
            LOGGER.debug("No existing metrics file found, starting fresh")
            return
            
        try:
            with open(self.metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            
            self.steps = metrics_data.get('steps', [])
            self.epochs = metrics_data.get('epochs', [])
            self.losses = metrics_data.get('losses', []) 
            self.learning_rates = metrics_data.get('learning_rates', [])
            self.perplexities = metrics_data.get('perplexities', [])
            self.grad_norms = metrics_data.get('grad_norms', [])
            
            LOGGER.info(f"Loaded {len(self.steps)} training metrics from {self.metrics_file}")
            
        except Exception as e:
            LOGGER.warning(f"Failed to load metrics from {self.metrics_file}: {e}")
            LOGGER.info("Initializing with empty metrics")
    
    def _detect_training_interruptions(self) -> List[int]:
        """
        Detect training interruptions by finding gaps in step sequence.
        
        Returns:
            List of indices where training was restarted
        """
        interruptions = []
        for i in range(1, len(self.steps)):
            # Consider a gap of more than 1 step as an interruption
            if self.steps[i] - self.steps[i-1] > 1:
                interruptions.append(i)
        return interruptions
    
    def plot_training_progress(self, save_path: str) -> None:
        """
        Create comprehensive training plots with interruption markers.
        
        Args:
            save_path: Path where to save the combined plot
        """
        if not self.steps:
            LOGGER.warning("No metrics to plot")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Progress', fontsize=16, fontweight='bold')
        
        # Detect training interruptions
        interruptions = self._detect_training_interruptions()
        
        # Plot 1: Loss over time
        ax1.plot(self.steps, self.losses, 'b-', linewidth=2, alpha=0.8, label='Training Loss')
        for interrupt_idx in interruptions:
            ax1.axvline(x=self.steps[interrupt_idx], color='red', linestyle='--', alpha=0.7, 
                       label='Restart' if interrupt_idx == interruptions[0] else "")
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        if interruptions:
            ax1.legend()
        
        # Plot 2: Learning rate schedule
        ax2.plot(self.steps, self.learning_rates, 'r-', linewidth=2, alpha=0.8, label='Learning Rate')
        for interrupt_idx in interruptions:
            ax2.axvline(x=self.steps[interrupt_idx], color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Perplexity over time
        ax3.plot(self.steps, self.perplexities, 'g-', linewidth=2, alpha=0.8, label='Perplexity')
        for interrupt_idx in interruptions:
            ax3.axvline(x=self.steps[interrupt_idx], color='red', linestyle='--', alpha=0.7)
        ax3.set_xlabel('Training Steps')
        ax3.set_ylabel('Perplexity')
        ax3.set_title('Training Perplexity')
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Plot 4: Gradient norm (if available)
        if self.grad_norms and len(self.grad_norms) > 0:
            # Handle case where gradient norms might be fewer than total steps
            grad_steps = self.steps[-len(self.grad_norms):]
            ax4.plot(grad_steps, self.grad_norms, 'purple', linewidth=2, alpha=0.8, label='Gradient Norm')
            
            # Add interruption markers for gradient norm plot
            for interrupt_idx in interruptions:
                if interrupt_idx >= len(self.steps) - len(self.grad_norms):
                    grad_interrupt_idx = interrupt_idx - (len(self.steps) - len(self.grad_norms))
                    if 0 <= grad_interrupt_idx < len(grad_steps):
                        ax4.axvline(x=grad_steps[grad_interrupt_idx], color='red', linestyle='--', alpha=0.7)
            
            ax4.set_xlabel('Training Steps')
            ax4.set_ylabel('Gradient Norm')
            ax4.set_title('Gradient Norm')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Gradient Norm\nNot Available', 
                    ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('Gradient Norm')
        
        # Add restart information
        if interruptions:
            restart_text = f"Training restarted {len(interruptions)} time(s)"
            fig.text(0.02, 0.02, restart_text, fontsize=10, style='italic', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        LOGGER.info(f"Training plots saved to {save_path}")
        
        # Generate individual plots for detailed analysis
        self._save_individual_plots(os.path.dirname(save_path))
    
    def _save_individual_plots(self, save_dir: str) -> None:
        """
        Save individual metric plots for detailed analysis.
        
        Args:
            save_dir: Directory where to save individual plots
        """
        if not self.steps:
            LOGGER.warning("No metrics available for individual plots")
            return
            
        # Individual loss plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.losses, 'b-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        loss_plot_path = os.path.join(save_dir, 'loss_curve.png')
        plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        LOGGER.debug(f"Individual loss plot saved to {loss_plot_path}")
        
        # Individual perplexity plot
        plt.figure(figsize=(10, 6))
        plt.plot(self.steps, self.perplexities, 'g-', linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Perplexity')
        plt.title('Training Perplexity Over Time')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        perplexity_plot_path = os.path.join(save_dir, 'perplexity_curve.png')
        plt.savefig(perplexity_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        LOGGER.debug(f"Individual perplexity plot saved to {perplexity_plot_path}")
    
    def save_plots_and_metrics(self, output_dir: str, plot_name: str = "training_progress.png") -> None:
        """
        Save both metrics data and plots.
        
        Args:
            output_dir: Directory where to save files
            plot_name: Name for the combined plot file
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics data
        self.save_metrics()
        
        # Generate and save plots if we have data
        if self.steps:
            plots_path = os.path.join(output_dir, plot_name)
            self.plot_training_progress(plots_path)
            LOGGER.info(f"Training metrics and plots saved to {output_dir}")
        else:
            LOGGER.warning("No training metrics available for plotting")