import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch_geometric.data import DataLoader, Data
from typing import List, Dict, Tuple, Optional
import os
from collections import defaultdict
import pandas as pd


class ModelVisualizer:
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 300):
        self.figsize = figsize
        self.dpi = dpi
        
    def plot_training_curves(self, 
                           train_losses: List[Dict], 
                           val_losses: List[Dict],
                           train_metrics: Dict[str, List[Dict]],
                           val_metrics: Dict[str, List[Dict]],
                           save_path: str = None, 
                           title_suffix: str = "") -> None:
            
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        epochs = range(1, len(train_losses) + 1)
        
        # Check what loss keys are actually available
        if train_losses:
            available_keys = train_losses[0].keys()
            print(f"Available loss keys: {list(available_keys)}")
        
        # Top row: MSE Losses - Updated for orbital naming convention
        loss_titles = ['Occupation Loss (Orbital Occupations)', 'KEI-BO Loss (Orbital Interactions)', 'Energy Loss (MCSCF Energy)']
        
        # Try both naming conventions for backward compatibility
        if train_losses and 'occupation_loss' in train_losses[0]:
            # Orbital naming convention
            loss_keys = ['occupation_loss', 'kei_bo_loss', 'energy_loss']
        else:
            # Fallback to original naming convention
            loss_keys = ['node_loss', 'edge_loss', 'global_loss']
        
        for i, (title, key) in enumerate(zip(loss_titles, loss_keys)):
            try:
                axes[0, i].plot(epochs, [l[key] for l in train_losses], 'b-', label='Train', linewidth=2)
                axes[0, i].plot(epochs, [l[key] for l in val_losses], 'r-', label='Val', linewidth=2)
                axes[0, i].set_title(f'{title}{title_suffix}', fontsize=12, fontweight='bold')
                axes[0, i].set_xlabel('Epoch')
                axes[0, i].set_ylabel('MSE Loss')
                axes[0, i].legend()
                axes[0, i].grid(True, alpha=0.3)
                axes[0, i].set_yscale('log')
            except KeyError as e:
                print(f"Warning: Loss key '{key}' not found. Available keys: {list(train_losses[0].keys()) if train_losses else 'None'}")
                axes[0, i].text(0.5, 0.5, f'Loss key "{key}"\nnot available', 
                               transform=axes[0, i].transAxes, ha='center', va='center')
                axes[0, i].set_title(f'{title}{title_suffix}', fontsize=12, fontweight='bold')
        
        # Bottom row: MAE Metrics - Updated for orbital naming convention
        mae_titles = ['Occupation MAE (Orbital Occupations)', 'KEI-BO MAE (Orbital Interactions)', 'Energy MAE (MCSCF Energy)']
        
        # Try both naming conventions for backward compatibility
        if train_metrics and 'occupation' in train_metrics:
            # Orbital naming convention
            metric_keys = ['occupation', 'kei_bo', 'energy']
        else:
            # Fallback to original naming convention
            metric_keys = ['node', 'edge', 'global']
        
        for i, (title, key) in enumerate(zip(mae_titles, metric_keys)):
            try:
                axes[1, i].plot(epochs, [m['mae'] for m in train_metrics[key]], 'b-', label='Train', linewidth=2)
                axes[1, i].plot(epochs, [m['mae'] for m in val_metrics[key]], 'r-', label='Val', linewidth=2)
                axes[1, i].set_title(f'{title}{title_suffix}', fontsize=12, fontweight='bold')
                axes[1, i].set_xlabel('Epoch')
                axes[1, i].set_ylabel('MAE')
                axes[1, i].legend()
                axes[1, i].grid(True, alpha=0.3)
                axes[1, i].set_yscale('log')
            except KeyError as e:
                print(f"Warning: Metric key '{key}' not found. Available keys: {list(train_metrics.keys()) if train_metrics else 'None'}")
                axes[1, i].text(0.5, 0.5, f'Metric key "{key}"\nnot available', 
                               transform=axes[1, i].transAxes, ha='center', va='center')
                axes[1, i].set_title(f'{title}{title_suffix}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        plt.close()
    
    def plot_cross_validation_summary(self, 
                                    all_results: List[Dict], 
                                    folder_names: List[str],
                                    save_path: str = None) -> None:
        """Create enhanced summary plots across all folds."""
        n_folds = len(all_results)
        
        # Helper function to convert to float
        def to_float(value):
            if hasattr(value, 'item'):
                return float(value.item())
            return float(value)
        
        # Detect naming convention from first result
        first_result = all_results[0] if all_results else {}
        train_metrics = first_result.get('train_metrics', {})
        val_metrics = first_result.get('val_metrics', {})
        
        if 'occupation' in train_metrics:
            # Orbital naming convention
            task_keys = ['occupation', 'kei_bo', 'energy']
            task_names = ['Occupation MSE (Orbital Occupations)', 'KEI-BO MSE (Orbital Interactions)', 'Energy MSE (MCSCF Energy)']
        else:
            # Fallback to original naming convention
            task_keys = ['node', 'edge', 'global']
            task_names = ['Node MSE (Partial Charges)', 'Edge MSE (KEI-BO)', 'Global MSE (MCSCF Energy)']
        
        # Extract final metrics for each fold
        metrics_data = {}
        for i, task_key in enumerate(task_keys):
            try:
                metrics_data[f'val_{task_key}_mse'] = [to_float(r['val_metrics'][task_key][-1]['mse']) for r in all_results]
                metrics_data[f'train_{task_key}_mse'] = [to_float(r['train_metrics'][task_key][-1]['mse']) for r in all_results]
            except (KeyError, IndexError) as e:
                print(f"Warning: Could not extract metrics for {task_key}: {e}")
                metrics_data[f'val_{task_key}_mse'] = [0.0] * n_folds
                metrics_data[f'train_{task_key}_mse'] = [0.0] * n_folds
        
        # Summary plots
        fig, axes = plt.subplots(1, 4, figsize=(24, 6))
        x_pos = np.arange(n_folds)
        width = 0.35
        
        # First three plots: MSE comparisons
        for i, task_key in enumerate(task_keys):
            val_key = f'val_{task_key}_mse'
            train_key = f'train_{task_key}_mse'
            
            if val_key in metrics_data and train_key in metrics_data:
                bars1 = axes[i].bar(x_pos - width/2, metrics_data[train_key], width, 
                                     label='Train', alpha=0.8)
                bars2 = axes[i].bar(x_pos + width/2, metrics_data[val_key], width, 
                                     label='Val', alpha=0.8)
                
                axes[i].set_title(task_names[i], fontsize=14, fontweight='bold')
                axes[i].set_ylabel('MSE')
                axes[i].set_xlabel('Validation Folder')
                axes[i].set_xticks(x_pos)
                axes[i].set_xticklabels(folder_names, rotation=45, ha='right')
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
                axes[i].set_yscale('log')
        
        # Fourth plot: Box plot for validation MSE distribution
        box_data = []
        box_labels = []
        for i, task_key in enumerate(task_keys):
            val_key = f'val_{task_key}_mse'
            if val_key in metrics_data:
                box_data.append(metrics_data[val_key])
                box_labels.append(f'{task_key.title()} MSE')
        
        if box_data:
            axes[3].boxplot(box_data, labels=box_labels)
            axes[3].set_title('Validation MSE Distribution Across Folds', fontsize=14, fontweight='bold')
            axes[3].set_ylabel('MSE (log scale)')
            axes[3].set_yscale('log')
            axes[3].grid(True, alpha=0.3)
        else:
            axes[3].text(0.5, 0.5, 'No data available\nfor box plot', 
                        transform=axes[3].transAxes, ha='center', va='center')
        
        plt.suptitle(f'Cross-Validation Results Summary ({n_folds} Folds)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"Cross-validation summary saved to {save_path}")
        plt.close()
    
def plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path=None, title_suffix=""):
    """Convenience function for plotting training curves."""
    visualizer = ModelVisualizer()
    visualizer.plot_training_curves(train_losses, val_losses, train_metrics, val_metrics, save_path, title_suffix)


def create_summary_plots(all_results, run_dir, folder_names):
    """Convenience function for creating cross-validation summary plots."""
    visualizer = ModelVisualizer()
    summary_path = os.path.join(run_dir, 'cross_validation_summary.png')
    visualizer.plot_cross_validation_summary(all_results, folder_names, summary_path)
    print(f"\nSummary plot saved to: {summary_path}")


def create_comprehensive_analysis(all_results, all_fold_info, folder_names, run_dir):
    """Create comprehensive analysis with enhanced visualizations."""
    visualizer = ModelVisualizer()