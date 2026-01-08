import torch
import numpy as np
from typing import Dict, Tuple, Optional
import pickle
import os


class DataNormalizer:
    """
    Handles normalization and denormalization of node, edge, and global targets.
    Supports both standardization (z-score) and min-max normalization.
    """
    
    def __init__(self, method: str = 'standardize'):
        """
        Args:
            method: 'standardize' for z-score normalization or 'minmax' for min-max scaling
        """
        if method not in ['standardize', 'minmax']:
            raise ValueError("method must be 'standardize' or 'minmax'")
        
        self.method = method
        self.stats = {
            'node': {'fitted': False},
            'edge': {'fitted': False},
            'global': {'fitted': False}
        }
    
    def fit(self, data_loader) -> None:
        """
        Compute normalization statistics from a DataLoader.
        
        Args:
            data_loader: PyTorch Geometric DataLoader containing training data
        """
        # Collect all values
        node_values = []
        edge_values = []
        global_values = []
        
        for batch in data_loader:
            node_values.append(batch.y.cpu().numpy().flatten())
            edge_values.append(batch.edge_y.cpu().numpy().flatten())
            global_values.append(batch.global_y.cpu().numpy().flatten())
        
        # Concatenate all values
        node_values = np.concatenate(node_values)
        edge_values = np.concatenate(edge_values)
        global_values = np.concatenate(global_values)
        
        # Compute statistics for each task
        for values, key in [(node_values, 'node'), 
                           (edge_values, 'edge'), 
                           (global_values, 'global')]:
            
            if self.method == 'standardize':
                mean = float(np.mean(values))
                std = float(np.std(values))
                # Avoid division by zero
                if std < 1e-8:
                    std = 1.0
                self.stats[key] = {
                    'mean': mean,
                    'std': std,
                    'fitted': True
                }
            else:  # minmax
                min_val = float(np.min(values))
                max_val = float(np.max(values))
                # Avoid division by zero
                if abs(max_val - min_val) < 1e-8:
                    max_val = min_val + 1.0
                self.stats[key] = {
                    'min': min_val,
                    'max': max_val,
                    'fitted': True
                }
        
        print("\nNormalization statistics computed:")
        self._print_stats()
    
    def _print_stats(self) -> None:
        """Print normalization statistics."""
        for key in ['node', 'edge', 'global']:
            if self.stats[key]['fitted']:
                print(f"\n{key.upper()}:")
                if self.method == 'standardize':
                    print(f"  Mean: {self.stats[key]['mean']:.6f}")
                    print(f"  Std:  {self.stats[key]['std']:.6f}")
                else:
                    print(f"  Min: {self.stats[key]['min']:.6f}")
                    print(f"  Max: {self.stats[key]['max']:.6f}")
    
    def normalize_value(self, value: torch.Tensor, task: str) -> torch.Tensor:
        """
        Normalize a value for a specific task.
        
        Args:
            value: Tensor to normalize
            task: One of 'node', 'edge', or 'global'
        
        Returns:
            Normalized tensor
        """
        if not self.stats[task]['fitted']:
            raise ValueError(f"Normalizer not fitted for task '{task}'. Call fit() first.")
        
        if self.method == 'standardize':
            mean = self.stats[task]['mean']
            std = self.stats[task]['std']
            return (value - mean) / std
        else:  # minmax
            min_val = self.stats[task]['min']
            max_val = self.stats[task]['max']
            return (value - min_val) / (max_val - min_val)
    
    def denormalize_value(self, value: torch.Tensor, task: str) -> torch.Tensor:
        """
        Denormalize a value for a specific task.
        
        Args:
            value: Normalized tensor to denormalize
            task: One of 'node', 'edge', or 'global'
        
        Returns:
            Denormalized tensor
        """
        if not self.stats[task]['fitted']:
            raise ValueError(f"Normalizer not fitted for task '{task}'. Call fit() first.")
        
        if self.method == 'standardize':
            mean = self.stats[task]['mean']
            std = self.stats[task]['std']
            return value * std + mean
        else:  # minmax
            min_val = self.stats[task]['min']
            max_val = self.stats[task]['max']
            return value * (max_val - min_val) + min_val
    
    def normalize_batch(self, batch) -> None:
        """
        Normalize targets in a batch in-place.
        
        Args:
            batch: PyTorch Geometric Batch object
        """
        batch.y = self.normalize_value(batch.y, 'node')
        batch.edge_y = self.normalize_value(batch.edge_y, 'edge')
        batch.global_y = self.normalize_value(batch.global_y, 'global')
    
    def denormalize_predictions(self, 
                                occupation_pred: torch.Tensor, 
                                keibo_pred: torch.Tensor, 
                                energy_pred: torch.Tensor,
                                s_percent_pred: torch.Tensor,
                                p_percent_pred: torch.Tensor,
                                d_percent_pred: torch.Tensor,
                                f_percent_pred: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Denormalize model predictions for 7-task architecture.
        
        Args:
            occupation_pred: Normalized occupation predictions (node-level)
            keibo_pred: Normalized KEI-BO predictions (edge-level)
            energy_pred: Normalized energy predictions (global-level)
            s_percent_pred: S% predictions (node-level, no normalization needed - already 0-1)
            p_percent_pred: P% predictions (node-level, no normalization needed - already 0-1)
            d_percent_pred: D% predictions (node-level, no normalization needed - already 0-1)
            f_percent_pred: F% predictions (node-level, no normalization needed - already 0-1)
        
        Returns:
            Tuple of (occupation, keibo, energy, s%, p%, d%, f%)
            Only occupation, keibo, and energy are denormalized; hybridization % are passed through
        """
        return (
            self.denormalize_value(occupation_pred, 'node'),
            self.denormalize_value(keibo_pred, 'edge'),
            self.denormalize_value(energy_pred, 'global'),
            s_percent_pred,  # No denormalization - already bounded 0-1
            p_percent_pred,  # No denormalization - already bounded 0-1
            d_percent_pred,  # No denormalization - already bounded 0-1
            f_percent_pred   # No denormalization - already bounded 0-1
        )
    
    def save(self, filepath: str) -> None:
        """Save normalization statistics to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'method': self.method,
                'stats': self.stats
            }, f)
        print(f"Normalization statistics saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load normalization statistics from a file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.method = data['method']
        self.stats = data['stats']
        print(f"Normalization statistics loaded from {filepath}")
        self._print_stats()
    
    def is_fitted(self) -> bool:
        """Check if normalizer has been fitted."""
        return all(self.stats[key]['fitted'] for key in ['node', 'edge', 'global'])


class NormalizedDataLoader:
    """
    Wrapper around DataLoader that applies normalization to batches.
    """
    
    def __init__(self, data_loader, normalizer: DataNormalizer):
        """
        Args:
            data_loader: PyTorch Geometric DataLoader
            normalizer: Fitted DataNormalizer instance
        """
        self.data_loader = data_loader
        self.normalizer = normalizer
        
        if not normalizer.is_fitted():
            raise ValueError("Normalizer must be fitted before use")
    
    def __iter__(self):
        for batch in self.data_loader:
            # Clone batch to avoid modifying original
            batch = batch.clone()
            self.normalizer.normalize_batch(batch)
            yield batch
    
    def __len__(self):
        return len(self.data_loader)
    
    @property
    def dataset(self):
        return self.data_loader.dataset


def create_normalized_loaders(train_loader, 
                              val_loader, 
                              method: str = 'standardize',
                              save_path: Optional[str] = None) -> Tuple[NormalizedDataLoader, 
                                                                         NormalizedDataLoader, 
                                                                         DataNormalizer]:
    """
    Create normalized data loaders from existing loaders.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        method: Normalization method ('standardize' or 'minmax')
        save_path: Optional path to save normalization statistics
    
    Returns:
        Tuple of (normalized_train_loader, normalized_val_loader, normalizer)
    """
    # Create and fit normalizer on training data
    normalizer = DataNormalizer(method=method)
    normalizer.fit(train_loader)
    
    # Save statistics if path provided
    if save_path:
        normalizer.save(save_path)
    
    # Create normalized loaders
    norm_train_loader = NormalizedDataLoader(train_loader, normalizer)
    norm_val_loader = NormalizedDataLoader(val_loader, normalizer)
    
    return norm_train_loader, norm_val_loader, normalizer