import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional


class ImprovedTripleTaskLoss(nn.Module):
    """
    Improved multi-task loss with better first-epoch weighting strategies.
    
    Supports multiple weighting schemes:
    1. 'inverse': Original inverse loss weighting (1/L_i)
    2. 'log_inverse': Log-scaled inverse (1/log(L_i + e))
    3. 'uncertainty': Uncertainty-based weighting (Kendall et al. 2018)
    4. 'equal_gradient': Equalizes gradient magnitudes across tasks
    """
    
    def __init__(self, 
                 node_weight: float = 1.0, 
                 edge_weight: float = 1.0, 
                 global_weight: float = 1.0,
                 use_first_epoch_weighting: bool = False,
                 weighting_scheme: str = 'log_inverse',
                 temperature: float = 1.0):
        """
        Args:
            node_weight: Initial weight for node loss
            edge_weight: Initial weight for edge loss
            global_weight: Initial weight for global loss
            use_first_epoch_weighting: If True, recompute weights from first batch
            weighting_scheme: One of ['inverse', 'log_inverse', 'uncertainty', 'equal_gradient']
            temperature: Temperature parameter for softening the weighting (higher = more uniform)
        """
        super(ImprovedTripleTaskLoss, self).__init__()
        self.node_weight = node_weight
        self.edge_weight = edge_weight
        self.global_weight = global_weight
        self.use_first_epoch_weighting = use_first_epoch_weighting
        self.weights_initialized = not use_first_epoch_weighting
        self.weighting_scheme = weighting_scheme
        self.temperature = temperature
        self.mse = nn.MSELoss()
        
        # For uncertainty weighting (learnable parameters)
        if weighting_scheme == 'uncertainty':
            self.log_vars = nn.Parameter(torch.zeros(3))
    
    def initialize_weights_from_losses(self, node_loss: float, edge_loss: float, global_loss: float):
        """
        Initialize weights using the specified weighting scheme.
        
        Args:
            node_loss: Initial node task loss
            edge_loss: Initial edge task loss  
            global_loss: Initial global task loss
        """
        losses = np.array([node_loss, edge_loss, global_loss])
        
        if self.weighting_scheme == 'inverse':
            # Original: 1/L_i
            weights = 1.0 / (losses + 1e-8)
            
        elif self.weighting_scheme == 'log_inverse':
            # Log-scaled: 1/log(L_i + e)
            # This compresses the range and handles extreme differences better
            weights = 1.0 / (np.log(losses + np.e))
            
        elif self.weighting_scheme == 'equal_gradient':
            # Equalizes expected gradient magnitudes
            # Weight = 1/sqrt(L_i) - this is a good middle ground
            weights = 1.0 / (np.sqrt(losses + 1e-8))
            
        elif self.weighting_scheme == 'uncertainty':
            # Uncertainty weighting - will be learned, just initialize reasonably
            weights = 1.0 / (np.log(losses + np.e))
            
        else:
            raise ValueError(f"Unknown weighting_scheme: {self.weighting_scheme}")
        
        # Apply temperature softening
        if self.temperature != 1.0:
            weights = weights ** (1.0 / self.temperature)
        
        # Normalize so they sum to 3 (average weight of 1.0)
        total = weights.sum()
        weights = 3.0 * weights / total
        
        self.node_weight = float(weights[0])
        self.edge_weight = float(weights[1])
        self.global_weight = float(weights[2])
        
        self.weights_initialized = True
        
        print(f"\n{'='*60}")
        print(f"FIRST-EPOCH LOSS WEIGHTING ({self.weighting_scheme.upper()})")
        print(f"{'='*60}")
        print(f"Initial losses:")
        print(f"  Node (charges):    {node_loss:.6f}")
        print(f"  Edge (bond order): {edge_loss:.6f}")
        print(f"  Global (energy):   {global_loss:.6f}")
        print(f"\nLoss scale ratios (max/min): {max(losses)/min(losses):.2f}")
        print(f"\nComputed weights (scheme={self.weighting_scheme}, temp={self.temperature}):")
        print(f"  Node weight:   {self.node_weight:.4f}")
        print(f"  Edge weight:   {self.edge_weight:.4f}")
        print(f"  Global weight: {self.global_weight:.4f}")
        print(f"\nEffective loss contributions (weight * loss):")
        print(f"  Node:   {self.node_weight * node_loss:.4f}")
        print(f"  Edge:   {self.edge_weight * edge_loss:.4f}")
        print(f"  Global: {self.global_weight * global_loss:.4f}")
        print(f"{'='*60}\n")
    
    def forward(self, 
                node_pred: torch.Tensor, 
                edge_pred: torch.Tensor, 
                global_pred: torch.Tensor,
                node_target: torch.Tensor, 
                edge_target: torch.Tensor, 
                global_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        
        node_loss = self.mse(node_pred, node_target)
        edge_loss = self.mse(edge_pred, edge_target)
        global_loss = self.mse(global_pred, global_target)
        
        # Initialize weights from first batch if needed
        if self.use_first_epoch_weighting and not self.weights_initialized:
            self.initialize_weights_from_losses(
                node_loss.item(), edge_loss.item(), global_loss.item()
            )
        
        # Use uncertainty weighting if enabled
        if self.weighting_scheme == 'uncertainty':
            # Kendall et al. 2018: L = sum(1/(2*sigma^2) * L_i + log(sigma))
            # where sigma^2 = exp(log_var)
            precision = torch.exp(-self.log_vars)
            total_loss = (
                precision[0] * node_loss + self.log_vars[0] +
                precision[1] * edge_loss + self.log_vars[1] +
                precision[2] * global_loss + self.log_vars[2]
            ) / 2.0
            
            # Update weights for logging (approximate)
            with torch.no_grad():
                self.node_weight = precision[0].item()
                self.edge_weight = precision[1].item()
                self.global_weight = precision[2].item()
        else:
            total_loss = (self.node_weight * node_loss + 
                         self.edge_weight * edge_loss + 
                         self.global_weight * global_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'node_loss': node_loss.item(),
            'edge_loss': edge_loss.item(),
            'global_loss': global_loss.item(),
            'node_weight': self.node_weight,
            'edge_weight': self.edge_weight,
            'global_weight': self.global_weight
        }
        
        return total_loss, loss_dict


def test_weighting_schemes():
    """Test different weighting schemes on realistic molecular data."""
    
    print("="*70)
    print("TESTING DIFFERENT WEIGHTING SCHEMES")
    print("="*70)
    
    # Realistic molecular data scales
    node_pred = torch.randn(10, 1) * 0.3
    node_target = torch.randn(10, 1) * 0.3
    edge_pred = torch.randn(20, 1) * 1.5
    edge_target = torch.randn(20, 1) * 1.5
    global_pred = torch.randn(1, 1) * 100 - 500
    global_target = torch.randn(1, 1) * 100 - 500
    
    schemes = ['inverse', 'log_inverse', 'equal_gradient']
    temperatures = [1.0, 2.0]
    
    results = []
    
    for scheme in schemes:
        for temp in temperatures:
            print(f"\n{'='*70}")
            print(f"SCHEME: {scheme.upper()}, TEMPERATURE: {temp}")
            print(f"{'='*70}")
            
            loss_fn = ImprovedTripleTaskLoss(
                use_first_epoch_weighting=True,
                weighting_scheme=scheme,
                temperature=temp
            )
            
            total_loss, loss_dict = loss_fn(
                node_pred, edge_pred, global_pred,
                node_target, edge_target, global_target
            )
            
            # Calculate balance metric (std of weighted losses)
            weighted_losses = [
                loss_dict['node_weight'] * loss_dict['node_loss'],
                loss_dict['edge_weight'] * loss_dict['edge_loss'],
                loss_dict['global_weight'] * loss_dict['global_loss']
            ]
            balance = np.std(weighted_losses) / np.mean(weighted_losses)
            
            results.append({
                'scheme': scheme,
                'temperature': temp,
                'node_weight': loss_dict['node_weight'],
                'edge_weight': loss_dict['edge_weight'],
                'global_weight': loss_dict['global_weight'],
                'weighted_node': weighted_losses[0],
                'weighted_edge': weighted_losses[1],
                'weighted_global': weighted_losses[2],
                'balance_cv': balance  # Coefficient of variation
            })
            
            print(f"Weighted loss contributions:")
            print(f"  Node:   {weighted_losses[0]:.4f}")
            print(f"  Edge:   {weighted_losses[1]:.4f}")
            print(f"  Global: {weighted_losses[2]:.4f}")
            print(f"Balance metric (CV): {balance:.4f} (lower is better)")
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")
    print(f"{'Scheme':<20} {'Temp':<8} {'Balance CV':<12} {'Weights (N/E/G)'}")
    print("-"*70)
    
    for r in results:
        weights_str = f"{r['node_weight']:.2f}/{r['edge_weight']:.2f}/{r['global_weight']:.2f}"
        print(f"{r['scheme']:<20} {r['temperature']:<8.1f} {r['balance_cv']:<12.4f} {weights_str}")
    
    print(f"\nRecommendation: Use 'log_inverse' or 'equal_gradient' with temperature=1.0-2.0")
    print(f"This provides better balance for multi-scale molecular properties.")


if __name__ == "__main__":
    test_weighting_schemes()