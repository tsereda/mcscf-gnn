import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional


class GradNormLoss(nn.Module):
    """
    GradNorm: Gradient Normalization for Adaptive Loss Balancing in Deep Multitask Networks
    Reference: https://arxiv.org/abs/1711.02257
    
    Automatically balances task weights by:
    1. Computing gradient norms for each task
    2. Adjusting weights to balance training rates across tasks
    3. Using a restoring force hyperparameter (alpha) to control balance
    """
    
    def __init__(self, 
                 num_tasks: int = 3,
                 alpha: float = 1.5,
                 learning_rate: float = 0.025,
                 initial_weights: Optional[List[float]] = None):
        """
        Args:
            num_tasks: Number of tasks (3 for our case: occupation, keibo, energy)
            alpha: Restoring force hyperparameter (controls how aggressively to balance)
                   - alpha = 0: No balancing (static weights)
                   - alpha = 1.5: Recommended default value
                   - Higher alpha: More aggressive balancing
            learning_rate: Learning rate for updating task weights
            initial_weights: Initial task weights (if None, uses equal weights)
        """
        super(GradNormLoss, self).__init__()
        
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.lr = learning_rate
        
        # Initialize task weights as learnable parameters
        if initial_weights is None:
            initial_weights = [1.0] * num_tasks
        
        self.task_weights = nn.Parameter(
            torch.tensor(initial_weights, dtype=torch.float32)
        )
        
        # Track initial losses for normalization
        self.initial_losses = None
        self.initial_loss_set = False
        
        # Statistics tracking
        self.weight_history = []
        self.loss_ratio_history = []
        self.grad_norm_history = []
        
        self.mse = nn.MSELoss()
    
    def set_initial_losses(self, losses: List[float]) -> None:
        """
        Set initial losses from the first training step.
        Used to normalize loss ratios.
        
        Args:
            losses: List of initial losses [occupation_loss, keibo_loss, energy_loss]
        """
        self.initial_losses = torch.tensor(losses, dtype=torch.float32)
        self.initial_loss_set = True
        print(f"\nGradNorm initial losses set:")
        print(f"  Occupation: {losses[0]:.6f}")
        print(f"  KEI-BO: {losses[1]:.6f}")
        print(f"  Energy: {losses[2]:.6f}")
    
    def compute_grad_norm(self, loss: torch.Tensor, 
                         model_parameters, 
                         retain_graph: bool = True) -> torch.Tensor:
        """
        Compute L2 norm of gradients for a specific task loss.
        
        Args:
            loss: Task-specific loss
            model_parameters: Model parameters to compute gradients for
            retain_graph: Whether to retain computation graph
            
        Returns:
            L2 norm of gradients
        """
        # Convert model_parameters to list to ensure it's reusable
        param_list = list(model_parameters)
        
        grads = torch.autograd.grad(
            loss, 
            param_list, 
            retain_graph=retain_graph,
            create_graph=False,
            allow_unused=True
        )
        
        # Filter out None gradients
        valid_grads = [g.flatten() for g in grads if g is not None]
        
        # Handle case where all gradients are None
        if len(valid_grads) == 0:
            return torch.tensor(1e-8, device=loss.device)
        
        # Compute norm
        grad_norm = torch.norm(torch.cat(valid_grads))
        
        return grad_norm
    
    def forward(self, 
                occupation_pred: torch.Tensor, 
                keibo_pred: torch.Tensor, 
                energy_pred: torch.Tensor,
                occupation_target: torch.Tensor, 
                keibo_target: torch.Tensor, 
                energy_target: torch.Tensor,
                model_parameters = None,
                update_weights: bool = True) -> Tuple[torch.Tensor, Dict]:
        """
        Compute weighted multi-task loss with GradNorm balancing.
        
        Args:
            occupation_pred, keibo_pred, energy_pred: Model predictions
            occupation_target, keibo_target, energy_target: Ground truth targets
            model_parameters: Model parameters (needed for gradient computation)
            update_weights: Whether to update task weights (set False during validation)
            
        Returns:
            total_loss: Weighted sum of task losses
            loss_dict: Dictionary with loss components and statistics
        """
        # Compute individual task losses
        occupation_loss = self.mse(occupation_pred, occupation_target)
        keibo_loss = self.mse(keibo_pred, keibo_target)
        energy_loss = self.mse(energy_pred, energy_target)
        
        losses = [occupation_loss, keibo_loss, energy_loss]
        
        # Set initial losses on first call
        if not self.initial_loss_set:
            self.set_initial_losses([l.item() for l in losses])
        
        # Ensure weights are positive
        weights = torch.softmax(self.task_weights, dim=0) * self.num_tasks
        
        # Update weights using GradNorm (only during training)
        # IMPORTANT: Do this BEFORE computing the weighted total loss
        if update_weights and model_parameters is not None and self.initial_loss_set:
            # Use detached weights for gradient norm computation
            with torch.no_grad():
                weights_for_gradnorm = weights.clone()
            self._update_weights_gradnorm(losses, weights_for_gradnorm, model_parameters)
            # Re-compute weights after update
            weights = torch.softmax(self.task_weights, dim=0) * self.num_tasks
        
        # Compute weighted total loss for main backward pass
        weighted_losses = [w * l for w, l in zip(weights, losses)]
        total_loss = sum(weighted_losses)
        
        # Prepare loss dictionary with orbital naming convention
        loss_dict = {
            'total_loss': total_loss.item(),
            'occupation_loss': occupation_loss.item(),
            'keibo_loss': keibo_loss.item(),
            'energy_loss': energy_loss.item(),
            'occupation_weight': weights[0].item(),
            'keibo_weight': weights[1].item(),
            'energy_weight': weights[2].item()
        }
        
        return total_loss, loss_dict
    
    def _update_weights_gradnorm(self, 
                                 losses: List[torch.Tensor],
                                 weights: torch.Tensor,
                                 model_parameters) -> None:
        """
        Update task weights using GradNorm algorithm.
        
        The algorithm:
        1. Compute loss ratios: L_i(t) / L_i(0)
        2. Compute average loss ratio across tasks
        3. Compute gradient norms for each task
        4. Compute target gradient norms based on inverse training rates
        5. Update weights to minimize difference between actual and target grad norms
        """
        # Compute loss ratios (current loss / initial loss)
        loss_ratios = torch.stack([
            l / l0 for l, l0 in zip(losses, self.initial_losses.to(losses[0].device))
        ])
        
        # Compute average loss ratio
        avg_loss_ratio = loss_ratios.mean()
        
        # Compute inverse training rate: r_i = L_i(t) / L_avg(t)
        inverse_train_rates = loss_ratios / avg_loss_ratio
        
        # Compute gradient norms for each task
        grad_norms = []
        
        for i, (loss, weight) in enumerate(zip(losses, weights)):
            weighted_loss = weight * loss
            
            grad_norm = self.compute_grad_norm(
                weighted_loss, 
                model_parameters,
                retain_graph=True
            )
            grad_norms.append(grad_norm)
        
        grad_norms = torch.stack(grad_norms)
        
        # Compute average gradient norm
        avg_grad_norm = grad_norms.mean()
        
        # Compute target gradient norms: G_target = avg_grad * (r_i)^alpha
        target_grad_norms = avg_grad_norm * (inverse_train_rates ** self.alpha)
        
        # Compute GradNorm loss (L1 distance between actual and target grad norms)
        gradnorm_loss = torch.abs(grad_norms - target_grad_norms).sum()
        
        # Update task weights
        try:
            weight_grads = torch.autograd.grad(
                gradnorm_loss,
                self.task_weights,
                retain_graph=False,
                allow_unused=True
            )[0]
            
            # Only update if gradients exist
            if weight_grads is not None:
                with torch.no_grad():
                    self.task_weights.data -= self.lr * weight_grads
                    # Renormalize to prevent drift
                    self.task_weights.data = torch.clamp(self.task_weights.data, min=-10, max=10)
        except RuntimeError as e:
            # If we can't compute gradients, skip this update
            print(f"Warning: GradNorm weight update failed: {e}")
            pass
        
        # Track statistics (using detached values)
        self.weight_history.append(weights.detach().cpu().numpy().copy())
        self.loss_ratio_history.append(loss_ratios.detach().cpu().numpy().copy())
        self.grad_norm_history.append(grad_norms.detach().cpu().numpy().copy())
    
    def get_statistics(self) -> Dict:
        """Get training statistics for analysis."""
        if not self.weight_history:
            return {}
        
        weights_array = np.array(self.weight_history)
        loss_ratios_array = np.array(self.loss_ratio_history)
        grad_norms_array = np.array(self.grad_norm_history)
        
        return {
            'weight_history': weights_array,
            'loss_ratio_history': loss_ratios_array,
            'grad_norm_history': grad_norms_array,
            'final_weights': weights_array[-1] if len(weights_array) > 0 else None,
            'avg_weights': weights_array.mean(axis=0) if len(weights_array) > 0 else None
        }
    
    def plot_statistics(self, save_path: str = None) -> None:
        """Plot GradNorm training statistics."""
        import matplotlib.pyplot as plt
        
        stats = self.get_statistics()
        if not stats:
            print("No statistics to plot")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        task_names = ['Occupation', 'KEI-BO', 'Energy']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        
        # Plot 1: Task weights over time
        for i, (name, color) in enumerate(zip(task_names, colors)):
            axes[0].plot(stats['weight_history'][:, i], 
                        label=name, color=color, linewidth=2)
        axes[0].set_title('Task Weights Evolution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Training Step')
        axes[0].set_ylabel('Weight')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Loss ratios over time
        for i, (name, color) in enumerate(zip(task_names, colors)):
            axes[1].plot(stats['loss_ratio_history'][:, i], 
                        label=name, color=color, linewidth=2)
        axes[1].set_title('Loss Ratios (L_t / L_0)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Training Step')
        axes[1].set_ylabel('Loss Ratio')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_yscale('log')
        
        # Plot 3: Gradient norms over time
        for i, (name, color) in enumerate(zip(task_names, colors)):
            axes[2].plot(stats['grad_norm_history'][:, i], 
                        label=name, color=color, linewidth=2)
        axes[2].set_title('Gradient Norms', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Training Step')
        axes[2].set_ylabel('Gradient L2 Norm')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_yscale('log')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GradNorm statistics saved to {save_path}")
        else:
            plt.show()
        
        plt.close()