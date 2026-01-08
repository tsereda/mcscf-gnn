import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
from typing import List, Dict, Optional
import os

from orbital_gnn import OrbitalTripleTaskGNN, OrbitalMultiTaskLoss, compute_metrics
from orbital_parser import OrbitalGAMESSParser
from visualization import plot_training_curves
from normalization import DataNormalizer
from gradnorm import GradNormLoss


class OrbitalGAMESSTrainer:
    """Trainer class for the orbital-centric triple-task GAMESS GNN"""
    
    def __init__(self, model: OrbitalTripleTaskGNN,
                 learning_rate: float,
                 weight_decay: float,
                 occupation_weight: float,
                 keibo_weight: float, 
                 energy_weight: float,
                 hybrid_weight: float = 1.0,
                 normalizer: Optional[DataNormalizer] = None,
                 use_uncertainty_weighting: bool = True,
                 use_gradnorm: bool = False,
                 gradnorm_alpha: float = 1.5,
                 gradnorm_lr: float = 0.025,
                 wandb_enabled: bool = False,
                 device: str = None):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.wandb_enabled = wandb_enabled
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        # Choose loss function based on settings
        self.use_gradnorm = use_gradnorm
        
        if use_gradnorm and use_uncertainty_weighting:
            raise ValueError("Cannot use both GradNorm and uncertainty weighting simultaneously. Choose one.")
        
        if use_gradnorm:
            self.loss_fn = GradNormLoss(
                num_tasks=7,  # Updated to 7 tasks
                alpha=gradnorm_alpha,
                learning_rate=gradnorm_lr,
                initial_weights=[occupation_weight, keibo_weight, energy_weight, 
                               hybrid_weight, hybrid_weight, hybrid_weight, hybrid_weight]
            )
            self.loss_fn = self.loss_fn.to(device)  # Move loss function to device
            print(f"Using GradNorm loss with 7 tasks (alpha={gradnorm_alpha}, lr={gradnorm_lr})")
        else:
            self.loss_fn = OrbitalMultiTaskLoss(
                use_uncertainty_weighting=use_uncertainty_weighting,
                occupation_weight=occupation_weight, 
                keibo_weight=keibo_weight, 
                energy_weight=energy_weight,
                hybrid_weight=hybrid_weight
            )
            self.loss_fn = self.loss_fn.to(device)  # Move loss function to device
            if use_uncertainty_weighting:
                print(f"Using uncertainty weighting for 7 tasks (automatic balancing)")
            else:
                print(f"Using static loss weights (occupation={occupation_weight}, keibo={keibo_weight}, energy={energy_weight}, hybrid={hybrid_weight})")
        
        # Initialize optimizer AFTER loss function so we can include loss parameters
        # Combine model and loss function parameters (now properly on device)
        params_to_optimize = list(model.parameters())
        if use_uncertainty_weighting or use_gradnorm:
            # Add loss function parameters after ensuring they're on the correct device
            params_to_optimize += list(self.loss_fn.parameters())
        
        self.optimizer = optim.Adam(
            params_to_optimize, 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalizer = normalizer
        
        self.task_types = ['occupation', 'keibo', 'energy', 's_percent', 'p_percent', 'd_percent', 'f_percent']
        self.train_losses, self.val_losses = [], []
        self.train_metrics = {task: [] for task in self.task_types}
        self.val_metrics = {task: [] for task in self.task_types}
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"Device: {self.device} | Parameters: {param_count:,}")
        print(f"Optimizer: Adam(lr={learning_rate}, weight_decay={weight_decay})")
        if normalizer:
            print(f"Normalization: ENABLED (method={normalizer.method})")
        else:
            print(f"Normalization: DISABLED")
    
    def _run_epoch(self, dataloader: DataLoader, is_training: bool = True):
        """Run one epoch and return results."""
        self.model.train() if is_training else self.model.eval()
        losses, predictions, targets = [], {task: [] for task in self.task_types}, {task: [] for task in self.task_types}
        
        with torch.enable_grad() if is_training else torch.no_grad():
            for data in dataloader:
                data = data.to(self.device)
                
                # Store original targets for metrics computation (before normalization)
                # 7 targets: occupation, keibo, energy, s%, p%, d%, f%
                original_targets = [
                    data.y.clone(), 
                    data.edge_y.clone(), 
                    data.global_y.clone(),
                    data.hybrid_y[:, 0:1].clone(),  # s%
                    data.hybrid_y[:, 1:2].clone(),  # p%
                    data.hybrid_y[:, 2:3].clone(),  # d%
                    data.hybrid_y[:, 3:4].clone()   # f%
                ]
                
                # Apply normalization if normalizer is provided
                if self.normalizer:
                    self.normalizer.normalize_batch(data)
                
                if is_training:
                    self.optimizer.zero_grad()
                
                # Forward pass: returns 7 predictions
                preds = self.model(data)
                
                # Prepare targets: occupation, keibo, energy, s%, p%, d%, f%
                targs = [
                    data.y, 
                    data.edge_y, 
                    data.global_y,
                    data.hybrid_y[:, 0:1],  # s%
                    data.hybrid_y[:, 1:2],  # p%
                    data.hybrid_y[:, 2:3],  # d%
                    data.hybrid_y[:, 3:4]   # f%
                ]
                
                # Compute loss based on whether GradNorm is enabled
                if self.use_gradnorm and is_training:
                    # GradNorm needs model parameters for gradient computation
                    model_params = list(self.model.parameters())
                    total_loss, loss_dict = self.loss_fn(
                        *preds, *targs,
                        model_parameters=model_params,
                        update_weights=True
                    )
                else:
                    # Standard loss or validation
                    if self.use_gradnorm:
                        total_loss, loss_dict = self.loss_fn(
                            *preds, *targs,
                            model_parameters=None,
                            update_weights=False
                        )
                    else:
                        total_loss, loss_dict = self.loss_fn(*preds, *targs)
                
                if torch.isnan(total_loss):
                    continue
                
                if is_training:
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                losses.append(loss_dict)
                
                # Denormalize predictions for metrics computation
                if self.normalizer:
                    denorm_preds = self.normalizer.denormalize_predictions(*preds)
                else:
                    denorm_preds = preds
                
                # Store denormalized predictions and original targets for metrics
                for i, task in enumerate(self.task_types):
                    predictions[task].append(denorm_preds[i].cpu().detach())
                    targets[task].append(original_targets[i].cpu())
        
        # Compute results
        avg_losses = {k: np.mean([l[k] for l in losses]) for k in losses[0].keys()} if losses else {}
        metrics = {}
        for task in self.task_types:
            if predictions[task]:
                all_preds, all_targets = torch.cat(predictions[task]), torch.cat(targets[task])
                metrics[task] = compute_metrics(all_preds, all_targets)
            else:
                metrics[task] = {'mse': float('inf'), 'mae': float('inf')}
        
        return {'losses': avg_losses, **{f'{task}_metrics': metrics[task] for task in self.task_types}}
    
    def train_epoch(self, dataloader: DataLoader) -> Dict:
        return self._run_epoch(dataloader, True)
    
    def validate(self, dataloader: DataLoader) -> Dict:
        return self._run_epoch(dataloader, False)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
            num_epochs: int, print_frequency: int) -> Dict:
        """Full training loop."""
        print(f"Training {num_epochs} epochs | Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):
            train_results = self.train_epoch(train_loader)
            val_results = self.validate(val_loader)
            
            self.train_losses.append(train_results['losses'])
            self.val_losses.append(val_results['losses'])
            for task in self.task_types:
                self.train_metrics[task].append(train_results[f'{task}_metrics'])
                self.val_metrics[task].append(val_results[f'{task}_metrics'])
            
            # Log to WandB if enabled
            if self.wandb_enabled:
                import wandb
                log_dict = {
                    'epoch': epoch + 1,
                    'train_loss': train_results['losses']['total_loss'],
                    'val_loss': val_results['losses']['total_loss'],
                    'train_occupation_mse': train_results['occupation_metrics']['mse'],
                    'train_keibo_mse': train_results['keibo_metrics']['mse'],
                    'train_energy_mse': train_results['energy_metrics']['mse'],
                    'val_occupation_mse': val_results['occupation_metrics']['mse'],
                    'val_keibo_mse': val_results['keibo_metrics']['mse'],
                    'val_energy_mse': val_results['energy_metrics']['mse'],
                }
                
                # Add weights if available
                if 'occupation_weight' in train_results['losses']:
                    log_dict.update({
                        'occupation_weight': train_results['losses']['occupation_weight'],
                        'keibo_weight': train_results['losses']['keibo_weight'],
                        'energy_weight': train_results['losses']['energy_weight']
                    })
                
                wandb.log(log_dict)
            
            if (epoch + 1) % print_frequency == 0:
                self._print_progress(epoch + 1, num_epochs, train_results, val_results)
        
        print("Orbital training completed!")
        return {'train_losses': self.train_losses, 'val_losses': self.val_losses,
                'train_metrics': self.train_metrics, 'val_metrics': self.val_metrics}
    
    def _print_progress(self, epoch, total, train_results, val_results):
        """Print training progress with compact, informative formatting."""
        losses = train_results['losses']
        val_losses = val_results['losses']
        
        # Extract task weights if available
        weights_str = ""
        if 'occupation_weight' in losses:
            weights_str = f" w=[{losses['occupation_weight']:.2f},{losses['keibo_weight']:.2f},{losses['energy_weight']:.2f}]"
        
        # Main progress line
        print(f"\nEpoch {epoch:3d}/{total} | "
              f"Loss: Tr={losses['total_loss']:.4f} Val={val_losses['total_loss']:.4f}{weights_str}")
        
        # Detailed metrics line - now showing individual loss components and MSE
        print(f"  Occ: L={losses['occupation_loss']:.4f}/{val_losses['occupation_loss']:.4f} "
              f"MSE={train_results['occupation_metrics']['mse']:.6f}/{val_results['occupation_metrics']['mse']:.6f} | "
              f"KEI: L={losses['keibo_loss']:.4f}/{val_losses['keibo_loss']:.4f} "
              f"MSE={train_results['keibo_metrics']['mse']:.6f}/{val_results['keibo_metrics']['mse']:.6f} | "
              f"Eng: L={losses['energy_loss']:.4f}/{val_losses['energy_loss']:.4f} "
              f"MSE={train_results['energy_metrics']['mse']:.1f}/{val_results['energy_metrics']['mse']:.1f}")
    
    def plot_training_curves(self, save_path: str = None, title_suffix: str = ""):
        """Plot training curves for orbital tasks."""
        if not self.train_losses:
            print("No training history to plot")
            return
        plot_training_curves(self.train_losses, self.val_losses, self.train_metrics, 
                            self.val_metrics, save_path, title_suffix)


def process_orbital_files(parser: OrbitalGAMESSParser, filepaths: List[str]) -> List:
    """Process multiple GAMESS files using orbital parser."""
    graphs, failed = [], []
    
    for filepath in filepaths:
        try:
            graphs.append(parser.parse_and_convert(filepath))
        except Exception as e:
            failed.append((filepath, str(e)))
    
    print(f"Processed {len(graphs)}/{len(filepaths)} orbital graphs")
    if failed:
        print(f"Failed ({len(failed)}): {', '.join([os.path.basename(f[0]) for f in failed[:3]])}"
              + (f" + {len(failed) - 3} more" if len(failed) > 3 else ""))
    
    return graphs