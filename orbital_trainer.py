import torch
import torch.optim as optim
from torch_geometric.data import DataLoader
import numpy as np
from typing import List, Dict, Optional
import os

from orbital_gnn import OrbitalTripleTaskGNN, OrbitalTripleTaskLoss, compute_metrics
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
                 normalizer: Optional[DataNormalizer] = None,
                 use_gradnorm: bool = False,
                 gradnorm_alpha: float = 1.5,
                 gradnorm_lr: float = 0.025,
                 use_first_epoch_weighting: bool = False,
                 wandb_enabled: bool = False,
                 device: str = None):
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.wandb_enabled = wandb_enabled
        self.use_first_epoch_weighting = use_first_epoch_weighting
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Choose loss function based on settings
        self.use_gradnorm = use_gradnorm
        
        if use_gradnorm and use_first_epoch_weighting:
            raise ValueError("Cannot use both GradNorm and first-epoch weighting simultaneously. Choose one.")
        
        if use_gradnorm:
            self.loss_fn = GradNormLoss(
                num_tasks=3,
                alpha=gradnorm_alpha,
                learning_rate=gradnorm_lr,
                initial_weights=[occupation_weight, keibo_weight, energy_weight]
            )
            print(f"Using GradNorm loss (alpha={gradnorm_alpha}, lr={gradnorm_lr})")
        elif use_first_epoch_weighting:
            self.loss_fn = OrbitalTripleTaskLoss(
                occupation_weight, keibo_weight, energy_weight,
                use_first_epoch_weighting=True
            )
            print(f"Using first-epoch loss weighting (initial: occupation={occupation_weight}, keibo={keibo_weight}, energy={energy_weight})")
            print(f"  ⚠️  Weights will be recomputed from first batch losses")
        else:
            self.loss_fn = OrbitalTripleTaskLoss(occupation_weight, keibo_weight, energy_weight)
            print(f"Using static loss weights (occupation={occupation_weight}, keibo={keibo_weight}, energy={energy_weight})")
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.normalizer = normalizer
        
        self.task_types = ['occupation', 'keibo', 'energy']
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
                original_targets = [data.y.clone(), data.edge_y.clone(), data.global_y.clone()]
                
                # Apply normalization if normalizer is provided
                if self.normalizer:
                    self.normalizer.normalize_batch(data)
                
                if is_training:
                    self.optimizer.zero_grad()
                
                # Forward pass: occupation_pred, keibo_pred, energy_pred
                preds = self.model(data)
                targs = [data.y, data.edge_y, data.global_y]
                
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
        """Print training progress with orbital-specific terminology."""
        print(f"\nEpoch {epoch}/{total}")
        for name, results in [("Train", train_results), ("Val", val_results)]:
            losses = results['losses']
            components = [
                f"occupation: {losses.get('occupation_loss', 0.0):.4f}",
                f"keibo: {losses.get('keibo_loss', 0.0):.4f}",
                f"energy: {losses.get('energy_loss', 0.0):.4f}"
            ]
            
            # Add task weights if available
            if 'occupation_weight' in losses:
                weights = f"(w: {losses['occupation_weight']:.2f}, {losses['keibo_weight']:.2f}, {losses['energy_weight']:.2f})"
                print(f"{name}: {losses['total_loss']:.4f} ({', '.join(components)}) {weights}")
            else:
                print(f"{name}: {losses['total_loss']:.4f} ({', '.join(components)})")
        
        for task in self.task_types:
            train_mse = train_results[f'{task}_metrics']['mse']
            val_mse = val_results[f'{task}_metrics']['mse']
            print(f"{task.title()}: {train_mse:.6f} / {val_mse:.6f}")
    
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