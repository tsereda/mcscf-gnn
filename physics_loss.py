import torch
import torch.nn as nn
from typing import Tuple, Dict, Optional

class PhysicsInformedLoss(nn.Module):
    """
    Physics-informed multi-task loss with quantum mechanical constraints.
    Enforces:
    1. KBO antisymmetry: BO(i,j) = -BO(j,i) (Pauli exclusion principle)
    2. Electron conservation: sum of orbital occupations = total electrons
    3. Occupation bounds: 0 ≤ occupation ≤ 2 (Pauli exclusion)
    4. Hybridization normalization: s% + p% + d% + f% = 1 (already enforced by softmax)
    """
    def __init__(self,
                 use_uncertainty_weighting: bool = True,
                 occupation_weight: float = 1.0,
                 kei_bo_weight: float = 1.0,
                 energy_weight: float = 1.0,
                 hybrid_weight: float = 1.0,
                 include_hybridization: bool = True,
                 antisymmetry_weight: float = 0.1,
                 electron_conservation_weight: float = 0.05,
                 occupation_bounds_weight: float = 0.01):
        super().__init__()
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.include_hybridization = include_hybridization
        self.mse = nn.MSELoss()
        self.antisymmetry_weight = antisymmetry_weight
        self.electron_conservation_weight = electron_conservation_weight
        self.occupation_bounds_weight = occupation_bounds_weight
        num_tasks = 7 if include_hybridization else 3
        if use_uncertainty_weighting:
            self.log_var_occupation = nn.Parameter(torch.zeros(1))
            self.log_var_kei_bo = nn.Parameter(torch.zeros(1))
            self.log_var_energy = nn.Parameter(torch.zeros(1))
            if include_hybridization:
                self.log_var_hybrid = nn.Parameter(torch.zeros(1))
        else:
            self.occupation_weight = occupation_weight
            self.kei_bo_weight = kei_bo_weight
            self.energy_weight = energy_weight
            self.hybrid_weight = hybrid_weight
    def compute_kbo_antisymmetry_loss(self, kei_bo_pred: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        edge_dict = {}
        for idx, (i, j) in enumerate(edge_index.t()):
            i_val, j_val = i.item(), j.item()
            edge_dict[(i_val, j_val)] = idx
        antisymmetry_violations = []
        for idx, (i, j) in enumerate(edge_index.t()):
            i_val, j_val = i.item(), j.item()
            reverse_key = (j_val, i_val)
            if reverse_key in edge_dict:
                reverse_idx = edge_dict[reverse_key]
                violation = kei_bo_pred[idx] + kei_bo_pred[reverse_idx]
                antisymmetry_violations.append(violation ** 2)
        if antisymmetry_violations:
            return torch.stack(antisymmetry_violations).mean()
        else:
            return torch.tensor(0.0, device=kei_bo_pred.device)
    def compute_electron_conservation_loss(self, occupation_pred: torch.Tensor, occupation_target: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        num_molecules = batch.max().item() + 1
        violations = []
        for mol_idx in range(num_molecules):
            mol_mask = (batch == mol_idx)
            pred_total = occupation_pred[mol_mask].sum()
            target_total = occupation_target[mol_mask].sum()
            violation = (pred_total - target_total) ** 2
            violations.append(violation)
        if violations:
            return torch.stack(violations).mean()
        else:
            return torch.tensor(0.0, device=occupation_pred.device)
    def compute_occupation_bounds_loss(self, occupation_pred: torch.Tensor) -> torch.Tensor:
        lower_violations = torch.clamp(-occupation_pred, min=0.0) ** 2
        upper_violations = torch.clamp(occupation_pred - 2.0, min=0.0) ** 2
        return (lower_violations.mean() + upper_violations.mean()) / 2.0
    def forward(self,
                occupation_pred: torch.Tensor,
                kei_bo_pred: torch.Tensor,
                energy_pred: torch.Tensor,
                s_percent_pred: torch.Tensor,
                p_percent_pred: torch.Tensor,
                d_percent_pred: torch.Tensor,
                f_percent_pred: torch.Tensor,
                occupation_target: torch.Tensor,
                kei_bo_target: torch.Tensor,
                energy_target: torch.Tensor,
                s_percent_target: torch.Tensor,
                p_percent_target: torch.Tensor,
                d_percent_target: torch.Tensor,
                f_percent_target: torch.Tensor,
                edge_index: Optional[torch.Tensor] = None,
                batch: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
        occupation_target = occupation_target.squeeze() if occupation_target.dim() > 1 else occupation_target
        kei_bo_target = kei_bo_target.squeeze() if kei_bo_target.dim() > 1 else kei_bo_target
        energy_target = energy_target.squeeze() if energy_target.dim() > 1 else energy_target
        s_percent_target = s_percent_target.squeeze() if s_percent_target.dim() > 1 else s_percent_target
        p_percent_target = p_percent_target.squeeze() if p_percent_target.dim() > 1 else p_percent_target
        d_percent_target = d_percent_target.squeeze() if d_percent_target.dim() > 1 else d_percent_target
        f_percent_target = f_percent_target.squeeze() if f_percent_target.dim() > 1 else f_percent_target
        occupation_loss = self.mse(occupation_pred, occupation_target)
        kei_bo_loss = self.mse(kei_bo_pred, kei_bo_target)
        energy_loss = self.mse(energy_pred, energy_target)
        if self.include_hybridization:
            pred_hybrid = torch.stack([s_percent_pred, p_percent_pred, d_percent_pred, f_percent_pred], dim=-1)
            target_hybrid = torch.stack([s_percent_target, p_percent_target, d_percent_target, f_percent_target], dim=-1)
            hybrid_loss = self.mse(pred_hybrid, target_hybrid)
        antisymmetry_loss = torch.tensor(0.0, device=occupation_pred.device)
        if self.antisymmetry_weight > 0 and edge_index is not None:
            antisymmetry_loss = self.compute_kbo_antisymmetry_loss(kei_bo_pred, edge_index)
        electron_conservation_loss = torch.tensor(0.0, device=occupation_pred.device)
        if self.electron_conservation_weight > 0 and batch is not None:
            electron_conservation_loss = self.compute_electron_conservation_loss(occupation_pred, occupation_target, batch)
        occupation_bounds_loss = torch.tensor(0.0, device=occupation_pred.device)
        if self.occupation_bounds_weight > 0:
            occupation_bounds_loss = self.compute_occupation_bounds_loss(occupation_pred)
        if self.use_uncertainty_weighting:
            device = self.log_var_occupation.device
            occupation_loss = occupation_loss.to(device)
            kei_bo_loss = kei_bo_loss.to(device)
            energy_loss = energy_loss.to(device)
            total_loss = (
                torch.exp(-self.log_var_occupation) * occupation_loss + self.log_var_occupation +
                torch.exp(-self.log_var_kei_bo) * kei_bo_loss + self.log_var_kei_bo +
                torch.exp(-self.log_var_energy) * energy_loss + self.log_var_energy
            )
            if self.include_hybridization:
                hybrid_loss = hybrid_loss.to(device)
                total_loss += torch.exp(-self.log_var_hybrid) * hybrid_loss + self.log_var_hybrid
            occupation_weight = torch.exp(-self.log_var_occupation).item()
            kei_bo_weight = torch.exp(-self.log_var_kei_bo).item()
            energy_weight = torch.exp(-self.log_var_energy).item()
            if self.include_hybridization:
                hybrid_weight = torch.exp(-self.log_var_hybrid).item()
        else:
            total_loss = (
                self.occupation_weight * occupation_loss +
                self.kei_bo_weight * kei_bo_loss +
                self.energy_weight * energy_loss
            )
            if self.include_hybridization:
                total_loss += self.hybrid_weight * hybrid_loss
            occupation_weight = self.occupation_weight
            kei_bo_weight = self.kei_bo_weight
            energy_weight = self.energy_weight
            if self.include_hybridization:
                hybrid_weight = self.hybrid_weight
        total_loss += self.antisymmetry_weight * antisymmetry_loss
        total_loss += self.electron_conservation_weight * electron_conservation_loss
        total_loss += self.occupation_bounds_weight * occupation_bounds_loss
        loss_dict = {
            'total_loss': total_loss.item(),
            'occupation_loss': occupation_loss.item(),
            'kei_bo_loss': kei_bo_loss.item(),
            'energy_loss': energy_loss.item(),
            'occupation_weight': occupation_weight,
            'kei_bo_weight': kei_bo_weight,
            'energy_weight': energy_weight,
            'antisymmetry_loss': antisymmetry_loss.item(),
            'electron_conservation_loss': electron_conservation_loss.item(),
            'occupation_bounds_loss': occupation_bounds_loss.item()
        }
        if self.include_hybridization:
            loss_dict.update({
                'hybrid_loss': hybrid_loss.item(),
                'hybrid_weight': hybrid_weight
            })
        return total_loss, loss_dict
