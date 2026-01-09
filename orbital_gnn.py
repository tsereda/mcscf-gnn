import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional
import numpy as np


class GaussianRBF(nn.Module):
    """
    Gaussian Radial Basis Function (RBF) distance expansion.
    
    Expands scalar distances into a vector of Gaussian basis functions,
    providing a smooth, learnable representation of distance-dependent 
    interactions (e.g., orbital kinetic energy exchange).
    
    Args:
        num_rbf: Number of Gaussian basis functions
        cutoff: Maximum distance for RBF centers
        learnable_centers: If True, RBF centers are learnable parameters
    """
    
    def __init__(self, num_rbf: int = 50, cutoff: float = 5.0, learnable_centers: bool = False):
        super(GaussianRBF, self).__init__()
        
        self.num_rbf = num_rbf
        self.cutoff = cutoff
        
        # Initialize RBF centers evenly spaced from 0 to cutoff
        centers = torch.linspace(0, cutoff, num_rbf)
        
        if learnable_centers:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer('centers', centers)
        
        # Gamma controls the width of Gaussians
        # Wider spacing → larger gamma for overlap
        self.gamma = 10.0 / cutoff
    
    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Expand distances using Gaussian RBF.
        
        Args:
            distances: [num_edges, 1] scalar distances
        
        Returns:
            rbf_expanded: [num_edges, num_rbf] expanded distances
        """
        # distances: [num_edges, 1] → [num_edges, num_rbf]
        # Compute Gaussian: exp(-gamma * (d - center)^2)
        diff = distances - self.centers.view(1, -1)  # Broadcasting
        rbf_expanded = torch.exp(-self.gamma * diff ** 2)
        
        return rbf_expanded


class OrbitalEmbedding(nn.Module):
    """Embedding layer for orbital features"""
    
    def __init__(self, max_atomic_num: int = 20, orbital_embedding_dim: int = 32):
        super(OrbitalEmbedding, self).__init__()
        
        # Store dimensions
        self.orbital_embedding_dim = orbital_embedding_dim
        self.atomic_embed_dim = orbital_embedding_dim
        self.orbital_type_embed_dim = max(1, orbital_embedding_dim // 2)  # Ensure at least 1
        self.m_quantum_embed_dim = max(1, orbital_embedding_dim // 4)     # Ensure at least 1
        
        # Embeddings for different orbital properties
        self.atomic_embedding = nn.Embedding(max_atomic_num + 1, self.atomic_embed_dim)
        self.orbital_type_embedding = nn.Embedding(4, self.orbital_type_embed_dim)  # S, P, D, F
        self.m_quantum_embedding = nn.Embedding(7, self.m_quantum_embed_dim)     # -3 to +3
        
        # Calculate actual total input features
        input_features = 1  # [occupation] - only continuous feature after embeddings
        embedding_features = self.atomic_embed_dim + self.orbital_type_embed_dim + self.m_quantum_embed_dim
        total_features = input_features + embedding_features
        
        self.feature_combiner = nn.Linear(total_features, orbital_embedding_dim)
        
        print(f"OrbitalEmbedding dimensions:")
        print(f"  atomic_embed_dim: {self.atomic_embed_dim}")
        print(f"  orbital_type_embed_dim: {self.orbital_type_embed_dim}")
        print(f"  m_quantum_embed_dim: {self.m_quantum_embed_dim}")
        print(f"  total_features: {total_features}")
        print(f"  output_dim: {orbital_embedding_dim}")
        
    def forward(self, orbital_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            orbital_features: [num_orbitals, 4] tensor with features:
                [atomic_num, orbital_type, m_quantum, occupation]
        
        Returns:
            embedded_features: [num_orbitals, orbital_embedding_dim]
        """
        # Extract discrete features for embedding
        atomic_nums = orbital_features[:, 0].long()
        orbital_types = orbital_features[:, 1].long()
        m_quantums = orbital_features[:, 2].long() + 3  # Shift to 0-6 range
        
        # Create embeddings
        atomic_embeds = self.atomic_embedding(atomic_nums)
        orbital_embeds = self.orbital_type_embedding(orbital_types)
        m_quantum_embeds = self.m_quantum_embedding(m_quantums)
        
        # Extract continuous feature (occupation)
        occupation = orbital_features[:, 3:4]  # Keep 2D shape [num_orbitals, 1]
        
        # Concatenate all features
        combined_features = torch.cat([
            occupation,
            atomic_embeds,
            orbital_embeds,
            m_quantum_embeds
        ], dim=1)
        
        # Final combination
        embedded_features = self.feature_combiner(combined_features)
        
        return embedded_features


class OrbitalMessagePassing(nn.Module):
    """Message passing layer for orbital interactions"""
    
    def __init__(self, hidden_dim: int, edge_input_dim: int = 1, dropout: float = 0.1):
        super(OrbitalMessagePassing, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.edge_input_dim = edge_input_dim
        
        # Edge network for NNConv
        # Note: edge_input_dim can be 1 (raw distance) or num_rbf (RBF-expanded)
        self.edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * hidden_dim)
        )
        
        # Message passing layer
        self.conv = NNConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            nn=self.edge_network,
            aggr='add'
        )
        
        self.batch_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_orbitals, hidden_dim] orbital features
            edge_index: [2, num_edges] edge connectivity
            edge_attr: [num_edges, edge_input_dim] edge features (distances)
        
        Returns:
            updated_x: [num_orbitals, hidden_dim] updated orbital features
        """
        # Message passing
        out = self.conv(x, edge_index, edge_attr)
        out = self.batch_norm(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        # Residual connection
        return x + out


class OrbitalAttentionPool(nn.Module):
    """Attention-based pooling for orbital-to-molecule aggregation"""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super(OrbitalAttentionPool, self).__init__()
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.global_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, orbital_embeddings: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            orbital_embeddings: [num_orbitals, hidden_dim]
            batch: [num_orbitals] batch assignment for each orbital
        
        Returns:
            global_pred: [num_molecules, 1]
        """
        # Compute attention scores
        attention_logits = self.attention(orbital_embeddings)
        
        # Apply softmax per molecule
        batch_size = batch.max().item() + 1
        attention_weights = torch.zeros_like(attention_logits)
        
        for i in range(batch_size):
            mask = (batch == i)
            if mask.sum() > 0:
                molecule_logits = attention_logits[mask]
                molecule_weights = torch.softmax(molecule_logits, dim=0)
                attention_weights[mask] = molecule_weights
        
        # Apply attention weights and pool
        weighted_orbitals = orbital_embeddings * attention_weights
        molecule_embedding = global_add_pool(weighted_orbitals, batch)
        
        # Predict global property
        global_pred = self.global_head(molecule_embedding)
        
        return global_pred


class OrbitalTripleTaskGNN(nn.Module):
    """Orbital-centric GNN for predicting orbital occupations, KEI-BO values, molecular energy, and hybridization"""
    
    def __init__(self, 
                 orbital_input_dim: int = 4,      # [atomic_num, orbital_type, m_quantum, occupation]
                 edge_input_dim: int = 1,         # distance (raw)
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 max_atomic_num: int = 20,
                 orbital_embedding_dim: int = 64,
                 global_pooling_method: str = 'attention',
                 use_rbf_distance: bool = False,
                 num_rbf: int = 50,
                 rbf_cutoff: float = 5.0,
                 include_hybridization: bool = True,
                 use_element_baselines: bool = True):
        super(OrbitalTripleTaskGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.global_pooling_method = global_pooling_method
        self.use_rbf_distance = use_rbf_distance
        self.include_hybridization = include_hybridization
        self.use_element_baselines = use_element_baselines
        
        # Per-element energy baselines (physics-informed inductive bias)
        # Learns characteristic atomic energies (e.g., H ≈ -0.5, C ≈ -37.8 hartrees)
        if use_element_baselines:
            self.element_energy_baseline = nn.Parameter(
                torch.zeros(max_atomic_num + 1)  # +1 to handle 0-indexing safely
            )
            print("✓ Using per-element energy baselines (physics-informed)")
        else:
            self.element_energy_baseline = None
            print("✗ NOT using element baselines (may struggle with size-extensivity)")
        
        # RBF distance encoding (optional)
        if use_rbf_distance:
            self.rbf_expansion = GaussianRBF(num_rbf=num_rbf, cutoff=rbf_cutoff)
            effective_edge_dim = num_rbf
            print(f"Using RBF distance encoding: {num_rbf} basis functions, cutoff={rbf_cutoff}")
        else:
            self.rbf_expansion = None
            effective_edge_dim = edge_input_dim
            print(f"Using raw distance encoding")
        
        # Orbital embedding layer
        self.orbital_embedding = OrbitalEmbedding(max_atomic_num, orbital_embedding_dim)
        
        # Initial projection to hidden dimension
        self.input_projection = nn.Linear(orbital_embedding_dim, hidden_dim)
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            OrbitalMessagePassing(hidden_dim, effective_edge_dim, dropout)
            for _ in range(num_layers)
        ])
        
        # Orbital occupation prediction head
        self.occupation_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # KEI-BO value prediction head (edge prediction)
        kei_bo_input_dim = hidden_dim * 2 + effective_edge_dim
        self.kei_bo_head = nn.Sequential(
            nn.Linear(kei_bo_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Hybridization prediction head (unified with softmax to ensure s+p+d+f=1)
        # This enforces the physical constraint that percentages must sum to 100%
        self.hybridization_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # Output 4 logits for s, p, d, f
        )
        # Softmax applied in forward pass to ensure s% + p% + d% + f% = 1.0
        
        # Global energy prediction
        if global_pooling_method == 'attention':
            self.global_pooling = OrbitalAttentionPool(hidden_dim, dropout)
        elif global_pooling_method == 'mean':
            self.global_pooling = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif global_pooling_method == 'sum':
            self.global_pooling = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1)
            )
        else:
            raise ValueError(f"Invalid global_pooling_method: {global_pooling_method}")
    
    def _compute_element_baseline(self, data: Data) -> torch.Tensor:
        """
        Compute per-molecule element baseline energy.
        
        This is the "boring" part of the energy: sum of atomic reference energies.
        The GNN will learn the "interesting" part: bonding, correlation, geometry effects.
        
        Args:
            data: PyG Data object with orbital features
            
        Returns:
            baseline_per_mol: [num_molecules] element baseline energies
        """
        # Extract atomic numbers from orbital features (first column)
        atomic_numbers = data.x[:, 0].long()  # [num_orbitals]
        
        # Look up baseline energy for each atom type
        # This learns e.g.: H→-0.5, C→-37.8, O→-75.0 hartrees
        element_energies = self.element_energy_baseline[atomic_numbers]  # [num_orbitals]
        
        # Sum per molecule
        num_molecules = data.batch.max().item() + 1
        baseline_per_mol = torch.zeros(num_molecules, device=data.x.device)
        
        # Scatter-add: sum element energies within each molecule
        baseline_per_mol.index_add_(0, data.batch, element_energies)
        
        return baseline_per_mol
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass for orbital-level predictions
        
        Args:
            data: PyTorch Geometric Data object with orbital features
        
        Returns:
            Tuple of predictions:
            - occupation_pred: [num_orbitals] predicted orbital occupations
            - kei_bo_pred: [num_edges] predicted KEI-BO values
            - energy_pred: [num_molecules] predicted molecular energies
            - s_percent_pred: [num_orbitals] predicted s character %
            - p_percent_pred: [num_orbitals] predicted p character %
            - d_percent_pred: [num_orbitals] predicted d character %
            - f_percent_pred: [num_orbitals] predicted f character %
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        
        # Apply RBF expansion to edge distances if enabled
        if self.use_rbf_distance:
            edge_attr = self.rbf_expansion(edge_attr)
        
        # Embed orbital features
        x = self.orbital_embedding(x)
        x = self.input_projection(x)
        
        # Message passing between orbitals
        for layer in self.message_layers:
            x = layer(x, edge_index, edge_attr)
        
        orbital_embeddings = x
        
        # Predict orbital occupations
        occupation_pred = self.occupation_head(orbital_embeddings).squeeze(-1)
        
        # Predict hybridization percentages with softmax constraint (conditionally)
        if self.include_hybridization:
            # Use unified hybridization head with softmax to enforce s+p+d+f=1
            hybridization_logits = self.hybridization_head(orbital_embeddings)
            hybridization_probs = torch.softmax(hybridization_logits, dim=-1)
            
            # Extract individual percentages (extract as 1D to match other predictions)
            s_percent_pred = hybridization_probs[:, 0]
            p_percent_pred = hybridization_probs[:, 1]
            d_percent_pred = hybridization_probs[:, 2]
            f_percent_pred = hybridization_probs[:, 3]
        else:
            # Return dummy predictions if not included
            s_percent_pred = torch.zeros_like(occupation_pred)
            p_percent_pred = torch.zeros_like(occupation_pred)
            d_percent_pred = torch.zeros_like(occupation_pred)
            f_percent_pred = torch.zeros_like(occupation_pred)
        
        # Predict KEI-BO values (edge predictions)
        kei_bo_pred = self._predict_kei_bo_values(orbital_embeddings, edge_index, edge_attr).squeeze(-1)
        
        # Predict global energy with element baselines
        if batch is None:
            batch = torch.zeros(orbital_embeddings.size(0), dtype=torch.long, device=orbital_embeddings.device)
        
        # Compute interaction energy from GNN (bonding, correlation, etc.)
        if self.global_pooling_method == 'attention':
            interaction_energy = self.global_pooling(orbital_embeddings, batch).squeeze(-1)
        elif self.global_pooling_method == 'mean':
            molecule_embedding = global_mean_pool(orbital_embeddings, batch)
            interaction_energy = self.global_pooling(molecule_embedding).squeeze(-1)
        elif self.global_pooling_method == 'sum':
            molecule_embedding = global_add_pool(orbital_embeddings, batch)
            interaction_energy = self.global_pooling(molecule_embedding).squeeze(-1)
        
        # Add element baseline correction if enabled
        if self.use_element_baselines:
            element_baseline = self._compute_element_baseline(data)
            energy_pred = element_baseline + interaction_energy
        else:
            energy_pred = interaction_energy
        
        return occupation_pred, kei_bo_pred, energy_pred, s_percent_pred, p_percent_pred, d_percent_pred, f_percent_pred
    
    def _predict_kei_bo_values(self, orbital_embeddings: torch.Tensor, 
                               edge_index: torch.Tensor, 
                               edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Predict KEI-BO values for orbital pairs
        
        Args:
            orbital_embeddings: [num_orbitals, hidden_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_input_dim]
        
        Returns:
            kei_bo_pred: [num_edges, 1]
        """
        row, col = edge_index
        source_orbitals = orbital_embeddings[row]
        target_orbitals = orbital_embeddings[col]
        
        # Combine orbital pair features with edge features
        edge_features = torch.cat([source_orbitals, target_orbitals, edge_attr], dim=1)
        
        # Predict KEI-BO value
        kei_bo_pred = self.kei_bo_head(edge_features)
        
        return kei_bo_pred
    
    def predict(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference mode prediction"""
        self.eval()
        with torch.no_grad():
            return self.forward(data)


class OrbitalMultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss for orbital predictions.
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics"
    (Kendall et al., 2018)
    
    Learns task-dependent uncertainty (noise) parameters that automatically balance losses.
    """
    
    def __init__(self, 
                 use_uncertainty_weighting: bool = True,
                 occupation_weight: float = 1.0, 
                 kei_bo_weight: float = 1.0, 
                 energy_weight: float = 1.0,
                 hybrid_weight: float = 1.0,
                 include_hybridization: bool = True):
        super(OrbitalMultiTaskLoss, self).__init__()
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.include_hybridization = include_hybridization
        self.mse = nn.MSELoss()
        
        num_tasks = 7 if include_hybridization else 3
        
        if use_uncertainty_weighting:
            # Learnable log variance parameters for each task
            # Using log variance for numerical stability
            self.log_var_occupation = nn.Parameter(torch.zeros(1))
            self.log_var_kei_bo = nn.Parameter(torch.zeros(1))
            self.log_var_energy = nn.Parameter(torch.zeros(1))
            if include_hybridization:
                self.log_var_s_percent = nn.Parameter(torch.zeros(1))
                self.log_var_p_percent = nn.Parameter(torch.zeros(1))
                self.log_var_d_percent = nn.Parameter(torch.zeros(1))
                self.log_var_f_percent = nn.Parameter(torch.zeros(1))
            print(f"Using uncertainty weighting for {num_tasks} tasks (automatic balancing)")
        else:
            # Manual weights (backward compatibility)
            self.occupation_weight = occupation_weight
            self.kei_bo_weight = kei_bo_weight
            self.energy_weight = energy_weight
            self.hybrid_weight = hybrid_weight  # Same weight for all 4 hybridization tasks
            if include_hybridization:
                print(f"Using manual weights: occ={occupation_weight}, kei_bo={kei_bo_weight}, energy={energy_weight}, hybrid={hybrid_weight}")
            else:
                print(f"Using manual weights (no hybridization): occ={occupation_weight}, kei_bo={kei_bo_weight}, energy={energy_weight}")
    
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
                f_percent_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        
        # Ensure targets have same shape as predictions (flatten if needed)
        occupation_target = occupation_target.squeeze() if occupation_target.dim() > 1 else occupation_target
        kei_bo_target = kei_bo_target.squeeze() if kei_bo_target.dim() > 1 else kei_bo_target
        energy_target = energy_target.squeeze() if energy_target.dim() > 1 else energy_target
        s_percent_target = s_percent_target.squeeze() if s_percent_target.dim() > 1 else s_percent_target
        p_percent_target = p_percent_target.squeeze() if p_percent_target.dim() > 1 else p_percent_target
        d_percent_target = d_percent_target.squeeze() if d_percent_target.dim() > 1 else d_percent_target
        f_percent_target = f_percent_target.squeeze() if f_percent_target.dim() > 1 else f_percent_target
        
        # Compute individual task losses
        occupation_loss = self.mse(occupation_pred, occupation_target)
        kei_bo_loss = self.mse(kei_bo_pred, kei_bo_target)
        energy_loss = self.mse(energy_pred, energy_target)
        
        if self.include_hybridization:
            s_percent_loss = self.mse(s_percent_pred, s_percent_target)
            p_percent_loss = self.mse(p_percent_pred, p_percent_target)
            d_percent_loss = self.mse(d_percent_pred, d_percent_target)
            f_percent_loss = self.mse(f_percent_pred, f_percent_target)
        
        if self.use_uncertainty_weighting:
            # Ensure all losses are on the same device as the parameters
            device = self.log_var_occupation.device
            occupation_loss = occupation_loss.to(device)
            kei_bo_loss = kei_bo_loss.to(device)
            energy_loss = energy_loss.to(device)
            
            # Uncertainty-weighted loss: L = (1/2σ²) * loss + log(σ)
            # Equivalent to: L = exp(-log_var) * loss + log_var
            total_loss = (
                torch.exp(-self.log_var_occupation) * occupation_loss + self.log_var_occupation +
                torch.exp(-self.log_var_kei_bo) * kei_bo_loss + self.log_var_kei_bo +
                torch.exp(-self.log_var_energy) * energy_loss + self.log_var_energy
            )
            
            if self.include_hybridization:
                s_percent_loss = s_percent_loss.to(device)
                p_percent_loss = p_percent_loss.to(device)
                d_percent_loss = d_percent_loss.to(device)
                f_percent_loss = f_percent_loss.to(device)
                total_loss += (
                    torch.exp(-self.log_var_s_percent) * s_percent_loss + self.log_var_s_percent +
                    torch.exp(-self.log_var_p_percent) * p_percent_loss + self.log_var_p_percent +
                    torch.exp(-self.log_var_d_percent) * d_percent_loss + self.log_var_d_percent +
                    torch.exp(-self.log_var_f_percent) * f_percent_loss + self.log_var_f_percent
                )
            
            # Extract learned weights for logging
            occupation_weight = torch.exp(-self.log_var_occupation).item()
            kei_bo_weight = torch.exp(-self.log_var_kei_bo).item()
            energy_weight = torch.exp(-self.log_var_energy).item()
            if self.include_hybridization:
                s_weight = torch.exp(-self.log_var_s_percent).item()
                p_weight = torch.exp(-self.log_var_p_percent).item()
                d_weight = torch.exp(-self.log_var_d_percent).item()
                f_weight = torch.exp(-self.log_var_f_percent).item()
            else:
                s_weight = p_weight = d_weight = f_weight = 0.0
        else:
            # Manual weighting
            total_loss = (
                self.occupation_weight * occupation_loss + 
                self.kei_bo_weight * kei_bo_loss + 
                self.energy_weight * energy_loss
            )
            
            if self.include_hybridization:
                total_loss += self.hybrid_weight * (s_percent_loss + p_percent_loss + d_percent_loss + f_percent_loss)
            
            occupation_weight = self.occupation_weight
            kei_bo_weight = self.kei_bo_weight
            energy_weight = self.energy_weight
            if self.include_hybridization:
                s_weight = p_weight = d_weight = f_weight = self.hybrid_weight
            else:
                s_weight = p_weight = d_weight = f_weight = 0.0
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'occupation_loss': occupation_loss.item(),
            'kei_bo_loss': kei_bo_loss.item(),
            'energy_loss': energy_loss.item(),
            'occupation_weight': occupation_weight,
            'kei_bo_weight': kei_bo_weight,
            'energy_weight': energy_weight,
        }
        
        if self.include_hybridization:
            loss_dict.update({
                's_percent_loss': s_percent_loss.item(),
                'p_percent_loss': p_percent_loss.item(),
                'd_percent_loss': d_percent_loss.item(),
                'f_percent_loss': f_percent_loss.item(),
                's_weight': s_weight,
                'p_weight': p_weight,
                'd_weight': d_weight,
                'f_weight': f_weight
            })
        
        return total_loss, loss_dict


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute evaluation metrics (MSE, MAE)"""
    with torch.no_grad():
        pred_np = pred.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        
        return {'mse': mse, 'mae': mae}


def create_orbital_model(orbital_input_dim: int = 4,
                        edge_input_dim: int = 1,
                        hidden_dim: int = 128,
                        num_layers: int = 4,
                        dropout: float = 0.1,
                        max_atomic_num: int = 20,
                        orbital_embedding_dim: int = 64,
                        global_pooling_method: str = 'attention',
                        use_rbf_distance: bool = False,
                        num_rbf: int = 50,
                        rbf_cutoff: float = 5.0,
                        include_hybridization: bool = True,
                        use_element_baselines: bool = True) -> OrbitalTripleTaskGNN:
    """Factory function to create the orbital-centric model"""
    return OrbitalTripleTaskGNN(
        orbital_input_dim=orbital_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_atomic_num=max_atomic_num,
        orbital_embedding_dim=orbital_embedding_dim,
        global_pooling_method=global_pooling_method,
        use_rbf_distance=use_rbf_distance,
        num_rbf=num_rbf,
        rbf_cutoff=rbf_cutoff,
        include_hybridization=include_hybridization,
        use_element_baselines=use_element_baselines
    )


# Backwards compatibility alias
OrbitalTripleTaskLoss = OrbitalMultiTaskLoss