import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import NNConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from typing import Tuple, Optional
import numpy as np


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
        input_features = 5  # [occupation, s%, p%, d%, f%] - continuous features only
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
            orbital_features: [num_orbitals, 8] tensor with features:
                [atomic_num, orbital_type, m_quantum, occupation, s%, p%, d%, f%]
        
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
        
        # Use only continuous features (skip atomic_num, orbital_type, m_quantum)
        continuous_features = orbital_features[:, 3:]  # occupation, s%, p%, d%, f%
        
        # Concatenate all features
        combined_features = torch.cat([
            continuous_features,
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
        
        # Edge network for NNConv
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
    """Orbital-centric GNN for predicting orbital occupations, KEI-BO values, and molecular energy"""
    
    def __init__(self, 
                 orbital_input_dim: int = 8,      # [atomic_num, orbital_type, m_quantum, occupation, s%, p%, d%, f%]
                 edge_input_dim: int = 1,         # distance
                 hidden_dim: int = 128,
                 num_layers: int = 4,
                 dropout: float = 0.1,
                 max_atomic_num: int = 20,
                 orbital_embedding_dim: int = 64,
                 global_pooling_method: str = 'attention'):
        super(OrbitalTripleTaskGNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.global_pooling_method = global_pooling_method
        
        # Orbital embedding layer
        self.orbital_embedding = OrbitalEmbedding(max_atomic_num, orbital_embedding_dim)
        
        # Initial projection to hidden dimension
        self.input_projection = nn.Linear(orbital_embedding_dim, hidden_dim)
        
        # Message passing layers
        self.message_layers = nn.ModuleList([
            OrbitalMessagePassing(hidden_dim, edge_input_dim, dropout)
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
        self.keibo_head = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
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
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for orbital-level predictions
        
        Args:
            data: PyTorch Geometric Data object with orbital features
        
        Returns:
            occupation_pred: [num_orbitals, 1] predicted orbital occupations
            keibo_pred: [num_edges, 1] predicted KEI-BO values
            energy_pred: [num_molecules, 1] predicted molecular energies
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        
        # Embed orbital features
        x = self.orbital_embedding(x)
        x = self.input_projection(x)
        
        # Message passing between orbitals
        for layer in self.message_layers:
            x = layer(x, edge_index, edge_attr)
        
        orbital_embeddings = x
        
        # Predict orbital occupations (keep [N, 1] shape to match targets)
        occupation_pred = self.occupation_head(orbital_embeddings).squeeze(-1)
        
        # Predict KEI-BO values (edge predictions, keep [N, 1] shape)
        keibo_pred = self._predict_keibo_values(orbital_embeddings, edge_index, edge_attr).squeeze(-1)
        
        # Predict global energy
        if batch is None:
            batch = torch.zeros(orbital_embeddings.size(0), dtype=torch.long, device=orbital_embeddings.device)
        
        if self.global_pooling_method == 'attention':
            energy_pred = self.global_pooling(orbital_embeddings, batch).squeeze(-1)
        elif self.global_pooling_method == 'mean':
            molecule_embedding = global_mean_pool(orbital_embeddings, batch)
            energy_pred = self.global_pooling(molecule_embedding).squeeze(-1)
        elif self.global_pooling_method == 'sum':
            molecule_embedding = global_add_pool(orbital_embeddings, batch)
            energy_pred = self.global_pooling(molecule_embedding).squeeze(-1)
        
        return occupation_pred, keibo_pred, energy_pred
    
    def _predict_keibo_values(self, orbital_embeddings: torch.Tensor, 
                             edge_index: torch.Tensor, 
                             edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Predict KEI-BO values for orbital pairs
        
        Args:
            orbital_embeddings: [num_orbitals, hidden_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_input_dim]
        
        Returns:
            keibo_pred: [num_edges, 1]
        """
        row, col = edge_index
        source_orbitals = orbital_embeddings[row]
        target_orbitals = orbital_embeddings[col]
        
        # Combine orbital pair features with edge features
        edge_features = torch.cat([source_orbitals, target_orbitals, edge_attr], dim=1)
        
        # Predict KEI-BO value
        keibo_pred = self.keibo_head(edge_features)
        
        return keibo_pred
    
    def predict(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inference mode prediction"""
        self.eval()
        with torch.no_grad():
            return self.forward(data)


class OrbitalTripleTaskLoss(nn.Module):
    """Loss function for orbital-level multi-task learning"""
    
    def __init__(self, 
                 occupation_weight: float = 1.0, 
                 keibo_weight: float = 1.0, 
                 energy_weight: float = 1.0,
                 use_first_epoch_weighting: bool = False):
        super(OrbitalTripleTaskLoss, self).__init__()
        
        self.occupation_weight = occupation_weight
        self.keibo_weight = keibo_weight
        self.energy_weight = energy_weight
        self.use_first_epoch_weighting = use_first_epoch_weighting
        self.weights_initialized = not use_first_epoch_weighting
        self.mse = nn.MSELoss()
    
    def initialize_weights_from_losses(self, occupation_loss: float, keibo_loss: float, energy_loss: float):
        """Initialize task weights based on first epoch losses"""
        # Use inverse losses (harder tasks get more weight)
        inv_occupation = 1.0 / (occupation_loss + 1e-8)
        inv_keibo = 1.0 / (keibo_loss + 1e-8)
        inv_energy = 1.0 / (energy_loss + 1e-8)
        
        # Normalize so they sum to 3 (average weight of 1.0)
        total = inv_occupation + inv_keibo + inv_energy
        self.occupation_weight = 3.0 * inv_occupation / total
        self.keibo_weight = 3.0 * inv_keibo / total
        self.energy_weight = 3.0 * inv_energy / total
        
        self.weights_initialized = True
        
        print(f"\n{'='*60}")
        print("ORBITAL FIRST-EPOCH LOSS WEIGHTING")
        print(f"{'='*60}")
        print(f"Initial losses:")
        print(f"  Occupation:  {occupation_loss:.6f}")
        print(f"  KEI-BO:      {keibo_loss:.6f}")
        print(f"  Energy:      {energy_loss:.6f}")
        print(f"\nComputed inverse-loss weights:")
        print(f"  Occupation weight: {self.occupation_weight:.4f}")
        print(f"  KEI-BO weight:     {self.keibo_weight:.4f}")
        print(f"  Energy weight:     {self.energy_weight:.4f}")
        print(f"{'='*60}\n")
    
    def forward(self, 
                occupation_pred: torch.Tensor, 
                keibo_pred: torch.Tensor, 
                energy_pred: torch.Tensor,
                occupation_target: torch.Tensor, 
                keibo_target: torch.Tensor, 
                energy_target: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        
        # Ensure targets have same shape as predictions (flatten if needed)
        occupation_target = occupation_target.squeeze() if occupation_target.dim() > 1 else occupation_target
        keibo_target = keibo_target.squeeze() if keibo_target.dim() > 1 else keibo_target
        energy_target = energy_target.squeeze() if energy_target.dim() > 1 else energy_target
        
        occupation_loss = self.mse(occupation_pred, occupation_target)
        keibo_loss = self.mse(keibo_pred, keibo_target)
        energy_loss = self.mse(energy_pred, energy_target)
        
        # Initialize weights from first batch if needed
        if self.use_first_epoch_weighting and not self.weights_initialized:
            self.initialize_weights_from_losses(
                occupation_loss.item(), keibo_loss.item(), energy_loss.item()
            )
        
        total_loss = (self.occupation_weight * occupation_loss + 
                     self.keibo_weight * keibo_loss + 
                     self.energy_weight * energy_loss)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'occupation_loss': occupation_loss.item(),
            'keibo_loss': keibo_loss.item(),
            'energy_loss': energy_loss.item(),
            'occupation_weight': self.occupation_weight,
            'keibo_weight': self.keibo_weight,
            'energy_weight': self.energy_weight
        }
        
        return total_loss, loss_dict


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """Compute evaluation metrics (MSE, MAE)"""
    with torch.no_grad():
        pred_np = pred.cpu().numpy().flatten()
        target_np = target.cpu().numpy().flatten()
        
        mse = np.mean((pred_np - target_np) ** 2)
        mae = np.mean(np.abs(pred_np - target_np))
        
        return {'mse': mse, 'mae': mae}


def create_orbital_model(orbital_input_dim: int = 8,
                        edge_input_dim: int = 1,
                        hidden_dim: int = 128,
                        num_layers: int = 4,
                        dropout: float = 0.1,
                        max_atomic_num: int = 20,
                        orbital_embedding_dim: int = 64,
                        global_pooling_method: str = 'attention') -> OrbitalTripleTaskGNN:
    """Factory function to create the orbital-centric model"""
    return OrbitalTripleTaskGNN(
        orbital_input_dim=orbital_input_dim,
        edge_input_dim=edge_input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        max_atomic_num=max_atomic_num,
        orbital_embedding_dim=orbital_embedding_dim,
        global_pooling_method=global_pooling_method
    )