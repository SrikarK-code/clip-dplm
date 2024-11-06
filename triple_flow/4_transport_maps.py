"""
Transport Maps for Single-Cell Analysis
====================================

Mathematical Foundation:
---------------------
1. Optimal Transport between spaces:
   W₂(μ, ν) = inf_T ∫||x - T(x)||² dμ(x)
   where T: X → Y is the transport map

2. Triple Flow Setup:
   - T_CP: Cell → Perturbation
   - T_CE: Cell → Protein
   - T_PE: Perturbation → Protein

3. Consistency Requirement:
   T_CE ≈ T_PE ∘ T_CP

Biological Relevance:
-------------------
- Preserves gene expression structure
- Maintains biological distances
- Handles sparse expression data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
from torch.cuda.amp import autocast

from config import ICNNConfig
from core import SingleCellICNN

logger = logging.getLogger(__name__)

@dataclass
class TransportOutput:
    """Transport map outputs"""
    transported: torch.Tensor
    cost: torch.Tensor
    metrics: Optional[Dict[str, float]] = None

class TransportCost(nn.Module):
    """
    Cost function for single-cell transport
    
    Combines:
    1. Wasserstein-2 distance
    2. Sparsity preservation
    3. Expression-level normalization
    """
    def __init__(self, regularization: float = 0.01):
        super().__init__()
        self.regularization = regularization
        
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute transport cost
        
        Args:
            source: Source points (batch_size, dim)
            target: Target points (batch_size, dim)
        """
        # Basic transport cost
        w2_cost = torch.norm(source - target, dim=-1).mean()
        
        # Sparsity preservation for gene expression
        sparsity_cost = self.regularization * (
            torch.norm(source, p=1, dim=-1).mean() +
            torch.norm(target, p=1, dim=-1).mean()
        )
        
        total_cost = w2_cost + sparsity_cost
        
        metrics = {
            'w2_cost': w2_cost.item(),
            'sparsity_cost': sparsity_cost.item()
        }
        
        return total_cost, metrics

class SingleCellTransport(nn.Module):
    """
    Single transport map between biological spaces
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: ICNNConfig
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Core transport network
        self.transport_net = SingleCellICNN(config)
        
        # Cost function
        self.cost_fn = TransportCost()
        
        # Input/output normalization
        self.input_norm = nn.LayerNorm(input_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        
    def forward(
        self,
        source: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, TransportOutput]:
        """
        Compute optimal transport
        
        Args:
            source: Source points (batch_size, input_dim)
            target: Optional target points for training
        """
        # Normalize inputs
        source = self.input_norm(source)
        
        # Compute transport
        transported = self.transport_net.gradient(source)
        
        # Normalize output
        transported = self.output_norm(transported)
        
        # If in training mode, compute cost
        if self.training and target is not None:
            target = self.output_norm(target)
            cost, metrics = self.cost_fn(transported, target)
            
            return TransportOutput(
                transported=transported,
                cost=cost,
                metrics=metrics
            )
            
        return transported

class TripleTransportMaps(nn.Module):
    """
    Complete triple transport system for single-cell analysis
    """
    def __init__(
        self,
        cell_dim: int,
        pert_dim: int,
        protein_dim: int,
        config: ICNNConfig
    ):
        super().__init__()
        
        # Individual transport maps
        self.cell_to_pert = SingleCellTransport(
            input_dim=cell_dim,
            output_dim=pert_dim,
            config=config
        )
        
        self.cell_to_protein = SingleCellTransport(
            input_dim=cell_dim,
            output_dim=protein_dim,
            config=config
        )
        
        self.pert_to_protein = SingleCellTransport(
            input_dim=pert_dim,
            output_dim=protein_dim,
            config=config
        )
        
        # Consistency checker
        self.consistency_loss = ConsistencyChecker()
        
    def forward(
        self,
        cell_states: torch.Tensor,
        pert_states: Optional[torch.Tensor] = None,
        protein_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Union[torch.Tensor, TransportOutput]]:
        """
        Compute optimal transport between spaces
        
        Args:
            cell_states: Single-cell gene expression (batch_size, cell_dim)
            pert_states: Optional perturbation states
            protein_states: Optional protein states
        """
        outputs = {}
        
        # Cell to perturbation
        if pert_states is not None:
            outputs['cell_to_pert'] = self.cell_to_pert(
                cell_states, pert_states
            )
        
        # Cell to protein
        if protein_states is not None:
            outputs['cell_to_protein'] = self.cell_to_protein(
                cell_states, protein_states
            )
            
        # Perturbation to protein
        if pert_states is not None and protein_states is not None:
            outputs['pert_to_protein'] = self.pert_to_protein(
                pert_states, protein_states
            )
            
            # Check consistency only when all modalities are present
            if self.training:
                outputs['consistency_loss'] = self.consistency_loss(
                    cell_protein=outputs['cell_to_protein'].transported,
                    cell_pert=outputs['cell_to_pert'].transported,
                    pert_protein=outputs['pert_to_protein'].transported
                )
                
        return outputs

class ConsistencyChecker(nn.Module):
    """
    Checks consistency between transport maps
    
    Ensures: T_CE ≈ T_PE ∘ T_CP
    """
    def forward(
        self,
        cell_protein: torch.Tensor,
        cell_pert: torch.Tensor,
        pert_protein: torch.Tensor,
    ) -> torch.Tensor:
        """Compute consistency loss"""
        # Direct path: cell → protein
        direct_path = cell_protein
        
        # Composed path: cell → pert → protein
        composed_path = pert_protein(cell_pert)
        
        # Consistency loss
        return F.mse_loss(direct_path, composed_path)

def create_transport_system(
    cell_dim: int,
    pert_dim: int,
    protein_dim: int,
    hidden_dims: Optional[List[int]] = None,
    **kwargs
) -> TripleTransportMaps:
    """Factory function for transport system"""
    if hidden_dims is None:
        # Default architecture based on input dimensions
        hidden_dims = [
            max(cell_dim, pert_dim, protein_dim),
            max(cell_dim, pert_dim, protein_dim) // 2
        ]
        
    config = ICNNConfig(
        input_dim=max(cell_dim, pert_dim, protein_dim),
        hidden_dims=hidden_dims,
        **kwargs
    )
    
    model = TripleTransportMaps(
        cell_dim=cell_dim,
        pert_dim=pert_dim,
        protein_dim=protein_dim,
        config=config
    )
    
    logger.info(
        f"Created transport system with dimensions: "
        f"cell={cell_dim}, pert={pert_dim}, protein={protein_dim}"
    )
    
    return model

# Utility functions
@torch.no_grad()
def compute_transport_error(
    transport: SingleCellTransport,
    source: torch.Tensor,
    target: torch.Tensor,
    batch_size: int = 128
) -> float:
    """Compute transport error in batches"""
    errors = []
    for i in range(0, len(source), batch_size):
        batch_source = source[i:i + batch_size]
        batch_target = target[i:i + batch_size]
        
        transported = transport(batch_source)
        error = F.mse_loss(transported, batch_target)
        errors.append(error.item())
        
    return sum(errors) / len(errors)
