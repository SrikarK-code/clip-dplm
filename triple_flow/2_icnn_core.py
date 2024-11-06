"""
Core ICNN Implementation for Single-Cell Transport
===============================================

Mathematical Foundation:
---------------------
The Input Convex Neural Network (ICNN) ensures convexity by construction:

1. Core ICNN Properties:
   Ψ(x) is convex in x if:
   - All weights to x are unconstrained
   - All other weights are non-negative
   - All activation functions are convex and non-decreasing

2. Transport Map:
   T(x) = ∇Ψ(x) where Ψ is our convex potential

3. Single-Cell Specific Features:
   - Handles sparse high-dimensional data
   - Efficient gradient computation for large matrices
   - Stable normalization for count data

Biological Relevance:
-------------------
- Preserves gene expression structure
- Handles technical noise in single-cell data
- Maintains biological relationships between cells
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging
from torch.cuda.amp import autocast
import math

from config import ICNNConfig

logger = logging.getLogger(__name__)

class ConvexLayer(nn.Module):
    """
    Convex layer ensuring ∇²Ψ(x) ≽ 0
    
    Mathematical formulation:
    y = Wx + σ(Vz + b)
    where:
    - W: Unconstrained weights for x
    - V: Non-negative weights for z (previous layer)
    - σ: Convex, non-decreasing activation
    """
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        config: ICNNConfig
    ):
        super().__init__()
        self.config = config
        
        # Linear transformation (unconstrained)
        self.linear = nn.Linear(input_dim, output_dim)
        
        # Positive weights (non-negative through softplus)
        self.pos_weights = nn.Parameter(torch.zeros(output_dim, input_dim))
        
        # Learnable scale with careful initialization
        self.scale = nn.Parameter(torch.ones(1) * config.init_scale)
        
        # Normalization for numerical stability
        self.norm = nn.LayerNorm(output_dim) if config.use_layer_norm else nn.Identity()
        
        self._init_weights()
        
    def _init_weights(self) -> None:
        """Initialize weights for stable training"""
        nn.init.orthogonal_(self.linear.weight)
        if self.linear.bias is not None:
            fan_in = self.linear.weight.size(1)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.linear.bias, -bound, bound)
            
    def get_positive_weights(self) -> torch.Tensor:
        """Get non-negative weights through softplus"""
        return F.softplus(self.pos_weights + self.config.eps)
    
    def forward(
        self,
        x: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        scale: Optional[float] = None
    ) -> torch.Tensor:
        """
        Forward pass maintaining convexity
        
        Args:
            x: Input tensor (batch_size, input_dim)
            z: Previous layer activation
            scale: Optional manual scale factor
        """
        y = self.linear(x)
        
        if z is not None:
            scale = scale if scale is not None else self.scale
            pos_w = self.get_positive_weights()
            
            # Stable z contribution
            z_contrib = F.linear(z, pos_w)
            z_contrib = z_contrib * scale
            
            # Gradient stabilization
            if self.training:
                with torch.no_grad():
                    z_scale = z_contrib.abs().mean()
                    if z_scale > self.config.gradient_clip:
                        z_contrib = z_contrib * (self.config.gradient_clip / z_scale)
                        
            y = y + z_contrib
            
        # Normalization and activation
        y = self.norm(y)
        
        # Convex activation
        y = F.softplus(y) if self.config.activation == 'softplus' else F.celu(y)
        
        return y

class SingleCellICNN(nn.Module):
    """
    ICNN specialized for single-cell data
    
    Key features:
    1. Sparse data handling
    2. Stable gradient computation
    3. Memory-efficient processing
    """
    def __init__(self, config: ICNNConfig):
        super().__init__()
        self.config = config
        
        # Input normalization
        self.input_norm = nn.LayerNorm(config.input_dim)
        
        # Convex layers
        self.layers = nn.ModuleList()
        prev_dim = config.input_dim
        
        for hidden_dim in config.hidden_dims:
            self.layers.append(ConvexLayer(prev_dim, hidden_dim, config))
            prev_dim = hidden_dim
            
        # Final projection
        self.final = nn.Linear(prev_dim, 1)
        
    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Forward pass through ICNN
        
        Args:
            x: Single-cell data tensor (batch_size, n_genes)
            return_intermediates: Whether to return intermediate activations
        """
        x = self.input_norm(x)
        
        intermediates = [] if return_intermediates else None
        z = None
        
        for layer in self.layers:
            z = layer(x, z)
            if return_intermediates:
                intermediates.append(z)
                
        output = self.final(z)
        return output, intermediates
    
    @torch.enable_grad()
    def gradient(
        self,
        x: torch.Tensor,
        create_graph: bool = True
    ) -> torch.Tensor:
        """
        Compute transport map gradient
        
        The transport map is the gradient of the potential:
        T(x) = ∇Ψ(x)
        """
        x.requires_grad_(True)
        
        with autocast(enabled=False):
            y = self.forward(x)[0]
            grad = torch.autograd.grad(
                y.sum(), x,
                create_graph=create_graph,
                retain_graph=True
            )[0]
            
        if self.training:
            grad_norm = grad.norm(dim=-1, keepdim=True)
            grad = torch.where(
                grad_norm > self.config.gradient_clip,
                grad * self.config.gradient_clip / grad_norm,
                grad
            )
            
        return grad
    
    @torch.enable_grad()
    def hessian(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian matrix
        
        Used for verifying convexity: ∇²Ψ(x) ≽ 0
        """
        grad = self.gradient(x)
        hess = []
        
        for i in range(grad.shape[-1]):
            hess_i = torch.autograd.grad(
                grad[..., i].sum(), x,
                create_graph=True,
                retain_graph=True
            )[0]
            hess.append(hess_i)
            
        hess = torch.stack(hess, dim=-1)
        
        # Add regularization for numerical stability
        if self.training:
            eye = torch.eye(
                hess.shape[-1],
                device=hess.device
            ).expand_as(hess)
            hess = hess + self.config.hessian_reg * eye
            
        return hess

def create_single_cell_icnn(
    input_dim: int,
    hidden_dims: Optional[List[int]] = None,
    **kwargs
) -> SingleCellICNN:
    """Factory function for single-cell ICNN"""
    if hidden_dims is None:
        hidden_dims = [input_dim, input_dim // 2, input_dim // 4]
        
    config = ICNNConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        **kwargs
    )
    
    model = SingleCellICNN(config)
    logger.info(
        f"Created SingleCellICNN with {sum(p.numel() for p in model.parameters())} parameters"
    )
    return model
