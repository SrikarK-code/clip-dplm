"""
ESM Integration for Single-Cell Transport
=======================================

Mathematical Foundation:
---------------------
1. ESM Embeddings:
   e(s) = ESM(s) ∈ ℝ^d where:
   - s: Protein/gene sequence
   - d: Embedding dimension (1280 for ESM-2)

2. Projection Functions:
   P_gene: ℝ^{1280} → ℝ^{512}  (Gene projection)
   P_prot: ℝ^{1280} → ℝ^{512}  (Protein projection)

3. Integration:
   - Gene embeddings inform perturbation effects
   - Protein embeddings provide structural context

Biological Relevance:
-------------------
- ESM captures protein structural information
- Gene embeddings encode functional relationships
- Projections preserve biological similarities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import logging
from transformers import AutoModel, AutoTokenizer
from torch.cuda.amp import autocast
from dataclasses import dataclass

from config import ESMConfig, BiologicalDataType

logger = logging.getLogger(__name__)

@dataclass
class ESMOutput:
    """Container for ESM model outputs"""
    embeddings: torch.Tensor
    attention_weights: Optional[torch.Tensor] = None

class ESMIntegration(nn.Module):
    """
    ESM integration for single-cell analysis
    
    Maps sequences to biologically meaningful embeddings:
    1. Genes → Perturbation space
    2. Proteins → Target space
    """
    def __init__(self, config: ESMConfig):
        super().__init__()
        self.config = config
        
        # Initialize ESM model
        self._setup_esm()
        
        # Projection networks
        self.protein_projection = ProteinProjection(
            esm_dim=config.esm_dim,
            output_dim=config.protein_dim
        )
        
        self.gene_projection = GeneProjection(
            esm_dim=config.esm_dim,
            output_dim=config.gene_dim
        )
        
        # Optional memory cache
        self.cache = {}
        
    def _setup_esm(self) -> None:
        """Initialize ESM model with error handling"""
        try:
            self.model = AutoModel.from_pretrained(self.config.model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_path or self.config.model_name
            )
            
            # Freeze ESM model
            for param in self.model.parameters():
                param.requires_grad = False
                
        except Exception as e:
            logger.error(f"ESM initialization failed: {str(e)}")
            raise
            
    @torch.no_grad()
    def get_embeddings(
        self,
        sequences: List[str],
        data_type: BiologicalDataType
    ) -> ESMOutput:
        """
        Get embeddings for sequences
        
        Args:
            sequences: List of protein/gene sequences
            data_type: Type of biological sequence
        """
        # Check cache
        cache_key = str(hash(tuple(sequences)))
        if cache_key in self.cache:
            return self.cache[cache_key]
            
        # Tokenize sequences
        tokens = self.tokenizer(
            sequences,
            padding=True,
            truncation=True,
            max_length=self.config.max_sequence_length,
            return_tensors="pt"
        ).to(next(self.model.parameters()).device)
        
        # Get ESM embeddings
        with autocast():
            outputs = self.model(**tokens)
            
        # Project based on data type
        embeddings = outputs.last_hidden_state
        if data_type == BiologicalDataType.PROTEIN_SEQUENCE:
            embeddings = self.protein_projection(embeddings)
        else:  # Gene sequence
            embeddings = self.gene_projection(embeddings)
            
        # Cache results
        result = ESMOutput(
            embeddings=embeddings,
            attention_weights=outputs.attentions
        )
        self.cache[cache_key] = result
        
        return result

class ProteinProjection(nn.Module):
    """
    Projects ESM protein embeddings to transport space
    
    Architecture ensures:
    1. Dimensionality reduction
    2. Preserved structural relationships
    3. Stable gradients
    """
    def __init__(self, esm_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(esm_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            ResidualBlock(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class GeneProjection(nn.Module):
    """
    Projects ESM gene embeddings to transport space
    
    Architecture ensures:
    1. Gene function preservation
    2. Relevant for perturbation effects
    """
    def __init__(self, esm_dim: int, output_dim: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(esm_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            AttentionBlock(output_dim * 2),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)

class ResidualBlock(nn.Module):
    """Residual block for stable training"""
    def __init__(self, dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)

class AttentionBlock(nn.Module):
    """Self-attention for capturing gene relationships"""
    def __init__(self, dim: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(x, x, x)
        return self.norm(x + attended)

def create_esm_integration(
    model_name: str = "esm2_t33_650M_UR50D",
    **kwargs
) -> ESMIntegration:
    """Factory function for ESM integration"""
    config = ESMConfig(
        model_name=model_name,
        **kwargs
    )
    
    model = ESMIntegration(config)
    logger.info(f"Created ESM integration with model: {model_name}")
    return model

# Utility functions
def get_embeddings_batch(
    sequences: List[str],
    esm_model: ESMIntegration,
    data_type: BiologicalDataType,
    batch_size: int = 32
) -> torch.Tensor:
    """Process sequences in batches"""
    embeddings = []
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i:i + batch_size]
        batch_embeddings = esm_model.get_embeddings(
            batch,
            data_type
        ).embeddings
        embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)
