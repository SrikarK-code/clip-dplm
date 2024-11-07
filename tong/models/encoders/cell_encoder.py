import torch
import torch.nn as nn
from ..layers.gnn import MultiLayerPiGNN

class CellStateEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Gene expression encoder
        self.gene_encoder = nn.Sequential(
            nn.Linear(config.gene_dim, config.latent_dim * 2),
            nn.LayerNorm(config.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.LayerNorm(config.latent_dim)
        )
        
        # DPT (pseudotime) encoder 
        if config.use_time_encoding:
            self.time_encoder = nn.Sequential(
                nn.Linear(1, config.time_encoding_dim),
                nn.LayerNorm(config.time_encoding_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.time_encoding_dim, config.latent_dim),
                nn.LayerNorm(config.latent_dim)
            )
        
        # Graph neural network
        if config.gnn_type == 'pignn':
            self.gnn = MultiLayerPiGNN(
                latent_dim=config.latent_dim,
                n_layers=config.n_layers,
                n_heads=config.n_heads,
                n_neighbors=config.n_neighbors,
                dropout=config.dropout
            )
            
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
    def forward(self, gene_expr, dpt, edge_index, batch_idx):
        # Encode gene expression
        h = self.gene_encoder(gene_expr)
        
        # Add time encoding if enabled
        if hasattr(self, 'time_encoder'):
            t = self.time_encoder(dpt.unsqueeze(-1))
            h = h + t
            
        # Apply GNN layers
        h = self.gnn(h, edge_index, batch_idx)
        
        # Global context
        h_global = torch.zeros_like(h)
        h_global[batch_idx] = scatter_mean(h, batch_idx, dim=0)
        
        # Final projection with skip connection
        h = torch.cat([h, h_global], dim=-1)
        h = self.output_proj(h) + h[:, :self.config.latent_dim]
        
        return h
