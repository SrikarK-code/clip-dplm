import torch
import torch.nn as nn

class ProteinEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Build layers based on config hidden dims
        layers = []
        dims = [config.protein_dim] + config.hidden_dims + [config.latent_dim]
        
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]) if config.use_layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(config.dropout)
            ])
        
        # Remove last activation and dropout
        layers = layers[:-2]
        
        self.encoder = nn.Sequential(*layers)
        
        # Optional residual connection
        self.use_residual = dims[0] == dims[-1]
        
    def forward(self, protein_embedding):
        h = self.encoder(protein_embedding)
        
        if self.use_residual:
            h = h + protein_embedding
            
        return h
