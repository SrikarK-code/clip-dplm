import torch
import torch.nn as nn

class PerturbationEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # ESM embedding encoder
        self.esm_encoder = nn.Sequential(
            nn.Linear(config.esm_dim, config.latent_dim * 2),
            nn.LayerNorm(config.latent_dim * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.LayerNorm(config.latent_dim)
        )
        
        # Perturbation values encoder
        self.value_encoder = nn.Sequential(
            nn.Linear(config.n_genes, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.latent_dim),
            nn.LayerNorm(config.latent_dim)
        )
        
        # Cross attention between ESM and values
        if config.use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=config.latent_dim,
                num_heads=config.n_heads,
                dropout=config.dropout,
                batch_first=True
            )
            
            self.attention_norm = nn.LayerNorm(config.latent_dim)
            
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.latent_dim, config.latent_dim)
        )
        
    def forward(self, esm_embeddings, perturbation_values):
        # Encode ESM embeddings
        h_esm = self.esm_encoder(esm_embeddings)
        
        # Encode perturbation values
        h_val = self.value_encoder(perturbation_values)
        
        # Apply cross attention if enabled
        if hasattr(self, 'cross_attention'):
            h_att, _ = self.cross_attention(
                query=h_esm.unsqueeze(1),
                key=h_val.unsqueeze(1),
                value=h_val.unsqueeze(1)
            )
            h_att = self.attention_norm(h_att.squeeze(1))
        else:
            h_att = h_val
            
        # Combine features
        h = torch.cat([h_esm, h_att], dim=-1)
        
        # Final projection with residual
        h = self.output_proj(h) + h_esm
        
        return h
