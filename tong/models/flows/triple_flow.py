import torch
import torch.nn as nn
from .ot_flow import ExactOTFlow, SchrodingerBridgeFlow, OTFlow

class TripleFlow(nn.Module):
    """Handles flows between cell state, perturbation, and protein spaces."""
    
    def __init__(self, config):
        super().__init__()
        
        # Select flow type based on config
        if config.flow_type == 'exact_ot':
            flow_class = ExactOTFlow
        elif config.flow_type == 'sb':
            flow_class = SchrodingerBridgeFlow
        else:
            flow_class = OTFlow
            
        # Initialize flows between spaces
        self.cell_to_pert = flow_class(config)
        self.cell_to_protein = flow_class(config)
        self.pert_to_protein = flow_class(config)
        
        # Feature mixing for guidance
        self.feature_mixer = nn.Sequential(
            nn.Linear(config.latent_dim * 2, config.latent_dim),
            nn.LayerNorm(config.latent_dim),
            nn.GELU(),
            nn.Linear(config.latent_dim, config.latent_dim)
        ) if config.use_feature_mixing else None
        
    def mix_features(self, x, y):
        if self.feature_mixer is None:
            return x
        return self.feature_mixer(torch.cat([x, y], dim=-1))
    
    def forward(self, embeddings, return_regularization=False):
        flows = {}
        reg_losses = {}
        
        cell_emb = embeddings['cell_emb']
        protein_emb = embeddings['protein_emb']
        
        # Always compute cell-protein flow
        flow_out = self.cell_to_protein(
            cell_emb, protein_emb, return_regularization)
        
        if return_regularization:
            flows['cell_protein'] = flow_out[:-1]
            reg_losses['cell_protein'] = flow_out[-1]
        else:
            flows['cell_protein'] = flow_out
            
        # Add perturbation flows if available
        if 'pert_emb' in embeddings:
            pert_emb = embeddings['pert_emb']
            
            # Mix features for guidance
            guided_cell = self.mix_features(cell_emb, protein_emb)
            guided_pert = self.mix_features(pert_emb, protein_emb)
            
            # Cell-perturbation flow
            flow_out = self.cell_to_pert(
                guided_cell, pert_emb, return_regularization)
            
            if return_regularization:
                flows['cell_pert'] = flow_out[:-1]
                reg_losses['cell_pert'] = flow_out[-1]
            else:
                flows['cell_pert'] = flow_out
                
            # Perturbation-protein flow
            flow_out = self.pert_to_protein(
                guided_pert, protein_emb, return_regularization)
                
            if return_regularization:
                flows['pert_protein'] = flow_out[:-1]
                reg_losses['pert_protein'] = flow_out[-1]
            else:
                flows['pert_protein'] = flow_out
                
        return (flows, reg_losses) if return_regularization else flows
