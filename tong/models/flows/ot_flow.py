import torch
import torch.nn as nn
from conditional_flow_matching import (
    ConditionalFlowMatcher,
    ExactOptimalTransportConditionalFlowMatcher,
    SchrodingerBridgeConditionalFlowMatcher
)

class OTFlow(nn.Module):
    """Base class for OT-based flows."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Time embedding
        if config.use_time_embedding:
            self.time_encoder = nn.Sequential(
                nn.Linear(1, config.time_embed_dim),
                nn.LayerNorm(config.time_embed_dim),
                nn.GELU(),
                nn.Linear(config.time_embed_dim, config.latent_dim)
            )
            
        # Flow network
        dims = [config.latent_dim * (3 if config.use_time_embedding else 2)] + \
               [config.hidden_dim] * config.n_layers + \
               [config.latent_dim]
               
        layers = []
        for i in range(len(dims)-1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.LayerNorm(dims[i+1]),
                nn.GELU() if i < len(dims)-2 else nn.Tanh(),
                nn.Dropout(config.dropout) if i < len(dims)-2 else nn.Identity()
            ])
            
        self.net = nn.Sequential(*layers)
            
    def get_regularization(self, v, xt):
        """Compute regularization losses."""
        reg_loss = 0
        
        # Path length regularization
        if self.config.use_path_length_reg:
            reg_loss += torch.mean(torch.norm(v, dim=-1) ** 2)
            
        # Jacobian regularization
        if self.config.use_jacobian_reg:
            jac = torch.autograd.functional.jacobian(
                lambda x: self.net(x), xt[0:1]
            )
            reg_loss += torch.norm(jac, p='fro')
            
        return reg_loss

class ExactOTFlow(OTFlow):
    """Exact optimal transport flow."""
    
    def __init__(self, config):
        super().__init__(config)
        self.cfm = ExactOptimalTransportConditionalFlowMatcher(
            sigma=config.sigma
        )
        
    def forward(self, source, target, return_regularization=False):
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(source, target)
        
        # Prepare input with time embedding if enabled
        if hasattr(self, 'time_encoder'):
            t_emb = self.time_encoder(t.view(-1, 1))
            h = torch.cat([xt, ut, t_emb], dim=-1)
        else:
            h = torch.cat([xt, ut], dim=-1)
            
        # Compute vector field
        v = self.net(h)
        
        # Get regularization if requested
        if return_regularization:
            reg_loss = self.get_regularization(v, xt)
            return v, xt, t, reg_loss
            
        return v, xt, t

class SchrodingerBridgeFlow(OTFlow):
    """Schrödinger bridge flow."""
    
    def __init__(self, config):
        super().__init__(config)
        self.cfm = SchrodingerBridgeConditionalFlowMatcher(
            sigma=config.sigma,
            ot_method="sinkhorn",
            reg=2 * (config.sigma ** 2)
        )
        
    def forward(self, source, target, return_regularization=False):
        t, xt, ut = self.cfm.sample_location_and_conditional_flow(source, target)
        
        if hasattr(self, 'time_encoder'):
            t_emb = self.time_encoder(t.view(-1, 1))
            h = torch.cat([xt, ut, t_emb], dim=-1)
        else:
            h = torch.cat([xt, ut], dim=-1)
            
        v = self.net(h)
        
        if return_regularization:
            reg_loss = self.get_regularization(v, xt)
            return v, xt, t, reg_loss
            
        return v, xt, t
