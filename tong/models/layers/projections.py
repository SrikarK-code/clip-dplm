import torch.nn as nn

def create_projection_layer(d_in, d_out, dropout=0.1, use_layernorm=True):
    """Creates a projection layer with optional layernorm."""
    return nn.Sequential(
        nn.Linear(d_in, d_out),
        nn.LayerNorm(d_out) if use_layernorm else nn.Identity(),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_out, d_out),
        nn.LayerNorm(d_out) if use_layernorm else nn.Identity(),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_out, d_out),
        nn.LayerNorm(d_out) if use_layernorm else nn.Identity()
    )
