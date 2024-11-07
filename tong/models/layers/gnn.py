import torch
import torch.nn as nn
from torch_scatter import scatter_mean
import math

class PiGNNLayer(nn.Module):
    """Protein-informed Graph Neural Network Layer."""
    
    def __init__(self, d_emb, n_heads, n_neighbors, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_emb // n_heads
        self.n_neighbors = n_neighbors
        
        # Multi-head attention
        self.q_proj = nn.Linear(d_emb, d_emb)
        self.k_proj = nn.Linear(d_emb, d_emb)
        self.v_proj = nn.Linear(d_emb, d_emb)
        self.o_proj = nn.Linear(d_emb, d_emb)
        
        # Edge features
        self.edge_mlp = nn.Sequential(
            nn.Linear(3 * d_emb, d_emb),
            nn.LayerNorm(d_emb),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, d_emb)
        )
        
        # Node update
        self.node_mlp = nn.Sequential(
            nn.Linear(2 * d_emb, d_emb * 2),
            nn.LayerNorm(d_emb * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb * 2, d_emb)
        )
        
        # Global attention
        self.global_gate = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LayerNorm(d_emb),
            nn.GELU(),
            nn.Linear(d_emb, d_emb),
            nn.Sigmoid()
        )
        
        self.layer_norm1 = nn.LayerNorm(d_emb)
        self.layer_norm2 = nn.LayerNorm(d_emb)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, h, e, edge_index, batch_idx):
        """
        Forward pass of PiGNN layer.
        
        Args:
            h: Node features [n_nodes, d_emb]
            e: Edge features [n_edges, d_emb]
            edge_index: Edge indices [2, n_edges]
            batch_idx: Batch indices [n_nodes]
            
        Returns:
            Updated node features and edge features
        """
        # Get source and target nodes
        src, dst = edge_index
        
        # Multi-head attention
        q = self.q_proj(h[dst]).view(-1, self.n_heads, self.d_head)
        k = self.k_proj(h[src]).view(-1, self.n_heads, self.d_head) 
        v = self.v_proj(h[src]).view(-1, self.n_heads, self.d_head)
        
        # Attention scores
        scores = torch.einsum('nhd,nhd->nh', q, k) / math.sqrt(self.d_head)
        attn = torch.softmax(scores, dim=1)
        attn = self.dropout(attn)
        
        # Compute message
        msg = torch.einsum('nh,nhd->nhd', attn, v)
        msg = msg.reshape(-1, self.n_heads * self.d_head)
        msg = self.o_proj(msg)
        
        # Update edge features
        e_in = torch.cat([h[src], e, h[dst]], dim=-1)
        e = self.layer_norm1(e + self.edge_mlp(e_in))
        
        # Update node features
        h_in = torch.cat([msg, h[dst]], dim=-1)
        h_update = self.node_mlp(h_in)
        h = self.layer_norm2(h + h_update)
        
        # Global pooling and gating
        h_global = scatter_mean(h, batch_idx, dim=0)
        gates = self.global_gate(h_global)
        h = h * gates[batch_idx]
        
        return h, e

class MultiLayerPiGNN(nn.Module):
    """Multiple layers of PiGNN with residual connections."""
    
    def __init__(self, latent_dim, n_layers, n_heads, n_neighbors, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PiGNNLayer(latent_dim, n_heads, n_neighbors, dropout)
            for _ in range(n_layers)
        ])
        
        self.skip_proj = nn.Linear(latent_dim * n_layers, latent_dim)
        self.layer_norm = nn.LayerNorm(latent_dim)
        
    def forward(self, h, edge_index, batch_idx):
        e = torch.zeros(edge_index.shape[1], h.shape[1], device=h.device)
        
        # Collect intermediate representations
        h_intermediates = []
        
        for layer in self.layers:
            h, e = layer(h, e, edge_index, batch_idx)
            h_intermediates.append(h)
            
        # Combine all intermediate representations
        h_cat = torch.cat(h_intermediates, dim=-1)
        h_skip = self.skip_proj(h_cat)
        
        h = self.layer_norm(h + h_skip)
        
        return h
