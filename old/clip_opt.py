import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import numpy as np

class OptimizedProjectionHead(nn.Module):
    """Improved projection head with skip connections and layer scaling"""
    def __init__(self, input_dim, output_dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim * 2  # Wider network
            
        self.skip = nn.Linear(input_dim, output_dim)
        self.layer_scale = nn.Parameter(torch.ones(1) * 1e-4)
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Initialize using custom scheme
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    
    def forward(self, x):
        projected = self.projection(x)
        skip = self.skip(x)
        return skip + self.layer_scale * projected

class OptimizedCLIPModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Larger embedding cache for hard negative mining
        self.protein_embedding_cache = torch.zeros(
            (config.cache_size, config.projection_dim),
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        self.cache_ptr = 0
        
        # Models and projections
        self.diffmap_model = CLIPEncoder(config.diffmap_config)
        self.protein_model = CLIPEncoder(config.protein_config)
        
        self.diffmap_projection = OptimizedProjectionHead(
            input_dim=config.diffmap_config.hidden_size,
            output_dim=config.projection_dim,
            hidden_dim=config.projection_dim * 4  # Wider projections
        )
        self.protein_projection = OptimizedProjectionHead(
            input_dim=config.protein_config.hidden_size,
            output_dim=config.projection_dim,
            hidden_dim=config.projection_dim * 4
        )
        
        # Learned temperature with careful initialization
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1/0.07))
    
    def update_cache(self, protein_embeds):
        batch_size = protein_embeds.size(0)
        if self.cache_ptr + batch_size > self.config.cache_size:
            self.cache_ptr = 0
        self.protein_embedding_cache[self.cache_ptr:self.cache_ptr + batch_size] = protein_embeds.detach()
        self.cache_ptr = (self.cache_ptr + batch_size) % self.config.cache_size
    
    def forward(self, diffmap_values, protein_values, gather_distributed=True):
        # Get embeddings
        diffmap_outputs = self.diffmap_model(diffmap_values)
        protein_outputs = self.protein_model(protein_values)
        
        # Project with improved heads
        diffmap_embeds = self.diffmap_projection(diffmap_outputs)
        protein_embeds = self.protein_projection(protein_outputs)
        
        # Normalize with custom temperature
        diffmap_embeds = F.normalize(diffmap_embeds, dim=-1)
        protein_embeds = F.normalize(protein_embeds, dim=-1)
        
        # Update cache for hard negatives
        self.update_cache(protein_embeds)
        
        # Compute similarity with main batch and cached embeddings
        logit_scale = self.logit_scale.exp().clamp(max=100)
        
        if gather_distributed and dist.is_initialized():
            # Gather embeddings from all GPUs
            world_size = dist.get_world_size()
            diffmap_embeds_gathered = [torch.zeros_like(diffmap_embeds) for _ in range(world_size)]
            protein_embeds_gathered = [torch.zeros_like(protein_embeds) for _ in range(world_size)]
            
            dist.all_gather(diffmap_embeds_gathered, diffmap_embeds)
            dist.all_gather(protein_embeds_gathered, protein_embeds)
            
            diffmap_embeds = torch.cat(diffmap_embeds_gathered, dim=0)
            protein_embeds = torch.cat(protein_embeds_gathered, dim=0)
        
        # Compute main similarity
        sim_d_p = torch.matmul(diffmap_embeds, protein_embeds.t()) * logit_scale
        
        # Add hard negatives from cache
        sim_d_cache = torch.matmul(
            diffmap_embeds, 
            self.protein_embedding_cache[:self.cache_ptr].t()
        ) * logit_scale
        
        return {
            "logits_per_diffmap_protein": sim_d_p,
            "logits_per_diffmap_cache": sim_d_cache,
            "diffmap_embeds": diffmap_embeds,
            "protein_embeds": protein_embeds,
        }

def optimized_clip_loss(outputs, temperature=0.07):
    """Enhanced CLIP loss with hard negative mining"""
    sim_d_p = outputs["logits_per_diffmap_protein"]
    sim_d_cache = outputs["logits_per_diffmap_cache"]
    
    # Combine main similarities with cache
    combined_sim = torch.cat([sim_d_p, sim_d_cache], dim=1)
    
    batch_size = sim_d_p.size(0)
    labels = torch.arange(batch_size).to(sim_d_p.device)
    
    # Use label smoothing
    smooth_factor = 0.1
    n_categories = combined_sim.size(1)
    smooth_labels = torch.full_like(combined_sim, smooth_factor / (n_categories - 1))
    smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smooth_factor)
    
    # Compute loss with hard negatives
    loss_d = F.cross_entropy(combined_sim, labels)
    loss_p = F.cross_entropy(sim_d_p.t(), labels)
    
    return (loss_d + loss_p) / 2

def train_with_optimizations(model, train_loader, optimizer, epochs=100):
    model = DDP(model)
    scaler = GradScaler()
    
    for epoch in range(epochs):
        for batch in train_loader:
            diffmap_batch, protein_batch = batch
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(diffmap_batch, protein_batch)
                loss = optimized_clip_loss(outputs)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
