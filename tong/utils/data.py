import torch
from torch.utils.data import Dataset
import scanpy as sc
import anndata
import numpy as np
from scipy.sparse import issparse

class TripleFlowDataset(Dataset):
    """Dataset for triple flow model training."""
    
    def __init__(
        self,
        adata_path,
        protein_embeddings=None,
        gene_to_esm=None,
        device='cuda',
        transform=None
    ):
        # Load data
        self.adata = anndata.read_h5ad(adata_path)
        self.protein_embeddings = protein_embeddings
        self.gene_to_esm = gene_to_esm
        self.device = device
        self.transform = transform
        
        # Compute PAGA and DPT if not present
        if 'dpt_pseudotime' not in self.adata.obs:
            self._compute_trajectory_info()
        
        # Get control mask
        self.ctrl_mask = self.adata.obs['gene'] == 'CTRL'
        
        # Prepare graph connectivity
        self._prepare_graph()
        
    def _compute_trajectory_info(self):
        """Compute trajectory information using scanpy."""
        # Set root as control cell
        ctrl_cells = np.where(self.adata.obs['gene']=='CTRL')[0]
        self.adata.uns['iroot'] = ctrl_cells[0]
        
        # Compute neighbors if needed
        if 'neighbors' not in self.adata.uns:
            sc.pp.neighbors(self.adata, use_rep='X_pca')
            
        # Compute trajectory
        sc.tl.paga(self.adata)
        sc.tl.diffmap(self.adata)
        sc.tl.dpt(self.adata)
        
    def _prepare_graph(self):
        """Prepare graph connectivity information."""
        # Get adjacency matrix
        adj = self.adata.uns['neighbors']['connectivities']
        
        # Convert to edge index format
        if issparse(adj):
            adj = adj.tocoo()
            self.edge_index = torch.tensor(
                [adj.row, adj.col],
                device=self.device
            )
        else:
            self.edge_index = torch.tensor(
                adj.nonzero(),
                device=self.device
            )
        
    def get_perturbed_genes(self, idx):
        """Get perturbation information for a cell."""
        if self.ctrl_mask[idx]:
            return None, None
            
        gene = self.adata.obs.iloc[idx]['gene']
        pert_class = self.adata.obs.iloc[idx]['pertclass']
        
        # Get differential expression results
        de_key = f'rank_genes_{pert_class}_{gene}'
        if de_key not in self.adata.uns:
            return None, None
            
        # Get top genes
        top_genes = self.adata.uns[de_key][:10]  # Top 10 genes
        gene_values = torch.tensor(
            self.adata.uns[de_key]['scores'][:10],
            device=self.device
        )
        
        # Get ESM embeddings
        if self.gene_to_esm is not None:
            gene_embeddings = []
            for g in top_genes:
                if g in self.gene_to_esm:
                    gene_embeddings.append(self.gene_to_esm[g])
                    
            gene_embeddings = torch.tensor(
                np.stack(gene_embeddings),
                device=self.device
            )
            
            return gene_embeddings, gene_values
            
        return None, None
        
    def __len__(self):
        return len(self.adata)
        
    def __getitem__(self, idx):
        # Get gene expression
        x = self.adata.X[idx]
        if issparse(x):
            x = x.toarray()
        gene_expr = torch.tensor(x[0], device=self.device)
        
        # Get pseudotime
        dpt = torch.tensor(
            [self.adata.obs['dpt_pseudotime'][idx]],
            device=self.device
        )
        
        # Get perturbation info
        gene_indices, gene_values = self.get_perturbed_genes(idx)
        
        # Get protein embedding if available
        protein_emb = None
        if self.protein_embeddings is not None:
            protein = self.adata.obs.iloc[idx]['protein']
            if protein in self.protein_embeddings:
                protein_emb = torch.tensor(
                    self.protein_embeddings[protein],
                    device=self.device
                )
        
        # Create output dictionary
        output = {
            'gene_expr': gene_expr,
            'dpt': dpt,
            'edge_index': self.edge_index,
            'batch_idx': torch.tensor([0], device=self.device)
        }
        
        if gene_indices is not None:
            output['gene_indices'] = gene_indices
            output['gene_values'] = gene_values
            
        if protein_emb is not None:
            output['protein_emb'] = protein_emb
            
        if self.transform:
            output = self.transform(output)
            
        return output

class MemoryQueue:
    """Memory queue for contrastive learning."""
    
    def __init__(self, size, dim):
        self.size = size
        self.dim = dim
        self.ptr = 0
        self.queue = torch.zeros(size, dim)
        
    def enqueue_dequeue(self, embeddings):
        """Update queue with new embeddings using FIFO."""
        batch_size = embeddings.shape[0]
        
        # Return if queue is not initialized
        if self.queue is None:
            self.queue = embeddings.detach()
            return self.queue
            
        # Enqueue new embeddings and dequeue old ones
        if self.ptr + batch_size > self.size:
            # Handle queue wrap-around
            first_part = self.size - self.ptr
            self.queue[self.ptr:] = embeddings[:first_part].detach()
            self.queue[:batch_size - first_part] = embeddings[first_part:].detach()
            self.ptr = batch_size - first_part
        else:
            # Normal enqueue
            self.queue[self.ptr:self.ptr + batch_size] = embeddings.detach()
            self.ptr = (self.ptr + batch_size) % self.size
            
        return self.queue

class MultiModalBatch:
    """Batch collation with support for missing modalities."""
    
    def __init__(self, device='cuda'):
        self.device = device
        
    def __call__(self, batch):
        """Collate batch while handling missing modalities."""
        # Initialize output dictionary
        output = {
            'gene_expr': [],
            'dpt': [],
            'batch_idx': [],
            'edge_index': [],
            'gene_indices': [],
            'gene_values': [],
            'protein_emb': []
        }
        
        # Collect all available data
        batch_size = 0
        edge_offset = 0
        
        for data in batch:
            # Always present data
            output['gene_expr'].append(data['gene_expr'])
            output['dpt'].append(data['dpt'])
            
            # Update batch indices
            batch_idx = torch.full(
                (len(data['gene_expr']),),
                batch_size,
                device=self.device
            )
            output['batch_idx'].append(batch_idx)
            
            # Update edge indices with offset
            edge_index = data['edge_index'].clone()
            edge_index[0] += edge_offset
            edge_index[1] += edge_offset
            output['edge_index'].append(edge_index)
            
            # Optional modalities
            if 'gene_indices' in data:
                output['gene_indices'].append(data['gene_indices'])
                output['gene_values'].append(data['gene_values'])
            
            if 'protein_emb' in data:
                output['protein_emb'].append(data['protein_emb'])
                
            # Update offsets
            batch_size += 1
            edge_offset += len(data['gene_expr'])
            
        # Concatenate all tensors
        for key in output:
            if output[key]:
                output[key] = torch.cat(output[key], dim=0)
            else:
                output.pop(key)  # Remove empty modalities
                
        return output

class DataAugmentation:
    """Data augmentation for triple flow training."""
    
    def __init__(self, config):
        self.config = config
        
    def __call__(self, data):
        """Apply augmentations to the data."""
        # Gene expression augmentation
        if self.config.augmentation.gene_dropout > 0:
            mask = torch.rand_like(data['gene_expr']) > self.config.augmentation.gene_dropout
            data['gene_expr'] = data['gene_expr'] * mask
            
        # Graph augmentation
        if self.config.augmentation.edge_dropout > 0:
            mask = torch.rand(data['edge_index'].shape[1]) > self.config.augmentation.edge_dropout
            data['edge_index'] = data['edge_index'][:, mask]
            
        # Perturbation augmentation
        if 'gene_indices' in data and self.config.augmentation.pert_noise > 0:
            noise = torch.randn_like(data['gene_values']) * self.config.augmentation.pert_noise
            data['gene_values'] = data['gene_values'] + noise
            
        return data

def get_dataloader(config, split='train'):
    """Create dataloader with all utilities."""
    # Load data
    dataset = TripleFlowDataset(
        adata_path=config.data[f'{split}_path'],
        protein_embeddings=config.data.get('protein_embeddings'),
        gene_to_esm=config.data.get('gene_to_esm'),
        device=config.device,
        transform=DataAugmentation(config) if split == 'train' else None
    )
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        shuffle=(split == 'train'),
        collate_fn=MultiModalBatch(config.device),
        num_workers=config.training.num_workers,
        pin_memory=True
    )
    
    return dataloader
