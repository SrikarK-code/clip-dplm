"""
Training Framework for Single-Cell Transport
=========================================

Mathematical Foundation:
---------------------
1. Loss Function:
   L = L_transport + λ_consistency L_consistency where:
   - L_transport: Σ W₂(μᵢ, νᵢ) for each space pair
   - L_consistency: ||T_CE - T_PE ∘ T_CP||²

2. Optimization:
   - Adam with cosine learning rate schedule
   - Gradient clipping for stability
   - Automatic mixed precision for efficiency

Biological Relevance:
-------------------
- Handles sparse gene expression data
- Preserves cell state relationships
- Efficient batch processing for large datasets
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np
from torch.cuda.amp import autocast, GradScaler
import wandb
from dataclasses import dataclass

from transport_maps import TripleTransportMaps, TransportOutput
from config import OptimConfig

logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """Tracks training progress and state"""
    epoch: int = 0
    step: int = 0
    best_loss: float = float('inf')
    patience_counter: int = 0
    
    def should_stop_early(self, patience: int) -> bool:
        """Check early stopping condition"""
        return self.patience_counter >= patience

class SingleCellDataset(Dataset):
    """
    Dataset for single-cell transport
    
    Handles:
    - Gene expression matrices
    - Protein embeddings
    - Perturbation states
    """
    def __init__(
        self,
        cell_states: torch.Tensor,
        protein_states: Optional[torch.Tensor] = None,
        pert_states: Optional[torch.Tensor] = None,
    ):
        self.cell_states = cell_states
        self.protein_states = protein_states
        self.pert_states = pert_states
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Ensure data consistency"""
        n_cells = len(self.cell_states)
        
        if self.protein_states is not None:
            assert len(self.protein_states) == n_cells, "Mismatched protein states"
        
        if self.pert_states is not None:
            assert len(self.pert_states) == n_cells, "Mismatched perturbation states"
            
    def __len__(self) -> int:
        return len(self.cell_states)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get single-cell data point"""
        item = {'cell_states': self.cell_states[idx]}
        
        if self.protein_states is not None:
            item['protein_states'] = self.protein_states[idx]
            
        if self.pert_states is not None:
            item['pert_states'] = self.pert_states[idx]
            
        return item

class Trainer:
    """
    Trainer for single-cell transport system
    """
    def __init__(
        self,
        model: TripleTransportMaps,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config: OptimConfig,
        exp_dir: Path,
        use_wandb: bool = False
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.exp_dir = exp_dir
        self.use_wandb = use_wandb
        
        # Training state
        self.state = TrainingState()
        
        # Setup
        self._setup_training()
        
    def _setup_training(self):
        """Initialize training components"""
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.max_epochs,
            eta_min=self.config.min_learning_rate
        )
        
        # Mixed precision
        self.scaler = GradScaler()
        
        # Create directories
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Initialize wandb
        if self.use_wandb:
            self._setup_wandb()
            
    def _setup_wandb(self):
        """Initialize weights & biases"""
        wandb.init(
            project="single-cell-transport",
            config={
                'model_config': self.model.config.__dict__,
                'optim_config': self.config.__dict__
            }
        )
        
    def train(self):
        """Main training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.config.max_epochs):
            self.state.epoch = epoch
            
            # Training epoch
            train_metrics = self._train_epoch()
            
            # Validation
            if self.val_loader is not None:
                val_metrics = self._validate()
                
                # Early stopping check
                if val_metrics['total_loss'] < self.state.best_loss:
                    self.state.best_loss = val_metrics['total_loss']
                    self.state.patience_counter = 0
                    self._save_checkpoint('best')
                else:
                    self.state.patience_counter += 1
                    
                if self.state.should_stop_early(self.config.early_stopping_patience):
                    logger.info("Early stopping triggered")
                    break
                    
            # Logging
            self._log_metrics(train_metrics, val_metrics if self.val_loader else None)
            
            # LR scheduling
            self.scheduler.step()
            
        logger.info("Training completed")
        
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        metrics_sum = self._init_metrics()
        
        for batch in self.train_loader:
            batch_metrics = self._train_step(batch)
            
            # Accumulate metrics
            for k, v in batch_metrics.items():
                metrics_sum[k] += v
                
        # Average metrics
        metrics = {k: v / len(self.train_loader) for k, v in metrics_sum.items()}
        return metrics
    
    def _train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.optimizer.zero_grad()
        
        with autocast():
            outputs = self.model(
                cell_states=batch['cell_states'],
                pert_states=batch.get('pert_states'),
                protein_states=batch.get('protein_states')
            )
            
            loss = self._compute_loss(outputs)
            
        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Gradient clipping
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.gradient_clip_val
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return self._get_step_metrics(outputs, loss)
    
    def _validate(self) -> Dict[str, float]:
        """Validation loop"""
        self.model.eval()
        metrics_sum = self._init_metrics()
        
        with torch.no_grad():
            for batch in self.val_loader:
                outputs = self.model(
                    cell_states=batch['cell_states'],
                    pert_states=batch.get('pert_states'),
                    protein_states=batch.get('protein_states')
                )
                
                loss = self._compute_loss(outputs)
                step_metrics = self._get_step_metrics(outputs, loss)
                
                # Accumulate metrics
                for k, v in step_metrics.items():
                    metrics_sum[k] += v
                    
        # Average metrics
        metrics = {k: v / len(self.val_loader) for k, v in metrics_sum.items()}
        return metrics
    
    def _compute_loss(
        self,
        outputs: Dict[str, TransportOutput]
    ) -> torch.Tensor:
        """Compute total loss"""
        # Transport losses
        transport_loss = sum(
            out.cost for name, out in outputs.items()
            if isinstance(out, TransportOutput)
        )
        
        # Consistency loss if available
        consistency_loss = outputs.get('consistency_loss', 0.0)
        
        return transport_loss + self.config.consistency_weight * consistency_loss
    
    def _get_step_metrics(
        self,
        outputs: Dict[str, TransportOutput],
        loss: torch.Tensor
    ) -> Dict[str, float]:
        """Get metrics for logging"""
        metrics = {
            'total_loss': loss.item(),
            'learning_rate': self.scheduler.get_last_lr()[0]
        }
        
        # Add individual transport costs
        for name, out in outputs.items():
            if isinstance(out, TransportOutput) and out.metrics:
                for metric_name, value in out.metrics.items():
                    metrics[f"{name}_{metric_name}"] = value
                    
        return metrics
    
    def _init_metrics(self) -> Dict[str, float]:
        """Initialize metrics accumulator"""
        return {
            'total_loss': 0.0,
            'learning_rate': 0.0,
            'cell_to_protein_w2_cost': 0.0,
            'cell_to_pert_w2_cost': 0.0,
            'pert_to_protein_w2_cost': 0.0
        }
    
    def _log_metrics(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]] = None
    ):
        """Log metrics"""
        metrics = {
            'train/' + k: v for k, v in train_metrics.items()
        }
        
        if val_metrics:
            metrics.update({
                'val/' + k: v for k, v in val_metrics.items()
            })
            
        if self.use_wandb:
            wandb.log(metrics, step=self.state.step)
            
        logger.info(
            f"Epoch {self.state.epoch} - "
            f"Train Loss: {train_metrics['total_loss']:.4f}"
            + (f" - Val Loss: {val_metrics['total_loss']:.4f}"
               if val_metrics else "")
        )
        
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_state': self.state.__dict__,
            'config': self.config.__dict__
        }
        
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint: {path}")
        
    def load_checkpoint(self, path: Path):
        """Load checkpoint"""
        checkpoint = torch.load(path)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        self.state = TrainingState(**checkpoint['training_state'])
        
        logger.info(f"Loaded checkpoint: {path}")

def create_data_loaders(
    cell_states: torch.Tensor,
    protein_states: Optional[torch.Tensor] = None,
    pert_states: Optional[torch.Tensor] = None,
    batch_size: int = 128,
    val_split: float = 0.1,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Create train/val data loaders"""
    # Split data
    n_samples = len(cell_states)
    n_val = int(n_samples * val_split)
    indices = torch.randperm(n_samples)
    
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]
    
    # Create datasets
    train_dataset = SingleCellDataset(
        cell_states=cell_states[train_idx],
        protein_states=protein_states[train_idx] if protein_states is not None else None,
        pert_states=pert_states[train_idx] if pert_states is not None else None
    )
    
    val_dataset = SingleCellDataset(
        cell_states=cell_states[val_idx],
        protein_states=protein_states[val_idx] if protein_states is not None else None,
        pert_states=pert_states[val_idx] if pert_states is not None else None
    )
    
    # Create loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader
