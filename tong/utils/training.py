import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .metrics import FlowEvaluator, BiologicalMetrics
from .visualization import Visualizer

class Trainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model.to(device)
        self.config = config
        self.device = device
        
        # Initialize optimizers
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
        
        # Initialize evaluators
        self.flow_evaluator = FlowEvaluator(config)
        self.bio_metrics = BiologicalMetrics()
        self.visualizer = Visualizer(config.save_dir)
        
        # Initialize metric tracking
        self.metrics_history = {}
        
    def train_epoch(self, dataloader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        metrics = []
        
        for batch in dataloader:
            self.optimizer.zero_grad()
            
            # Forward pass
            flows, outputs = self.model(batch)
            loss = compute_all_losses(flows, outputs, self.config)
            
            # Backward pass
            loss.backward()
            if self.config.training.get('clip_grad_norm'):
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.clip_grad_norm
                )
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Compute metrics
            batch_metrics = self.flow_evaluator.compute_all_metrics(flows, outputs)
            metrics.append(batch_metrics)
            
        return total_loss / len(dataloader), metrics
        
    def evaluate(self, dataloader):
        """Evaluate model."""
        self.model.eval()
        total_loss = 0
        metrics = []
        
        with torch.no_grad():
            for batch in dataloader:
                flows, outputs = self.model(batch)
                loss = compute_all_losses(flows, outputs, self.config)
                
                total_loss += loss.item()
                metrics.append(
                    self.flow_evaluator.compute_all_metrics(flows, outputs)
                )
                
        return total_loss / len(dataloader), metrics
        
    def train(self, train_loader, val_loader, n_epochs):
        """Complete training loop."""
        best_val_loss = float('inf')
        patience = self.config.training.early_stopping_patience
        patience_counter = 0
        
        for epoch in range(n_epochs):
            # Training
            train_loss, train_metrics = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_metrics = self.evaluate(val_loader)
            
            # Update metrics history
            self._update_metrics_history(
                train_loss, val_loss,
                train_metrics, val_metrics
            )
            
            # Visualization
            if epoch % self.config.eval.vis_frequency == 0:
                self._create_visualizations(epoch)
                
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint('best.pt')
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
                
            # Save checkpoint
            if epoch % self.config.training.checkpoint_frequency == 0:
                self._save_checkpoint(f'checkpoint_{epoch}.pt')
                
            self._print_epoch_summary(epoch, train_loss, val_loss)
                
    def _update_metrics_history(self, train_loss, val_loss, train_metrics, val_metrics):
        """Update metrics history."""
        if 'loss' not in self.metrics_history:
            self.metrics_history['loss'] = {'train': [], 'val': []}
            
        self.metrics_history['loss']['train'].append(train_loss)
        self.metrics_history['loss']['val'].append(val_loss)
        
        # Update other metrics
        for metric in train_metrics[0].keys():
            if metric not in self.metrics_history:
                self.metrics_history[metric] = {'train': [], 'val': []}
                
            self.metrics_history[metric]['train'].append(
                np.mean([m[metric] for m in train_metrics])
            )
            self.metrics_history[metric]['val'].append(
                np.mean([m[metric] for m in val_metrics])
            )
            
    def _create_visualizations(self, epoch):
        """Create and save visualizations."""
        # Plot metrics
        fig = self.visualizer.plot_training_progress(self.metrics_history)
        fig.savefig(f'{self.config.save_dir}/metrics_{epoch}.png')
        plt.close(fig)
        
    def _save_checkpoint(self, filename):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics_history': self.metrics_history,
            'config': self.config
        }, f'{self.config.save_dir}/{filename}')
        
    def _print_epoch_summary(self, epoch, train_loss, val_loss):
        """Print epoch summary."""
        print(f
