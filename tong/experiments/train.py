import os
import yaml
import torch
import wandb
from pathlib import Path
from utils.training import Trainer
from utils.data import get_dataloader
from models.triple_flow import TripleFlowModel

def train(config_path):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config['device'] = device
    
    # Initialize wandb if enabled
    if config['logging']['wandb']:
        wandb.init(
            project=config['logging']['project_name'],
            config=config
        )
        
    # Create save directory
    save_dir = Path(config['logging']['save_dir'])
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create dataloaders
    train_loader = get_dataloader(config, 'train')
    val_loader = get_dataloader(config, 'val')
    
    # Initialize model
    model = TripleFlowModel(config['model'])
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        config=config,
        device=device
    )
    
    # Train model
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=config['training']['max_epochs']
    )
    
    # Save final model
    trainer.save_checkpoint('final.pt')
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    args = parser.parse_args()
    train(args.config)
