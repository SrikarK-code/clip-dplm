import torch
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from utils.visualization import Visualizer
from utils.data import get_dataloader
from models.triple_flow import TripleFlowModel

def visualize(config_path, checkpoint_path):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize model and visualizer
    model = TripleFlowModel(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config['device'])
    model.eval()
    
    visualizer = Visualizer(config['logging']['save_dir'])
    
    # Get test dataloader
    test_loader = get_dataloader(config, 'test')
    
    # Create visualizations
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # Forward pass
            flows, embeddings = model(batch)
            
            # Visualize embeddings
            fig = visualizer.plot_embeddings(
                embeddings,
                labels=batch.get('labels'),
                method='tsne'
            )
            fig.savefig(f'embeddings_batch_{batch_idx}.png')
            plt.close(fig)
            
            # Visualize flows
            for flow_name, flow_outputs in flows.items():
                fig = visualizer.plot_flow_field(flow_outputs)
                fig.savefig(f'flow_{flow_name}_batch_{batch_idx}.png')
                plt.close(fig)
                
            # Visualize attention
            if hasattr(model.cell_encoder, 'gnn'):
                fig = visualizer.plot_attention_weights(
                    model.cell_encoder.gnn.layers[0].attention_weights,
                    labels=batch.get('gene_names')
                )
                fig.savefig(f'attention_batch_{batch_idx}.png')
                plt.close(fig)
                
            if batch_idx >= config['eval']['vis_frequency']:
                break

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    visualize(args.config, args.checkpoint)
