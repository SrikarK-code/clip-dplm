import torch
import yaml
import pandas as pd
from pathlib import Path
from utils.metrics import FlowEvaluator, BiologicalMetrics
from utils.data import get_dataloader
from models.triple_flow import TripleFlowModel

def evaluate(config_path, checkpoint_path):
    # Load configuration
    with open(config_path) as f:
        config = yaml.safe_load(f)
        
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path)
    
    # Initialize model
    model = TripleFlowModel(config['model'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config['device'])
    model.eval()
    
    # Initialize evaluators
    flow_evaluator = FlowEvaluator(config)
    bio_metrics = BiologicalMetrics()
    
    # Get test dataloader
    test_loader = get_dataloader(config, 'test')
    
    # Collect predictions and metrics
    all_metrics = []
    all_embeddings = []
    
    with torch.no_grad():
        for batch in test_loader:
            # Forward pass
            flows, embeddings = model(batch)
            
            # Compute metrics
            metrics = flow_evaluator.compute_all_metrics(flows, embeddings)
            
            if config['eval']['compute_biological_metrics']:
                bio_metrics_batch = bio_metrics.compute_metrics(
                    batch, flows, embeddings)
                metrics.update(bio_metrics_batch)
                
            all_metrics.append(metrics)
            
            # Save embeddings if requested
            if config['eval']['save_embeddings']:
                all_embeddings.append(embeddings)
    
    # Aggregate metrics
    metrics_df = pd.DataFrame(all_metrics)
    mean_metrics = metrics_df.mean()
    std_metrics = metrics_df.std()
    
    # Save results
    results_dir = Path(config['logging']['save_dir']) / 'evaluation'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    mean_metrics.to_csv(results_dir / 'mean_metrics.csv')
    std_metrics.to_csv(results_dir / 'std_metrics.csv')
    
    if config['eval']['save_embeddings']:
        torch.save(all_embeddings, results_dir / 'embeddings.pt')
        
    return mean_metrics, std_metrics

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    args = parser.parse_args()
    
    mean_metrics, std_metrics = evaluate(args.config, args.checkpoint)
    print("\nEvaluation Results:")
    for metric, value in mean_metrics.items():
        print(f"{metric}: {value:.4f} ± {std_metrics[metric]:.4f}")
