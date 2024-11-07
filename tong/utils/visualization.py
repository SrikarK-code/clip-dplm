import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import plotly.express as px

class Visualizer:
    def __init__(self, save_dir=None):
        self.save_dir = save_dir
        
    def plot_embeddings(self, embeddings, labels=None, method='tsne'):
        """Plot embeddings using dimensionality reduction."""
        spaces = ['cell_emb', 'pert_emb', 'protein_emb']
        fig, axes = plt.subplots(1, len(spaces), figsize=(15, 5))
        
        for i, space in enumerate(spaces):
            if space not in embeddings:
                continue
                
            emb = embeddings[space].detach().cpu().numpy()
            
            # Reduce dimensionality
            if method == 'tsne':
                reducer = TSNE(n_components=2)
                emb_2d = reducer.fit_transform(emb)
                
            # Plot
            ax = axes[i]
            if labels is not None and f'{space}_labels' in labels:
                scatter = ax.scatter(emb_2d[:, 0], emb_2d[:, 1], 
                                  c=labels[f'{space}_labels'])
                plt.colorbar(scatter, ax=ax)
            else:
                ax.scatter(emb_2d[:, 0], emb_2d[:, 1])
                
            ax.set_title(space)
                
        return fig
        
    def plot_flow_field(self, flow_outputs, n_points=20):
        """Plot the vector field of a flow."""
        v, xt, t = flow_outputs
        
        # Create grid
        x = np.linspace(xt.min(), xt.max(), n_points)
        y = np.linspace(xt.min(), xt.max(), n_points)
        X, Y = np.meshgrid(x, y)
        
        # Get vectors
        grid_points = torch.tensor(np.stack([X.flatten(), Y.flatten()]).T)
        with torch.no_grad():
            vectors = v(grid_points).reshape(n_points, n_points, 2)
            
        # Plot
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.streamplot(X, Y, vectors[..., 0], vectors[..., 1])
        ax.set_title('Flow Vector Field')
        
        return fig
        
    def plot_attention_weights(self, attention_weights, labels=None):
        """Plot attention weights between nodes."""
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(attention_weights.detach().cpu().numpy(), 
                   ax=ax, cmap='viridis')
        
        if labels is not None:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            
        ax.set_title('Attention Weights')
        
        return fig
        
    def plot_training_progress(self, metrics_history):
        """Plot training metrics over time."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 15))
        axes = axes.flatten()
        
        for i, (metric, values) in enumerate(metrics_history.items()):
            if i >= len(axes):
                break
                
            axes[i].plot(values['train'], label='train')
            axes[i].plot(values['val'], label='val')
            axes[i].set_title(metric)
            axes[i].legend()
            
        return fig
