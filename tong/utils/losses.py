import torch
import torch.nn.functional as F

def contrastive_loss(x, y, temperature=0.1, queue=None):
    """InfoNCE contrastive loss with optional memory queue."""
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    
    # Include queue samples if provided
    if queue is not None:
        y = torch.cat([y, queue.clone().detach()], dim=0)
    
    # Compute similarities
    sim = torch.matmul(x, y.T) / temperature
    
    # Labels are the positives on the diagonal
    labels = torch.arange(len(x), device=x.device)
    
    return F.cross_entropy(sim, labels)

def path_length_regularization(v):
    """Regularize path length of flows."""
    return torch.mean(torch.norm(v, dim=-1) ** 2)

def jacobian_regularization(v, x):
    """Regularize Jacobian of vector field."""
    jac = torch.autograd.functional.jacobian(lambda x: v, x)
    return torch.norm(jac, p='fro')

def flow_matching_loss(v, target_v):
    """L2 loss between predicted and target vector fields."""
    return F.mse_loss(v, target_v)

def compute_all_losses(flows, embeddings, config):
    """Compute all losses for training."""
    total_loss = 0
    loss_dict = {}
    
    # Contrastive losses between spaces
    if config.loss_weights.contrastive > 0:
        spaces = ['cell_emb', 'pert_emb', 'protein_emb']
        for i, space1 in enumerate(spaces):
            if space1 not in embeddings:
                continue
            for space2 in spaces[i+1:]:
                if space2 not in embeddings:
                    continue
                    
                loss = contrastive_loss(
                    embeddings[space1],
                    embeddings[space2],
                    temperature=config.training.temperature,
                    queue=embeddings.get(f'{space2}_queue')
                )
                total_loss += config.loss_weights.contrastive * loss
                loss_dict[f'contrastive_{space1}_{space2}'] = loss.item()
                
    # Flow matching losses
    if config.loss_weights.flow > 0:
        for flow_name, (v, xt, t, target_v) in flows.items():
            loss = flow_matching_loss(v, target_v)
            total_loss += config.loss_weights.flow * loss
            loss_dict[f'flow_{flow_name}'] = loss.item()
            
    # Regularization losses
    if config.loss_weights.regularization > 0:
        for flow_name, (v, xt, t, _) in flows.items():
            # Path length regularization
            if config.regularization.path_length:
                path_loss = path_length_regularization(v)
                total_loss += config.loss_weights.regularization * path_loss
                loss_dict[f'path_length_{flow_name}'] = path_loss.item()
                
            # Jacobian regularization
            if config.regularization.jacobian:
                jac_loss = jacobian_regularization(v, xt)
                total_loss += config.loss_weights.regularization * jac_loss
                loss_dict[f'jacobian_{flow_name}'] = jac_loss.item()
                
    return total_loss, loss_dict
