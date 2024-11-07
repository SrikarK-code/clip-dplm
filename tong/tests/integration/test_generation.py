import torch
from models import TripleFlowModel
def test_protein_generation(model,cell_state_1,cell_state_2):
    with torch.no_grad():
        cell_emb_1=model.cell_encoder(cell_state_1['gene_expr'],
                                    cell_state_1['dpt'],
                                    cell_state_1['edge_index'],
                                    cell_state_1['batch_idx'])
        cell_emb_2=model.cell_encoder(cell_state_2['gene_expr'],
                                    cell_state_2['dpt'],
                                    cell_state_2['edge_index'],
                                    cell_state_2['batch_idx'])
        v_cell,x_cell,t_cell=model.cell_to_cell(cell_emb_1,cell_emb_2)
        trajectory=model.cell_to_protein(x_cell)
        visualize_trajectory(trajectory)
def visualize_trajectory(trajectory):
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.plot(trajectory.detach()[:,0],trajectory.detach()[:,1],'.-')
    plt.title('Latent Space Trajectory')
    plt.subplot(122)
    plt.imshow(trajectory.detach().T)
    plt.title('Feature Evolution')
