import torch
from models.flows import OTFlow
def test_ot_flow(cell_embeddings):
    flow=OTFlow(latent_dim=512)
    src=cell_embeddings[0:50]
    tgt=cell_embeddings[50:100]
    v,xt,t=flow(src,tgt)
    print(f"Vector field shape: {v.shape}")
    print(f"Sample points shape: {xt.shape}")
    print(f"Time points shape: {t.shape}")
    visualize_flow_field(v,xt,t)
def visualize_flow_field(v,xt,t):
    plt.figure(figsize=(10,10))
    plt.quiver(xt[:,0].detach(),xt[:,1].detach(),
               v[:,0].detach(),v[:,1].detach(),t.detach())
    plt.colorbar(label='Time')
