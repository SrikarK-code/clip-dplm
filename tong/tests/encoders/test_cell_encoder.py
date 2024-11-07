import torch
from models.encoders import CellStateEncoder
def test_cell_encoder(adata):
    gene_dim=adata.n_vars
    encoder=CellStateEncoder(gene_dim=gene_dim,latent_dim=512)
    expr=torch.tensor(adata.X.toarray()[0:100]).float()
    dpt=torch.tensor(adata.obs['dpt_pseudotime'].values[0:100]).float()
    edge_idx=torch.tensor(adata.uns['neighbors']['connectivities'].nonzero())[:,0:100]
    batch_idx=torch.zeros(100)
    out=encoder(expr,dpt,edge_idx,batch_idx)
    print(f"Output shape: {out.shape}")
    return out
def analyze_cell_embeddings(embeddings,adata):
    import umap
    reducer=umap.UMAP()
    emb_2d=reducer.fit_transform(embeddings.detach().numpy())
    sc.pl.scatter(adata,basis='X_umap',color='cell_type')
    plt.figure()
    plt.scatter(emb_2d[:,0],emb_2d[:,1],c=adata.obs['cell_type'].cat.codes)
