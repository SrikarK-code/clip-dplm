import torch
import scanpy as sc
def test_dataset_basic():
    adata=sc.read_h5ad("data.h5ad")
    print(f"Cells: {adata.n_obs}")
    print(f"Genes: {adata.n_vars}")
    print(f"Sparsity: {1-adata.X.data.size/np.prod(adata.X.shape)}")
    sc.pl.highest_expr_genes(adata,n_top=20)
    return adata
def test_cell_types(adata):
    if 'cell_type' in adata.obs:
        sc.pl.umap(adata,color='cell_type')
        print(adata.obs['cell_type'].value_counts())
def test_graph_connectivity(adata):
    sc.pp.neighbors(adata)
    G=sc.neighbors.umap.umap_init_graph(adata.obsp['connectivities'],adata.shape[0])
    print(f"Graph edges: {G.number_of_edges()}")
    print(f"Average degree: {G.number_of_edges()*2/G.number_of_nodes()}")
