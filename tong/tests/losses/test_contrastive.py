import torch
from utils.losses import contrastive_loss
def test_contrastive(cell_emb,protein_emb,temperature=0.1):
    loss=contrastive_loss(cell_emb,protein_emb,temperature)
    print(f"Contrastive loss: {loss.item()}")
    similarity=torch.matmul(cell_emb,protein_emb.T)
    plt.figure()
    plt.imshow(similarity.detach())
    plt.colorbar()
    return loss
def analyze_embedding_alignment(cell_emb,protein_emb):
    from sklearn.metrics import adjusted_rand_score
    cell_clusters=KMeans(n_clusters=10).fit_predict(cell_emb.detach())
    prot_clusters=KMeans(n_clusters=10).fit_predict(protein_emb.detach())
    ari=adjusted_rand_score(cell_clusters,prot_clusters)
    print(f"Adjusted Rand Index: {ari}")
