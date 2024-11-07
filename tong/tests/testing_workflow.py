# Data inspection:
adata = test_dataset_basic()
test_cell_types(adata)
test_graph_connectivity(adata)

# Test encoders:
cell_emb = test_cell_encoder(adata)
analyze_cell_embeddings(cell_emb, adata)

# Test flows:
test_ot_flow(cell_emb)

# Test losses:
loss = test_contrastive(cell_emb, protein_emb)
analyze_embedding_alignment(cell_emb, protein_emb)

# Test generation:
test_protein_generation(model, cell_state_1, cell_state_2)
