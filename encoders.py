import re
import numpy as np
import pandas as pd
import ast
import torch
import torch.nn as nn

import scanpy as sc
import scvi

from transformers import T5EncoderModel, T5Tokenizer, T5ForConditionalGeneration



class ProtT5EncodingModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.protT5_model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc")
        self.protT5_tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_bfd")

    def forward(self, sequence):
        processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        ids = self.protT5_tokenizer(processed_seq, add_special_tokens=True, return_tensors="pt", padding='longest')
        input_ids = ids['input_ids'].to(self.protT5_model.device)
        attention_mask = ids['attention_mask'].to(self.protT5_model.device)

        with torch.no_grad():
            embedding_repr = self.protT5_model(input_ids=input_ids, attention_mask=attention_mask)

        seq_emb = embedding_repr.last_hidden_state
        return seq_emb




class DiffMapEncodingModule:
    def __init__(self):
        self.latent_representations = {}

    def encode(self, adata_dict):
        for cell_type, adata in adata_dict.items():
            print(f"Encoding for cell type: {cell_type}...")
            adata_copy = adata.copy()
            latent = adata_copy.obsm['X_diffmap']
            self.latent_representations[cell_type] = latent

        print("Encoding completed.")
        return self.latent_representations




class RNABERTEncodingModule:
    def __init__(self, pretraining_path='bert_mul_2.pth', batch_size=1):
        self.pretraining_path = pretraining_path
        self.batch_size = batch_size
        self.embeddings = None

    def run_model(self, rna_motif_sequence, embedding_output='embeddings_rnabert.txt'):
        with open('rna_motif_input.txt', 'w') as file:
            file.write(f"rna1\n{rna_motif_sequence}")

        command = f"python MLM_SFP.py --pretraining {self.pretraining_path} --data_embedding rna_motif_input.txt --embedding_output {embedding_output} --batch {self.batch_size}"
        os.system(command)

    def load_embeddings(self, embedding_file='embeddings_rnabert.txt'):
        self.embeddings = pd.read_table(embedding_file, names=['rna_motif_emb'])
        self.embeddings['rna_motif_emb'] = self.embeddings['rna_motif_emb'].apply(lambda x: np.array(ast.literal_eval(x)))
        return self.embeddings

    def encode(self, rna_motif_sequence):
        self.run_model(rna_motif_sequence)
        return self.load_embeddings()
