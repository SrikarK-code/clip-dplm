import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from clip import RNAProteinCLIP, DiffMapProteinCLIP
from classifiers import MLPClassifier, TransformerClassifier, LinearClassifier, SimpleNonLinearClassifier
from configuration_hybrid_clip import HybridCLIPConfig

def train_clip(clip_model, train_loader, optimizer, num_epochs, device):
    clip_model.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            rna, protein, _ = [t.to(device) for t in batch]
            optimizer.zero_grad()
            outputs = clip_model(rna, protein)
            loss = nn.CrossEntropyLoss()(outputs["logits_per_rna_protein"], torch.arange(rna.shape[0]).to(device))
            loss.backward()
            optimizer.step()

def train_classifier(clip_model, classifier, train_loader, optimizer, num_epochs, device):
    clip_model.eval()
    classifier.train()
    for epoch in range(num_epochs):
        for batch in train_loader:
            rna, protein, labels = [t.to(device) for t in batch]
            optimizer.zero_grad()
            with torch.no_grad():
                clip_outputs = clip_model(rna, protein)
            latent = torch.cat([clip_outputs["rna_embeds"], clip_outputs["protein_embeds"]], dim=-1)
            logits = classifier(latent)
            loss = nn.CrossEntropyLoss()(logits, labels)
            loss.backward()
            optimizer.step()

def evaluate(clip_model, classifier, test_loader, device):
    clip_model.eval()
    classifier.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            rna, protein, labels = [t.to(device) for t in batch]
            clip_outputs = clip_model(rna, protein)
            latent = torch.cat([clip_outputs["rna_embeds"], clip_outputs["protein_embeds"]], dim=-1)
            logits = classifier(latent)
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

def ablation_study(config, rna_data, protein_data, diffmap_data, labels, num_classes, device):
    clip_models = {
        'RNA-Protein CLIP': RNAProteinCLIP(config).to(device),
        'DiffMap-Protein CLIP': DiffMapProteinCLIP(config).to(device)
    }
    
    classifiers = {
        'MLP': MLPClassifier(config.projection_dim * 2, [256, 128], num_classes).to(device),
        'Transformer': TransformerClassifier(config.projection_dim * 2, 256, num_classes, num_layers=2, num_heads=8).to(device),
        'Linear': LinearClassifier(config.projection_dim * 2, num_classes).to(device),
        'SimpleNonLinear': SimpleNonLinearClassifier(config.projection_dim * 2, 256, num_classes).to(device)
    }
    
    results = {}
    
    for clip_name, clip_model in clip_models.items():
        clip_optimizer = optim.Adam(clip_model.parameters(), lr=1e-4)
        
        if clip_name == 'RNA-Protein CLIP':
            train_data = TensorDataset(rna_data, protein_data, labels)
        else:
            train_data = TensorDataset(diffmap_data, protein_data, labels)
        
        train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
        train_clip(clip_model, train_loader, clip_optimizer, num_epochs=10, device=device)
        
        for clf_name, classifier in classifiers.items():
            clf_optimizer = optim.Adam(classifier.parameters(), lr=1e-4)
            train_classifier(clip_model, classifier, train_loader, clf_optimizer, num_epochs=10, device=device)
            accuracy = evaluate(clip_model, classifier, DataLoader(train_data, batch_size=32), device)
            results[f"{clip_name} + {clf_name}"] = accuracy
    
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # example config
    config = HybridCLIPConfig(
        rna_config={"hidden_size": 768, "num_hidden_layers": 3, "layer_norm_eps": 1e-12},
        protein_config
