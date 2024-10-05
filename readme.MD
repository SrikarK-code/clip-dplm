# RNA-Protein and DiffMap-Protein CLIP Models

This project implements CLIP (Contrastive Language-Image Pre-training) inspired models for RNA-Protein and DiffMap-Protein interactions.

## Overview

1. **CLIP Models**: We have two main CLIP models:
   - RNA-Protein CLIP
   - DiffMap-Protein CLIP

2. **How CLIP Works**:
   - Encodes RNA/DiffMap and Protein inputs separately
   - Projects encoded features into a shared space
   - Learns to match correct pairs through contrastive learning

3. **Classifiers**: After training CLIP, we use it to extract features for classification:
   - MLP Classifier
   - Transformer Classifier
   - Linear Classifier
   - Simple Non-Linear Classifier

4. **Ablation Study**: We test different combinations of CLIP models and classifiers to find the best performing setup.

## Project Structure

- `configuration_hybrid_clip.py`: Configuration for CLIP models
- `clip.py`: Implementation of CLIP models
- `classifiers.py`: Different classifier architectures
- `ablation.py`: Code for training, evaluation, and ablation studies

## How to Use

1. Prepare your RNA, Protein, and DiffMap data
2. Set up the configuration in `ablation.py`
3. Run `python ablation.py`

## Detailed Explanation

1. **CLIP Training**:
   - Input: RNA/DiffMap and Protein pairs
   - Encode inputs using separate encoders
   - Project encoded features to a shared space
   - Compute similarity between all pairs in a batch
   - Train the model to give high similarity to correct pairs and low similarity to incorrect pairs

2. **Feature Extraction**:
   - After training, use CLIP to encode RNA/DiffMap and Protein inputs
   - Concatenate the encoded features

3. **Classification**:
   - Use the concatenated features as input to various classifiers
   - Train classifiers to predict desired outputs

4. **Ablation Study**:
   - Train both CLIP models (RNA-Protein and DiffMap-Protein)
   - For each CLIP model, train all classifier types
   - Evaluate performance of each combination
