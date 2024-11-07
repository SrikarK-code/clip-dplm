# Triple Flow Model

An optimal transport-based flow model for aligning cell states, perturbation effects, and protein embeddings in biological spaces.

## Mathematical Framework

### 1. Conditional Flow Matching (CFM)

The core idea is to learn vector fields that transport probability distributions. Given source and target distributions p₀ and p₁, we learn a time-dependent vector field v_θ(x,t) such that:

```
dx/dt = v_θ(x,t)
x(0) ~ p₀
x(1) ~ p₁
```

The CFM objective is:
```math
L(θ) = E_{t~U(0,1), x₀~p₀, x₁~p₁}[‖v_θ(x_t,t) - u_t(x_t|x₀,x₁)‖²]
```
where:
- x_t = tx₁ + (1-t)x₀ + σε, ε ~ N(0,I)
- u_t is the target vector field

### 2. Optimal Transport (OT)

We use OT to find optimal pairings between spaces. For distributions μ and ν, we solve:
```math
W₂(μ,ν) = inf_{π ∈ Π(μ,ν)} ∫∫ ‖x-y‖² dπ(x,y)
```
where Π(μ,ν) is the set of joint distributions with marginals μ and ν.

### 3. Triple Space Alignment

We align three spaces:
- Cell state space (C)
- Perturbation effect space (P)
- Protein embedding space (E)

using pairwise flows:
```math
F_CP: C → P
F_CE: C → E
F_PE: P → E
```

## Component Architecture

### 1. Encoders

#### Cell State Encoder
- Input: Gene expression matrix X ∈ ℝⁿˣᵍ, DPT τ ∈ ℝⁿ, Graph G
- Architecture:
  - Gene Expression MLP
  - Pseudotime Embedding
  - PiGNN for graph structure
- Output: Cell embeddings h_c ∈ ℝⁿˣᵈ

#### Perturbation Encoder
- Input: Top k genes and their values
- Architecture:
  - ESM embeddings for genes
  - Cross-attention mechanism
- Output: Perturbation embeddings h_p ∈ ℝⁿˣᵈ

#### Protein Encoder
- Input: ESM2 protein embeddings 
- Architecture:
  - Multi-layer projection
- Output: Protein embeddings h_e ∈ ℝⁿˣᵈ

### 2. Flow Components

#### Optimal Transport Flow
```python
class OTFlow:
    def forward(self, source, target):
        # 1. Compute OT plan
        π = compute_ot_plan(source, target)
        
        # 2. Sample locations
        x_t = sample_path_location(source, target, t)
        
        # 3. Compute vector field
        v = predict_vector_field(x_t, t)
        
        return v, x_t, t
```

#### Vector Field Network
```python
class VectorField:
    def forward(self, x, t):
        # 1. Time embedding
        t_emb = self.time_encoder(t)
        
        # 2. Concatenate with position
        h = torch.cat([x, t_emb], dim=-1)
        
        # 3. Predict velocity
        v = self.mlp(h)
        
        return v
```

## Pipeline Workflow

1. **Data Preprocessing**
   ```
   Raw Data → Filtering → Normalization → HVG Selection → Graph Construction
   ```

2. **Embedding Generation**
   ```
   Sequences → ESM2 → Protein/Gene Embeddings
   ```

3. **Training Flow**
   ```
   Input Data → Encoders → OT Matching → Flow Prediction → Loss Computation
   ```

4. **Inference**
   ```
   Cell State → Cell Encoder → Flow Networks → Generated Perturbations/Proteins
   ```

## Loss Components

1. **Contrastive Loss**
```math
L_cont = -log(exp(sim(h₁,h₂)/τ) / Σᵢexp(sim(h₁,hᵢ)/τ))
```

2. **Flow Matching Loss**
```math
L_flow = E[‖v_θ(x_t,t) - u_t(x_t|x₀,x₁)‖²]
```

3. **Regularization**
```math
L_reg = λ₁E[‖v_θ‖²] + λ₂E[‖∇ₓv_θ‖²]
```
