"""
Triple Flow Configuration Framework
=================================

Mathematical Foundations:
-----------------------
The triple flow framework operates in three fundamental spaces:
1. Cell State Space (C ⊂ ℝ^512) 
   - Equipped with pseudotime τ: C → [0,1]
   - Graph structure G = (V,E) where V ⊂ C

2. Perturbation Space (P ⊂ ℝ^512)
   - Sparse representation of differential effects
   - ESM embeddings of affected genes

3. Protein Space (E ⊂ ℝ^512)
   - ESM language model embeddings 
   - Preserves protein structural information

Transport System:
---------------
The system learns three transport maps:
T_CP: C → P  (Cell to Perturbation)
T_CE: C → E  (Cell to Protein)
T_PE: P → E  (Perturbation to Protein)

With consistency constraint:
T_CE ≈ T_PE ∘ T_CP

Biological Constraints:
---------------------
1. Smooth trajectories: ∫||∇_t c_t||² dt
2. Development hierarchy: Σ_{i<j} φ(c_i,c_j)
3. Cell plasticity: ψ(c_t, c_{t+δ})

Cost Functions:
-------------
ℒ_total = ℒ_transport + λ_bio ℒ_bio + λ_consistency ℒ_consistency

where:
- ℒ_transport: Wasserstein-2 cost
- ℒ_bio: Biological constraints
- ℒ_consistency: Map composition consistency
"""

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
import yaml
import logging
from enum import Enum
import threading
from datetime import datetime

# [Previous imports...]

class BiologicalDataType(Enum):
    """
    Biological data types with corresponding mathematical spaces
    
    Each type represents a specific biological modality with its own
    mathematical structure and constraints.
    """
    GENE_EXPRESSION = "gene_expression"  # ℝ^n with sparsity
    PROTEIN_SEQUENCE = "protein_sequence"  # ESM embedding space
    PERTURBATION = "perturbation"  # Sparse differential space
    PSEUDOTIME = "pseudotime"  # [0,1] ordering
    CONNECTIVITY = "connectivity"  # Graph adjacency

class BiologicalScale(Enum):
    """
    Biological scales with corresponding statistical properties
    
    Each scale requires different normalization and handling:
    - Single cell: High noise, sparse
    - Pseudo bulk: Averaged, reduced noise
    - Population: Aggregate statistics
    """
    SINGLE_CELL = "single_cell"
    PSEUDO_BULK = "pseudo_bulk"
    POPULATION = "population"
    TIME_SERIES = "time_series"

class CellTrajectoryType(Enum):
    """
    Cell trajectory topologies
    
    Mathematical structures:
    - LINEAR: Total order
    - BRANCHING: Tree structure
    - CYCLIC: Circle topology
    - BIFURCATION: Critical points
    """
    LINEAR = "linear"
    BRANCHING = "branching"
    CYCLIC = "cyclic"
    BIFURCATION = "bifurcation"

@dataclass
class BiologicalLossConfig:
    """
    Configuration for biological loss terms
    
    Loss = Σ w_i L_i where:
    - L_i: Individual loss components
    - w_i: Learnable weights
    """
    # Core biological losses
    expression_loss_weight: float = 1.0  # Gene expression fidelity
    protein_loss_weight: float = 1.0    # Protein structure preservation
    perturbation_loss_weight: float = 1.0  # Perturbation effect accuracy
    
    # Trajectory-specific losses
    pseudotime_weight: float = 1.0  # Temporal ordering
    branching_weight: float = 0.5   # Lineage preservation
    cycle_weight: float = 0.5       # Cycle stability
    
    # Quality control losses
    sparsity_penalty: float = 0.1   # L1 regularization
    dropout_weight: float = 0.2     # Dropout robustness
    library_size_weight: float = 0.3  # Size normalization
    
    # Scale-specific weights
    scale_weights: Dict[BiologicalScale, float] = field(
        default_factory=lambda: {
            BiologicalScale.SINGLE_CELL: 1.0,
            BiologicalScale.PSEUDO_BULK: 0.8,
            BiologicalScale.POPULATION: 0.6,
            BiologicalScale.TIME_SERIES: 0.9
        }
    )

@dataclass
class ICNNConfig:
    """
    Input Convex Neural Network Configuration
    
    Ensures ∇²Ψ(x) ≽ 0 through architecture:
    - Positive weights in hidden layers
    - Convex activation functions
    """
    input_dim: int
    hidden_dims: List[int]
    dropout: float = 0.1
    use_time: bool = True
    eps: float = 1e-6
    init_scale: float = 0.01
    gradient_clip: float = 1.0
    hessian_reg: float = 1e-4
    use_layer_norm: bool = True
    activation: str = 'celu'
    weight_decay: float = 1e-5
    sparse_mode: bool = False
    biological_activation: bool = True
    stable_gradient: bool = True



@dataclass
class OptimConfig:
    """
    Optimization configuration for biological transport
    
    Uses adaptive learning with warmup:
    lr(t) = min_lr + (max_lr - min_lr) * min(t/warmup_steps, 1)
    
    Gradient accumulation for effective batch size:
    B_eff = batch_size * accumulation_steps
    """
    learning_rate: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 100
    early_stopping_patience: int = 10
    gradient_clip_val: float = 1.0
    warmup_steps: int = 1000
    min_learning_rate: float = 1e-6
    scheduler: str = "cosine"  # Learning rate schedule
    optimizer: str = "adamw"   # Optimizer type
    loss_scaling: bool = True  # Automatic mixed precision
    gradient_accumulation_steps: int = 1
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size with accumulation"""
        return self.batch_size * self.gradient_accumulation_steps

@dataclass
class ESMConfig:
    """
    ESM Language Model Configuration
    
    ESM provides contextual embeddings:
    e(s) = ESM(s) ∈ ℝ^{L×d}
    
    where:
    - s: Amino acid sequence
    - L: Sequence length
    - d: Embedding dimension
    """
    model_name: str = "esm2_t33_650M_UR50D"
    esm_dim: int = 1280  # Base ESM embedding dimension
    protein_dim: int = 512  # Projected protein dimension
    gene_dim: int = 512    # Projected gene dimension
    num_attention_heads: int = 8
    dropout: float = 0.1
    use_sequence_context: bool = True
    max_sequence_length: int = 1024
    tokenizer_path: Optional[str] = None
    
    def validate_model(self):
        """Validate ESM model compatibility"""
        valid_models = {
            "esm2_t33_650M_UR50D",
            "esm2_t36_3B_UR50D",
            "esm2_t48_15B_UR50D"
        }
        if self.model_name not in valid_models:
            raise ValueError(f"Invalid ESM model: {self.model_name}")

@dataclass
class QualityControlConfig:
    """
    Quality control configuration for biological data
    
    Filters based on:
    1. Coverage: min_cells × min_genes matrix density
    2. Signal: noise ratio estimation
    3. Library complexity metrics
    """
    min_cells_per_gene: int = 10
    min_genes_per_cell: int = 200
    max_mt_percent: float = 20.0  # Maximum mitochondrial percentage
    min_library_size: int = 1000
    max_library_size: int = 1000000
    max_noise_ratio: float = 0.5
    
    def validate_data(self, data: torch.Tensor, data_type: BiologicalDataType) -> bool:
        """
        Validate biological data quality
        
        Returns True if data meets quality thresholds:
        - Sufficient coverage
        - Acceptable noise levels
        - Appropriate library size
        """
        if data_type == BiologicalDataType.GENE_EXPRESSION:
            return (
                (data.sum(1) >= self.min_genes_per_cell).all() and
                (data.sum(0) >= self.min_cells_per_gene).all()
            )
        return True

@dataclass
class MemoryConfig:
    """
    Memory management configuration
    
    Handles:
    1. Gradient checkpointing
    2. Memory-efficient backprop
    3. Automatic mixed precision
    """
    mode: str = "standard"  # or "low_memory", "checkpoint"
    gradient_checkpoint_layers: List[str] = field(default_factory=list)
    max_batch_memory_gb: float = 16.0
    pin_memory: bool = True
    empty_cache_freq: int = 1
    
    def validate(self):
        """Validate memory configuration"""
        if self.max_batch_memory_gb <= 0:
            raise ValueError("Max batch memory must be positive")
            
    def get_memory_status(self) -> Dict[str, float]:
        """Get current GPU memory status"""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1e9,
                'reserved': torch.cuda.memory_reserved() / 1e9,
                'max_allocated': torch.cuda.max_memory_allocated() / 1e9
            }
        return {}

@dataclass
class ExperimentConfig:
    """
    Experiment tracking configuration
    
    Tracks:
    1. Training metrics
    2. Validation performance
    3. Model checkpoints
    4. Hyperparameters
    """
    name: str
    base_dir: Path
    save_frequency: int = 10
    keep_last_k: int = 5
    log_level: str = "INFO"
    use_wandb: bool = False
    wandb_project: Optional[str] = None
    
    def __post_init__(self):
        self.exp_dir = self.base_dir / self.name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir = self.exp_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging"""
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.exp_dir / "experiment.log"),
                logging.StreamHandler()
            ]
        )

class TripleFlowConfig:
    """
    Master configuration for triple flow system
    
    Combines all sub-configurations and provides:
    1. Validation
    2. Persistence
    3. Runtime management
    """
    def __init__(
        self,
        icnn: ICNNConfig,
        biological: BiologicalLossConfig,
        esm: ESMConfig,
        optim: OptimConfig,
        memory: MemoryConfig,
        quality_control: QualityControlConfig,
        experiment: ExperimentConfig,
        seed: int = 42
    ):
        self.icnn = icnn
        self.biological = biological
        self.esm = esm
        self.optim = optim
        self.memory = memory
        self.quality_control = quality_control
        self.experiment = experiment
        self.seed = seed
        
        # Runtime state
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._setup()
        
    def _setup(self):
        """Complete setup procedure"""
        self._set_seed()
        self._validate_all()
        self._optimize_runtime()
        
    def _set_seed(self):
        """Set random seeds for reproducibility"""
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            
    def _validate_all(self):
        """Comprehensive validation"""
        self.esm.validate_model()
        self.memory.validate()
        self._validate_compatibility()
        
    def _validate_compatibility(self):
        """Validate component compatibility"""
        if self.optim.batch_size > 1 and not self.memory.pin_memory:
            logging.warning("Pin memory disabled with batch size > 1")
            
    def _optimize_runtime(self):
        """Optimize runtime settings"""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if self.memory.mode == "low_memory":
                torch.cuda.empty_cache()
                
    def save(self, path: Union[str, Path]):
        """Save configuration"""
        config_dict = {
            'icnn': self.icnn.__dict__,
            'biological': self.biological.__dict__,
            'esm': self.esm.__dict__,
            'optim': self.optim.__dict__,
            'memory': self.memory.__dict__,
            'quality_control': self.quality_control.__dict__,
            'experiment': {
                k: str(v) if isinstance(v, Path) else v 
                for k, v in self.experiment.__dict__.items()
            },
            'seed': self.seed
        }
        
        with open(path, 'w') as f:
            yaml.dump(config_dict, f)
            
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'TripleFlowConfig':
        """Load configuration"""
        with open(path) as f:
            config_dict = yaml.safe_load(f)
            
        return cls(
            icnn=ICNNConfig(**config_dict['icnn']),
            biological=BiologicalLossConfig(**config_dict['biological']),
            esm=ESMConfig(**config_dict['esm']),
            optim=OptimConfig(**config_dict['optim']),
            memory=MemoryConfig(**config_dict['memory']),
            quality_control=QualityControlConfig(**config_dict['quality_control']),
            experiment=ExperimentConfig(**config_dict['experiment']),
            seed=config_dict['seed']
        )

class MetricTracker:
    """
    Tracks training metrics and handles checkpointing
    
    Implements:
    1. Moving averages
    2. Best model selection
    3. Early stopping
    """
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.metrics: Dict[str, List[float]] = {}
        self.best_values: Dict[str, float] = {}
        self.patience_counters: Dict[str, int] = {}
        self._lock = threading.Lock()
        
    def update(
        self,
        metrics: Dict[str, float],
        epoch: int,
        save_checkpoint: bool = True
    ) -> bool:
        """
        Update metrics and check stopping criteria
        
        Returns: True if should stop training
        """
        with self._lock:
            # Update metrics
            for name, value in metrics.items():
                if name not in self.metrics:
                    self.metrics[name] = []
                self.metrics[name].append(value)
                
                # Update best values
                if name not in self.best_values or value < self.best_values[name]:
                    self.best_values[name] = value
                    self.patience_counters[name] = 0
                    if save_checkpoint:
                        self._save_checkpoint(epoch, name)
                else:
                    self.patience_counters[name] = self.patience_counters.get(name, 0) + 1
                    
            # Check early stopping
            return any(
                counter >= self.config.keep_last_k
                for counter in self.patience_counters.values()
            )
            
    def _save_checkpoint(self, epoch: int, metric_name: str):
        """Save checkpoint"""
        checkpoint_path = self.config.checkpoint_dir / f"epoch_{epoch}_{metric_name}.pt"
        torch.save({
            'epoch': epoch,
            'metrics': self.metrics,
            'best_values': self.best_values,
            'timestamp': datetime.now().isoformat()
        }, checkpoint_path)
        
        # Clean old checkpoints
        self._cleanup_checkpoints()
        
    def _cleanup_checkpoints(self):
        """Remove old checkpoints"""
        checkpoints = sorted(self.config.checkpoint_dir.glob("*.pt"))
        if len(checkpoints) > self.config.keep_last_k:
            for checkpoint in checkpoints[:-self.config.keep_last_k]:
                checkpoint.unlink()
                
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary"""
        return {
            'metrics': {k: np.mean(v[-5:]) for k, v in self.metrics.items()},
            'best_values': self.best_values,
            'patience_status': self.patience_counters
        }
