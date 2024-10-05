import jax
import jax.numpy as jnp
import flax.linen as nn

class MLPClassifier(nn.Module):
    hidden_dims: list
    output_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = False):
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        return nn.Dense(self.output_dim)(x)

class TransformerClassifier(nn.Module):
    num_layers: int
    num_heads: int
    hidden_dim: int
    output_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Dense(self.hidden_dim)(x)
        for _ in range(self.num_layers):
            y = nn.LayerNorm()(x)
            y = nn.MultiHeadDotProductAttention(num_heads=self.num_heads)(y, y)
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
            x = x + y
            y = nn.LayerNorm()(x)
            y = nn.Dense(self.hidden_dim * 4)(y)
            y = nn.relu(y)
            y = nn.Dense(self.hidden_dim)(y)
            y = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(y)
            x = x + y
        x = nn.LayerNorm()(x)
        return nn.Dense(self.output_dim)(x)

class LinearClassifier(nn.Module):
    output_dim: int

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.output_dim)(x)

class SimpleNonLinearClassifier(nn.Module):
    hidden_dim: int
    output_dim: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, train: bool = False):
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        x = nn.LayerNorm()(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        return nn.Dense(self.output_dim)(x)
