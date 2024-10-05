import jax
import jax.numpy as jnp
import optax
from flax.training import train_state
from typing import Dict, Any
from clip import FlaxRNAProteinCLIP, FlaxDiffMapProteinCLIP
from classifiers import MLPClassifier, TransformerClassifier, LinearClassifier, SimpleNonLinearClassifier
from configuration_hybrid_clip import HybridCLIPConfig

def create_train_state(rng, config, model, learning_rate):
    params = model.init(rng, jnp.ones((1, config.max_position_embeddings)))
    tx = optax.adam(learning_rate)
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)

def train_step(state, batch, labels):
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, batch)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return state, loss, accuracy

def eval_step(state, batch, labels):
    logits = state.apply_fn({'params': state.params}, batch)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return loss, accuracy

def train_classifier(rng, config, model, train_data, train_labels, eval_data, eval_labels, num_epochs, batch_size):
    train_state = create_train_state(rng, config, model, learning_rate=1e-4)
    
    for epoch in range(num_epochs):
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i+batch_size]
            batch_labels = train_labels[i:i+batch_size]
            train_state, loss, accuracy = train_step(train_state, batch, batch_
