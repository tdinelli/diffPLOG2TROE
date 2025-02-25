from typing import Any, Callable, Dict, Optional, Union

import equinox as eqx
import jax.numpy as jnp
import optax
from jax import jit
from jaxtyping import Array, Float64


class LossType(eqx.Enumeration):
    """Enumeration of available loss functions for rate constant fitting."""

    MSE = "mse"  # Mean squared error
    MAE = "mae"  # Mean absolute error
    HUBER = "huber"  # Huber loss (robust to outliers)
    LOG_MSE = "log_mse"  # MSE in log space
    RELATIVE = "relative"  # Relative error


class Loss(eqx.Module):
    """A unified interface for loss functions."""

    type: LossType
    huber_delta: Optional[Float64] = 1.0

    def __init__(self, type: LossType = LossType.MSE, huber_delta: float = 1.0):
        """
        Initialize the loss function.

        Args:
            type: Type of loss function to use
            huber_delta: Delta parameter for Huber loss (if applicable)
        """
        self.type = type
        self.huber_delta = huber_delta

    @classmethod
    def mse(cls) -> "Loss":
        """Create a mean squared error loss."""
        return cls(LossType.MSE)

    @classmethod
    def mae(cls) -> "Loss":
        """Create a mean absolute error loss."""
        return cls(LossType.MAE)

    @classmethod
    def huber(cls, delta: float = 1.0) -> "Loss":
        """Create a Huber loss with the specified delta parameter."""
        return cls(LossType.HUBER, huber_delta=delta)

    @classmethod
    def log_mse(cls) -> "Loss":
        """Create a mean squared error loss in log space."""
        return cls(LossType.LOG_MSE)

    @classmethod
    def relative(cls) -> "Loss":
        """Create a relative error loss."""
        return cls(LossType.RELATIVE)

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "Loss":
        """Create a Loss instance from a configuration dictionary."""
        loss_type = config.get("type", "mse")
        huber_delta = config.get("huber_delta", 1.0)
        return cls(type=loss_type, huber_delta=huber_delta)

    def __call__(self, k_model: Array, k_ref: Array) -> Float64:
        """Compute the loss between model and reference values."""
        return compute_loss(k_model, k_ref, self)


def get_optax_loss_fn(loss: Loss) -> Callable:
    """Get the appropriate Optax loss function based on the loss type."""
    if loss.type == LossType.MSE:
        return optax.l2_loss
    elif loss.type == LossType.HUBER:
        return lambda preds, targets: optax.huber_loss(preds, targets, delta=loss.huber_delta)
    elif loss.type == LossType.LOG_MSE:
        return lambda preds, targets: optax.l2_loss(jnp.log(preds + 1e-30), jnp.log(targets + 1e-30))
    elif loss.type == LossType.RELATIVE:
        return lambda preds, targets: jnp.mean(((preds - targets) / (targets + 1e-30)) ** 2)
    else:
        return optax.l2_loss  # Default to MSE


def compute_loss(k_model: Array, k_ref: Array, loss: Loss) -> Float64:
    """Compute loss between model and reference rate constants."""
    k_model_flat = k_model.flatten()
    k_ref_flat = k_ref.flatten()

    loss_fn = get_optax_loss_fn(loss)

    if loss.type in [LossType.MSE, LossType.LOG_MSE]:
        per_example_loss = loss_fn(k_model_flat, k_ref_flat)
        return jnp.sum(per_example_loss)
    elif isinstance(loss_fn(k_model_flat[:1], k_ref_flat[:1]), jnp.ndarray):
        per_example_loss = loss_fn(k_model_flat, k_ref_flat)
        return jnp.sum(per_example_loss)
    else:
        return loss_fn(k_model_flat, k_ref_flat)


compute_loss_jit = jit(compute_loss, static_argnames=("loss",))
