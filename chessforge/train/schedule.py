import jax.numpy as jnp


def maia3_lr_schedule(peak: float, minimum: float, warmup: int, cycle: int):
    """Linear warmup then cyclic cosine annealing peak -> minimum every `cycle` steps."""
    def schedule(step):
        step = jnp.asarray(step, jnp.float32)
        warm = peak * jnp.minimum(1.0, (step + 1.0) / max(warmup, 1))
        t = jnp.maximum(step - warmup, 0.0)
        frac = jnp.mod(t, cycle) / cycle
        cos = minimum + 0.5 * (peak - minimum) * (1.0 + jnp.cos(jnp.pi * frac))
        return jnp.where(step < warmup, warm, cos)
    return schedule