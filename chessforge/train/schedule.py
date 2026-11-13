import jax.numpy as jnp


def maia3_lr_schedule(peak: float, minimum: float, warmup: int, cycle: int):
    """Linear warmup then cyclic cosine annealing peak -> minimum every `cycle` steps."""

    def schedule(step):
        step = jnp.asarray(step, jnp.float32)
        warm = peak * jnp.minimum(1.0, (step + 1.0) / max(warmup, 1))
        t = jnp.maximum(step - warmup, 0.0)
        frac = jnp.mod(t, max(cycle, 1)) / max(cycle, 1)
        cos = minimum + 0.5 * (peak - minimum) * (1.0 + jnp.cos(jnp.pi * frac))
        return jnp.where(step < warmup, warm, cos)

    return schedule


def wsd_lr_schedule(
    peak: float,
    minimum: float,
    warmup: int,
    total_steps: int,
    decay_frac: float = 0.20,
):
    """Warmup -> Stable -> Cosine-Decay schedule (pairs with Stage-2 titled data annealing)."""
    decay_steps = max(1, int(round(total_steps * decay_frac)))
    decay_start = max(warmup, total_steps - decay_steps)

    def schedule(step):
        step = jnp.asarray(step, jnp.float32)
        warm = peak * jnp.minimum(1.0, (step + 1.0) / max(warmup, 1))
        prog = jnp.clip((step - decay_start) / decay_steps, 0.0, 1.0)
        decayed = minimum + 0.5 * (peak - minimum) * (1.0 + jnp.cos(jnp.pi * prog))
        return jnp.where(
            step < warmup, warm, jnp.where(step < decay_start, peak, decayed)
        )

    return schedule


def cosine_lr_schedule(peak: float, minimum: float, warmup: int, total_steps: int):
    span = max(1, total_steps - warmup)

    def schedule(step):
        step = jnp.asarray(step, jnp.float32)
        warm = peak * jnp.minimum(1.0, (step + 1.0) / max(warmup, 1))
        prog = jnp.clip((step - warmup) / span, 0.0, 1.0)
        cos = minimum + 0.5 * (peak - minimum) * (1.0 + jnp.cos(jnp.pi * prog))
        return jnp.where(step < warmup, warm, cos)

    return schedule


def build_lr_schedule(o):
    kind = getattr(o, "schedule", "cyclic_cosine").lower()
    if kind == "wsd":
        return wsd_lr_schedule(
            o.lr, o.min_lr, o.warmup_steps, o.total_steps, o.decay_frac
        )
    if kind == "cosine":
        return cosine_lr_schedule(o.lr, o.min_lr, o.warmup_steps, o.total_steps)
    if kind == "cyclic_cosine":
        return maia3_lr_schedule(o.lr, o.min_lr, o.warmup_steps, o.cosine_cycle_steps)
    raise ValueError(f"Unknown schedule {o.schedule!r}")
