"""Probability operations for p(move, effort | board, pre-move context).

These helpers only consume predictions. Observed thinking time belongs in the
training loss, never in the inputs used to marginalize or sample the policy.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def joint_move_log_probs(
    ponder_logits: jax.Array,
    conditional_move_logits: jax.Array,
    legal: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Return log p(move) and log p(move | effort), with shapes (B,V), (B,K,V).

    Normalize over legal moves *within each effort bucket* before mixing with
    p(effort). Masking an already mixed policy changes the effort prior and is
    not equivalent. An all-false legal row falls back to all moves, solely to
    keep padded/terminal rows numerically defined; it is not a playable policy.
    """
    if conditional_move_logits.ndim != 3 or ponder_logits.ndim != 2:
        raise ValueError("Expected conditional logits (B,K,V) and ponder logits (B,K)")
    if conditional_move_logits.shape[:2] != ponder_logits.shape:
        raise ValueError("Conditional move and ponder logits must agree on (B,K)")
    if min(conditional_move_logits.shape[1:]) < 1:
        raise ValueError("At least one effort bucket and move are required")

    logits = conditional_move_logits.astype(jnp.float32)
    if legal is not None:
        if legal.shape != (logits.shape[0], logits.shape[-1]):
            raise ValueError("Legal mask must have shape (B,V)")
        mask = legal.astype(jnp.bool_)
        mask = mask | ~jnp.any(mask, axis=-1, keepdims=True)
        logits = jnp.where(mask[:, None, :], logits, -1e9)
    conditional_logp = jax.nn.log_softmax(logits, axis=-1)
    effort_logp = jax.nn.log_softmax(ponder_logits.astype(jnp.float32), axis=-1)
    marginal_logp = jax.scipy.special.logsumexp(
        conditional_logp + effort_logp[:, :, None], axis=1
    )
    return marginal_logp, conditional_logp


def sample_joint_move_time(
    rng: jax.Array,
    ponder_logits: jax.Array,
    conditional_move_logits: jax.Array,
    clk_left: jax.Array,
    legal: jax.Array | None = None,
) -> dict[str, jax.Array]:
    """Sample an effort bucket, then its move, from the predicted joint model.

    Supply finite positive pre-move clock seconds and at least one legal move
    for each playable row. Seconds use the sampled bucket's center, matching
    V2's normalized effort convention. Invalid clocks fall back to one second;
    returned times for such rows have no calibrated clock interpretation.
    """
    _, conditional_logp = joint_move_log_probs(
        ponder_logits, conditional_move_logits, legal
    )
    if clk_left.shape != (ponder_logits.shape[0],):
        raise ValueError("clk_left must have shape (B,)")
    effort_rng, move_rng = jax.random.split(rng)
    bucket = jax.random.categorical(effort_rng, ponder_logits.astype(jnp.float32))
    chosen_logp = conditional_logp[jnp.arange(bucket.shape[0]), bucket, :]
    move = jax.random.categorical(move_rng, chosen_logp)
    effort = (bucket.astype(jnp.float32) + 0.5) / ponder_logits.shape[-1]
    clean_clock = jnp.where(
        jnp.isfinite(clk_left) & (clk_left > 0), clk_left, 1.0
    ).astype(jnp.float32)
    seconds = jnp.expm1(effort * jnp.log1p(jnp.maximum(clean_clock, 1.0)))
    return {
        "move": move,
        "effort_bucket": bucket,
        "ponder_effort": effort,
        "ponder_sec": seconds,
    }
