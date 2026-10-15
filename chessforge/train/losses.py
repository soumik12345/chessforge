import jax.numpy as jnp
import optax


def maia3_loss(
    logits_move,
    logits_value,
    ponder,
    batch,
    *,
    value_coef=0.1,
    ponder_coef=0.0,
    mask_illegal=False,
):
    w = batch["weight"]
    wsum = jnp.maximum(w.sum(), 1.0)
    logits = logits_move
    if mask_illegal and "legal" in batch:
        logits = jnp.where(batch["legal"], logits_move, -1e9)
    pol = optax.softmax_cross_entropy_with_integer_labels(logits, batch["move"])
    val = optax.softmax_cross_entropy_with_integer_labels(logits_value, batch["value"])
    loss_p = (pol * w).sum() / wsum
    loss_v = (val * w).sum() / wsum
    total = loss_p + value_coef * loss_v
    metrics = {"policy_loss": loss_p, "value_loss": loss_v}
    if ponder_coef > 0:
        tgt = batch["ponder"]
        ok = jnp.isfinite(tgt) & (w > 0)
        err = jnp.where(ok, (ponder - jnp.nan_to_num(tgt) / 100.0) ** 2, 0.0)
        loss_t = err.sum() / jnp.maximum(ok.sum(), 1)
        total = total + ponder_coef * loss_t
        metrics["ponder_loss"] = loss_t
    metrics["loss"] = total
    metrics["policy_acc"] = (
        (jnp.argmax(logits_move, -1) == batch["move"]) * w
    ).sum() / wsum
    if "legal" in batch:
        masked = jnp.where(batch["legal"], logits_move, -jnp.inf)
        metrics["policy_acc_legal"] = (
            (jnp.argmax(masked, -1) == batch["move"]) * w
        ).sum() / wsum
    metrics["value_acc"] = (
        (jnp.argmax(logits_value, -1) == batch["value"]) * w
    ).sum() / wsum
    return total, metrics
