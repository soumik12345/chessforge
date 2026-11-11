from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax


def hl_gauss_ponder_loss(
    ponder_logits: jax.Array,
    ponder_sec: jax.Array,
    clk_left: jax.Array,
    has_clock: jax.Array,
    w: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """HL-Gauss distributional loss on normalized human effort u = log(1+ponder)/log(1+clk)."""
    K = ponder_logits.shape[-1]
    ok = (
        jnp.isfinite(ponder_sec)
        & (ponder_sec >= 0.0)
        & (has_clock > 0.5)
        & (clk_left > 0.5)
        & (w > 0)
    )
    clean_p = jnp.where(ok, ponder_sec, 0.0)
    clean_c = jnp.where(ok, jnp.maximum(clk_left, 1.0), 1.0)
    u = jnp.clip(jnp.log1p(clean_p) / jnp.log1p(clean_c), 0.0, 1.0)[:, None]

    edges = jnp.linspace(0.0, 1.0, K + 1, dtype=jnp.float32)[None, :]
    sigma = 0.75 / K
    cdf = 0.5 * (1.0 + jax.lax.erf((edges - u) / (sigma * math.sqrt(2.0))))
    tgt_probs = cdf[:, 1:] - cdf[:, :-1]
    tgt_probs = tgt_probs / jnp.maximum(tgt_probs.sum(axis=-1, keepdims=True), 1e-8)

    logp = jax.nn.log_softmax(ponder_logits.astype(jnp.float32), axis=-1)
    nll = -(tgt_probs * logp).sum(axis=-1)
    mask_w = jnp.where(ok, w, 0.0)
    return (nll * mask_w).sum() / jnp.maximum(mask_w.sum(), 1.0), mask_w.sum()


def maia3_loss(
    logits_move: jax.Array,
    logits_value: jax.Array,
    ponder: jax.Array,
    batch: dict,
    *,
    aux: dict | None = None,
    ref_logits_move: jax.Array | None = None,
    value_coef: float = 0.1,
    ponder_coef: float = 0.0,
    reply_coef: float = 0.0,
    label_smoothing: float = 0.0,
    dpo_coef: float = 0.0,
    dpo_beta: float = 0.2,
    sf_max_cp_loss: float = 35.0,
    mask_illegal: bool = True,
):
    w = batch["weight"].astype(jnp.float32)
    wsum = jnp.maximum(w.sum(), 1.0)
    logits_move_f32 = logits_move.astype(jnp.float32)
    logits_value_f32 = logits_value.astype(jnp.float32)

    logits = logits_move_f32
    if mask_illegal and "legal" in batch:
        logits = jnp.where(batch["legal"], logits_move_f32, -1e9)

    logp = jax.nn.log_softmax(logits, axis=-1)
    nll_move = -jnp.take_along_axis(logp, batch["move"][:, None], axis=1)[:, 0]
    if label_smoothing > 0.0 and "legal" in batch:
        leg_f = batch["legal"].astype(jnp.float32)
        n_leg = jnp.maximum(leg_f.sum(axis=-1), 1.0)
        smooth_nll = -(logp * leg_f).sum(axis=-1) / n_leg
        pol = (1.0 - label_smoothing) * nll_move + label_smoothing * smooth_nll
    else:
        pol = nll_move

    val = optax.softmax_cross_entropy_with_integer_labels(
        logits_value_f32, batch["value"]
    )
    loss_p = (pol * w).sum() / wsum
    loss_v = (val * w).sum() / wsum
    total = loss_p + value_coef * loss_v
    metrics = {"policy_loss": loss_p, "value_loss": loss_v}

    if ponder_coef > 0:
        if aux is not None and "ponder_logits" in aux:
            has_clk = batch.get("has_clock", jnp.ones_like(w))
            clk_l = batch.get("clk_left", jnp.full_like(w, 60.0))
            loss_t, _ = hl_gauss_ponder_loss(
                aux["ponder_logits"],
                batch["ponder"].astype(jnp.float32),
                clk_l.astype(jnp.float32),
                has_clk.astype(jnp.float32),
                w,
            )
        else:
            tgt = batch["ponder"].astype(jnp.float32)
            ok = jnp.isfinite(tgt) & (w > 0)
            err = jnp.where(
                ok, (ponder.astype(jnp.float32) - jnp.nan_to_num(tgt) / 100.0) ** 2, 0.0
            )
            loss_t = err.sum() / jnp.maximum(ok.sum(), 1.0)
        total = total + ponder_coef * loss_t
        metrics["ponder_loss"] = loss_t

    if (
        reply_coef > 0
        and aux is not None
        and "reply_logits" in aux
        and "next_move" in batch
    ):
        nxt = batch["next_move"]
        ok_r = (nxt >= 0) & (w > 0)
        rw = jnp.where(ok_r, w, 0.0)
        r_ce = optax.softmax_cross_entropy_with_integer_labels(
            aux["reply_logits"].astype(jnp.float32), jnp.maximum(nxt, 0)
        )
        loss_r = (r_ce * rw).sum() / jnp.maximum(rw.sum(), 1.0)
        total = total + reply_coef * loss_r
        metrics["reply_loss"] = loss_r

    if dpo_coef > 0 and "sf_move" in batch:
        sf_m = batch["sf_move"]
        sf_cp = batch.get("sf_cp_loss", jnp.zeros_like(w))
        valid_dpo = (
            (sf_m >= 0)
            & (sf_m != batch["move"])
            & ((sf_cp < 0) | (sf_cp <= sf_max_cp_loss))
            & (w > 0)
        )
        dw = jnp.where(valid_dpo, w, 0.0)
        lp_hum = jnp.take_along_axis(logp, batch["move"][:, None], axis=1)[:, 0]
        lp_sf = jnp.take_along_axis(logp, jnp.maximum(sf_m, 0)[:, None], axis=1)[:, 0]
        margin = lp_hum - lp_sf

        if ref_logits_move is not None:
            ref_l = ref_logits_move.astype(jnp.float32)
            if mask_illegal and "legal" in batch:
                ref_l = jnp.where(batch["legal"], ref_l, -1e9)
            ref_logp = jax.nn.log_softmax(ref_l, axis=-1)
            rlp_hum = jnp.take_along_axis(ref_logp, batch["move"][:, None], axis=1)[:, 0]
            rlp_sf = jnp.take_along_axis(
                ref_logp, jnp.maximum(sf_m, 0)[:, None], axis=1
            )[:, 0]
            margin = margin - (rlp_hum - rlp_sf)
        else:
            margin = margin - 0.5  # SimPO-style reference-free margin fallback

        dpo_elem = -jax.nn.log_sigmoid(dpo_beta * margin)
        loss_dpo = (dpo_elem * dw).sum() / jnp.maximum(dw.sum(), 1.0)
        total = total + dpo_coef * loss_dpo
        metrics["dpo_loss"] = loss_dpo

    metrics["loss"] = total
    metrics["policy_acc"] = (
        (jnp.argmax(logits_move_f32, -1) == batch["move"]) * w
    ).sum() / wsum
    if "legal" in batch:
        masked = jnp.where(batch["legal"], logits_move_f32, -jnp.inf)
        metrics["policy_acc_legal"] = (
            (jnp.argmax(masked, -1) == batch["move"]) * w
        ).sum() / wsum
    metrics["value_acc"] = (
        (jnp.argmax(logits_value_f32, -1) == batch["value"]) * w
    ).sum() / wsum
    return total, metrics