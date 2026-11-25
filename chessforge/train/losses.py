from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax

from chessforge.models.joint_time import joint_move_log_probs


def hl_gauss_ponder_targets(
    num_buckets: int,
    ponder_sec: jax.Array,
    clk_left: jax.Array,
    has_clock: jax.Array,
    w: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Return smooth effort labels (B,K) and valid clock-label weights (B,)."""
    K = num_buckets
    if K < 1:
        raise ValueError("num_buckets must be positive")
    ok = (
        jnp.isfinite(ponder_sec)
        & (ponder_sec >= 0.0)
        & (has_clock > 0.5)
        & jnp.isfinite(clk_left)
        & jnp.isfinite(has_clock)
        & (clk_left > 0.5)
        & jnp.isfinite(w)
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
    return tgt_probs, jnp.where(ok, w, 0.0)


def hl_gauss_ponder_loss(
    ponder_logits: jax.Array,
    ponder_sec: jax.Array,
    clk_left: jax.Array,
    has_clock: jax.Array,
    w: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """HL-Gauss loss on effort u = log(1+ponder)/log(1+clk)."""
    tgt_probs, mask_w = hl_gauss_ponder_targets(
        ponder_logits.shape[-1], ponder_sec, clk_left, has_clock, w
    )

    logp = jax.nn.log_softmax(ponder_logits.astype(jnp.float32), axis=-1)
    nll = -(tgt_probs * logp).sum(axis=-1)
    return (nll * mask_w).sum() / jnp.maximum(mask_w.sum(), 1.0), mask_w.sum()


def joint_time_nll_sums(
    ponder_logits: jax.Array,
    conditional_move_logp: jax.Array,
    batch: dict,
    w: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Return weighted joint NLL, posterior effort NLL, and valid clock weight."""
    targets, time_w = hl_gauss_ponder_targets(
        ponder_logits.shape[-1],
        batch.get("ponder", jnp.full_like(w, jnp.nan)),
        batch.get("clk_left", jnp.zeros_like(w)),
        batch.get("has_clock", jnp.zeros_like(w)),
        w,
    )
    chosen_logp = jnp.take_along_axis(
        conditional_move_logp, batch["move"][:, None, None], axis=-1
    )[:, :, 0]
    joint_logp = chosen_logp + jax.nn.log_softmax(
        ponder_logits.astype(jnp.float32), axis=-1
    )
    posterior_logp = jax.nn.log_softmax(joint_logp, axis=-1)
    joint_nll = -(targets * joint_logp).sum(axis=-1)
    posterior_nll = -(targets * posterior_logp).sum(axis=-1)
    return (joint_nll * time_w).sum(), (posterior_nll * time_w).sum(), time_w.sum()


def maia3_loss(
    logits_move: jax.Array,
    logits_value: jax.Array,
    ponder: jax.Array,
    batch: dict,
    *,
    aux: dict | None = None,
    ref_logits_move: jax.Array | None = None,
    policy_coef: float = 1.0,
    value_coef: float = 0.1,
    ponder_coef: float = 0.0,
    joint_time_coef: float = 0.0,
    reply_coef: float = 0.0,
    label_smoothing: float = 0.0,
    dpo_coef: float = 0.0,
    dpo_beta: float = 0.2,
    kl_coef: float = 0.0,
    sf_max_cp_loss: float = 35.0,
    mask_illegal: bool = True,
):
    joint = aux is not None and "conditional_move_logits" in aux
    w = batch["weight"].astype(jnp.float32)
    if joint:
        w = jnp.where(jnp.isfinite(w) & (w > 0), w, 0.0)
    wsum = jnp.maximum(w.sum(), 1.0)
    logits_move_f32 = logits_move.astype(jnp.float32)
    logits_value_f32 = logits_value.astype(jnp.float32)

    legal = batch.get("legal") if mask_illegal else None
    if joint:
        if "ponder_logits" not in aux:
            raise ValueError("Joint move-time loss requires ponder_logits")
        logp, conditional_logp = joint_move_log_probs(
            aux["ponder_logits"], aux["conditional_move_logits"], legal
        )
    else:
        if joint_time_coef > 0:
            raise ValueError("joint_time_coef requires a joint move-time model")
        logits = (
            jnp.where(legal, logits_move_f32, -1e9)
            if legal is not None
            else logits_move_f32
        )
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
    total = policy_coef * loss_p + value_coef * loss_v
    metrics = {"policy_loss": loss_p, "value_loss": loss_v}

    if joint:
        joint_nll, posterior_nll, time_weight = joint_time_nll_sums(
            aux["ponder_logits"], conditional_logp, batch, w
        )
        # At policy_coef=1 and joint_time_coef=1, unsmoothed policy + posterior
        # losses sum to exact joint NLL on clocked rows and marginal move NLL otherwise.
        loss_joint_time = posterior_nll / wsum
        total = total + joint_time_coef * loss_joint_time
        metrics["joint_time_loss"] = loss_joint_time
        metrics["joint_nll_sum"] = joint_nll
        metrics["joint_nll"] = joint_nll / jnp.maximum(time_weight, 1.0)
        metrics["joint_time_weight"] = time_weight

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

    ref_logp = None
    if ref_logits_move is not None and (dpo_coef > 0 or kl_coef > 0):
        ref_l = ref_logits_move.astype(jnp.float32)
        if legal is not None:
            ref_mask = legal | ~jnp.any(legal, axis=-1, keepdims=True)
            ref_l = jnp.where(ref_mask, ref_l, -1e9)
        ref_logp = jax.nn.log_softmax(ref_l, axis=-1)

    if kl_coef > 0 and ref_logp is not None:
        if legal is not None:
            ref_mask = legal | ~jnp.any(legal, axis=-1, keepdims=True)
            ref_p = jnp.where(ref_mask, jnp.exp(ref_logp), 0.0)
        else:
            ref_p = jnp.exp(ref_logp)
        safe_diff = jnp.where(ref_p > 0.0, ref_logp - logp, 0.0)
        kl_elem = jnp.sum(ref_p * safe_diff, axis=-1)
        loss_kl = (kl_elem * w).sum() / wsum
        total = total + kl_coef * loss_kl
        metrics["kl_loss"] = loss_kl

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

        if ref_logp is not None:
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
    if joint:
        logits_move_f32, _ = joint_move_log_probs(
            aux["ponder_logits"], aux["conditional_move_logits"]
        )
    metrics["policy_acc"] = (
        (jnp.argmax(logits_move_f32, -1) == batch["move"]) * w
    ).sum() / wsum
    if "legal" in batch:
        if joint:
            masked = logp
            if legal is None:
                masked, _ = joint_move_log_probs(
                    aux["ponder_logits"], aux["conditional_move_logits"], batch["legal"]
                )
            masked = jnp.where(batch["legal"], masked, -jnp.inf)
        else:
            masked = jnp.where(batch["legal"], logits_move_f32, -jnp.inf)
        metrics["policy_acc_legal"] = (
            (jnp.argmax(masked, -1) == batch["move"]) * w
        ).sum() / wsum
    metrics["value_acc"] = (
        (jnp.argmax(logits_value_f32, -1) == batch["value"]) * w
    ).sum() / wsum
    return total, metrics