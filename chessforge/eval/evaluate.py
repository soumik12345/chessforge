"""Move-matching accuracy, top-3, entropy, per-rating breakdown, and Anti-Engine (Eval-AntiSF)."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chessforge.models import forward_model

NUM_RATING_BINS = 32  # round(game_elo / 100), clipped


def make_eval_step(graphdef: nnx.GraphDef, data_sharding, repl, sf_max_cp_loss: float = 35.0):
    def step(params: nnx.State, batch: dict):
        model = nnx.merge(graphdef, params)
        lm, lv, _ = forward_model(model, batch, deterministic=True, return_aux=False)
        lm = lm.astype(jnp.float32)
        lv = lv.astype(jnp.float32)

        w = batch["weight"]
        legal = batch["legal"] | (w[:, None] == 0)
        masked = jnp.where(legal, lm, -jnp.inf)
        logp = jax.nn.log_softmax(masked, axis=-1)
        probs = jnp.where(legal, jnp.exp(logp), 0.0)
        entropy = -jnp.sum(jnp.where(probs > 0, probs * logp, 0.0), axis=-1) * w

        nll = -jnp.take_along_axis(logp, batch["move"][:, None], axis=1)[:, 0]
        nll = jnp.where(w > 0, nll, 0.0)
        pred_move = jnp.argmax(masked, -1)
        correct = (pred_move == batch["move"]).astype(jnp.float32) * w
        top3 = jax.lax.top_k(masked, k=3)[1]
        correct_top3 = (
            jnp.any(top3 == batch["move"][:, None], axis=-1).astype(jnp.float32) * w
        )
        correct_unmasked = (jnp.argmax(lm, -1) == batch["move"]).astype(jnp.float32) * w

        vnll = (
            -jnp.take_along_axis(
                jax.nn.log_softmax(lv, -1), batch["value"][:, None], axis=1
            )[:, 0]
            * w
        )
        vacc = (jnp.argmax(lv, -1) == batch["value"]).astype(jnp.float32) * w

        # Anti-Engine disagreement slice (where human move != Stockfish #1 and loses <= sf_max_cp_loss)
        sf_m = batch.get("sf_move", jnp.full_like(batch["move"], -1))
        sf_cp = batch.get("sf_cp_loss", jnp.full_like(w, -1.0))
        disagree = (
            (sf_m >= 0)
            & (sf_m != batch["move"])
            & ((sf_cp < 0) | (sf_cp <= sf_max_cp_loss))
            & (w > 0)
        )
        dw = jnp.where(disagree, w, 0.0)
        hum_logit = jnp.take_along_axis(masked, batch["move"][:, None], axis=1)[:, 0]
        sf_logit = jnp.take_along_axis(
            masked, jnp.maximum(sf_m, 0)[:, None], axis=1
        )[:, 0]
        hum_pref = jax.nn.sigmoid(jnp.where(disagree, hum_logit - sf_logit, 0.0)) * dw
        anti_sf_hum_pick = (pred_move == batch["move"]).astype(jnp.float32) * dw
        anti_sf_eng_pick = (pred_move == jnp.maximum(sf_m, 0)).astype(jnp.float32) * dw

        bins = jnp.clip(
            jnp.round(batch["game_elo"] / 100.0).astype(jnp.int32),
            0,
            NUM_RATING_BINS - 1,
        )
        oh = jax.nn.one_hot(bins, NUM_RATING_BINS) * w[:, None]
        return {
            "n": w.sum(),
            "correct": correct.sum(),
            "correct_top3": correct_top3.sum(),
            "correct_unmasked": correct_unmasked.sum(),
            "nll": nll.sum(),
            "entropy": entropy.sum(),
            "value_nll": vnll.sum(),
            "value_correct": vacc.sum(),
            "anti_sf_n": dw.sum(),
            "anti_sf_human_pick": anti_sf_hum_pick.sum(),
            "anti_sf_engine_pick": anti_sf_eng_pick.sum(),
            "anti_sf_human_pref_prob": hum_pref.sum(),
            "bin_n": oh.sum(0),
            "bin_correct": (oh * correct[:, None]).sum(0),
            "bin_nll": (oh * nll[:, None]).sum(0),
        }

    return jax.jit(step, in_shardings=(repl, data_sharding), out_shardings=repl)


def summarize(s: dict) -> dict:
    n = max(float(s["n"]), 1.0)
    acc = float(s["correct"]) / n
    overall = {
        "positions": int(s["n"]),
        "move_acc": acc,
        "move_acc_top3": float(s["correct_top3"]) / n,
        "move_acc_ci95": 1.96 * math.sqrt(acc * (1 - acc) / n),
        "move_acc_unmasked": float(s["correct_unmasked"]) / n,
        "perplexity": math.exp(float(s["nll"]) / n),
        "policy_entropy": float(s["entropy"]) / n,
        "value_nll": float(s["value_nll"]) / n,
        "value_acc": float(s["value_correct"]) / n,
    }
    anti_n = float(s.get("anti_sf_n", 0.0))
    if anti_n > 0:
        overall["anti_sf_positions"] = int(anti_n)
        overall["anti_sf_human_acc"] = float(s["anti_sf_human_pick"]) / anti_n
        overall["anti_sf_engine_pick_rate"] = float(s["anti_sf_engine_pick"]) / anti_n
        overall["anti_sf_human_pref_prob"] = (
            float(s["anti_sf_human_pref_prob"]) / anti_n
        )

    per_bin = {}
    for b in range(NUM_RATING_BINS):
        nb = float(s["bin_n"][b])
        if nb > 0:
            per_bin[b * 100] = {
                "n": int(nb),
                "acc": float(s["bin_correct"][b]) / nb,
                "perplexity": math.exp(float(s["bin_nll"][b]) / nb),
            }
    return {"overall": overall, "per_rating": per_bin}


def run_eval(step_fn, params: nnx.State, global_batches) -> dict:
    total = None
    for gb in global_batches:
        out = step_fn(params, gb)
        total = out if total is None else jax.tree.map(jnp.add, total, out)
    if total is None:
        raise RuntimeError("Evaluation dataset yielded 0 batches.")
    return summarize(jax.tree.map(np.asarray, jax.device_get(total)))