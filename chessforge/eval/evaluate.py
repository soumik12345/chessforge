"""Move-matching accuracy (legal-masked argmax), perplexity, per-rating-bin breakdown."""
from __future__ import annotations

import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

NUM_RATING_BINS = 32  # round(game_elo / 100), clipped


def make_eval_step(graphdef: nnx.GraphDef, data_sharding, repl):
    def step(params: nnx.State, batch: dict):
        model = nnx.merge(graphdef, params)
        lm, lv, _ = model(
            batch["board"],
            batch["self_elo"],
            batch["oppo_elo"],
            time_info=batch.get("time_info"),
            deterministic=True,
        )
        w = batch["weight"]
        legal = batch["legal"] | (w[:, None] == 0)
        masked = jnp.where(legal, lm, -jnp.inf)
        logp = jax.nn.log_softmax(masked, axis=-1)
        nll = -jnp.take_along_axis(logp, batch["move"][:, None], axis=1)[:, 0]
        nll = jnp.where(w > 0, nll, 0.0)
        correct = (jnp.argmax(masked, -1) == batch["move"]).astype(jnp.float32) * w
        correct_unmasked = (jnp.argmax(lm, -1) == batch["move"]).astype(jnp.float32) * w
        vnll = -jnp.take_along_axis(jax.nn.log_softmax(lv, -1), batch["value"][:, None], axis=1)[:, 0] * w
        vacc = (jnp.argmax(lv, -1) == batch["value"]).astype(jnp.float32) * w
        bins = jnp.clip(jnp.round(batch["game_elo"] / 100.0).astype(jnp.int32), 0, NUM_RATING_BINS - 1)
        oh = jax.nn.one_hot(bins, NUM_RATING_BINS) * w[:, None]
        return {
            "n": w.sum(), "correct": correct.sum(), "correct_unmasked": correct_unmasked.sum(),
            "nll": nll.sum(), "value_nll": vnll.sum(), "value_correct": vacc.sum(),
            "bin_n": oh.sum(0), "bin_correct": (oh * correct[:, None]).sum(0),
            "bin_nll": (oh * nll[:, None]).sum(0),
        }

    return jax.jit(step, in_shardings=(repl, data_sharding), out_shardings=repl)


def summarize(s: dict) -> dict:
    n = max(float(s["n"]), 1.0)
    acc = float(s["correct"]) / n
    overall = {
        "positions": int(s["n"]),
        "move_acc": acc,
        "move_acc_ci95": 1.96 * math.sqrt(acc * (1 - acc) / n),
        "move_acc_unmasked": float(s["correct_unmasked"]) / n,
        "perplexity": math.exp(float(s["nll"]) / n),
        "value_nll": float(s["value_nll"]) / n,
        "value_acc": float(s["value_correct"]) / n,
    }
    per_bin = {}
    for b in range(NUM_RATING_BINS):
        nb = float(s["bin_n"][b])
        if nb > 0:
            per_bin[b * 100] = {"n": int(nb), "acc": float(s["bin_correct"][b]) / nb,
                                "perplexity": math.exp(float(s["bin_nll"][b]) / nb)}
    return {"overall": overall, "per_rating": per_bin}


def run_eval(step_fn, params: nnx.State, global_batches) -> dict:
    total = None
    for gb in global_batches:
        out = step_fn(params, gb)
        total = out if total is None else jax.tree.map(jnp.add, total, out)
    return summarize(jax.tree.map(np.asarray, jax.device_get(total)))