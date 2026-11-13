"""Fit a tiny joint model to artificial move/time pairs; no dataset required.

Run: python -m chessforge.cli.demo_joint_time --steps 100
This is a mechanics demonstration, not a human-move accuracy benchmark.
"""

from __future__ import annotations

import argparse
import json

import chess
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from chessforge import chess_utils as cu
from chessforge.models import build_model, build_model_config, forward_model
from chessforge.models.joint_time import joint_move_log_probs, sample_joint_move_time
from chessforge.train.losses import maia3_loss


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.steps < 1:
        ap.error("--steps must be positive")

    cfg = build_model_config(
        "chessforge-v2-5m",
        {
            "dim_vit": 32,
            "head_hid_dim": 32,
            "history": 2,
            "num_blocks": 1,
            "num_heads": 2,
            "compute_dtype": "float32",
            "gab_per_square_dim": 4,
            "gab_gen_size": 8,
            "gab_intermediate_dim": 16,
            "use_joint_move_time": True,
            "joint_move_time_rank": 4,
            "ponder_buckets": 4,
            "use_opponent_reply_head": False,
        },
    )
    graphdef, params = nnx.split(build_model(cfg, rngs=args.seed))
    board = chess.Board()
    moves = [cu.move_index("e2e4"), cu.move_index("d2d4")]
    legal = np.zeros((2, cu.VOCAB_SIZE), bool)
    legal[:, cu.legal_move_indices(board).astype(int)] = True
    # Identical pre-move inputs: only training labels differ. The artificial
    # association is e4/0.1 seconds vs d4/50 seconds on a 60-second clock.
    batch = {
        "board": jnp.asarray(np.tile(cu.board_codes(board), (2, 2, 1))),
        "self_elo": jnp.full(2, 1800.0),
        "oppo_elo": jnp.full(2, 1800.0),
        "clk_left": jnp.full(2, 60.0),
        "has_clock": jnp.ones(2),
        "move": jnp.asarray(moves),
        "ponder": jnp.array([0.1, 50.0]),
        "value": jnp.ones(2, jnp.int32),
        "weight": jnp.ones(2),
        "legal": jnp.asarray(legal),
    }

    def objective(p):
        lm, lv, lp, aux = forward_model(nnx.merge(graphdef, p), batch, return_aux=True)
        return maia3_loss(
            lm,
            lv,
            lp,
            batch,
            aux=aux,
            value_coef=0.0,
            joint_time_coef=1.0,
        )[0]

    tx = optax.adam(3e-3)
    opt_state = tx.init(params)

    @jax.jit
    def step(p, opt_state):
        loss, grad = jax.value_and_grad(objective)(p)
        updates, opt_state = tx.update(grad, opt_state, p)
        return optax.apply_updates(p, updates), opt_state, loss

    before = float(objective(params))
    for _ in range(args.steps):
        params, opt_state, _ = step(params, opt_state)
    after = float(objective(params))
    lm, _, _, aux = forward_model(nnx.merge(graphdef, params), batch, return_aux=True)
    _, conditional = joint_move_log_probs(
        aux["ponder_logits"], aux["conditional_move_logits"], batch["legal"]
    )
    samples = sample_joint_move_time(
        jax.random.PRNGKey(args.seed + 1),
        jnp.repeat(aux["ponder_logits"][:1], 8, axis=0),
        jnp.repeat(aux["conditional_move_logits"][:1], 8, axis=0),
        jnp.full(8, 60.0),
        jnp.repeat(batch["legal"][:1], 8, axis=0),
    )
    print(
        json.dumps(
            {
                "demo": "synthetic associations only; no human accuracy claim",
                "steps": args.steps,
                "joint_nll_before": before,
                "joint_nll_after": after,
                "marginal_move_probabilities": {
                    cu.index_to_uci(m): float(jnp.exp(lm[0, m])) for m in moves
                },
                "conditional_move_probabilities": [
                    {
                        "effort_bucket": k,
                        **{
                            cu.index_to_uci(m): float(jnp.exp(conditional[0, k, m]))
                            for m in moves
                        },
                    }
                    for k in range(cfg.ponder_buckets)
                ],
                "joint_samples": [
                    {
                        "move": cu.index_to_uci(int(m)),
                        "effort_bucket": int(k),
                        "seconds": round(float(t), 3),
                    }
                    for m, k, t in zip(
                        samples["move"], samples["effort_bucket"], samples["ponder_sec"]
                    )
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
