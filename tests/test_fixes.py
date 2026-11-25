"""Tests for exact 1-pass FastShardIterator, KL-anchored DPO, frozen value head, and MI eval."""

from pathlib import Path

import chess
import jax
import numpy as np
import pytest
from flax import nnx

from chessforge import chess_utils as cu
from chessforge.config import DataConfig, OptimConfig, TrainConfig
from chessforge.data import pgn, pipeline, records
from chessforge.eval.evaluate import summarize
from chessforge.train.trainer import Trainer


@pytest.mark.parametrize("buffer_shards", [2, 4])
@pytest.mark.parametrize("batch_size", [1, 7, 32])
def test_fast_shard_iterator_exact_one_pass_zero_duplicates(
    tmp_path: Path, buffer_shards, batch_size
):
    uid = 0
    for shard, size in enumerate([1, 4, 2, 3]):
        writer = pgn.ShardedWriter(str(tmp_path), f"sh{shard}", records_per_shard=size)
        for _ in range(size):
            writer.write(
                records.encode_position(
                    boards=np.zeros((2, 64), np.int8),
                    n_hist=1,
                    stm=0,
                    move=uid,
                    self_elo=2000,
                    oppo_elo=2000,
                    value=1,
                    ply=uid,
                    legal=[uid],
                )
            )
            uid += 1
        writer.close()

    files = sorted(str(p) for p in tmp_path.glob("*.array_record"))
    it = pipeline.FastShardIterator(
        files=files,
        batch_size=batch_size,
        history=2,
        seed=0,
        buffer_shards=buffer_shards,
    )
    seen = []
    while len(seen) < 3 * uid:
        moves = next(it)["move"].tolist()
        assert len(moves) == batch_size
        seen.extend(moves)
    for start in range(0, 3 * uid, uid):
        assert sorted(seen[start : start + uid]) == list(range(uid))

    state = it.get_state()
    expected = [next(it)["move"] for _ in range(3)]
    it.set_state(state)
    for batch in expected:
        np.testing.assert_array_equal(next(it)["move"], batch)


def test_kl_anchored_dpo_freeze_value_head_and_mi_eval(tmp_path: Path):
    base_opts = {
        "history": 2,
        "num_blocks": 1,
        "dim_vit": 32,
        "head_hid_dim": 32,
        "num_heads": 2,
        "compute_dtype": "float32",
        "gab_per_square_dim": 4,
        "gab_gen_size": 8,
        "gab_intermediate_dim": 16,
        "ponder_buckets": 4,
        "use_joint_move_time": True,
        "joint_move_time_rank": 4,
    }
    cfg = TrainConfig(
        model="chessforge-v2-5m",
        model_overrides=base_opts,
        data=DataConfig(global_batch_size=4),
        optim=OptimConfig(lr=1e-3, weight_decay=1e-2, grad_accum_steps=1),
        policy_coef=0.25,
        value_coef=0.10,
        joint_time_coef=0.25,
        kl_coef=0.50,
        dpo_coef=2.0,
        dpo_beta=0.25,
        freeze_value_head=True,
        workdir=str(tmp_path),
    )
    board = chess.Board()
    legal = np.zeros((4, cu.VOCAB_SIZE), dtype=bool)
    legal[:, cu.legal_move_indices(board)] = True
    e4, d4 = cu.move_index("e2e4"), cu.move_index("d2d4")
    batch = {
        "board": np.tile(cu.board_codes(board), (4, 2, 1)),
        "legal": legal,
        "move": np.array([e4, d4, e4, d4], np.int32),
        "sf_move": np.array([d4, e4, d4, e4], np.int32),
        "sf_cp_loss": np.array([10.0, 15.0, 20.0, 5.0], np.float32),
        "self_elo": np.full(4, 2500.0, np.float32),
        "oppo_elo": np.full(4, 2500.0, np.float32),
        "game_elo": np.full(4, 2500.0, np.float32),
        "value": np.array([0, 1, 2, 1], np.int32),
        "ponder": np.array([1.0, 8.0, 2.0, 25.0], np.float32),
        "clk_left": np.full(4, 180.0, np.float32),
        "has_clock": np.ones(4, np.float32),
        "weight": np.ones(4, np.float32),
    }

    trainer = Trainer(cfg)
    try:
        rng = jax.random.PRNGKey(0)
        state = trainer.p_init(rng)
        # Re-compile with ref_params active (simulating init_from_checkpoint)
        trainer.cfg.init_from_checkpoint = "dummy_ref"
        trainer._build_fns()
        ref_params = jax.tree.map(lambda x: x.copy(), state.params)
        gb = trainer.to_global(batch)

        before_model = nnx.merge(trainer.graphdef, state.params)
        v_before = np.asarray(before_model.fc_value.kernel[...]).copy()
        vh_before = np.asarray(before_model.fc_value_hid.kernel[...]).copy()
        p_before = np.asarray(before_model.policy_head.proj_from.kernel[...]).copy()

        updated, m = trainer.p_train_step(state, ref_params, gb, rng)
        assert np.isfinite(float(m["loss"]))
        assert np.isfinite(float(m["kl_loss"]))
        assert float(m["kl_loss"]) < 1e-5  # params == ref_params at step 0
        assert np.isfinite(float(m["dpo_loss"]))

        after_model = nnx.merge(trainer.graphdef, updated.params)
        # Value head must be strictly unchanged (zero grad + zero weight decay)
        np.testing.assert_array_equal(np.asarray(after_model.fc_value.kernel[...]), v_before)
        np.testing.assert_array_equal(np.asarray(after_model.fc_value_hid.kernel[...]), vh_before)
        # Policy head must have updated
        assert not np.allclose(np.asarray(after_model.policy_head.proj_from.kernel[...]), p_before)

        ev = summarize(trainer.eval_step(updated.params, gb))["overall"]
        assert "marginal_effort_nll" in ev
        assert "ponder_effort_mae" in ev
        assert "ponder_effort_r" in ev
        assert "effort_move_mi_nats" in ev
        assert np.isfinite(ev["effort_move_mi_nats"])
    finally:
        trainer.ckpt.close()
