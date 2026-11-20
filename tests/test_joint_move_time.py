from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chessforge import chess_utils as cu
from chessforge.cli.uci import ChessforgeUCIEngine
from chessforge.config import DataConfig, OptimConfig, TrainConfig, load_train_config
from chessforge.eval.evaluate import summarize
from chessforge.models import build_model, build_model_config, forward_model
from chessforge.models.joint_time import joint_move_log_probs, sample_joint_move_time
from chessforge.train.losses import maia3_loss
from chessforge.train.trainer import Trainer, summarize_train_metrics


def test_masked_joint_policy_and_sampling():
    effort_logits = jnp.log(jnp.array([[0.25, 0.75]]))
    move_logits = jnp.log(jnp.array([[[1.0, 3.0, 900.0], [4.0, 1.0, 1.0]]]))
    legal = jnp.array([[True, True, False]])
    marginal, conditional = joint_move_log_probs(effort_logits, move_logits, legal)
    np.testing.assert_allclose(
        jnp.exp(marginal), [[0.6625, 0.3375, 0.0]], rtol=1e-6
    )
    np.testing.assert_allclose(jnp.exp(conditional).sum(-1), 1.0, rtol=1e-6)

    rng = jax.random.PRNGKey(0)
    clock = jnp.array([60.0])
    samples = sample_joint_move_time(rng, effort_logits, move_logits, clock, legal)
    assert int(samples["move"][0]) in (0, 1)
    center = (int(samples["effort_bucket"][0]) + 0.5) / 2
    np.testing.assert_allclose(samples["ponder_sec"], 61.0**center - 1, rtol=1e-6)

    empty, _ = joint_move_log_probs(effort_logits, move_logits, jnp.zeros_like(legal))
    assert bool(jnp.isfinite(empty).all())
    np.testing.assert_allclose(jnp.exp(empty).sum(-1), 1.0, rtol=1e-6)


def test_joint_training_with_clockless_and_padded_positions(tmp_path):
    options = {
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
    }
    preset = "chessforge-v2-5m"
    baseline = build_model(build_model_config(preset, options), rngs=0)
    cfg = TrainConfig(
        model=preset,
        model_overrides={
            **options,
            "use_joint_move_time": True,
            "joint_move_time_rank": 4,
        },
        data=DataConfig(global_batch_size=4),
        optim=OptimConfig(grad_accum_steps=2),
        joint_time_coef=1.0,
        value_coef=0.0,
        workdir=str(tmp_path),
    )
    board = chess.Board()
    legal = np.zeros((4, cu.VOCAB_SIZE), dtype=bool)
    legal[:3, cu.legal_move_indices(board)] = True
    batch = {
        "board": np.tile(cu.board_codes(board), (4, 2, 1)),
        "legal": legal,
        "move": np.array([cu.move_index(m) for m in ("e2e4", "e2e4", "d2d4", "a1a1")]),
        "self_elo": np.full(4, 1800.0, np.float32),
        "oppo_elo": np.full(4, 1800.0, np.float32),
        "game_elo": np.full(4, 1800.0, np.float32),
        "value": np.ones(4, np.int32),
        "ponder": np.array([2.0, np.nan, np.nan, np.nan], np.float32),
        "clk_left": np.array([60.0, 0.0, 0.0, 0.0], np.float32),
        "has_clock": np.array([1.0, 0.0, 0.0, 0.0], np.float32),
        "weight": np.array([1.0, 1.0, 1.0, 0.0], np.float32),
    }
    lm_base, lv_base, _, aux_base = forward_model(baseline, batch, return_aux=True)
    assert "conditional_move_logits" not in aux_base

    trainer = Trainer(cfg)
    try:
        rng = jax.random.PRNGKey(0)
        state = trainer.p_init(rng)
        model = build_model(trainer.model_cfg, rngs=0)
        lm, lv, lp, aux = forward_model(model, batch, return_aux=True)
        expected = jax.nn.log_softmax(jnp.where(legal[:3], lm_base[:3], -1e9), -1)
        np.testing.assert_allclose(lm[:3], expected, atol=1e-6)
        np.testing.assert_array_equal(lv, lv_base)
        changed_labels = {**batch, "move": batch["move"][::-1], "ponder": np.ones(4)}
        np.testing.assert_array_equal(forward_model(model, changed_labels)[0], lm)

        loss, metrics = maia3_loss(
            lm, lv, lp, batch, aux=aux, value_coef=0.0, joint_time_coef=1.0
        )
        clockless_nll = -lm[1, batch["move"][1]] - lm[2, batch["move"][2]]
        np.testing.assert_allclose(
            loss * 3, metrics["joint_nll_sum"] + clockless_nll, rtol=1e-6
        )

        global_batch = trainer.to_global(batch)
        updated, train_metrics = trainer.p_train_step(state, None, global_batch, rng)
        assert float(train_metrics["joint_time_weight"]) == 1.0
        assert np.isfinite(float(train_metrics["loss"]))
        summary = summarize_train_metrics(jax.device_get(train_metrics), 1)
        np.testing.assert_allclose(summary["joint_nll"], train_metrics["joint_nll_sum"])
        updated_model = nnx.merge(trainer.graphdef, updated.params)
        assert np.any(np.asarray(updated_model.effort_move_residual.effort_gates[...]))

        evaluation = trainer.eval_step(updated.params, global_batch)
        overall = summarize(evaluation)["overall"]
        assert overall["positions"] == 3
        assert overall["joint_time_weight"] == 1.0
        assert overall["joint_nll"] > overall["effort_given_move_nll"] > 0
    finally:
        trainer.ckpt.close()


def test_default_configs_and_original_models_unaffected(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    for name in (
        "maia3_5m.yaml",
        "maia3_79m.yaml",
        "ablate_5m_relative.yaml",
        "chessforge_master_28m.yaml",
        "chessforge_master_dpo.yaml",
    ):
        tcfg = load_train_config(str(repo_root / "configs" / name))
        mcfg = build_model_config(tcfg.model, tcfg.model_overrides)
        assert mcfg.use_joint_move_time is False
        assert tcfg.joint_time_coef == 0.0

    master_cfg = load_train_config(
        str(repo_root / "configs" / "chessforge_master_28m.yaml")
    )
    joint_cfg = load_train_config(
        str(repo_root / "configs" / "chessforge_joint_time_28m.yaml")
    )
    joint_mcfg = build_model_config(joint_cfg.model, joint_cfg.model_overrides)
    assert joint_mcfg.use_joint_move_time is True
    assert joint_cfg.joint_time_coef == 1.0
    assert joint_cfg.data.anneal_glob == master_cfg.data.anneal_glob
    assert joint_cfg.data.anneal_start_frac == master_cfg.data.anneal_start_frac

    with pytest.raises(ValueError, match="only supported for chessformer_v2"):
        Trainer(
            TrainConfig(
                model="maia3-5m",
                model_overrides={"use_joint_move_time": True},
                data=DataConfig(global_batch_size=2),
                workdir=str(tmp_path / "invalid_v1"),
            )
        )

    with pytest.raises(ValueError, match="use_joint_move_time=true"):
        Trainer(
            TrainConfig(
                model="chessforge-v2-5m",
                data=DataConfig(global_batch_size=2),
                joint_time_coef=1.0,
                workdir=str(tmp_path / "invalid_v2"),
            )
        )

    board = chess.Board()
    legal = np.zeros((2, cu.VOCAB_SIZE), dtype=bool)
    legal[:, cu.legal_move_indices(board)] = True
    batch = {
        "board": np.tile(cu.board_codes(board), (2, 2, 1)),
        "legal": legal,
        "move": np.array([cu.move_index("e2e4"), cu.move_index("d2d4")], np.int32),
        "self_elo": np.full(2, 2400.0, np.float32),
        "oppo_elo": np.full(2, 2400.0, np.float32),
        "game_elo": np.full(2, 2400.0, np.float32),
        "value": np.ones(2, np.int32),
        "ponder": np.array([2.0, 5.0], np.float32),
        "clk_left": np.array([180.0, 180.0], np.float32),
        "has_clock": np.ones(2, np.float32),
        "weight": np.ones(2, np.float32),
    }

    for preset, overrides in (
        (
            "maia3-5m",
            {
                "history": 2,
                "num_blocks": 1,
                "dim_vit": 32,
                "head_hid_dim": 32,
                "num_heads": 2,
                "compute_dtype": "float32",
            },
        ),
        (
            "chessforge-v2-5m",
            {
                "history": 2,
                "num_blocks": 1,
                "dim_vit": 32,
                "head_hid_dim": 32,
                "num_heads": 2,
                "compute_dtype": "float32",
                "gab_per_square_dim": 4,
                "gab_gen_size": 8,
                "gab_intermediate_dim": 16,
            },
        ),
    ):
        trainer = Trainer(
            TrainConfig(
                model=preset,
                model_overrides=overrides,
                data=DataConfig(global_batch_size=2),
                workdir=str(tmp_path / preset),
            )
        )
        try:
            rng = jax.random.PRNGKey(0)
            state = trainer.p_init(rng)
            model = nnx.merge(trainer.graphdef, state.params)
            assert not hasattr(model, "effort_move_residual")
            gb = trainer.to_global(batch)
            _, train_metrics = trainer.p_train_step(state, None, gb, rng)
            train_summary = summarize_train_metrics(jax.device_get(train_metrics), 1)
            assert not any(k.startswith("joint_") for k in train_summary)
            eval_summary = summarize(trainer.eval_step(state.params, gb))["overall"]
            assert not any(
                k.startswith("joint_") or k == "effort_given_move_nll"
                for k in eval_summary
            )
        finally:
            trainer.ckpt.close()


def test_uci_engine_isolation_for_original_and_joint_models():
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
    }
    board = chess.Board()
    checkmate_board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    codes_long = [cu.board_codes(board)] * 4  # intentionally > history (2)

    for preset, overrides, expect_legal in (
        ("maia3-5m", {k: base_opts[k] for k in ("history", "num_blocks", "dim_vit", "head_hid_dim", "num_heads", "compute_dtype")}, False),
        ("chessforge-v2-5m", base_opts, False),
        ("chessforge-v2-5m", {**base_opts, "use_joint_move_time": True, "joint_move_time_rank": 4}, True),
    ):
        eng = object.__new__(ChessforgeUCIEngine)
        eng.model_cfg = build_model_config(preset, overrides)
        eng.base_time = 180.0
        eng.inc_time = 2.0
        eng.game_format = 0

        batch = eng._make_single_batch(
            board, codes_long, 0, 2500.0, 2500.0, 4, 4, 120.0, 120.0, 1.0
        )
        assert batch["board"].shape == (1, 2, 64)
        assert ("legal" in batch) is expect_legal

        mate_batch = eng._make_single_batch(
            checkmate_board, codes_long, 0, 2500.0, 2500.0, 4, 4, 120.0, 120.0, 1.0
        )
        model = build_model(eng.model_cfg, rngs=0)
        lm, lv, lp, _ = forward_model(model, mate_batch, deterministic=True, return_aux=True)
        assert bool(jnp.isfinite(lm).all())
        assert bool(jnp.isfinite(lv).all())
        assert bool(jnp.isfinite(lp).all())