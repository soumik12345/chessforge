"""Tests for official titles, OTB/v3, ChessformerV2, DPO, and opt-in joint time."""

import math
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from chessforge import chess_utils as cu
from chessforge.cli.uci import ChessforgeUCIEngine
from chessforge.config import DataConfig, TrainConfig, load_train_config
from chessforge.data import pgn, pipeline, records
from chessforge.eval.evaluate import make_eval_step, run_eval
from chessforge.models import build_model, build_model_config, forward_model
from chessforge.models.joint_time import sample_joint_move_time
from chessforge.train.losses import maia3_loss
from chessforge.train.schedule import wsd_lr_schedule
from chessforge.train.trainer import Trainer


def test_official_titles_and_board_geometry():
    for t in ("GM", "WGM", "IM", "WIM", "FM", "WFM", "CM", "WCM", "NM", "WNM", "LM"):
        assert cu.is_official_title(t)
    assert not cu.is_official_title("BOT")
    assert not cu.is_official_title("")
    assert not cu.is_official_title(None)

    b = chess.Board("r3k2r/pppq1ppp/2npbn2/4p3/2B1P3/2NP1N2/PPPQ1PPP/R3K2R b KQkq - 4 8")
    assert cu.castling_mask(b) == 0b1111
    b_ep = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    assert cu.ep_file(b_ep) == 3  # d-file


def test_multiline_otb_pgn_and_v2_v3_compat(tmp_path: Path):
    otb_pgn = (
        '[Event "86th Tata Steel Masters"]\n'
        '[Site "Wijk aan Zee NED"]\n'
        '[Result "1-0"]\n'
        '[WhiteTitle "GM"]\n'
        '[BlackTitle "IM"]\n'
        '[WhiteElo "2740"]\n'
        '[BlackElo "2485"]\n\n'
        "1. d4 Nf6 2. c4 e6 3. Nf3 d5\n"
        "4. Nc3 Be7 (4... Bb4 5. cxd5) 5. Bf4 O-O 6. e3 c5 1-0\n"
    )
    games = list(pgn.iter_games(otb_pgn.splitlines(keepends=True)))
    assert len(games) == 1
    filt = pgn.GameFilter(
        game_type="", require_rated=False, require_clock=False, titled_mode="both"
    )
    assert filt(*games[0])
    recs = pgn.game_to_records(
        games[0][0],
        games[0][1],
        pgn.PositionConfig(positions_per_game=0, allow_no_clock=True),
    )
    assert len(recs) == 12
    dec, leg = records.decode_position(recs[0])
    assert dec["self_title"] == cu.TITLE_GRANDMASTER
    assert dec["oppo_title"] == cu.TITLE_INTL_MASTER
    assert dec["game_format"] == cu.FORMAT_OTB_CLASSICAL
    assert dec["has_clock"] == 0
    assert dec["next_move"] >= 0
    assert leg.size == 20

    # Verify legacy v2 binary record still decodes and upgrades cleanly
    v2 = np.zeros(1, records.position_dtype_v2(2))
    v2["self_elo"] = 2400
    v2["oppo_elo"] = 2350
    v2["n_legal"] = 1
    raw_v2 = bytes([2, 2]) + v2.tobytes() + np.asarray([5], np.int16).tobytes()
    dec2, leg2 = records.decode_position(raw_v2)
    assert dec2["self_elo"] == 2400
    assert dec2["sf_move"] == -1
    np.testing.assert_array_equal(leg2, [5])


def test_chessformer_v2_forward_grad_dpo_and_eval(tmp_path: Path):
    writer = pgn.ShardedWriter(str(tmp_path), "v3", records_per_shard=4)
    for i in range(4):
        writer.write(
            records.encode_position(
                boards=np.ones((4, 64), np.int8),
                n_hist=2,
                stm=i % 2,
                move=10 + i,
                self_elo=2550,
                oppo_elo=2450,
                value=2,
                ply=18,
                clk_left=120.0,
                opp_clk_left=95.0,
                ponder=4.5,
                prev_opp_ponder=3.0,
                base_time=180,
                increment=2,
                castling=0b0101,
                ep_file=4 if i == 0 else -1,
                rep_count=i % 2,
                self_title=cu.TITLE_GRANDMASTER,
                oppo_title=cu.TITLE_INTL_MASTER,
                game_format=cu.FORMAT_BLITZ,
                has_clock=1,
                next_move=20 + i,
                sf_move=30 + i,
                sf_cp_loss=15,
                legal=[10 + i, 30 + i, 40],
            )
        )
    writer.close()

    pattern = str(tmp_path / "*.array_record")
    dcfg = DataConfig(train_glob=pattern, global_batch_size=4, history_mask_prob=0.0)
    it, _ = pipeline.make_train_iterator(dcfg, history=4)
    batch = next(it)

    mcfg = build_model_config(
        "chessforge-v2-5m",
        {
            "history": 4,
            "num_blocks": 2,
            "dim_vit": 64,
            "head_hid_dim": 64,
            "num_heads": 4,
            "compute_dtype": "float32",
            "remat_blocks": True,
        },
    )
    model = build_model(mcfg, rngs=0)
    graphdef, params = nnx.split(model)

    def loss_fn(p):
        m = nnx.merge(graphdef, p)
        lm, lv, lp, aux = forward_model(
            m, batch, deterministic=False, rng=jax.random.PRNGKey(0), return_aux=True
        )
        return maia3_loss(
            lm,
            lv,
            lp,
            batch,
            aux=aux,
            ref_logits_move=jax.lax.stop_gradient(lm),
            value_coef=0.1,
            ponder_coef=0.05,
            reply_coef=0.15,
            dpo_coef=0.25,
        )

    (loss_val, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    assert np.isfinite(float(loss_val))
    assert "ponder_loss" in metrics and "reply_loss" in metrics and "dpo_loss" in metrics
    assert "joint_time_loss" not in metrics and "joint_nll" not in metrics
    grad_norms = [float(jnp.linalg.norm(g)) for g in jax.tree.leaves(grads)]
    assert any(gn > 0.0 for gn in grad_norms)

    # Verify Eval-AntiSF metrics populate cleanly without leaking joint-time keys
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    dshard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    estep = make_eval_step(graphdef, dshard, repl)
    res = run_eval(
        estep,
        jax.device_put(params, repl),
        (
            jax.device_put(b, dshard)
            for b in pipeline.eval_batches(pattern, history=4, global_batch_size=4)
        ),
    )
    assert res["overall"]["positions"] == 4
    assert res["overall"]["anti_sf_positions"] == 4
    assert "joint_time_weight" not in res["overall"]
    assert "joint_nll" not in res["overall"]
    assert "effort_given_move_nll" not in res["overall"]

    sched = wsd_lr_schedule(4e-4, 1e-6, warmup=10, total_steps=100, decay_frac=0.2)
    assert float(sched(50)) == 4e-4
    assert float(sched(100)) < 1e-5


def test_configs_and_tracking_backwards_compat():
    cfg_dir = Path(__file__).resolve().parents[1] / "configs"
    original_configs = [
        "ablate_5m_relative.yaml",
        "chessforge_master_28m.yaml",
        "chessforge_master_dpo.yaml",
        "maia3_5m.yaml",
        "maia3_79m.yaml",
    ]
    for name in original_configs:
        tcfg = load_train_config(
            str(cfg_dir / name), ["wandb_project=chessforge-test"]
        )
        mcfg = build_model_config(tcfg.model, tcfg.model_overrides)
        assert not mcfg.use_joint_move_time
        assert tcfg.joint_time_coef == 0.0
        assert tcfg.wandb_project == "chessforge-test"
        assert tcfg.trackio_project == ""

    jcfg = load_train_config(
        str(cfg_dir / "chessforge_joint_time_28m.yaml"),
        ["trackio_project=chessforge-ablation", "wandb_project=chessforge-test"],
    )
    jmcfg = build_model_config(jcfg.model, jcfg.model_overrides)
    assert jmcfg.use_joint_move_time
    assert jcfg.joint_time_coef == 1.0
    assert jcfg.trackio_project == "chessforge-ablation"
    assert jcfg.wandb_project == "chessforge-test"


def test_joint_move_time_ablation_isolated(tmp_path: Path):
    writer = pgn.ShardedWriter(str(tmp_path), "joint_v3", records_per_shard=4)
    for i in range(4):
        writer.write(
            records.encode_position(
                boards=np.ones((4, 64), np.int8),
                n_hist=2,
                stm=i % 2,
                move=10 + i,
                self_elo=2500,
                oppo_elo=2450,
                value=1,
                ply=14 + i,
                clk_left=120.0,
                opp_clk_left=110.0,
                ponder=0.5 + 10.0 * i,
                prev_opp_ponder=2.0,
                base_time=180,
                increment=2,
                castling=0b1111,
                ep_file=-1,
                rep_count=0,
                self_title=cu.TITLE_GRANDMASTER,
                oppo_title=cu.TITLE_GRANDMASTER,
                game_format=cu.FORMAT_BLITZ,
                has_clock=1,
                next_move=20 + i,
                legal=[10 + i, 30 + i, 40],
            )
        )
    writer.close()

    pattern = str(tmp_path / "*.array_record")
    dcfg = DataConfig(train_glob=pattern, global_batch_size=4, history_mask_prob=0.0)
    it, _ = pipeline.make_train_iterator(dcfg, history=4)
    batch = next(it)

    shared_overrides = {
        "history": 4,
        "num_blocks": 2,
        "dim_vit": 64,
        "head_hid_dim": 64,
        "num_heads": 4,
        "ponder_buckets": 4,
        "compute_dtype": "float32",
    }
    base_cfg = build_model_config("chessforge-v2-5m", shared_overrides)
    joint_cfg = build_model_config(
        "chessforge-v2-5m",
        {**shared_overrides, "use_joint_move_time": True, "joint_move_time_rank": 4},
    )
    base_model = build_model(base_cfg, rngs=0)
    joint_model = build_model(joint_cfg, rngs=0)
    base_def, base_params = nnx.split(base_model)
    joint_def, joint_params = nnx.split(joint_model)

    def _key_map(state):
        out = {}
        for path, val in jax.tree_util.tree_leaves_with_path(state):
            k = tuple(
                str(p.key if hasattr(p, "key") else getattr(p, "name", p)) for p in path
            )
            out[k] = np.asarray(val.value if hasattr(val, "value") else val)
        return out

    base_map = _key_map(base_params)
    joint_map = _key_map(joint_params)
    assert not any("effort_move_residual" in k for k in base_map)
    assert any("effort_move_residual" in k for k in joint_map)
    assert set(base_map) < set(joint_map)
    for k, arr in base_map.items():
        np.testing.assert_array_equal(arr, joint_map[k])

    # Zero-initialized effort gates match the baseline legal policy at step 0
    lm_base, lv_base, _ = forward_model(base_model, batch, deterministic=True)
    lm_joint, lv_joint, _, aux_joint = forward_model(
        joint_model, batch, deterministic=True, return_aux=True
    )
    expected_logp = jax.nn.log_softmax(
        jnp.where(batch["legal"], lm_base.astype(jnp.float32), -1e9), axis=-1
    )
    np.testing.assert_allclose(
        np.asarray(lm_joint), np.asarray(expected_logp), atol=1e-5, rtol=1e-5
    )
    np.testing.assert_allclose(
        np.asarray(lv_joint), np.asarray(lv_base), atol=1e-5, rtol=1e-5
    )

    def joint_loss_fn(p):
        m = nnx.merge(joint_def, p)
        lm, lv, lp, aux = forward_model(m, batch, deterministic=True, return_aux=True)
        return maia3_loss(
            lm, lv, lp, batch, aux=aux, value_coef=0.1, joint_time_coef=1.0
        )

    (loss_val, metrics), grads = jax.value_and_grad(joint_loss_fn, has_aux=True)(
        joint_params
    )
    assert np.isfinite(float(loss_val))
    assert "joint_time_loss" in metrics and "joint_nll" in metrics
    effort_grads = [
        float(jnp.linalg.norm(v))
        for p, v in jax.tree_util.tree_leaves_with_path(grads)
        if any(
            str(k.key if hasattr(k, "key") else getattr(k, "name", k))
            == "effort_move_residual"
            for k in p
        )
    ]
    assert any(gn > 0.0 for gn in effort_grads)

    # Eval emits joint metrics only for joint models and obeys Bayes decomposition
    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    dshard = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    eval_gen = lambda: (
        jax.device_put(b, dshard)
        for b in pipeline.eval_batches(pattern, history=4, global_batch_size=4)
    )
    base_res = run_eval(
        make_eval_step(base_def, dshard, repl),
        jax.device_put(base_params, repl),
        eval_gen(),
    )
    joint_res = run_eval(
        make_eval_step(joint_def, dshard, repl),
        jax.device_put(joint_params, repl),
        eval_gen(),
    )
    assert "joint_nll" not in base_res["overall"]
    assert joint_res["overall"]["joint_time_weight"] == 4.0
    move_nll = math.log(joint_res["overall"]["perplexity"])
    np.testing.assert_allclose(
        joint_res["overall"]["joint_nll"],
        move_nll + joint_res["overall"]["effort_given_move_nll"],
        atol=1e-5,
    )

    samples = sample_joint_move_time(
        jax.random.PRNGKey(1),
        aux_joint["ponder_logits"],
        aux_joint["conditional_move_logits"],
        batch["clk_left"],
        batch["legal"],
    )
    for i, mv in enumerate(np.asarray(samples["move"])):
        assert batch["legal"][i, int(mv)]

    with pytest.raises(ValueError, match="use_joint_move_time"):
        Trainer(
            TrainConfig(
                model="chessforge-v2-5m",
                model_overrides={"use_joint_move_time": False},
                joint_time_coef=1.0,
                workdir=str(tmp_path / "bad_joint"),
            )
        )

    # UCI batch construction adds legal masks only for joint models
    board = chess.Board()
    codes = [cu.board_codes(board)]
    for use_joint in (False, True):
        eng = ChessforgeUCIEngine.__new__(ChessforgeUCIEngine)
        eng.model_cfg = build_model_config(
            "chessforge-v2-5m", {"history": 4, "use_joint_move_time": use_joint}
        )
        eng.base_time = 180.0
        eng.inc_time = 2.0
        eng.game_format = cu.FORMAT_BLITZ
        uci_batch = eng._make_single_batch(
            board, codes, 0, 2500.0, 2500.0, 4, 4, 180.0, 180.0, 1.0
        )
        assert ("legal" in uci_batch) is use_joint