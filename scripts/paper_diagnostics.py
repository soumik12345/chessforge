"""Comprehensive statistical & qualitative diagnostics for Tier-1 paper submission.

Computes:
  1. Exact ArrayRecord shard & position counts across Pool B, Pool C (Online vs OTB), DPO, and Eval.
  2. Exact paired McNemar's tests (chi2, two-sided p-value, wins/losses) on identical held-out
     positions comparing ChessForge models vs MAIA-3-5M, 23M, and 79M.
  3. 5-fold cross-validated affine+temperature recalibration of MAIA-3 value heads on FIDE OTB.
  4. High-mutual-information qualitative Grandmaster case studies from JointTime models
     (reconstructed FEN, observed move & think time, fast-effort vs slow-effort policy shift).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from pathlib import Path

import chess
import jax
import jax.numpy as jnp
import numpy as np
import optax
from array_record.python.array_record_module import ArrayRecordReader
from flax import nnx

from chessforge import chess_utils as cu
from chessforge.cli.annotate_anti_engine import _reconstruct_board
from chessforge.config import load_train_config
from chessforge.data import pipeline, records
from chessforge.distributed import init_distributed
from chessforge.models import ChessformerV2, build_model, forward_model
from chessforge.models.convert_torch import load_released
from chessforge.models.joint_time import joint_move_log_probs
from chessforge.train.trainer import Trainer


def count_shards_and_records(pattern: str) -> dict[str, int]:
    files = sorted(glob.glob(pattern))
    total = 0
    for f in files:
        r = ArrayRecordReader(f)
        try:
            total += int(r.num_records())
        finally:
            r.close()
    return {"shards": len(files), "positions": total}


def audit_corpus(data_root: str) -> dict:
    root = Path(data_root)
    audit: dict = {}

    # Pool B breakdown
    pool_b_dir = root / "mix_1800plus_titled_otb"
    if pool_b_dir.exists():
        b_train = count_shards_and_records(str(pool_b_dir / "*" / "train" / "*.array_record"))
        b_val = count_shards_and_records(str(pool_b_dir / "*" / "val" / "*.array_record"))
        audit["pool_b"] = {"train": b_train, "val": b_val}

    # Pool C breakdown per subdir (separates Pure Titled Online vs FIDE OTB accurately!)
    pool_c_dir = root / "pure_titled_and_otb"
    if pool_c_dir.exists():
        sub_breakdown = {}
        tot_tr_sh, tot_tr_pos, tot_va_pos = 0, 0, 0
        otb_pos, online_pos = 0, 0
        for sub in sorted(p for p in pool_c_dir.iterdir() if p.is_dir()):
            tr = count_shards_and_records(str(sub / "train" / "*.array_record"))
            va = count_shards_and_records(str(sub / "val" / "*.array_record"))
            if tr["shards"] == 0 and va["shards"] == 0:
                continue
            sub_breakdown[sub.name] = {"train": tr, "val": va}
            tot_tr_sh += tr["shards"]
            tot_tr_pos += tr["positions"]
            tot_va_pos += va["positions"]
            lname = sub.name.lower()
            if any(k in lname for k in ("otb", "fide", "twic", "lumbras", "classical")):
                otb_pos += tr["positions"]
            else:
                online_pos += tr["positions"]
        audit["pool_c"] = {
            "train": {"shards": tot_tr_sh, "positions": tot_tr_pos},
            "val": {"positions": tot_va_pos},
            "otb_train_positions": otb_pos,
            "online_titled_train_positions": online_pos,
            "otb_fraction": otb_pos / max(tot_tr_pos, 1),
            "sub_count": len(sub_breakdown),
        }

    for name, pat in [
        ("dpo_titled_antisf_train", "dpo_titled_antisf/train/*.array_record"),
        ("dpo_titled_antisf_val", "dpo_titled_antisf/val/*.array_record"),
        ("dpo_mixed_antisf_train", "dpo_mixed_antisf/train/*.array_record"),
        ("dpo_mixed_antisf_val", "dpo_mixed_antisf/val/*.array_record"),
        ("eval_titled_antisf", "eval_titled_antisf/*.array_record"),
        ("eval_fide_otb_antisf", "eval_fide_otb_antisf/*.array_record"),
        ("allie_test", "allie_test/*.array_record"),
    ]:
        res = count_shards_and_records(str(root / pat))
        if res["shards"] > 0:
            audit[name] = res
    return audit


def collect_per_position_preds(
    graphdef, params, mesh, data_sharding, repl, history: int, pattern: str, batch_size: int = 1024
) -> dict[str, np.ndarray]:
    @jax.jit
    def _pred_step(p, batch):
        m = nnx.merge(graphdef, p)
        lm, lv, _ = forward_model(m, batch, deterministic=True, return_aux=False)[:3]
        lm = lm.astype(jnp.float32)
        lv = lv.astype(jnp.float32)
        w = batch["weight"]
        legal = batch["legal"] | (w[:, None] == 0)
        masked = jnp.where(legal, lm, -jnp.inf)
        pred = jnp.argmax(masked, axis=-1)
        correct = pred == batch["move"]
        return correct, lv, batch["value"], w > 0

    correct_all, lv_all, val_all = [], [], []
    for b in pipeline.eval_batches(pattern, history, batch_size):
        gb = jax.tree.map(
            lambda x: jax.make_array_from_process_local_data(data_sharding, x), b
        )
        c, lv, v, valid = _pred_step(params, gb)
        c_np, lv_np, v_np, ok_np = jax.device_get((c, lv, v, valid))
        correct_all.append(np.asarray(c_np)[ok_np])
        lv_all.append(np.asarray(lv_np, np.float32)[ok_np])
        val_all.append(np.asarray(v_np, np.int32)[ok_np])

    return {
        "correct": np.concatenate(correct_all, axis=0),
        "value_logits": np.concatenate(lv_all, axis=0),
        "value": np.concatenate(val_all, axis=0),
    }


def mcnemar_test(a_correct: np.ndarray, b_correct: np.ndarray) -> dict:
    assert a_correct.shape == b_correct.shape
    n10 = int(np.sum(a_correct & ~b_correct))  # A right, B wrong
    n01 = int(np.sum(~a_correct & b_correct))  # A wrong, B right
    denom = n10 + n01
    if denom == 0:
        return {"n10": 0, "n01": 0, "delta_acc": 0.0, "chi2": 0.0, "p_value": 1.0}
    chi2 = ((abs(n10 - n01) - 1.0) ** 2) / denom if abs(n10 - n01) >= 1 else 0.0
    p_val = float(math.erfc(math.sqrt(max(chi2, 0.0) / 2.0)))
    return {
        "n10": n10,
        "n01": n01,
        "delta_acc": float(a_correct.mean() - b_correct.mean()),
        "chi2": float(chi2),
        "p_value": p_val,
    }


def cv_recalibrate_value_logits(
    logits: np.ndarray, labels: np.ndarray, n_folds: int = 5
) -> dict:
    """5-fold CV temperature + 3-class affine bias recalibration for OTB value head."""
    N = len(labels)
    idx = np.arange(N)
    raw_logp = logits - np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1, keepdims=True)) - logits.max(1, keepdims=True)
    # Exact log_softmax in numpy
    lmax = logits.max(axis=1, keepdims=True)
    lse = lmax + np.log(np.exp(logits - lmax).sum(axis=1, keepdims=True))
    raw_nll = float(-(logits[np.arange(N), labels] - lse[:, 0]).mean())
    raw_acc = float((logits.argmax(axis=1) == labels).mean())

    cal_nlls, cal_accs = [], []
    tx = optax.adam(0.05)

    @jax.jit
    def _fit_fold(tr_l, tr_y):
        params = {"log_t": jnp.zeros(()), "bias": jnp.zeros((3,))}
        opt_state = tx.init(params)

        def loss_fn(p):
            scaled = tr_l * jnp.exp(-p["log_t"]) + p["bias"][None, :]
            return optax.softmax_cross_entropy_with_integer_labels(scaled, tr_y).mean()

        def body(carry, _):
            p, os_ = carry
            _, g = jax.value_and_grad(loss_fn)(p)
            u, os_ = tx.update(g, os_, p)
            return (optax.apply_updates(p, u), os_), None

        (params, _), _ = jax.lax.scan(body, (params, opt_state), None, length=400)
        return params

    for f in range(n_folds):
        val_mask = (idx % n_folds) == f
        tr_l, tr_y = jnp.asarray(logits[~val_mask]), jnp.asarray(labels[~val_mask])
        va_l, va_y = logits[val_mask], labels[val_mask]
        p = _fit_fold(tr_l, tr_y)
        inv_t = float(jnp.exp(-p["log_t"]))
        bias = np.asarray(p["bias"], np.float32)
        scaled = va_l * inv_t + bias[None, :]
        sm = scaled.max(axis=1, keepdims=True)
        slse = sm + np.log(np.exp(scaled - sm).sum(axis=1, keepdims=True))
        cal_nlls.append(float(-(scaled[np.arange(len(va_y)), va_y] - slse[:, 0]).mean()))
        cal_accs.append(float((scaled.argmax(axis=1) == va_y).mean()))

    return {
        "raw_value_nll": raw_nll,
        "raw_value_acc": raw_acc,
        "cv5_recalibrated_value_nll": float(np.mean(cal_nlls)),
        "cv5_recalibrated_value_acc": float(np.mean(cal_accs)),
    }


def extract_joint_time_case_studies(
    cfg_path: str,
    overrides: list[str],
    step: int | None,
    eval_glob: str,
    max_scan: int = 8192,
    top_k: int = 15,
) -> list[dict]:
    """Find high-MI GM positions where fast vs slow effort buckets flip the top predicted move."""
    tcfg = load_train_config(cfg_path, overrides)
    trainer = Trainer(tcfg)
    if not (
        isinstance(build_model(trainer.model_cfg, rngs=0), ChessformerV2)
        and trainer.model_cfg.use_joint_move_time
    ):
        return []
    params = trainer.restore_params(step)
    graphdef = trainer.graphdef

    @jax.jit
    def _infer_joint(p, batch):
        m = nnx.merge(graphdef, p)
        _, _, _, aux = forward_model(m, batch, deterministic=True, return_aux=True)
        _, cond_logp = joint_move_log_probs(
            aux["ponder_logits"], aux["conditional_move_logits"], batch["legal"]
        )
        eff_p = jax.nn.softmax(aux["ponder_logits"].astype(jnp.float32), axis=-1)
        return eff_p, cond_logp

    files = sorted(glob.glob(eval_glob))
    feat = pipeline.FeaturizeTransform(trainer.model_cfg.history, 0.0, with_legal=True)
    rng = np.random.default_rng(0)

    raw_meta, feat_rows = [], []
    for f in files:
        r = ArrayRecordReader(f)
        try:
            for raw in r.read_all():
                rec, _ = records.decode_position(raw)
                if int(rec["has_clock"]) == 0 or not np.isfinite(float(rec["ponder"])):
                    continue
                if int(rec["self_elo"]) < 2400 or int(rec["ply"]) < 14:
                    continue
                raw_meta.append(rec)
                feat_rows.append(feat.random_map(raw, rng))
                if len(feat_rows) >= max_scan:
                    break
        finally:
            r.close()
        if len(feat_rows) >= max_scan:
            break

    if not feat_rows:
        return []

    tier_names = {0: "Untitled", 1: "CM/NM", 2: "FM", 3: "IM", 4: "GM"}
    candidates = []
    bs = 256
    for i0 in range(0, len(feat_rows), bs):
        chunk = feat_rows[i0 : i0 + bs]
        meta_chunk = raw_meta[i0 : i0 + bs]
        batch = {k: np.stack([row[k] for row in chunk], axis=0) for k in chunk[0]}
        eff_p_j, cond_lp_j = _infer_joint(params, jax.device_put(batch))
        eff_p = np.asarray(eff_p_j)
        cond_p = np.exp(np.asarray(cond_lp_j))  # (B, K, V)

        for j, rec in enumerate(meta_chunk):
            K = cond_p.shape[1]
            # Fast buckets (bottom 20%) vs Slow buckets (top 25%)
            fast_idx = slice(0, max(1, K // 5))
            slow_idx = slice(max(1, (3 * K) // 4), K)
            w_fast = eff_p[j, fast_idx]
            w_slow = eff_p[j, slow_idx]
            p_fast = (cond_p[j, fast_idx] * (w_fast / max(w_fast.sum(), 1e-8))[:, None]).sum(0)
            p_slow = (cond_p[j, slow_idx] * (w_slow / max(w_slow.sum(), 1e-8))[:, None]).sum(0)

            top_fast = int(np.argmax(p_fast))
            top_slow = int(np.argmax(p_slow))
            if top_fast == top_slow:
                continue

            m_mix = 0.5 * (p_fast + p_slow)
            jsd = 0.5 * (
                np.sum(np.where(p_fast > 1e-9, p_fast * np.log((p_fast + 1e-12) / (m_mix + 1e-12)), 0.0))
                + np.sum(np.where(p_slow > 1e-9, p_slow * np.log((p_slow + 1e-12) / (m_mix + 1e-12)), 0.0))
            )
            board = _reconstruct_board(rec)
            black = board.turn == chess.BLACK

            def _fmt_move(midx: int) -> str:
                u = cu.index_to_uci(midx)
                if black:
                    u = cu.mirror_uci(u)
                try:
                    return board.san(chess.Move.from_uci(u))
                except Exception:  # noqa: BLE001
                    return u

            hum_idx = int(rec["move"])
            sf_idx = int(rec["sf_move"])
            fast_top3 = [
                {"san": _fmt_move(int(k)), "prob": round(float(p_fast[k]), 4)}
                for k in np.argsort(p_fast)[::-1][:3]
            ]
            slow_top3 = [
                {"san": _fmt_move(int(k)), "prob": round(float(p_slow[k]), 4)}
                for k in np.argsort(p_slow)[::-1][:3]
            ]
            candidates.append(
                {
                    "jsd_nats": round(float(jsd), 4),
                    "fen": board.fen(),
                    "ply": int(rec["ply"]),
                    "self_elo": int(rec["self_elo"]),
                    "oppo_elo": int(rec["oppo_elo"]),
                    "self_title": tier_names.get(int(rec["self_title"]), "?"),
                    "oppo_title": tier_names.get(int(rec["oppo_title"]), "?"),
                    "clk_left_sec": round(float(rec["clk_left"]), 1),
                    "human_ponder_sec": round(float(rec["ponder"]), 1),
                    "human_move": _fmt_move(hum_idx),
                    "sf_move": _fmt_move(sf_idx) if sf_idx >= 0 else None,
                    "sf_cp_loss": int(rec["sf_cp_loss"]),
                    "human_matches_slow": hum_idx == top_slow,
                    "human_matches_fast": hum_idx == top_fast,
                    "fast_top3": fast_top3,
                    "slow_top3": slow_top3,
                }
            )

    candidates.sort(
        key=lambda d: (d["human_matches_slow"] and d["human_ponder_sec"] >= 8.0, d["jsd_nats"]),
        reverse=True,
    )
    return candidates[:top_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/mnt/maia3data")
    ap.add_argument("--out", default="/mnt/maia3data/paper_results/diagnostics/paper_diagnostics.json")
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument(
        "--models",
        nargs="*",
        default=[],
        help="Specs as LABEL|CONFIG|WORKDIR|STEP|EXTRA_OVERRIDES (comma-separated)",
    )
    ap.add_argument("--joint_case_study_spec", default="")
    a = ap.parse_args()

    init_distributed()
    from absl import flags

    if not flags.FLAGS.is_parsed():
        flags.FLAGS(["paper-diagnostics"])

    root = Path(a.data_root)
    online_glob = str(root / "eval_titled_antisf" / "*.array_record")
    otb_glob = str(root / "eval_fide_otb_antisf" / "*.array_record")

    report: dict = {"corpus_audit": audit_corpus(a.data_root), "mcnemar": {}, "otb_value_recalibration": {}}
    print("Corpus audit complete.", flush=True)

    mesh = jax.sharding.Mesh(np.asarray(jax.devices()), ("data",))
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("data"))
    repl = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    # 1. Evaluate MAIA-3 baselines for paired predictions + OTB value recalibration
    preds_online: dict[str, dict[str, np.ndarray]] = {}
    preds_otb: dict[str, dict[str, np.ndarray]] = {}
    for m_name in ("maia3-5m", "maia3-23m", "maia3-79m"):
        print(f"Collecting paired predictions for {m_name}...", flush=True)
        mcfg, mparams = load_released(m_name, "float32")
        gdef, _ = nnx.split(build_model(mcfg, rngs=0))
        mparams = jax.device_put(mparams, repl)
        preds_online[m_name] = collect_per_position_preds(
            gdef, mparams, mesh, data_sharding, repl, mcfg.history, online_glob, a.batch_size
        )
        preds_otb[m_name] = collect_per_position_preds(
            gdef, mparams, mesh, data_sharding, repl, mcfg.history, otb_glob, a.batch_size
        )
        report["otb_value_recalibration"][m_name] = cv_recalibrate_value_logits(
            preds_otb[m_name]["value_logits"], preds_otb[m_name]["value"]
        )

    # 2. Evaluate ChessForge models passed via --models
    cf_labels = []
    for spec in a.models:
        parts = spec.split("|")
        label, cfg_path, workdir, step_s = parts[0], parts[1], parts[2], parts[3]
        extra = [x for x in (parts[4].split(",") if len(parts) > 4 and parts[4] else []) if x]
        step = None if step_s in ("", "latest", "None") else int(step_s)
        if not os.path.exists(os.path.join(workdir, "checkpoints")):
            print(f"[skip] {label}: checkpoint dir missing at {workdir}", flush=True)
            continue
        print(f"Collecting paired predictions for {label} ({workdir}, step={step})...", flush=True)
        ov = [f"workdir={workdir}", "model_overrides.compute_dtype=float32", *extra]
        trainer = Trainer(load_train_config(cfg_path, ov))
        params = trainer.restore_params(step)
        preds_online[label] = collect_per_position_preds(
            trainer.graphdef, params, mesh, data_sharding, repl, trainer.model_cfg.history, online_glob, a.batch_size
        )
        preds_otb[label] = collect_per_position_preds(
            trainer.graphdef, params, mesh, data_sharding, repl, trainer.model_cfg.history, otb_glob, a.batch_size
        )
        cf_labels.append(label)

    # 3. Compute paired McNemar's tests (every CF model vs MAIA-3 baselines + pairwise CF)
    for domain, store in [("online", preds_online), ("otb", preds_otb)]:
        report["mcnemar"][domain] = {}
        for cf in cf_labels:
            for base in ("maia3-5m", "maia3-23m", "maia3-79m"):
                report["mcnemar"][domain][f"{cf}_vs_{base}"] = mcnemar_test(
                    store[cf]["correct"], store[base]["correct"]
                )
        for i, a_lbl in enumerate(cf_labels):
            for b_lbl in cf_labels[i + 1 :]:
                report["mcnemar"][domain][f"{a_lbl}_vs_{b_lbl}"] = mcnemar_test(
                    store[a_lbl]["correct"], store[b_lbl]["correct"]
                )

    # 4. Qualitative Joint-Time Case Studies
    if a.joint_case_study_spec:
        parts = a.joint_case_study_spec.split("|")
        cfg_path, workdir, step_s = parts[0], parts[1], parts[2]
        extra = [x for x in (parts[3].split(",") if len(parts) > 3 and parts[3] else []) if x]
        step = None if step_s in ("", "latest", "None") else int(step_s)
        if os.path.exists(os.path.join(workdir, "checkpoints")):
            print("Extracting qualitative Joint-Time GM case studies...", flush=True)
            ov = [f"workdir={workdir}", "model_overrides.compute_dtype=float32", *extra]
            report["joint_time_case_studies"] = extract_joint_time_case_studies(
                cfg_path, ov, step, online_glob
            )

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2))
    print(f"Wrote {a.out}", flush=True)


if __name__ == "__main__":
    main()