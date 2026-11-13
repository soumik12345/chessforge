from __future__ import annotations

import json
import time
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from etils import epath
from flax import nnx, struct
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from chessforge.config import TrainConfig, config_to_dict
from chessforge.data import pipeline
from chessforge.eval.evaluate import make_eval_step, run_eval
from chessforge.models import (
    build_model,
    build_model_config,
    count_params,
    forward_model,
)
from chessforge.train.losses import maia3_loss
from chessforge.train.schedule import build_lr_schedule


@struct.dataclass
class TrainState:
    step: jnp.ndarray
    params: Any
    opt_state: Any

    def __repr__(self) -> str:
        return f"TrainState(step={int(self.step)})"


def make_optimizer(o, schedule):
    return optax.chain(
        optax.clip_by_global_norm(o.grad_clip_norm),
        optax.adamw(schedule, b1=o.b1, b2=o.b2, eps=o.eps, weight_decay=o.weight_decay),
    )


def log0(msg: str):
    if jax.process_index() == 0:
        print(msg, flush=True)


def summarize_train_metrics(sums: dict, steps: int) -> dict[str, float]:
    """Average update metrics, but weight joint NLL by valid clock examples."""
    metrics = {k: float(v) / steps for k, v in sums.items()}
    if "joint_nll_sum" in sums:
        weight = float(sums["joint_time_weight"])
        metrics["joint_time_weight"] = weight
        metrics["joint_nll"] = float(sums["joint_nll_sum"]) / max(weight, 1e-8)
        metrics.pop("joint_nll_sum")
    return metrics


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.model_cfg = build_model_config(cfg.model, cfg.model_overrides)
        if cfg.joint_time_coef < 0:
            raise ValueError("joint_time_coef must be nonnegative")
        if cfg.joint_time_coef > 0 and not (
            self.model_cfg.arch == "chessformer_v2"
            and self.model_cfg.use_joint_move_time
        ):
            raise ValueError(
                "joint_time_coef requires ChessformerV2 use_joint_move_time=true"
            )
        self.graphdef, _ = nnx.split(build_model(self.model_cfg, rngs=0))
        ndev = jax.device_count()
        accum = cfg.optim.grad_accum_steps
        if cfg.data.global_batch_size % (ndev * accum) != 0:
            raise ValueError(
                f"global_batch_size ({cfg.data.global_batch_size}) must be divisible by "
                f"device_count ({ndev}) * grad_accum_steps ({accum}) = {ndev * accum}"
            )
        self.mesh = Mesh(np.asarray(jax.devices()), ("data",))
        self.data_sharding = NamedSharding(self.mesh, P("data"))
        self.repl = NamedSharding(self.mesh, P())
        self.schedule = build_lr_schedule(cfg.optim)
        self.tx = make_optimizer(cfg.optim, self.schedule)
        self.workdir = epath.Path(cfg.workdir)
        self.ckpt = ocp.CheckpointManager(
            self.workdir / "checkpoints",
            options=ocp.CheckpointManagerOptions(
                max_to_keep=cfg.keep_checkpoints, create=True
            ),
        )
        self.ref_params: Any = None
        self._build_fns()

    def _init_state(self, rng):
        model = build_model(self.model_cfg, rngs=nnx.Rngs(params=rng))
        _, params = nnx.split(model)
        return TrainState(
            step=jnp.zeros((), jnp.int32), params=params, opt_state=self.tx.init(params)
        )

    def _build_fns(self):
        cfg, graphdef = self.cfg, self.graphdef
        accum = cfg.optim.grad_accum_steps
        deterministic = self.model_cfg.dropout == 0.0
        use_dpo_ref = cfg.dpo_coef > 0 and bool(cfg.init_from_checkpoint)

        def loss_fn(params, ref_params, batch, rng):
            model = nnx.merge(graphdef, params)
            lm, lv, lp, aux = forward_model(
                model,
                batch,
                deterministic=deterministic,
                rng=rng,
                return_aux=True,
                mask_legal=cfg.mask_illegal_in_loss,
            )
            ref_lm = None
            if ref_params is not None:
                ref_f32 = jax.tree.map(lambda x: x.astype(jnp.float32), ref_params)
                ref_model = nnx.merge(graphdef, ref_f32)
                ref_lm, _, _ = jax.lax.stop_gradient(
                    forward_model(
                        ref_model,
                        batch,
                        deterministic=True,
                        return_aux=False,
                        mask_legal=cfg.mask_illegal_in_loss,
                    )
                )
            return maia3_loss(
                lm,
                lv,
                lp,
                batch,
                aux=aux,
                ref_logits_move=ref_lm,
                value_coef=cfg.value_coef,
                ponder_coef=cfg.ponder_coef,
                joint_time_coef=cfg.joint_time_coef,
                reply_coef=cfg.reply_coef,
                label_smoothing=cfg.label_smoothing,
                dpo_coef=cfg.dpo_coef,
                dpo_beta=cfg.dpo_beta,
                sf_max_cp_loss=cfg.sf_max_cp_loss,
                mask_illegal=cfg.mask_illegal_in_loss,
            )

        grad_fn = jax.value_and_grad(loss_fn, argnums=0, has_aux=True)

        def train_step(state, ref_params, batch, base_rng):
            rng = jax.random.fold_in(base_rng, state.step)
            if accum == 1:
                (_, metrics), grads = grad_fn(state.params, ref_params, batch, rng)
            else:
                mbs = jax.tree.map(
                    lambda x: x.reshape(-1, accum, *x.shape[1:]).swapaxes(0, 1), batch
                )
                mbs = jax.lax.with_sharding_constraint(
                    mbs, NamedSharding(self.mesh, P(None, "data"))
                )

                def body(carry, xs):
                    g_acc, m_acc = carry
                    mb, i = xs
                    (_, m), g = grad_fn(
                        state.params, ref_params, mb, jax.random.fold_in(rng, i)
                    )
                    return (
                        jax.tree.map(jnp.add, g_acc, g),
                        jax.tree.map(jnp.add, m_acc, m),
                    ), None

                zeros_g = jax.tree.map(jnp.zeros_like, state.params)
                m0 = jax.eval_shape(
                    lambda: grad_fn(
                        state.params,
                        ref_params,
                        jax.tree.map(lambda x: x[0], mbs),
                        rng,
                    )[0][1]
                )
                zeros_m = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), m0)
                (grads, metrics), _ = jax.lax.scan(
                    body, (zeros_g, zeros_m), (mbs, jnp.arange(accum))
                )
                grads = jax.tree.map(lambda g: g / accum, grads)
                metrics = jax.tree.map(lambda m: m / accum, metrics)
            if "joint_nll_sum" in metrics:
                # Counts and numerators must sum across microbatches, including
                # those with no clock labels; averaging their means is biased.
                metrics["joint_time_weight"] *= accum
                metrics["joint_nll_sum"] *= accum
                metrics["joint_nll"] = metrics["joint_nll_sum"] / jnp.maximum(
                    metrics["joint_time_weight"], 1e-8
                )
            updates, opt_state = self.tx.update(grads, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)
            metrics = dict(
                metrics,
                grad_norm=optax.global_norm(grads),
                lr=self.schedule(state.step),
            )
            return TrainState(
                step=state.step + 1, params=params, opt_state=opt_state
            ), metrics

        ref_in_sharding = self.repl if use_dpo_ref else None
        self.p_train_step = jax.jit(
            train_step,
            in_shardings=(self.repl, ref_in_sharding, self.data_sharding, self.repl),
            out_shardings=(self.repl, self.repl),
            donate_argnums=(0,),
        )
        self.p_init = jax.jit(self._init_state, out_shardings=self.repl)
        self.p_reinit_opt = jax.jit(
            lambda params: TrainState(
                step=jnp.zeros((), jnp.int32),
                params=params,
                opt_state=self.tx.init(params),
            ),
            in_shardings=self.repl,
            out_shardings=self.repl,
        )
        self.p_add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
        self.eval_step = make_eval_step(
            self.graphdef, self.data_sharding, self.repl, cfg.sf_max_cp_loss
        )

    def to_global(self, batch):
        return jax.tree.map(
            lambda x: jax.make_array_from_process_local_data(self.data_sharding, x),
            batch,
        )

    def _load_external_params(self, ckpt_dir: str, step: int | None) -> nnx.State:
        mgr = ocp.CheckpointManager(epath.Path(ckpt_dir))
        target_step = mgr.latest_step() if step is None else step
        if target_step is None:
            raise FileNotFoundError(f"No checkpoints found in {ckpt_dir!r}")
        state_shape = jax.eval_shape(self._init_state, jax.random.PRNGKey(0))
        abstract = jax.tree.map(
            lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=self.repl),
            state_shape,
        )
        restored = cast(
            TrainState,
            mgr.restore(target_step, args=ocp.args.StandardRestore(abstract)),
        )
        log0(f"Loaded external weights from {ckpt_dir} at step {target_step}")
        return restored.params

    def _data_state_path(self, step):
        return (
            self.workdir
            / "data_state"
            / str(step)
            / f"process_{jax.process_index()}.json"
        )

    def save(self, state, data_iter):
        step = int(state.step)
        self.ckpt.save(step, args=ocp.args.StandardSave(state))
        try:
            p = self._data_state_path(step)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(
                    data_iter.get_state(),
                    default=lambda o: o.item() if hasattr(o, "item") else str(o),
                )
            )
        except Exception as e:  # noqa: BLE001
            log0(f"[warn] could not save data iterator state: {e}")

    def restore_or_init(self):
        state = self.p_init(jax.random.PRNGKey(self.cfg.seed))
        latest = self.ckpt.latest_step()
        ext_params = None
        if self.cfg.init_from_checkpoint:
            ext_params = self._load_external_params(
                self.cfg.init_from_checkpoint, self.cfg.init_from_step
            )
            if self.cfg.dpo_coef > 0:
                self.ref_params = jax.device_put(
                    jax.tree.map(lambda x: x.astype(jnp.bfloat16), ext_params),
                    self.repl,
                )
        if latest is not None:
            abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
            state = cast(
                TrainState,
                self.ckpt.restore(latest, args=ocp.args.StandardRestore(abstract)),
            )
            log0(f"Restored checkpoint at step {latest}")
        elif ext_params is not None:
            state = self.p_reinit_opt(ext_params)
            log0("Initialized fresh optimizer state from init_from_checkpoint")
        return state, latest

    def restore_params(self, step: int | None = None) -> nnx.State:
        state = jax.eval_shape(self._init_state, jax.random.PRNGKey(0))
        abstract = jax.tree.map(
            lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=self.repl), state
        )
        step = self.ckpt.latest_step() if step is None else step
        restored = cast(
            TrainState,
            self.ckpt.restore(step, args=ocp.args.StandardRestore(abstract)),
        )
        return restored.params

    def evaluate(
        self, params: nnx.State, pattern: str, max_batches: int | None = None
    ) -> dict:
        d = self.cfg.data
        batches = pipeline.eval_batches(
            pattern,
            self.model_cfg.history,
            d.eval_batch_size,
            d.local_cache_dir,
            max_batches,
            d.read_threads,
        )
        return run_eval(self.eval_step, params, (self.to_global(b) for b in batches))

    def fit(self):
        cfg, o, d = self.cfg, self.cfg.optim, self.cfg.data
        micro_bs = d.global_batch_size // (jax.device_count() * o.grad_accum_steps)
        log0(
            f"model={self.model_cfg.name} arch={self.model_cfg.arch} "
            f"params={count_params(self.model_cfg):,} devices={jax.device_count()} "
            f"hosts={jax.process_count()} global_batch={d.global_batch_size} "
            f"accum={o.grad_accum_steps} microbatch_per_device={micro_bs}"
        )
        if jax.process_index() == 0:
            self.workdir.mkdir(parents=True, exist_ok=True)
            (self.workdir / "config.json").write_text(
                json.dumps(config_to_dict(cfg), indent=2)
            )

        state, restored_step = self.restore_or_init()
        step = int(state.step)
        anneal_step = (
            int(round(o.total_steps * d.anneal_start_frac))
            if d.anneal_glob
            else o.total_steps + 1
        )
        in_anneal = step >= anneal_step
        active_glob = d.anneal_glob if in_anneal else d.train_glob

        train_it, n_local = pipeline.make_train_iterator(
            d, self.model_cfg.history, pattern=active_glob
        )
        if restored_step is not None:
            p = self._data_state_path(restored_step)
            if p.exists():
                try:
                    train_it.set_state(json.loads(p.read_text()))
                    log0("Restored data iterator state")
                except Exception as e:  # noqa: BLE001
                    log0(f"[warn] data state restore failed ({e}); continuing fresh")
        n_total = n_local * jax.process_count()
        log0(
            f"~{n_total:,} active positions; {o.total_steps} steps = "
            f"{o.total_steps * d.global_batch_size / max(n_total, 1):.2f} epochs"
        )

        tracking_run = None
        if cfg.trackio_project and jax.process_index() == 0:
            import trackio

            tracking_run = trackio.init(
                project=cfg.trackio_project,
                space_id=cfg.trackio_space_id,
                config=config_to_dict(cfg),
                name=cfg.trackio_run_name or self.model_cfg.name,
                group=cfg.trackio_group,
                embed=False,
            )

        try:
            base_rng = jax.device_put(jax.random.PRNGKey(cfg.seed + 1), self.repl)
            acc, n_acc, t0 = None, 0, time.time()
            while step < o.total_steps:
                if not in_anneal and step >= anneal_step and d.anneal_glob:
                    log0(
                        f"[curriculum] step {step}: switching to anneal_glob={d.anneal_glob}"
                    )
                    train_it, _ = pipeline.make_train_iterator(
                        d,
                        self.model_cfg.history,
                        pattern=d.anneal_glob,
                        seed=d.shuffle_seed + 99991,
                    )
                    in_anneal = True

                batch = self.to_global(next(train_it))
                state, metrics = self.p_train_step(
                    state, self.ref_params, batch, base_rng
                )
                step += 1
                acc = metrics if acc is None else self.p_add(acc, metrics)
                n_acc += 1

                if step % cfg.log_every == 0:
                    m = summarize_train_metrics(jax.device_get(acc), n_acc)
                    dt = time.time() - t0
                    m["positions_per_sec"] = n_acc * d.global_batch_size / max(dt, 1e-6)
                    log0(
                        f"step {step} " + " ".join(f"{k}={v:.5g}" for k, v in m.items())
                    )
                    if tracking_run is not None:
                        tracking_run.log(
                            {f"train/{k}": v for k, v in m.items()}, step=step
                        )
                    acc, n_acc, t0 = None, 0, time.time()

                last = step == o.total_steps
                if d.val_glob and (step % cfg.eval_every == 0 or last):
                    r = self.evaluate(state.params, d.val_glob, d.val_batches)
                    log0(f"[val] step {step} {json.dumps(r['overall'])}")
                    if tracking_run is not None:
                        tracking_run.log(
                            {f"val/{k}": v for k, v in r["overall"].items()}, step=step
                        )
                if d.test_glob and (step % cfg.test_every == 0 or last):
                    r = self.evaluate(state.params, d.test_glob)
                    log0(f"[test] step {step} {json.dumps(r['overall'])}")
                    if tracking_run is not None:
                        tracking_run.log(
                            {f"test/{k}": v for k, v in r["overall"].items()}, step=step
                        )
                if step % cfg.ckpt_every == 0 or last:
                    self.save(state, train_it)
            self.ckpt.wait_until_finished()
            return state
        finally:
            if tracking_run is not None:
                tracking_run.finish()
