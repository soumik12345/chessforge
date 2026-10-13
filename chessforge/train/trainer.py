from __future__ import annotations

import json
import time
from typing import Any

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from etils import epath
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from ..config import TrainConfig, config_to_dict
from ..data import pipeline
from ..eval.evaluate import make_eval_step, run_eval
from ..models import build_model, build_model_config, count_params
from ..models.chessformer import dummy_inputs
from .losses import maia3_loss
from .schedule import maia3_lr_schedule


@flax.struct.dataclass
class TrainState:
    step: jnp.ndarray
    params: Any
    opt_state: Any


def make_optimizer(o, schedule):
    return optax.chain(
        optax.clip_by_global_norm(o.grad_clip_norm),
        optax.adamw(schedule, b1=o.b1, b2=o.b2, eps=o.eps, weight_decay=o.weight_decay),
    )


def log0(msg: str):
    if jax.process_index() == 0:
        print(msg, flush=True)


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.model_cfg = build_model_config(cfg.model, cfg.model_overrides)
        self.model = build_model(self.model_cfg)
        self.mesh = Mesh(np.asarray(jax.devices()), ("data",))
        self.data_sharding = NamedSharding(self.mesh, P("data"))
        self.repl = NamedSharding(self.mesh, P())
        o = cfg.optim
        self.schedule = maia3_lr_schedule(o.lr, o.min_lr, o.warmup_steps, o.cosine_cycle_steps)
        self.tx = make_optimizer(o, self.schedule)
        self.workdir = epath.Path(cfg.workdir)
        self.ckpt = ocp.CheckpointManager(
            self.workdir / "checkpoints",
            options=ocp.CheckpointManagerOptions(max_to_keep=cfg.keep_checkpoints, create=True))
        self._build_fns()

    # ------------------------------------------------------------------ setup
    def _init_state(self, rng):
        params = self.model.init(rng, *dummy_inputs(self.model_cfg))["params"]
        return TrainState(step=jnp.zeros((), jnp.int32), params=params, opt_state=self.tx.init(params))

    def _build_fns(self):
        cfg, model = self.cfg, self.model
        accum = cfg.optim.grad_accum_steps
        deterministic = self.model_cfg.dropout == 0.0

        def loss_fn(params, batch, rng):
            lm, lv, lp = model.apply({"params": params}, batch["board"], batch["self_elo"], batch["oppo_elo"],
                                     deterministic=deterministic, rngs={"dropout": rng})
            return maia3_loss(lm, lv, lp, batch, value_coef=cfg.value_coef, ponder_coef=cfg.ponder_coef,
                              mask_illegal=cfg.mask_illegal_in_loss)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

        def train_step(state, batch, base_rng):
            rng = jax.random.fold_in(base_rng, state.step)
            if accum == 1:
                (_, metrics), grads = grad_fn(state.params, batch, rng)
            else:
                # interleave so every device contributes equally to each micro-batch
                mbs = jax.tree.map(lambda x: x.reshape(-1, accum, *x.shape[1:]).swapaxes(0, 1), batch)
                mbs = jax.lax.with_sharding_constraint(mbs, NamedSharding(self.mesh, P(None, "data")))

                def body(carry, xs):
                    g_acc, m_acc = carry
                    mb, i = xs
                    (_, m), g = grad_fn(state.params, mb, jax.random.fold_in(rng, i))
                    return (jax.tree.map(jnp.add, g_acc, g), jax.tree.map(jnp.add, m_acc, m)), None

                zeros_g = jax.tree.map(jnp.zeros_like, state.params)
                m0 = jax.eval_shape(lambda: grad_fn(state.params, jax.tree.map(lambda x: x[0], mbs), rng)[0][1])
                zeros_m = jax.tree.map(lambda s: jnp.zeros(s.shape, s.dtype), m0)
                (grads, metrics), _ = jax.lax.scan(body, (zeros_g, zeros_m), (mbs, jnp.arange(accum)))
                grads = jax.tree.map(lambda g: g / accum, grads)
                metrics = jax.tree.map(lambda m: m / accum, metrics)
            updates, opt_state = self.tx.update(grads, state.opt_state, state.params)
            params = optax.apply_updates(state.params, updates)
            metrics = dict(metrics, grad_norm=optax.global_norm(grads), lr=self.schedule(state.step))
            return TrainState(step=state.step + 1, params=params, opt_state=opt_state), metrics

        self.p_train_step = jax.jit(train_step, in_shardings=(self.repl, self.data_sharding, self.repl),
                                    out_shardings=(self.repl, self.repl), donate_argnums=(0,))
        self.p_init = jax.jit(self._init_state, out_shardings=self.repl)
        self.p_add = jax.jit(lambda a, b: jax.tree.map(jnp.add, a, b))
        self.eval_step = make_eval_step(self.model, self.data_sharding, self.repl)

    def to_global(self, batch):
        return jax.tree.map(lambda x: jax.make_array_from_process_local_data(self.data_sharding, x), batch)

    # -------------------------------------------------------------- checkpoint
    def _data_state_path(self, step):
        return self.workdir / "data_state" / str(step) / f"process_{jax.process_index()}.json"

    def save(self, state, data_iter):
        step = int(state.step)
        self.ckpt.save(step, args=ocp.args.StandardSave(state))
        try:
            p = self._data_state_path(step)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(data_iter.get_state(), default=lambda o: o.item() if hasattr(o, "item") else str(o)))
        except Exception as e:  # noqa: BLE001
            log0(f"[warn] could not save data iterator state: {e}")

    def restore_or_init(self):
        state = self.p_init(jax.random.PRNGKey(self.cfg.seed))
        latest = self.ckpt.latest_step()
        if latest is not None:
            abstract = jax.tree.map(ocp.utils.to_shape_dtype_struct, state)
            state = self.ckpt.restore(latest, args=ocp.args.StandardRestore(abstract))
            log0(f"Restored checkpoint at step {latest}")
        return state, latest

    def restore_params(self, step: int | None = None):
        state = jax.eval_shape(self._init_state, jax.random.PRNGKey(0))
        abstract = jax.tree.map(lambda s: jax.ShapeDtypeStruct(s.shape, s.dtype, sharding=self.repl), state)
        step = self.ckpt.latest_step() if step is None else step
        return self.ckpt.restore(step, args=ocp.args.StandardRestore(abstract)).params

    # -------------------------------------------------------------------- eval
    def evaluate(self, params, pattern: str, max_batches: int | None = None) -> dict:
        d = self.cfg.data
        batches = pipeline.eval_batches(pattern, self.model_cfg.history, d.eval_batch_size,
                                        d.local_cache_dir, max_batches, d.read_threads)
        return run_eval(self.eval_step, params, (self.to_global(b) for b in batches))

    # --------------------------------------------------------------------- fit
    def fit(self):
        cfg, o, d = self.cfg, self.cfg.optim, self.cfg.data
        log0(f"model={self.model_cfg.name} params={count_params(self.model_cfg):,} devices={jax.device_count()} "
             f"hosts={jax.process_count()} global_batch={d.global_batch_size}")
        if jax.process_index() == 0:
            self.workdir.mkdir(parents=True, exist_ok=True)
            (self.workdir / "config.json").write_text(json.dumps(config_to_dict(cfg), indent=2))

        state, restored_step = self.restore_or_init()
        train_it, n_local = pipeline.make_train_iterator(d, self.model_cfg.history)
        if restored_step is not None:
            p = self._data_state_path(restored_step)
            if p.exists():
                try:
                    train_it.set_state(json.loads(p.read_text()))
                    log0("Restored data iterator state")
                except Exception as e:  # noqa: BLE001
                    log0(f"[warn] data state restore failed ({e}); continuing with fresh iterator")
        n_total = n_local * jax.process_count()
        log0(f"~{n_total:,} training positions; {o.total_steps} steps = "
             f"{o.total_steps * d.global_batch_size / max(n_total, 1):.2f} epochs")

        wandb = None
        if cfg.wandb_project and jax.process_index() == 0:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(project=cfg.wandb_project, config=config_to_dict(cfg), name=self.model_cfg.name)

        base_rng = jax.device_put(jax.random.PRNGKey(cfg.seed + 1), self.repl)
        step = int(state.step)
        acc, n_acc, t0 = None, 0, time.time()
        while step < o.total_steps:
            batch = self.to_global(next(train_it))
            state, metrics = self.p_train_step(state, batch, base_rng)
            step += 1
            acc = metrics if acc is None else self.p_add(acc, metrics)
            n_acc += 1

            if step % cfg.log_every == 0:
                m = {k: float(v) / n_acc for k, v in jax.device_get(acc).items()}
                dt = time.time() - t0
                m["positions_per_sec"] = n_acc * d.global_batch_size / dt
                log0(f"step {step} " + " ".join(f"{k}={v:.5g}" for k, v in m.items()))
                if wandb:
                    wandb.log({f"train/{k}": v for k, v in m.items()}, step=step)
                acc, n_acc, t0 = None, 0, time.time()

            last = step == o.total_steps
            if d.val_glob and (step % cfg.eval_every == 0 or last):
                r = self.evaluate(state.params, d.val_glob, d.val_batches)
                log0(f"[val] step {step} {json.dumps(r['overall'])}")
                if wandb:
                    wandb.log({f"val/{k}": v for k, v in r["overall"].items()}, step=step)
            if d.test_glob and (step % cfg.test_every == 0 or last):
                r = self.evaluate(state.params, d.test_glob)
                log0(f"[test] step {step} {json.dumps(r['overall'])}")
                if wandb:
                    wandb.log({f"test/{k}": v for k, v in r["overall"].items()}, step=step)
            if step % cfg.ckpt_every == 0 or last:
                self.save(state, train_it)
        self.ckpt.wait_until_finished()
        return state