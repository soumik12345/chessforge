"""Training callbacks and the default console/experiment metrics logger."""

from __future__ import annotations

import importlib
import json
import time
from typing import Any

import jax

from chessforge.config import ModelConfig, TrainConfig, config_to_dict
from chessforge.models import count_params


class Callback:
    """Override hooks as needed. Hooks run on every host, outside JIT.

    Metrics are device arrays at step end and evaluation result dictionaries at
    evaluation end. Callback errors propagate; started callbacks are finalized
    in reverse order, including when training or callback startup fails.
    """

    def on_train_start(
        self, cfg: TrainConfig, model_cfg: ModelConfig, num_positions: int
    ) -> None:
        pass

    def on_train_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        pass

    def on_eval_end(self, step: int, split: str, results: dict[str, Any]) -> None:
        pass

    def on_train_end(self) -> None:
        pass


@jax.jit
def _add_metrics(acc, metrics):
    return jax.tree.map(lambda a, b: a + b, acc, metrics)


class MetricsLogger(Callback):
    """Average training metrics and report them on process zero.

    ``tracking_backend`` selects Trackio (the default) or W&B. Only the selected
    backend is imported, and only when its project is configured. Each invocation
    of ``Trainer.fit`` starts a fresh accumulation window and finishes its own
    run, including on failure. Partial windows retain the existing behavior:
    they are reported only when the absolute step reaches ``log_every``.
    """

    def __init__(self) -> None:
        self._enabled = False
        self._primary = False
        self._run = None
        self._acc = None
        self._n_acc = 0
        self._t0 = 0.0

    def on_train_start(
        self, cfg: TrainConfig, model_cfg: ModelConfig, num_positions: int
    ) -> None:
        if cfg.tracking_backend not in ("wandb", "trackio"):
            raise ValueError(
                f"Unsupported tracking backend {cfg.tracking_backend!r}; "
                "expected 'wandb' or 'trackio'"
            )
        self._enabled = True
        self._primary = jax.process_index() == 0
        self._cfg = cfg
        self._acc, self._n_acc = None, 0
        if not self._primary:
            return

        print(
            f"model={model_cfg.name} params={count_params(model_cfg):,} devices={jax.device_count()} "
            f"hosts={jax.process_count()} global_batch={cfg.data.global_batch_size}",
            flush=True,
        )
        print(
            f"~{num_positions:,} training positions; {cfg.optim.total_steps} steps = "
            f"{cfg.optim.total_steps * cfg.data.global_batch_size / max(num_positions, 1):.2f} epochs",
            flush=True,
        )
        project = (
            cfg.wandb_project
            if cfg.tracking_backend == "wandb"
            else cfg.trackio_project
        )
        if project:
            backend = importlib.import_module(cfg.tracking_backend)
            self._run = backend.init(
                project=project,
                config=config_to_dict(cfg),
                name=model_cfg.name,
            )
        self._t0 = time.perf_counter()

    def on_train_step_end(self, step: int, metrics: dict[str, Any]) -> None:
        if not self._enabled:
            return
        # Every host must participate in JAX computations on global arrays.
        # Only reporting and host transfers are restricted to process zero.
        self._acc = metrics if self._acc is None else _add_metrics(self._acc, metrics)
        self._n_acc += 1
        if step % self._cfg.log_every:
            return
        if not self._primary:
            self._acc, self._n_acc = None, 0
            return

        averaged = {
            k: float(v) / self._n_acc for k, v in jax.device_get(self._acc).items()
        }
        elapsed = time.perf_counter() - self._t0
        averaged["positions_per_sec"] = (
            self._n_acc * self._cfg.data.global_batch_size / elapsed
        )
        print(
            f"step {step} " + " ".join(f"{k}={v:.5g}" for k, v in averaged.items()),
            flush=True,
        )
        self._log(step, "train", averaged)
        self._acc, self._n_acc = None, 0
        self._t0 = time.perf_counter()

    def on_eval_end(self, step: int, split: str, results: dict[str, Any]) -> None:
        if not self._enabled or not self._primary:
            return
        overall = results["overall"]
        print(f"[{split}] step {step} {json.dumps(overall)}", flush=True)
        self._log(step, split, overall)

    def _log(self, step: int, split: str, metrics: dict[str, Any]) -> None:
        if self._run is not None:
            self._run.log({f"{split}/{k}": v for k, v in metrics.items()}, step=step)

    def on_train_end(self) -> None:
        self._enabled = False
        self._acc, self._n_acc = None, 0
        run, self._run = self._run, None
        if run is not None:
            run.finish()
