"""Metric reporting and Trainer callback lifecycle, without external services."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, call

import jax.numpy as jnp
import numpy as np
import pytest

from chessforge.config import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainConfig,
    load_train_config,
)
from chessforge.train import (
    Callback,
    MetricsLogger,
    callbacks as callbacks_module,
    trainer as trainer_module,
)
from chessforge.train.trainer import Trainer


@pytest.fixture(params=["wandb", "trackio"])
def logging_env(monkeypatch, request):
    backend = request.param
    tracker = Mock()
    for name in ("wandb", "trackio"):
        monkeypatch.setitem(sys.modules, name, tracker if name == backend else None)
    monkeypatch.setattr(callbacks_module.jax, "process_index", lambda: 0)
    monkeypatch.setattr(callbacks_module, "count_params", lambda cfg: 123)
    clock = iter([0.0, 4.0, 4.0, 6.0, 8.0, 8.0])
    monkeypatch.setattr(callbacks_module.time, "perf_counter", lambda: next(clock))
    cfg = TrainConfig(
        data=DataConfig(global_batch_size=16), log_every=2, tracking_backend=backend
    )
    setattr(cfg, f"{backend}_project", "test")
    return cfg, tracker


def test_metrics_logger_averages_windows_and_prefixes_evaluations(logging_env, capsys):
    cfg, tracker = logging_env
    logger = MetricsLogger()
    logger.on_train_start(cfg, ModelConfig(name="test-model"), 100)
    tracker.init.assert_called_once_with(
        project="test", config=trainer_module.config_to_dict(cfg), name="test-model"
    )
    run = tracker.init.return_value

    for step, loss in enumerate([2.0, 4.0, 6.0, 8.0], start=1):
        logger.on_train_step_end(step, {"loss": jnp.array(loss)})
        assert run.log.call_count == step // 2

    results = {"overall": {"move_acc": 0.5}, "per_rating": {1500: {"move_acc": 0.6}}}
    logger.on_eval_end(4, "val", results)
    logger.on_eval_end(4, "test", results)
    assert run.log.call_args_list == [
        call({"train/loss": 3.0, "train/positions_per_sec": 8.0}, step=2),
        call({"train/loss": 7.0, "train/positions_per_sec": 16.0}, step=4),
        call({"val/move_acc": 0.5}, step=4),
        call({"test/move_acc": 0.5}, step=4),
    ]
    logger.on_train_end()
    logger.on_train_end()
    run.finish.assert_called_once_with()
    output = capsys.readouterr().out
    assert "model=test-model params=123" in output
    assert "~100 training positions" in output
    assert "step 2 loss=3 positions_per_sec=8" in output
    assert '[val] step 4 {"move_acc": 0.5}' in output
    assert '[test] step 4 {"move_acc": 0.5}' in output


def test_resumed_logging_window_and_reuse(logging_env):
    cfg, tracker = logging_env
    cfg.log_every = 4
    logger = MetricsLogger()
    logger.on_train_start(cfg, ModelConfig(), 100)
    logger.on_train_step_end(3, {"loss": jnp.array(2.0)})
    logger.on_train_step_end(4, {"loss": jnp.array(4.0)})
    logger.on_train_step_end(5, {"loss": jnp.array(100.0)})
    logger.on_train_end()
    run = tracker.init.return_value
    run.log.assert_called_once_with(
        {"train/loss": 3.0, "train/positions_per_sec": 8.0}, step=4
    )

    # A second fit must discard the prior, unreported partial window.
    logger.on_train_start(cfg, ModelConfig(), 100)
    logger.on_train_step_end(4, {"loss": jnp.array(6.0)})
    assert run.log.call_args == call(
        {"train/loss": 6.0, "train/positions_per_sec": 8.0}, step=4
    )
    logger.on_train_end()
    assert run.finish.call_count == 2


def test_console_logging_without_tracking_project(logging_env, monkeypatch, capsys):
    cfg, _ = logging_env
    # Configuring the other backend must not activate it or select its project.
    other = "trackio" if cfg.tracking_backend == "wandb" else "wandb"
    setattr(cfg, f"{cfg.tracking_backend}_project", "")
    setattr(cfg, f"{other}_project", "other-project")
    monkeypatch.setitem(sys.modules, cfg.tracking_backend, None)
    logger = MetricsLogger()
    logger.on_train_start(cfg, ModelConfig(), 100)
    logger.on_train_step_end(2, {"loss": jnp.array(2.0)})
    logger.on_eval_end(2, "val", {"overall": {"move_acc": 0.5}})
    logger.on_train_end()
    output = capsys.readouterr().out
    assert "step 2 loss=2" in output
    assert "[val] step 2" in output


def test_nonzero_process_accumulates_without_reporting(
    logging_env, monkeypatch, capsys
):
    cfg, tracker = logging_env
    monkeypatch.setattr(callbacks_module.jax, "process_index", lambda: 1)
    device_get = Mock(side_effect=AssertionError("must not fetch metrics"))
    monkeypatch.setattr(callbacks_module.jax, "device_get", device_get)
    logger = MetricsLogger()
    logger.on_train_start(cfg, ModelConfig(), 100)
    add_metrics = Mock(wraps=callbacks_module._add_metrics)
    monkeypatch.setattr(callbacks_module, "_add_metrics", add_metrics)
    logger.on_train_step_end(1, {"loss": jnp.array(2.0)})
    logger.on_train_step_end(2, {"loss": jnp.array(4.0)})
    logger.on_eval_end(2, "val", {})
    logger.on_train_end()
    tracker.init.assert_not_called()
    add_metrics.assert_called_once()
    device_get.assert_not_called()
    assert capsys.readouterr().out == ""


@pytest.fixture
def trainer(monkeypatch, tmp_path):
    """Exercise the real fit loop with deterministic training/evaluation stubs."""
    instance = Trainer.__new__(Trainer)
    instance.cfg = TrainConfig(
        workdir=str(tmp_path),
        data=DataConfig(val_glob="val", test_glob="test"),
        optim=OptimConfig(total_steps=5),
        eval_every=2,
        test_every=3,
        ckpt_every=4,
    )
    instance.model_cfg = ModelConfig()
    instance.callbacks = [Mock(spec=Callback)]
    instance.workdir = tmp_path
    instance.repl = None
    instance.ckpt = Mock()
    initial = SimpleNamespace(step=1, params="params")
    instance.restore_or_init = Mock(return_value=(initial, 1))
    iterator = Mock()
    iterator.__next__ = Mock(return_value="batch")
    monkeypatch.setattr(
        trainer_module.pipeline, "make_train_iterator", lambda *args: (iterator, 100)
    )
    instance.to_global = lambda batch: batch
    instance.p_train_step = Mock(
        side_effect=lambda state, batch, rng: (
            SimpleNamespace(step=state.step + 1, params="params"),
            {"loss": jnp.array(2.0)},
        )
    )
    instance.evaluate = Mock(return_value={"overall": {"move_acc": 0.5}})
    instance.save = Mock()
    return instance


def test_fit_dispatches_callbacks_with_resume_and_final_evaluation(trainer):
    callback = trainer.callbacks[0]
    state = trainer.fit()
    assert state.step == 5
    callback.on_train_start.assert_called_once_with(trainer.cfg, trainer.model_cfg, 100)
    assert [c.args[0] for c in callback.on_train_step_end.call_args_list] == [
        2,
        3,
        4,
        5,
    ]
    assert [(c.args[0], c.args[1]) for c in callback.on_eval_end.call_args_list] == [
        (2, "val"),
        (3, "test"),
        (4, "val"),
        (5, "val"),
        (5, "test"),
    ]
    assert all(
        c.args[2] is trainer.evaluate.return_value
        for c in callback.on_eval_end.call_args_list
    )
    assert [c.args[0].step for c in trainer.save.call_args_list] == [4, 5]
    trainer.ckpt.wait_until_finished.assert_called_once_with()
    callback.on_train_end.assert_called_once_with()
    assert json.loads((trainer.workdir / "config.json").read_text())["eval_every"] == 2


@pytest.mark.parametrize("failure", ["start", "train", "step", "eval", "end"])
def test_fit_finalizes_callbacks_on_failure(trainer, failure):
    first, second = Mock(spec=Callback), Mock(spec=Callback)
    trainer.callbacks = [first, second]
    target = {
        "start": second.on_train_start,
        "train": trainer.p_train_step,
        "step": second.on_train_step_end,
        "eval": trainer.evaluate,
        "end": second.on_train_end,
    }[failure]
    target.side_effect = RuntimeError("failed")
    with pytest.raises(RuntimeError, match="failed"):
        trainer.fit()
    first.on_train_end.assert_called_once_with()
    second.on_train_end.assert_called_once_with()


@pytest.mark.parametrize("failure", [False, True])
def test_fit_finishes_tracking_run(trainer, logging_env, failure):
    cfg, tracker = logging_env
    trainer.cfg.tracking_backend = cfg.tracking_backend
    setattr(trainer.cfg, f"{cfg.tracking_backend}_project", "test")
    trainer.callbacks = [MetricsLogger()]
    if failure:
        trainer.p_train_step.side_effect = RuntimeError("failed")
        with pytest.raises(RuntimeError, match="failed"):
            trainer.fit()
    else:
        trainer.fit()
    tracker.init.return_value.finish.assert_called_once_with()


@pytest.mark.parametrize("process_index", [0, 1])
def test_unknown_backend_fails_before_import(monkeypatch, process_index):
    monkeypatch.setattr(callbacks_module.jax, "process_index", lambda: process_index)
    cfg = TrainConfig(tracking_backend="unknown")
    with pytest.raises(ValueError, match="Unsupported tracking backend 'unknown'"):
        MetricsLogger().on_train_start(cfg, ModelConfig(), 100)


@pytest.mark.parametrize("source", ["yaml", "overrides"])
def test_trackio_configuration_loading(tmp_path, source):
    assert TrainConfig().tracking_backend == "trackio"
    if source == "yaml":
        path = tmp_path / "train.yaml"
        path.write_text("trackio_project: chessforge\n")
        cfg = load_train_config(str(path))
    else:
        cfg = load_train_config(None, ["trackio_project=chessforge"])
    assert cfg.tracking_backend == "trackio"
    assert cfg.trackio_project == "chessforge"
    assert cfg.wandb_project == ""


def test_empty_callbacks_disable_tracking(trainer, monkeypatch):
    monkeypatch.setitem(sys.modules, "trackio", None)
    trainer.cfg.trackio_project = "test"
    trainer.callbacks = []
    assert trainer.fit().step == 5


def test_callbacks_finish_in_reverse_start_order(trainer):
    events = []

    class RecordingCallback(Callback):
        def __init__(self, name):
            self.name = name

        def on_train_start(self, cfg, model_cfg, num_positions):
            events.append((self.name, "start"))

        def on_train_end(self):
            events.append((self.name, "end"))

    trainer.callbacks = [RecordingCallback("first"), RecordingCallback("second")]
    trainer.fit()
    assert events == [
        ("first", "start"),
        ("second", "start"),
        ("second", "end"),
        ("first", "end"),
    ]


@pytest.mark.parametrize("default_logger", [True, False])
def test_real_training_with_default_or_empty_callbacks(
    tmp_path, monkeypatch, capsys, default_logger
):
    """Run actual JIT updates and checkpointing through the public constructor."""
    cfg = TrainConfig(
        workdir=str(tmp_path),
        model_overrides={
            "history": 1,
            "dim_emb": 4,
            "dim_vit": 8,
            "num_blocks": 1,
            "num_heads": 2,
            "head_hid_dim": 8,
            "pos_encoding": "relative",
            "compute_dtype": "float32",
        },
        data=DataConfig(global_batch_size=2),
        optim=OptimConfig(total_steps=2),
        log_every=2,
    )
    batch = {
        "board": np.zeros((2, 1, 64), dtype=np.int8),
        "self_elo": np.full(2, 1500, dtype=np.float32),
        "oppo_elo": np.full(2, 1500, dtype=np.float32),
        "weight": np.ones(2, dtype=np.float32),
        "move": np.zeros(2, dtype=np.int32),
        "value": np.ones(2, dtype=np.int32),
    }
    iterator = Mock()
    iterator.__next__ = Mock(return_value=batch)
    iterator.get_state.return_value = {"offset": 2}
    monkeypatch.setattr(
        trainer_module.pipeline, "make_train_iterator", lambda *args: (iterator, 100)
    )
    monkeypatch.setitem(sys.modules, "trackio", None)
    instance = Trainer(cfg) if default_logger else Trainer(cfg, callbacks=[])
    try:
        state = instance.fit()
        assert int(state.step) == 2
        assert instance.ckpt.latest_step() == 2
        assert json.loads((tmp_path / "data_state/2/process_0.json").read_text()) == {
            "offset": 2
        }
        if default_logger:
            assert len(instance.callbacks) == 1
            assert isinstance(instance.callbacks[0], MetricsLogger)
        else:
            assert instance.callbacks == []
        assert ("step 2 " in capsys.readouterr().out) == default_logger
    finally:
        instance.ckpt.close()
