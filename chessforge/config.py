from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Architecture config. Field semantics mirror maia3/model_registry.py."""
    name: str = "maia3-5m"
    arch: str = "chessformer"
    history: int = 8               # boards incl. current (paper's n = history - 1)
    include_time_info: bool = False
    dim_emb: int = 128             # Elo soft-embedding dim
    dim_vit: int = 256
    num_blocks: int = 8
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    head_hid_dim: int = 256
    pos_encoding: str = "gab"      # gab | relative | absolute
    gab_gen_size: int = 64
    gab_per_square_dim: int = 0    # 0 => average pooling variant (3M/5M)
    gab_intermediate_dim: int = 64
    use_rms_norm: bool = True
    omit_qkv_biases: bool = True
    activation: str = "gelu"
    elo_max: float = 5000.0
    compute_dtype: str = "bfloat16"
    param_dtype: str = "float32"

    @property
    def dtype(self):
        import jax.numpy as jnp
        return jnp.dtype(self.compute_dtype)

    @property
    def pdtype(self):
        import jax.numpy as jnp
        return jnp.dtype(self.param_dtype)


@dataclass
class DataConfig:
    train_glob: str = ""
    val_glob: str = ""
    test_glob: str = ""            # ALLIE test set converted with cli/prepare_eval.py
    global_batch_size: int = 1024  # positions per optimizer step (all hosts)
    eval_batch_size: int = 2048
    history_mask_prob: float = 0.05
    shuffle_seed: int = 0
    num_workers: int = 16          # grain multiprocessing workers per host
    read_threads: int = 16
    prefetch_buffer_size: int = 256
    local_cache_dir: str = ""      # stage gs:// shards to local disk if set
    val_batches: int = 50


@dataclass
class OptimConfig:
    lr: float = 5e-5
    min_lr: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 3.5
    warmup_steps: int = 1000
    cosine_cycle_steps: int = 50_000
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    total_steps: int = 1_000_000
    grad_accum_steps: int = 1


@dataclass
class TrainConfig:
    model: str = "maia3-5m"
    model_overrides: dict = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    value_coef: float = 0.1
    ponder_coef: float = 0.0
    mask_illegal_in_loss: bool = False
    seed: int = 42
    workdir: str = "/tmp/maia3_run"
    log_every: int = 100
    eval_every: int = 10_000
    test_every: int = 50_000
    ckpt_every: int = 10_000
    keep_checkpoints: int = 3
    wandb_project: str = ""


def _from_dict(cls, d: dict[str, Any]):
    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        v, t = d[f.name], hints[f.name]
        if dataclasses.is_dataclass(t) and isinstance(v, dict):
            v = _from_dict(t, v)
        kwargs[f.name] = v
    unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
    if unknown:
        raise ValueError(f"Unknown keys for {cls.__name__}: {sorted(unknown)}")
    return cls(**kwargs)


def _apply_override(d: dict, override: str):
    key, _, raw = override.partition("=")
    if not _:
        raise ValueError(f"Override must look like a.b=value, got {override!r}")
    parts = key.split(".")
    cur = d
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = yaml.safe_load(raw)


def load_train_config(path: str | None, overrides: list[str] | None = None) -> TrainConfig:
    d: dict = {}
    if path:
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
    for o in overrides or []:
        _apply_override(d, o)
    return _from_dict(TrainConfig, d)


def config_to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)