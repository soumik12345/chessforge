from __future__ import annotations

import dataclasses
import typing
from dataclasses import dataclass, field
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    """Architecture config supporting both Maia-3 v1 and ChessformerV2."""

    name: str = "maia3-5m"
    arch: str = "chessformer"  # chessformer | chessformer_v2
    history: int = 8  # boards incl. current (paper's n = history - 1)
    include_time_info: bool = False
    dim_emb: int = 128  # Elo / context embedding dim
    dim_vit: int = 256
    num_blocks: int = 8
    num_heads: int = 8
    mlp_ratio: float = 2.0
    dropout: float = 0.0
    head_hid_dim: int = 256
    pos_encoding: str = "gab"  # gab | relative | absolute
    gab_gen_size: int = 64
    gab_per_square_dim: int = 0  # 0 => average pooling variant (3M/5M v1)
    gab_intermediate_dim: int = 64
    use_rms_norm: bool = True
    omit_qkv_biases: bool = True
    activation: str = "gelu"
    elo_max: float = 5000.0
    compute_dtype: str = "bfloat16"
    param_dtype: str = "float32"

    # ChessformerV2 extensions (ignored by v1 Chessformer)
    num_registers: int = 4  # [CLS_VALUE, CLS_PONDER, REG_PLAN_1, REG_PLAN_2]
    use_swiglu: bool = True
    use_qk_norm: bool = True
    use_adaln: bool = True
    use_ray_gab: bool = True
    use_attn_policy_residual: bool = True
    use_opponent_reply_head: bool = True
    ponder_buckets: int = 16
    use_joint_move_time: bool = False
    joint_move_time_rank: int = 16  # rank of the effort-conditioned policy residual
    elo_num_basis: int = 16
    remat_blocks: bool = False  # gradient checkpointing for v5e-1 / 16GB GPUs

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
    anneal_glob: str = ""  # optional pure-titled/OTB mix used after anneal_start_frac
    anneal_start_frac: float = 0.80
    val_glob: str = ""
    test_glob: str = ""  # ALLIE / Titled / OTB / Anti-SF eval shards
    global_batch_size: int = 1024  # positions per optimizer step (all hosts)
    eval_batch_size: int = 2048
    history_mask_prob: float = 0.05
    shuffle_seed: int = 0
    num_workers: int = 16  # grain multiprocessing workers per host
    read_threads: int = 16
    prefetch_buffer_size: int = 256
    local_cache_dir: str = ""  # stage gs:// shards to local disk if set
    val_batches: int = 50


@dataclass
class OptimConfig:
    lr: float = 5e-5
    min_lr: float = 1e-5
    weight_decay: float = 1e-6
    grad_clip_norm: float = 3.5
    warmup_steps: int = 1000
    cosine_cycle_steps: int = 50_000
    schedule: str = "cyclic_cosine"  # cyclic_cosine | wsd | cosine
    decay_frac: float = 0.20  # used when schedule == "wsd"
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
    joint_time_coef: float = (
        0.0  # posterior effort CE; 1 gives joint NLL with unsmoothed policy
    )
    reply_coef: float = 0.0  # 2-ply human opponent reply auxiliary loss
    label_smoothing: float = 0.0
    dpo_coef: float = 0.0  # Anti-Engine DPO / contrastive loss weight
    dpo_beta: float = 0.2
    sf_max_cp_loss: float = (
        35.0  # only prefer human over SF when human move loses <= N cp
    )
    init_from_checkpoint: str = (
        ""  # initialize weights (and DPO ref_params) from checkpoint dir
    )
    init_from_step: int | None = None
    mask_illegal_in_loss: bool = True
    seed: int = 42
    workdir: str = "/tmp/maia3_run"
    log_every: int = 100
    eval_every: int = 10_000
    test_every: int = 50_000
    ckpt_every: int = 10_000
    keep_checkpoints: int = 3
    trackio_project: str = ""  # empty disables experiment tracking
    trackio_run_name: str | None = None  # falls back to the model name
    trackio_group: str | None = None  # group related runs, e.g. ablation seeds
    trackio_space_id: str | None = None  # optional Hugging Face Space


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


def load_train_config(
    path: str | None, overrides: list[str] | None = None
) -> TrainConfig:
    d: dict = {}
    if path:
        with open(path) as fh:
            d = yaml.safe_load(fh) or {}
    for o in overrides or []:
        _apply_override(d, o)
    return _from_dict(TrainConfig, d)


def config_to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)
