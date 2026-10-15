"""Model garden: named presets + NNX architecture registry + released-checkpoint index."""

from __future__ import annotations

import jax
import numpy as np
from flax import nnx

from chessforge.config import ModelConfig
from chessforge.models.chessformer import Chessformer

_ARCHS = {"chessformer": Chessformer}
_PRESETS: dict[str, dict] = {}

HF_CHECKPOINTS = {
    "maia3-3m": ("UofTCSSLab/Maia3-ablate-3M", "maia3-3m.pt"),
    "maia3-5m": ("UofTCSSLab/Maia3-5M", "maia3-5m.pt"),
    "maia3-23m": ("UofTCSSLab/Maia3-23M", "maia3-23m.pt"),
    "maia3-79m": ("UofTCSSLab/Maia3-79M", "maia3-79m.pt"),
}


def register_architecture(name: str, module_cls):
    _ARCHS[name] = module_cls


def register_preset(name: str, **kwargs):
    _PRESETS[name] = dict(name=name, **kwargs)


register_preset(
    "maia3-3m",
    dim_vit=192,
    head_hid_dim=192,
    num_heads=6,
    gab_gen_size=64,
    gab_per_square_dim=0,
    gab_intermediate_dim=64,
)
register_preset(
    "maia3-5m",
    dim_vit=256,
    head_hid_dim=256,
    num_heads=8,
    gab_gen_size=64,
    gab_per_square_dim=0,
    gab_intermediate_dim=64,
)
register_preset(
    "maia3-23m",
    dim_vit=512,
    head_hid_dim=512,
    num_heads=16,
    gab_gen_size=128,
    gab_per_square_dim=32,
    gab_intermediate_dim=128,
)
register_preset(
    "maia3-79m",
    dim_vit=1024,
    head_hid_dim=1024,
    num_heads=32,
    gab_gen_size=128,
    gab_per_square_dim=32,
    gab_intermediate_dim=128,
)
register_preset(
    "maia3-5m-absolute",
    **{**_PRESETS["maia3-5m"], "name": "maia3-5m-absolute", "pos_encoding": "absolute"},
)
register_preset(
    "maia3-5m-relative",
    **{**_PRESETS["maia3-5m"], "name": "maia3-5m-relative", "pos_encoding": "relative"},
)
register_preset(
    "maia3-5m-hist1", **{**_PRESETS["maia3-5m"], "name": "maia3-5m-hist1", "history": 1}
)
register_preset(
    "maia3-5m-hist32",
    **{**_PRESETS["maia3-5m"], "name": "maia3-5m-hist32", "history": 32},
)


def list_presets() -> list[str]:
    return sorted(_PRESETS)


def build_model_config(name: str, overrides: dict | None = None) -> ModelConfig:
    if name not in _PRESETS:
        raise KeyError(f"Unknown preset {name!r}; available: {list_presets()}")
    kw = dict(_PRESETS[name])
    kw.update(overrides or {})
    return ModelConfig(**kw)


def build_model(cfg: ModelConfig, *, rngs: nnx.Rngs | int = 0) -> nnx.Module:
    if isinstance(rngs, int):
        rngs = nnx.Rngs(rngs)
    return _ARCHS[cfg.arch](cfg, rngs=rngs)


def count_params(cfg: ModelConfig) -> int:
    def _make():
        _, state = nnx.split(build_model(cfg, rngs=0))
        return state

    shapes = jax.eval_shape(_make)
    return int(sum(np.prod(x.shape) for x in jax.tree.leaves(shapes)))
