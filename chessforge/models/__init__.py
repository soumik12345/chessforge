"""Model garden: Maia-3 v1 + ChessformerV2 presets & unified forward helper."""

from __future__ import annotations

import jax
import numpy as np
from flax import nnx

from chessforge.config import ModelConfig
from chessforge.models.chessformer import Chessformer
from chessforge.models.chessformer_v2 import ChessformerV2

_ARCHS = {
    "chessformer": Chessformer,
    "chessformer_v2": ChessformerV2,
}
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


# Official Maia-3 v1 presets (exact 1:1 reproduction)
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
_maia3_5m_options = {
    key: value for key, value in _PRESETS["maia3-5m"].items() if key != "name"
}
register_preset("maia3-5m-absolute", **_maia3_5m_options, pos_encoding="absolute")
register_preset("maia3-5m-relative", **_maia3_5m_options, pos_encoding="relative")
register_preset("maia3-5m-hist1", **_maia3_5m_options, history=1)
register_preset("maia3-5m-hist32", **_maia3_5m_options, history=32)

# ChessformerV2 presets (deep Pre-Norm + Ray-GAB + Registers + Official-Title AdaLN)
register_preset(
    "chessforge-v2-5m",
    arch="chessformer_v2",
    dim_vit=256,
    head_hid_dim=256,
    num_blocks=8,
    num_heads=8,
    mlp_ratio=2.67,
    gab_gen_size=64,
    gab_per_square_dim=16,
    gab_intermediate_dim=64,
)
register_preset(
    "chessforge-v2-12m",
    arch="chessformer_v2",
    dim_vit=320,
    head_hid_dim=320,
    num_blocks=12,
    num_heads=10,
    mlp_ratio=2.67,
    gab_gen_size=64,
    gab_per_square_dim=16,
    gab_intermediate_dim=96,
)
register_preset(
    "chessforge-v2-28m",
    arch="chessformer_v2",
    dim_vit=384,
    head_hid_dim=384,
    num_blocks=16,
    num_heads=12,
    mlp_ratio=2.67,
    gab_gen_size=96,
    gab_per_square_dim=24,
    gab_intermediate_dim=128,
)
register_preset(
    "chessforge-v2-60m",
    arch="chessformer_v2",
    dim_vit=512,
    head_hid_dim=512,
    num_blocks=20,
    num_heads=16,
    mlp_ratio=2.67,
    gab_gen_size=128,
    gab_per_square_dim=32,
    gab_intermediate_dim=128,
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


def forward_model(
    model: nnx.Module,
    batch: dict,
    *,
    deterministic: bool = True,
    rng: jax.Array | None = None,
    return_aux: bool = False,
    mask_legal: bool = True,
):
    """Unified forward call for both Chessformer (v1) and ChessformerV2."""
    if isinstance(model, ChessformerV2):
        return model(
            batch["board"],
            batch["self_elo"],
            batch["oppo_elo"],
            rich_time=batch.get("rich_time"),
            has_clock=batch.get("has_clock"),
            board_meta=batch.get("board_meta"),
            ep_square_idx=batch.get("ep_square_idx"),
            self_title=batch.get("self_title"),
            oppo_title=batch.get("oppo_title"),
            game_format=batch.get("game_format"),
            clk_left=batch.get("clk_left"),
            legal=batch.get("legal") if mask_legal else None,
            deterministic=deterministic,
            rng=rng,
            return_aux=return_aux,
        )
    lm, lv, lp = model(
        batch["board"],
        batch["self_elo"],
        batch["oppo_elo"],
        time_info=batch.get("time_info"),
        deterministic=deterministic,
        rng=rng,
    )
    if return_aux:
        return lm, lv, lp, {}
    return lm, lv, lp


def count_params(cfg: ModelConfig) -> int:
    def _make():
        _, state = nnx.split(build_model(cfg, rngs=0))
        return state

    shapes = jax.eval_shape(_make)
    return int(sum(np.prod(x.shape) for x in jax.tree.leaves(shapes)))
