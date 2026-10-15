"""Convert released Maia-3 PyTorch checkpoints into Flax NNX state/models."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chessforge.config import ModelConfig
from chessforge.models import build_model


def load_torch_state_dict(path: str, trust: bool = False) -> dict[str, np.ndarray]:
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=not trust)
    sd = (
        ckpt["model_state_dict"]
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt
        else ckpt
    )
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "_orig_mod."):
            k = k.removeprefix(pre)
        out[k.replace("smolgen", "gab")] = v.detach().float().cpu().numpy()
    return out


def torch_state_to_pure_dict(sd: dict[str, np.ndarray], c: ModelConfig) -> dict:
    def lin(prefix, bias=True):
        d = {"kernel": sd[prefix + ".weight"].T}
        if bias and prefix + ".bias" in sd:
            d["bias"] = sd[prefix + ".bias"]
        return d

    def ln(prefix):
        return {"scale": sd[prefix + ".weight"], "bias": sd[prefix + ".bias"]}

    def norm(prefix):
        return {"scale": sd[prefix + ".weight"]} if c.use_rms_norm else ln(prefix)

    D = c.dim_vit
    p = {
        "elo_embedding_low": sd["elo_embedding_low.weight"][0],
        "elo_embedding_high": sd["elo_embedding_high.weight"][0],
        "token_projection": lin("token_projection"),
        "final_norm": ln("transformer.norm"),
        "last_ln": ln("last_ln"),
        "proj_sq_from": {"kernel": sd["proj_sq_from.weight"].T},
        "proj_sq_to": {"kernel": sd["proj_sq_to.weight"].T},
        "promo_bias_proj": {"kernel": sd["promo_bias_proj.weight"].T},
        "fc_value_hid": lin("fc_value_hid"),
        "fc_value": lin("fc_value"),
        "fc_ponder_hid": lin("fc_ponder_hid"),
        "fc_ponder": lin("fc_ponder"),
    }
    if c.pos_encoding == "gab":
        p["gab_shared_weight"] = sd["gab_shared_weight"]
    if c.pos_encoding == "absolute":
        p["abs_pe"] = sd["abs_pe.bias"]
    for i in range(c.num_blocks):
        t = f"transformer.layers.{i}"
        W = sd[f"{t}.self_attn.mha.in_proj_weight"]
        attn = {n: {"kernel": W[j * D : (j + 1) * D].T} for j, n in enumerate("qkv")}
        if f"{t}.self_attn.mha.in_proj_bias" in sd:
            bvec = sd[f"{t}.self_attn.mha.in_proj_bias"]
            for j, n in enumerate("qkv"):
                attn[n]["bias"] = bvec[j * D : (j + 1) * D]
        attn["out"] = lin(f"{t}.self_attn.mha.out_proj")
        blk = {
            "attn": attn,
            "linear1": lin(f"{t}.linear1"),
            "linear2": lin(f"{t}.linear2"),
            "norm1": norm(f"{t}.norm1"),
            "norm2": norm(f"{t}.norm2"),
        }
        if c.pos_encoding == "gab":
            g = {}
            if c.gab_per_square_dim > 0:
                g["sm1"] = lin(f"{t}.self_attn.sm1")
            g.update(
                sm2=lin(f"{t}.self_attn.sm2"),
                ln1=ln(f"{t}.self_attn.ln1"),
                sm3=lin(f"{t}.self_attn.sm3"),
                ln2=ln(f"{t}.self_attn.ln2"),
            )
            blk["gab"] = g
        if c.pos_encoding == "relative":
            blk["relative_bias"] = {"gate": sd[f"{t}.self_attn.relative_bias.gate"]}
        p[f"block_{i}"] = blk
    return p


def _path_keys(jax_path) -> tuple[str, ...]:
    keys = [
        str(k.key if hasattr(k, "key") else getattr(k, "name", k)) for k in jax_path
    ]
    if keys and keys[-1] == "value":
        keys = keys[:-1]
    return tuple(keys)


def _flatten_pure_dict(
    d: dict, prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], np.ndarray]:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flatten_pure_dict(v, prefix + (str(k),)))
        elif v is not None:
            out[prefix + (str(k),)] = np.asarray(v)
    return out


def convert_to_nnx_state(sd: dict[str, np.ndarray], c: ModelConfig) -> nnx.State:
    """Convert a PyTorch state_dict into an `nnx.State` matching `build_model(c)`."""
    pure = _flatten_pure_dict(torch_state_to_pure_dict(sd, c))
    model = build_model(c, rngs=0)
    _, state = nnx.split(model)
    ref_paths = {
        _path_keys(p): v.shape for p, v in jax.tree_util.tree_leaves_with_path(state)
    }
    if ref_paths.keys() != pure.keys():
        raise ValueError(
            f"Param tree mismatch.\nmissing: {sorted(ref_paths.keys() - pure.keys())}\n"
            f"extra: {sorted(pure.keys() - ref_paths.keys())}"
        )
    for k, s in ref_paths.items():
        if tuple(s) != tuple(pure[k].shape):
            raise ValueError(f"Shape mismatch at {'.'.join(k)}: {pure[k].shape} vs {s}")

    def _replace(path, leaf):
        arr = jnp.asarray(pure[_path_keys(path)], dtype=c.pdtype)
        return (
            leaf.replace(arr)
            if hasattr(leaf, "replace") and not isinstance(leaf, jax.Array)
            else arr
        )

    return jax.tree_util.tree_map_with_path(_replace, state)


def convert(sd: dict[str, np.ndarray], c: ModelConfig) -> nnx.Module:
    """Convert a PyTorch state_dict and return a ready-to-call `nnx.Module`."""
    model = build_model(c, rngs=0)
    graphdef, _ = nnx.split(model)
    return nnx.merge(graphdef, convert_to_nnx_state(sd, c))


def load_released(
    preset: str, compute_dtype: str = "float32"
) -> tuple[ModelConfig, nnx.State]:
    """Download + convert a released Maia-3 checkpoint. Returns (ModelConfig, nnx.State)."""
    from huggingface_hub import hf_hub_download

    from . import HF_CHECKPOINTS, build_model_config

    repo, fname = HF_CHECKPOINTS[preset]
    c = build_model_config(preset, {"compute_dtype": compute_dtype})
    path = hf_hub_download(repo_id=repo, filename=fname)
    return c, convert_to_nnx_state(load_torch_state_dict(path), c)
