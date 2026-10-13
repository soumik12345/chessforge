"""Convert released Maia-3 PyTorch checkpoints into Flax params for this implementation."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..config import ModelConfig
from . import build_model
from .chessformer import dummy_inputs


def load_torch_state_dict(path: str, trust: bool = False) -> dict[str, np.ndarray]:
    import torch
    ckpt = torch.load(path, map_location="cpu", weights_only=not trust)
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    out = {}
    for k, v in sd.items():
        for pre in ("module.", "_orig_mod."):
            k = k.removeprefix(pre)
        out[k.replace("smolgen", "gab")] = v.detach().float().cpu().numpy()
    return out


def torch_state_to_flax(sd: dict[str, np.ndarray], c: ModelConfig) -> dict:
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
        attn = {n: {"kernel": W[j * D:(j + 1) * D].T} for j, n in enumerate("qkv")}
        if f"{t}.self_attn.mha.in_proj_bias" in sd:
            bvec = sd[f"{t}.self_attn.mha.in_proj_bias"]
            for j, n in enumerate("qkv"):
                attn[n]["bias"] = bvec[j * D:(j + 1) * D]
        attn["out"] = lin(f"{t}.self_attn.mha.out_proj")
        blk = {"attn": attn, "linear1": lin(f"{t}.linear1"), "linear2": lin(f"{t}.linear2"),
               "norm1": norm(f"{t}.norm1"), "norm2": norm(f"{t}.norm2")}
        if c.pos_encoding == "gab":
            g = {}
            if c.gab_per_square_dim > 0:
                g["sm1"] = lin(f"{t}.self_attn.sm1")
            g.update(sm2=lin(f"{t}.self_attn.sm2"), ln1=ln(f"{t}.self_attn.ln1"),
                     sm3=lin(f"{t}.self_attn.sm3"), ln2=ln(f"{t}.self_attn.ln2"))
            blk["gab"] = g
        if c.pos_encoding == "relative":
            blk["relative_bias"] = {"gate": sd[f"{t}.self_attn.relative_bias.gate"]}
        p[f"block_{i}"] = blk
    return p


def convert(sd: dict[str, np.ndarray], c: ModelConfig) -> dict:
    params = torch_state_to_flax(sd, c)
    model = build_model(c)
    ref = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), *dummy_inputs(c)))["params"]
    ref_paths = {jax.tree_util.keystr(k): v.shape for k, v in jax.tree_util.tree_leaves_with_path(ref)}
    got_paths = {jax.tree_util.keystr(k): np.shape(v) for k, v in jax.tree_util.tree_leaves_with_path(params)}
    if ref_paths.keys() != got_paths.keys():
        raise ValueError(f"Param tree mismatch.\nmissing: {sorted(ref_paths.keys() - got_paths.keys())}\n"
                         f"extra: {sorted(got_paths.keys() - ref_paths.keys())}")
    for k, s in ref_paths.items():
        if tuple(s) != tuple(got_paths[k]):
            raise ValueError(f"Shape mismatch at {k}: {got_paths[k]} vs {s}")
    return jax.tree.map(lambda x: jnp.asarray(x, c.pdtype), params)


def load_released(preset: str, compute_dtype: str = "float32"):
    """Download + convert a released Maia-3 checkpoint. Returns (ModelConfig, params)."""
    from huggingface_hub import hf_hub_download
    from . import HF_CHECKPOINTS, build_model_config
    repo, fname = HF_CHECKPOINTS[preset]
    c = build_model_config(preset, {"compute_dtype": compute_dtype})
    path = hf_hub_download(repo_id=repo, filename=fname)
    return c, convert(load_torch_state_dict(path), c)