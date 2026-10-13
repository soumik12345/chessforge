"""Chessformer / Maia-3 in Flax NNX. Parameter layout maps 1:1 onto maia3/models.py."""
from __future__ import annotations

import math

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from ..chess_utils import VOCAB_SIZE
from ..config import ModelConfig

LN_EPS = 1e-5
RMS_EPS = float(np.finfo(np.float32).eps)  # torch.nn.RMSNorm(eps=None) default


def _uniform(bound: float):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.uniform(key, shape, dtype, -bound, bound)
    return init


def _normal(std: float):
    def init(key, shape, dtype=jnp.float32):
        return std * jax.random.normal(key, shape, dtype)
    return init


def torch_linear(in_features: int, out_features: int, c: ModelConfig, *, rngs: nnx.Rngs,
                 use_bias: bool = True, bound: float | None = None, zero_bias: bool = False) -> nnx.Linear:
    """nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias."""
    b = (1.0 / math.sqrt(in_features)) if bound is None else bound
    b_init = nnx.initializers.zeros if zero_bias else _uniform(1.0 / math.sqrt(in_features))
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=use_bias,
        dtype=c.dtype,
        param_dtype=c.pdtype,
        kernel_init=_uniform(b),
        bias_init=b_init,
        rngs=rngs,
    )


def layer_norm(features: int, c: ModelConfig, *, rngs: nnx.Rngs) -> nnx.LayerNorm:
    return nnx.LayerNorm(features, epsilon=LN_EPS, dtype=jnp.float32, param_dtype=c.pdtype, rngs=rngs)


def block_norm(features: int, c: ModelConfig, *, rngs: nnx.Rngs) -> nnx.Module:
    if c.use_rms_norm:
        return nnx.RMSNorm(features, epsilon=RMS_EPS, dtype=jnp.float32, param_dtype=c.pdtype, rngs=rngs)
    return layer_norm(features, c, rngs=rngs)


def _dropout(x: jax.Array, rate: float, deterministic: bool, rng: jax.Array | None) -> jax.Array:
    if deterministic or rate <= 0.0:
        return x
    if rng is None:
        raise ValueError("Dropout requires an RNG key when deterministic=False and dropout > 0.")
    keep = 1.0 - rate
    mask = jax.random.bernoulli(rng, p=keep, shape=x.shape)
    return jnp.where(mask, x / keep, 0.0)


def _rel_index() -> np.ndarray:
    idx = np.zeros((64, 64), np.int32)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    idx[i * 8 + j, k * 8 + l] = 15 * (i - k + 7) + (j - l + 7)
    return idx


REL_INDEX = _rel_index()


class RelativeBias(nnx.Module):
    def __init__(self, num_heads: int, param_dtype: jnp.dtype = jnp.float32):
        self.num_heads = num_heads
        self.gate = nnx.Param(jnp.zeros((num_heads, 225), dtype=param_dtype))

    def __call__(self) -> jax.Array:
        return self.gate.value[:, REL_INDEX].astype(jnp.float32)  # (H, 64, 64)


class GABGenerator(nnx.Module):
    """Geometric Attention Bias: compressed board state -> mix of shared bias templates."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.dim_vit
        if cfg.gab_per_square_dim == 0:
            self.sm1 = None
            self.sm2 = torch_linear(D, cfg.gab_intermediate_dim, cfg, rngs=rngs)
        else:
            self.sm1 = torch_linear(D, cfg.gab_per_square_dim, cfg, rngs=rngs)
            self.sm2 = torch_linear(64 * cfg.gab_per_square_dim, cfg.gab_intermediate_dim, cfg, rngs=rngs)
        self.ln1 = layer_norm(cfg.gab_intermediate_dim, cfg, rngs=rngs)
        self.sm3 = torch_linear(cfg.gab_intermediate_dim, cfg.num_heads * cfg.gab_gen_size, cfg, rngs=rngs)
        self.ln2 = layer_norm(cfg.num_heads * cfg.gab_gen_size, cfg, rngs=rngs)

    def __call__(self, x: jax.Array, gab_weight: jax.Array) -> jax.Array:
        c = self.cfg
        B = x.shape[0]
        gelu = lambda t: nnx.gelu(t, approximate=False)
        if self.sm1 is None:
            y = gelu(self.sm2(x.mean(axis=1)))
        else:
            y = self.sm1(x).reshape(B, -1)
            y = gelu(self.sm2(y))
        y = self.ln1(y)
        y = gelu(self.sm3(y))
        y = self.ln2(y).reshape(B, c.num_heads, c.gab_gen_size)
        b = jnp.einsum("bhg,og->bho", y.astype(c.dtype), gab_weight.astype(c.dtype),
                       preferred_element_type=jnp.float32)
        return b.reshape(B, c.num_heads, 64, 64)


class MultiHeadAttention(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.dim_vit
        use_bias = not cfg.omit_qkv_biases
        in_bound = math.sqrt(6.0 / (D + 3 * D))  # xavier_uniform on torch's packed (3D, D) in_proj
        self.q = torch_linear(D, D, cfg, rngs=rngs, use_bias=use_bias, bound=in_bound, zero_bias=True)
        self.k = torch_linear(D, D, cfg, rngs=rngs, use_bias=use_bias, bound=in_bound, zero_bias=True)
        self.v = torch_linear(D, D, cfg, rngs=rngs, use_bias=use_bias, bound=in_bound, zero_bias=True)
        self.out = torch_linear(D, D, cfg, rngs=rngs, use_bias=use_bias, bound=1.0 / math.sqrt(D), zero_bias=True)

    def __call__(self, x: jax.Array, bias: jax.Array | None, deterministic: bool,
                 rng: jax.Array | None = None) -> jax.Array:
        c = self.cfg
        B, S, D = x.shape
        H = c.num_heads
        hd = D // H
        q = self.q(x).reshape(B, S, H, hd)
        k = self.k(x).reshape(B, S, H, hd)
        v = self.v(x).reshape(B, S, H, hd)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / math.sqrt(hd)
        if bias is not None:
            logits = logits + bias
        w = jax.nn.softmax(logits, axis=-1)
        w = _dropout(w, c.dropout, deterministic, rng)
        out = jnp.einsum("bhqk,bkhd->bqhd", w.astype(c.dtype), v).reshape(B, S, D)
        return self.out(out)


class EncoderBlock(nnx.Module):
    """Post-norm transformer block (maia3 EncoderOnlyBlock)."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.dim_vit
        hidden = int(D * cfg.mlp_ratio)
        if cfg.pos_encoding == "gab":
            self.gab = GABGenerator(cfg, rngs=rngs)
        elif cfg.pos_encoding == "relative":
            self.relative_bias = RelativeBias(cfg.num_heads, cfg.pdtype)
        self.attn = MultiHeadAttention(cfg, rngs=rngs)
        self.norm1 = block_norm(D, cfg, rngs=rngs)
        self.linear1 = torch_linear(D, hidden, cfg, rngs=rngs)
        self.linear2 = torch_linear(hidden, D, cfg, rngs=rngs)
        self.norm2 = block_norm(D, cfg, rngs=rngs)

    def __call__(self, x: jax.Array, gab_weight: jax.Array | None, deterministic: bool,
                 rng: jax.Array | None = None) -> jax.Array:
        c = self.cfg
        r_attn, r_d1, r_mlp, r_d2 = (None, None, None, None) if rng is None else jax.random.split(rng, 4)
        bias = None
        if c.pos_encoding == "gab":
            bias = self.gab(x, gab_weight)
        elif c.pos_encoding == "relative":
            bias = self.relative_bias()[None]
        a = self.attn(x, bias, deterministic, r_attn)
        a = _dropout(a, c.dropout, deterministic, r_d1)
        x = self.norm1(x + a.astype(jnp.float32))
        act = (lambda t: nnx.gelu(t, approximate=False)) if c.activation == "gelu" else nnx.relu
        h = act(self.linear1(x))
        h = _dropout(h, c.dropout, deterministic, r_mlp)
        h = self.linear2(h)
        h = _dropout(h, c.dropout, deterministic, r_d2)
        return self.norm2(x + h.astype(jnp.float32))


class Chessformer(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D, hh = cfg.dim_vit, cfg.head_hid_dim
        std = math.sqrt(2.0 / (cfg.dim_emb + 1))  # xavier_normal on nn.Embedding(1, dim_emb)
        self.elo_embedding_low = nnx.Param(_normal(std)(rngs.params(), (cfg.dim_emb,), cfg.pdtype))
        self.elo_embedding_high = nnx.Param(_normal(std)(rngs.params(), (cfg.dim_emb,), cfg.pdtype))

        extra_time = 3 if cfg.include_time_info else 0
        in_dim = 12 * cfg.history + extra_time + 2 * cfg.dim_emb
        self.token_projection = torch_linear(in_dim, D, cfg, rngs=rngs)

        if cfg.pos_encoding == "absolute":
            self.abs_pe = nnx.Param(jnp.zeros((64, D), dtype=cfg.pdtype))
        if cfg.pos_encoding == "gab":
            gstd = math.sqrt(2.0 / (64 * 64 + cfg.gab_gen_size))
            self.gab_shared_weight = nnx.Param(
                _normal(gstd)(rngs.params(), (64 * 64, cfg.gab_gen_size), cfg.pdtype))

        for i in range(cfg.num_blocks):
            setattr(self, f"block_{i}", EncoderBlock(cfg, rngs=rngs))
        self.final_norm = layer_norm(D, cfg, rngs=rngs)
        self.last_ln = layer_norm(D, cfg, rngs=rngs)

        self.proj_sq_from = torch_linear(D, hh, cfg, rngs=rngs, use_bias=False)
        self.proj_sq_to = torch_linear(D, hh, cfg, rngs=rngs, use_bias=False)
        self.promo_bias_proj = torch_linear(hh, 4, cfg, rngs=rngs, use_bias=False)

        self.fc_value_hid = torch_linear(D, hh, cfg, rngs=rngs)
        self.fc_value = torch_linear(hh, 3, cfg, rngs=rngs)
        self.fc_ponder_hid = torch_linear(D, hh, cfg, rngs=rngs)
        self.fc_ponder = torch_linear(hh, 1, cfg, rngs=rngs)

    def __call__(
        self,
        boards: jax.Array,
        self_elo: jax.Array,
        oppo_elo: jax.Array,
        *,
        time_info: jax.Array | None = None,
        deterministic: bool = True,
        rng: jax.Array | None = None,
    ):
        """boards: int8 (B, history, 64) relative piece codes (0 empty, 1..6 own, 7..12 opp).
        Returns logits_move (B, 4352), logits_value (B, 3) [loss, draw, win], ponder (B,)."""
        c = self.cfg
        cdt = c.dtype
        B, H, S = boards.shape
        assert H == c.history and S == 64, (boards.shape, c.history)

        onehot = (boards[..., None] == jnp.arange(1, 13, dtype=boards.dtype)).astype(cdt)
        tokens = onehot.transpose(0, 2, 1, 3).reshape(B, 64, H * 12)  # oldest board first

        if c.include_time_info:
            if time_info is None:
                t_feat = jnp.zeros((B, 64, 3), dtype=cdt)
            else:
                t_feat = jnp.broadcast_to(time_info[:, None, :].astype(cdt), (B, 64, 3))
            tokens = jnp.concatenate([tokens, t_feat], axis=-1)

        e_low = self.elo_embedding_low.value.astype(jnp.float32)
        e_high = self.elo_embedding_high.value.astype(jnp.float32)

        def interp(elo):
            elo = jnp.clip(elo.astype(jnp.float32), 0.0, c.elo_max)
            w = (elo / c.elo_max)[:, None]
            emb = w * e_low + (1.0 - w) * e_high
            return jnp.broadcast_to(emb[:, None, :], (B, 64, c.dim_emb)).astype(cdt)

        x = jnp.concatenate([tokens, interp(self_elo), interp(oppo_elo)], axis=-1)
        x = self.token_projection(x).astype(jnp.float32)  # keep residual stream in fp32

        if c.pos_encoding == "absolute":
            x = x + self.abs_pe.value.astype(jnp.float32)
        gab_w = self.gab_shared_weight.value if c.pos_encoding == "gab" else None

        blk_rngs = [None] * c.num_blocks if rng is None else list(jax.random.split(rng, c.num_blocks))
        for i in range(c.num_blocks):
            blk: EncoderBlock = getattr(self, f"block_{i}")
            x = blk(x, gab_w, deterministic, blk_rngs[i])
        x = self.final_norm(x)

        hh = c.head_hid_dim
        sq_from = self.proj_sq_from(x)
        sq_to = self.proj_sq_to(x)
        scores = jnp.einsum("bid,bjd->bij", sq_from, sq_to, preferred_element_type=jnp.float32) / math.sqrt(hh)
        promo_bias = self.promo_bias_proj(sq_to[:, 56:64]).astype(jnp.float32) * math.sqrt(hh)  # (B, 8, 4)
        base = scores[:, 48:56, 56:64]                                                           # rank7 -> rank8
        promo = (base[..., None] + promo_bias[:, None, :, :]).reshape(B, 256)
        logits_move = jnp.concatenate([scores.reshape(B, 64 * 64), promo], axis=-1)
        assert logits_move.shape[-1] == VOCAB_SIZE

        pooled = self.last_ln(x.mean(axis=1))
        v = nnx.relu(self.fc_value_hid(pooled))
        logits_value = self.fc_value(v).astype(jnp.float32)
        p = nnx.relu(self.fc_ponder_hid(pooled))
        ponder = self.fc_ponder(p).astype(jnp.float32)[:, 0]
        return logits_move, logits_value, ponder


def dummy_inputs(c: ModelConfig, batch: int = 2):
    return (jnp.zeros((batch, c.history, 64), jnp.int8),
            jnp.full((batch,), 1500.0, jnp.float32),
            jnp.full((batch,), 1500.0, jnp.float32))