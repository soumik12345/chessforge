"""Chessformer / Maia-3 in Flax linen. Parameter layout maps 1:1 onto maia3/models.py."""
from __future__ import annotations

import math

import flax.linen as nn
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


def torch_dense(features: int, fan_in: int, c: ModelConfig, name: str, use_bias: bool = True):
    """nn.Linear default init: U(-1/sqrt(fan_in), 1/sqrt(fan_in)) for weight and bias."""
    b = 1.0 / math.sqrt(fan_in)
    return nn.Dense(features, use_bias=use_bias, dtype=c.dtype, param_dtype=c.pdtype,
                    kernel_init=_uniform(b), bias_init=_uniform(b), name=name)


def layer_norm(c: ModelConfig, name: str):
    return nn.LayerNorm(epsilon=LN_EPS, dtype=jnp.float32, param_dtype=c.pdtype, name=name)


def block_norm(c: ModelConfig, name: str):
    if c.use_rms_norm:
        return nn.RMSNorm(epsilon=RMS_EPS, dtype=jnp.float32, param_dtype=c.pdtype, name=name)
    return layer_norm(c, name)


def _rel_index() -> np.ndarray:
    idx = np.zeros((64, 64), np.int32)
    for i in range(8):
        for j in range(8):
            for k in range(8):
                for l in range(8):
                    idx[i * 8 + j, k * 8 + l] = 15 * (i - k + 7) + (j - l + 7)
    return idx


REL_INDEX = _rel_index()


class RelativeBias(nn.Module):
    num_heads: int
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self):
        gate = self.param("gate", nn.initializers.zeros, (self.num_heads, 225), self.param_dtype)
        return gate[:, REL_INDEX].astype(jnp.float32)  # (H, 64, 64)


class GABGenerator(nn.Module):
    """Geometric Attention Bias: compressed board state -> mix of shared bias templates."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, gab_weight):
        c = self.cfg
        B, _, D = x.shape
        gelu = lambda t: nn.gelu(t, approximate=False)
        if c.gab_per_square_dim == 0:
            y = gelu(torch_dense(c.gab_intermediate_dim, D, c, "sm2")(x.mean(axis=1)))
        else:
            y = torch_dense(c.gab_per_square_dim, D, c, "sm1")(x).reshape(B, -1)
            y = gelu(torch_dense(c.gab_intermediate_dim, 64 * c.gab_per_square_dim, c, "sm2")(y))
        y = layer_norm(c, "ln1")(y)
        y = gelu(torch_dense(c.num_heads * c.gab_gen_size, c.gab_intermediate_dim, c, "sm3")(y))
        y = layer_norm(c, "ln2")(y).reshape(B, c.num_heads, c.gab_gen_size)
        b = jnp.einsum("bhg,og->bho", y.astype(c.dtype), gab_weight.astype(c.dtype),
                       preferred_element_type=jnp.float32)
        return b.reshape(B, c.num_heads, 64, 64)


class MultiHeadAttention(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, bias, deterministic: bool):
        c = self.cfg
        B, S, D = x.shape
        H = c.num_heads
        hd = D // H
        use_bias = not c.omit_qkv_biases
        in_bound = math.sqrt(6.0 / (D + 3 * D))  # xavier_uniform on torch's packed (3D, D) in_proj

        def proj(name, bound):
            return nn.Dense(D, use_bias=use_bias, dtype=c.dtype, param_dtype=c.pdtype,
                            kernel_init=_uniform(bound), bias_init=nn.initializers.zeros, name=name)

        q = proj("q", in_bound)(x).reshape(B, S, H, hd)
        k = proj("k", in_bound)(x).reshape(B, S, H, hd)
        v = proj("v", in_bound)(x).reshape(B, S, H, hd)
        logits = jnp.einsum("bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32) / math.sqrt(hd)
        if bias is not None:
            logits = logits + bias
        w = jax.nn.softmax(logits, axis=-1)
        w = nn.Dropout(c.dropout)(w, deterministic=deterministic)
        out = jnp.einsum("bhqk,bkhd->bqhd", w.astype(c.dtype), v).reshape(B, S, D)
        return proj("out", 1.0 / math.sqrt(D))(out)


class EncoderBlock(nn.Module):
    """Post-norm transformer block (maia3 EncoderOnlyBlock)."""
    cfg: ModelConfig

    @nn.compact
    def __call__(self, x, gab_weight, deterministic: bool):
        c = self.cfg
        bias = None
        if c.pos_encoding == "gab":
            bias = GABGenerator(c, name="gab")(x, gab_weight)
        elif c.pos_encoding == "relative":
            bias = RelativeBias(c.num_heads, c.pdtype, name="relative_bias")()[None]
        a = MultiHeadAttention(c, name="attn")(x, bias, deterministic)
        a = nn.Dropout(c.dropout)(a, deterministic=deterministic)
        x = block_norm(c, "norm1")(x + a)  # residual stream kept in fp32
        hidden = int(c.dim_vit * c.mlp_ratio)
        act = (lambda t: nn.gelu(t, approximate=False)) if c.activation == "gelu" else nn.relu
        h = act(torch_dense(hidden, c.dim_vit, c, "linear1")(x))
        h = nn.Dropout(c.dropout)(h, deterministic=deterministic)
        h = torch_dense(c.dim_vit, hidden, c, "linear2")(h)
        h = nn.Dropout(c.dropout)(h, deterministic=deterministic)
        return block_norm(c, "norm2")(x + h)


class Chessformer(nn.Module):
    cfg: ModelConfig

    @nn.compact
    def __call__(self, boards, self_elo, oppo_elo, deterministic: bool = True):
        """boards: int8 (B, history, 64) relative piece codes (0 empty, 1..6 own, 7..12 opp).
        Returns logits_move (B, 4352), logits_value (B, 3) [loss, draw, win], ponder (B,)."""
        c = self.cfg
        cdt = c.dtype
        B, H, S = boards.shape
        assert H == c.history and S == 64, (boards.shape, c.history)

        onehot = (boards[..., None] == jnp.arange(1, 13, dtype=boards.dtype)).astype(cdt)
        tokens = onehot.transpose(0, 2, 1, 3).reshape(B, 64, H * 12)  # oldest board first

        std = math.sqrt(2.0 / (c.dim_emb + 1))  # xavier_normal on nn.Embedding(1, dim_emb)
        e_low = self.param("elo_embedding_low", _normal(std), (c.dim_emb,), c.pdtype)
        e_high = self.param("elo_embedding_high", _normal(std), (c.dim_emb,), c.pdtype)

        def interp(elo):
            elo = jnp.clip(elo.astype(jnp.float32), 0.0, c.elo_max)
            w = (elo / c.elo_max)[:, None]
            emb = w * e_low.astype(jnp.float32) + (1.0 - w) * e_high.astype(jnp.float32)
            return jnp.broadcast_to(emb[:, None, :], (B, 64, c.dim_emb)).astype(cdt)

        x = jnp.concatenate([tokens, interp(self_elo), interp(oppo_elo)], axis=-1)
        x = torch_dense(c.dim_vit, x.shape[-1], c, "token_projection")(x)

        if c.pos_encoding == "absolute":
            x = x + self.param("abs_pe", nn.initializers.zeros, (64, c.dim_vit), c.pdtype).astype(cdt)
        gab_w = None
        if c.pos_encoding == "gab":
            gstd = math.sqrt(2.0 / (64 * 64 + c.gab_gen_size))
            gab_w = self.param("gab_shared_weight", _normal(gstd), (64 * 64, c.gab_gen_size), c.pdtype)

        for i in range(c.num_blocks):
            x = EncoderBlock(c, name=f"block_{i}")(x, gab_w, deterministic)
        x = layer_norm(c, "final_norm")(x)

        # attention-style from-to policy head with promotion biases
        hh = c.head_hid_dim
        sq_from = torch_dense(hh, c.dim_vit, c, "proj_sq_from", use_bias=False)(x)
        sq_to = torch_dense(hh, c.dim_vit, c, "proj_sq_to", use_bias=False)(x)
        scores = jnp.einsum("bid,bjd->bij", sq_from, sq_to, preferred_element_type=jnp.float32) / math.sqrt(hh)
        promo_bias = torch_dense(4, hh, c, "promo_bias_proj", use_bias=False)(sq_to[:, 56:64])
        promo_bias = promo_bias.astype(jnp.float32) * math.sqrt(hh)                 # (B, 8 to-file, 4)
        base = scores[:, 48:56, 56:64]                                               # rank7 -> rank8
        promo = (base[..., None] + promo_bias[:, None, :, :]).reshape(B, 256)        # from, to, qrbn
        logits_move = jnp.concatenate([scores.reshape(B, 64 * 64), promo], axis=-1)
        assert logits_move.shape[-1] == VOCAB_SIZE

        pooled = layer_norm(c, "last_ln")(x.mean(axis=1))
        v = nn.relu(torch_dense(hh, c.dim_vit, c, "fc_value_hid")(pooled))
        logits_value = torch_dense(3, hh, c, "fc_value")(v).astype(jnp.float32)
        p = nn.relu(torch_dense(hh, c.dim_vit, c, "fc_ponder_hid")(pooled))
        ponder = torch_dense(1, hh, c, "fc_ponder")(p).astype(jnp.float32)[:, 0]
        return logits_move, logits_value, ponder


def dummy_inputs(c: ModelConfig, batch: int = 2):
    return (jnp.zeros((batch, c.history, 64), jnp.int8),
            jnp.full((batch,), 1500.0, jnp.float32),
            jnp.full((batch,), 1500.0, jnp.float32))