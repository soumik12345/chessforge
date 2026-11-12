"""ChessformerV2: Deep Pre-Norm + Zero-Copy Ray-GAB + Cross-Attn Pool + Official-Title AdaLN."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from chessforge import chess_utils as cu
from chessforge.config import ModelConfig
from chessforge.models.chessformer import (
    GABGenerator,
    RelativeBias,
    _dropout,
    _normal,
    layer_norm,
    torch_linear,
)

RMS_EPS = float(np.finfo(np.float32).eps)
ADALN_BOTTLENECK = 128


def rms_norm(features: int, c: ModelConfig, *, rngs: nnx.Rngs) -> nnx.RMSNorm:
    return nnx.RMSNorm(
        features, epsilon=RMS_EPS, dtype=jnp.float32, param_dtype=c.pdtype, rngs=rngs
    )


def zero_linear(in_features: int, out_features: int, c: ModelConfig, *, rngs: nnx.Rngs):
    return nnx.Linear(
        in_features,
        out_features,
        use_bias=True,
        dtype=c.dtype,
        param_dtype=c.pdtype,
        kernel_init=nnx.initializers.zeros,
        bias_init=nnx.initializers.zeros,
        rngs=rngs,
    )


class ContextConditioner(nnx.Module):
    """Non-linear RBF Elo basis + rating gap + official title + format + clock + board meta."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.dim_vit
        tdim = 32
        self.title_emb = nnx.Embed(
            cu.NUM_TITLE_TIERS, tdim, dtype=cfg.dtype, param_dtype=cfg.pdtype, rngs=rngs
        )
        self.format_emb = nnx.Embed(
            cu.NUM_GAME_FORMATS,
            tdim,
            dtype=cfg.dtype,
            param_dtype=cfg.pdtype,
            rngs=rngs,
        )
        in_dim = 2 * cfg.elo_num_basis + 3 + 3 * tdim + 8 + 7
        self.fc1 = torch_linear(in_dim, D, cfg, rngs=rngs)
        self.fc2 = torch_linear(D, D, cfg, rngs=rngs)
        self.adaln_bottleneck = torch_linear(D, ADALN_BOTTLENECK, cfg, rngs=rngs)

    def _rbf_elo(self, elo: jax.Array) -> jax.Array:
        K = self.cfg.elo_num_basis
        centers = jnp.linspace(800.0, 2900.0, K, dtype=jnp.float32)
        sigma = 160.0
        e = jnp.clip(elo.astype(jnp.float32), 400.0, 3300.0)[:, None]
        return jnp.exp(-0.5 * ((e - centers[None, :]) / sigma) ** 2)

    def __call__(
        self,
        self_elo: jax.Array,
        oppo_elo: jax.Array,
        self_title: jax.Array,
        oppo_title: jax.Array,
        game_format: jax.Array,
        board_meta: jax.Array,
        rich_time: jax.Array,
        has_clock: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        cdt = self.cfg.dtype
        se_rbf = self._rbf_elo(self_elo)
        oe_rbf = self._rbf_elo(oppo_elo)
        elo_scalars = jnp.stack(
            [
                (self_elo - 1800.0) / 600.0,
                (oppo_elo - 1800.0) / 600.0,
                jnp.tanh((self_elo - oppo_elo) / 400.0),
            ],
            axis=-1,
        )
        st = self.title_emb(jnp.clip(self_title, 0, cu.NUM_TITLE_TIERS - 1))
        ot = self.title_emb(jnp.clip(oppo_title, 0, cu.NUM_TITLE_TIERS - 1))
        gf = self.format_emb(jnp.clip(game_format, 0, cu.NUM_GAME_FORMATS - 1))
        feat = jnp.concatenate(
            [
                se_rbf.astype(cdt),
                oe_rbf.astype(cdt),
                elo_scalars.astype(cdt),
                st.astype(cdt),
                ot.astype(cdt),
                gf.astype(cdt),
                board_meta.astype(cdt),
                rich_time.astype(cdt),
                has_clock[:, None].astype(cdt),
            ],
            axis=-1,
        )
        cond = self.fc2(jax.nn.silu(self.fc1(feat))).astype(jnp.float32)
        cond_low = jax.nn.silu(self.adaln_bottleneck(cond))  # (B, 128) in cdt
        return cond, cond_low


class RayGABBias(nnx.Module):
    """Learned spatial GAB + zero-copy exact line-of-sight & pin/X-ray bias on (B, H, 64, 64)."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        self.gab = GABGenerator(cfg, rngs=rngs)
        if cfg.use_ray_gab:
            self.ray_w = nnx.Param(jnp.zeros((8, cfg.num_heads), dtype=cfg.pdtype))
            self.rel_table = nnx.Param(
                jnp.zeros((cu.NUM_PAIR_RELATIONS, cfg.num_heads), dtype=cfg.pdtype)
            )

    def __call__(
        self, x_sq: jax.Array, gab_weight: jax.Array, ray_feats: jax.Array | None
    ) -> jax.Array:
        bias = self.gab(x_sq, gab_weight)  # (B, H, 64, 64)
        if self.cfg.use_ray_gab and ray_feats is not None:
            # ray_feats is already (B, 8, 64, 64) in compute_dtype -> zero transpose copies!
            rb = jnp.einsum(
                "bcij,ch->bhij",
                ray_feats,
                self.ray_w.value.astype(self.cfg.dtype),
                preferred_element_type=jnp.float32,
            )
            static_b = self.rel_table.value[jnp.asarray(cu.PAIR_REL_INDEX)].astype(
                jnp.float32
            ).transpose(2, 0, 1)
            bias = bias + rb + static_b[None]
        return bias


class MultiHeadAttentionV2(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.dim_vit
        H = cfg.num_heads
        hd = D // H
        use_bias = not cfg.omit_qkv_biases
        in_bound = math.sqrt(6.0 / (4 * D))
        self.q = torch_linear(
            D, D, cfg, rngs=rngs, use_bias=use_bias, bound=in_bound, zero_bias=True
        )
        self.k = torch_linear(
            D, D, cfg, rngs=rngs, use_bias=use_bias, bound=in_bound, zero_bias=True
        )
        self.v = torch_linear(
            D, D, cfg, rngs=rngs, use_bias=use_bias, bound=in_bound, zero_bias=True
        )
        self.out = torch_linear(
            D,
            D,
            cfg,
            rngs=rngs,
            use_bias=use_bias,
            bound=1.0 / math.sqrt(D),
            zero_bias=True,
        )
        if cfg.use_qk_norm:
            self.q_norm = rms_norm(hd, cfg, rngs=rngs)
            self.k_norm = rms_norm(hd, cfg, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        bias: jax.Array | None,
        deterministic: bool,
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        c = self.cfg
        B, S, D = x.shape
        H = c.num_heads
        hd = D // H
        q = self.q(x).reshape(B, S, H, hd)
        k = self.k(x).reshape(B, S, H, hd)
        v = self.v(x).reshape(B, S, H, hd)
        if c.use_qk_norm:
            q = self.q_norm(q).astype(c.dtype)
            k = self.k_norm(k).astype(c.dtype)
        logits = jnp.einsum(
            "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32
        ) / math.sqrt(hd)
        if bias is not None:
            logits = logits + bias
        w = jax.nn.softmax(logits, axis=-1)
        w = _dropout(w, c.dropout, deterministic, rng)
        out = jnp.einsum("bhqk,bkhd->bqhd", w.astype(c.dtype), v).reshape(B, S, D)
        return self.out(out), logits


class SwiGLUMLP(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.dim_vit
        raw_h = (
            int(D * cfg.mlp_ratio * 2.0 / 3.0)
            if cfg.mlp_ratio <= 3.0
            else int(D * cfg.mlp_ratio)
        )
        hidden = ((max(raw_h, 64) + 63) // 64) * 64
        self.gate_up = torch_linear(D, 2 * hidden, cfg, rngs=rngs, use_bias=False)
        self.down = torch_linear(hidden, D, cfg, rngs=rngs, use_bias=False)

    def __call__(
        self, x: jax.Array, deterministic: bool, rng: jax.Array | None = None
    ) -> jax.Array:
        gate, up = jnp.split(self.gate_up(x), 2, axis=-1)
        h = _dropout(jax.nn.silu(gate) * up, self.cfg.dropout, deterministic, rng)
        return self.down(h)


class EncoderBlockV2(nnx.Module):
    """Strict S=64 Pre-RMSNorm + Bottleneck AdaLN + Zero-Copy Ray-GAB + QK-Norm MHA + SwiGLU."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.dim_vit
        self.norm1 = (
            rms_norm(D, cfg, rngs=rngs)
            if cfg.use_rms_norm
            else layer_norm(D, cfg, rngs=rngs)
        )
        self.norm2 = (
            rms_norm(D, cfg, rngs=rngs)
            if cfg.use_rms_norm
            else layer_norm(D, cfg, rngs=rngs)
        )
        if cfg.use_adaln:
            self.adaln = zero_linear(ADALN_BOTTLENECK, 6 * D, cfg, rngs=rngs)
        if cfg.pos_encoding == "gab":
            self.sq_bias = RayGABBias(cfg, rngs=rngs)
        elif cfg.pos_encoding == "relative":
            self.relative_bias = RelativeBias(cfg.num_heads, cfg.pdtype)
        self.attn = MultiHeadAttentionV2(cfg, rngs=rngs)
        if cfg.use_swiglu:
            self.mlp = SwiGLUMLP(cfg, rngs=rngs)
        else:
            hidden = int(D * cfg.mlp_ratio)
            self.linear1 = torch_linear(D, hidden, cfg, rngs=rngs)
            self.linear2 = torch_linear(hidden, D, cfg, rngs=rngs)

    def __call__(
        self,
        x: jax.Array,
        cond_low: jax.Array,
        gab_weight: jax.Array | None,
        ray_feats: jax.Array | None,
        deterministic: bool,
        rng: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        c = self.cfg
        r_attn, r_d1, r_mlp, r_d2 = (
            (None, None, None, None) if rng is None else jax.random.split(rng, 4)
        )
        if c.use_adaln:
            mods = self.adaln(cond_low).astype(jnp.float32)[:, None, :]
            sh_a, sc_a, gt_a, sh_m, sc_m, gt_m = jnp.split(mods, 6, axis=-1)
        else:
            sh_a = sc_a = gt_a = sh_m = sc_m = gt_m = 0.0

        h1 = self.norm1(x) * (1.0 + sc_a) + sh_a
        bias = None
        if c.pos_encoding == "gab":
            if gab_weight is None:
                raise ValueError("GAB positional encoding requires gab_shared_weight")
            bias = self.sq_bias(h1, gab_weight, ray_feats)
        elif c.pos_encoding == "relative":
            bias = self.relative_bias()[None]

        a_out, attn_sq = self.attn(h1, bias, deterministic, r_attn)
        a_out = _dropout(a_out, c.dropout, deterministic, r_d1)
        x = x + (1.0 + gt_a) * a_out.astype(jnp.float32)

        h2 = self.norm2(x) * (1.0 + sc_m) + sh_m
        if c.use_swiglu:
            m_out = self.mlp(h2, deterministic, r_mlp)
        else:
            act = (
                (lambda t: nnx.gelu(t, approximate=False))
                if c.activation == "gelu"
                else nnx.relu
            )
            m_out = self.linear2(
                _dropout(act(self.linear1(h2)), c.dropout, deterministic, r_mlp)
            )
        m_out = _dropout(m_out, c.dropout, deterministic, r_d2)
        x = x + (1.0 + gt_m) * m_out.astype(jnp.float32)
        return x, attn_sq


class CrossAttentionPool(nnx.Module):
    """Selective [CLS_VALUE, CLS_PONDER] readout after trunk while keeping trunk S=64."""

    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D = cfg.dim_vit
        self.queries = nnx.Param(_normal(0.02)(rngs.params(), (2, D), cfg.pdtype))
        self.q_proj = torch_linear(D, D, cfg, rngs=rngs, use_bias=False)
        self.k_proj = torch_linear(D, D, cfg, rngs=rngs, use_bias=False)
        self.v_proj = torch_linear(D, D, cfg, rngs=rngs, use_bias=False)
        self.out_proj = torch_linear(D, D, cfg, rngs=rngs)

    def __call__(self, x_boards: jax.Array, cond: jax.Array) -> tuple[jax.Array, jax.Array]:
        c = self.cfg
        B, S, D = x_boards.shape
        H = c.num_heads
        hd = D // H
        q_in = self.queries.value.astype(jnp.float32)[None] + cond[:, None, :]
        q = self.q_proj(q_in).reshape(B, 2, H, hd)
        k = self.k_proj(x_boards).reshape(B, S, H, hd)
        v = self.v_proj(x_boards).reshape(B, S, H, hd)
        logits = jnp.einsum(
            "bqhd,bkhd->bhqk", q, k, preferred_element_type=jnp.float32
        ) / math.sqrt(hd)
        w = jax.nn.softmax(logits, axis=-1).astype(c.dtype)
        pooled = self.out_proj(
            jnp.einsum("bhqk,bkhd->bqhd", w, v).reshape(B, 2, D)
        ).astype(jnp.float32)
        return pooled[:, 0, :] + cond, pooled[:, 1, :] + cond


class BilinearMoveHead(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, with_attn_res: bool, rngs: nnx.Rngs):
        self.cfg = cfg
        D, hh = cfg.dim_vit, cfg.head_hid_dim
        self.with_attn_res = with_attn_res
        self.proj_from = torch_linear(D, hh, cfg, rngs=rngs, use_bias=False)
        self.proj_to = torch_linear(D, hh, cfg, rngs=rngs, use_bias=False)
        self.from_prior = torch_linear(D, 1, cfg, rngs=rngs, use_bias=True)
        self.promo_bias_proj = torch_linear(hh, 4, cfg, rngs=rngs, use_bias=False)
        if with_attn_res:
            res_heads = cfg.num_heads * min(2, cfg.num_blocks)
            self.attn_w = nnx.Param(jnp.zeros((res_heads,), dtype=cfg.pdtype))

    def __call__(
        self, x_sq: jax.Array, recent_attn: jax.Array | None = None
    ) -> jax.Array:
        B = x_sq.shape[0]
        hh = self.cfg.head_hid_dim
        sq_from = self.proj_from(x_sq)
        sq_to = self.proj_to(x_sq)
        scores = jnp.einsum(
            "bid,bjd->bij", sq_from, sq_to, preferred_element_type=jnp.float32
        ) / math.sqrt(hh)
        scores = scores + self.from_prior(x_sq).astype(jnp.float32)

        if self.with_attn_res and recent_attn is not None:
            scores = scores + jnp.einsum(
                "bhij,h->bij",
                recent_attn.astype(jnp.float32),
                self.attn_w.value.astype(jnp.float32),
            )

        promo_bias = self.promo_bias_proj(sq_to[:, 56:64]).astype(
            jnp.float32
        ) * math.sqrt(hh)
        base = scores[:, 48:56, 56:64]
        promo = (base[..., None] + promo_bias[:, None, :, :]).reshape(B, 256)
        return jnp.concatenate([scores.reshape(B, 64 * 64), promo], axis=-1)


class ChessformerV2(nnx.Module):
    def __init__(self, cfg: ModelConfig, *, rngs: nnx.Rngs):
        self.cfg = cfg
        D, hh = cfg.dim_vit, cfg.head_hid_dim
        if cfg.gab_per_square_dim == 0 and cfg.pos_encoding == "gab":
            raise ValueError(
                "ChessformerV2 requires gab_per_square_dim > 0 so GAB is spatially aware."
            )

        self.cond_net = ContextConditioner(cfg, rngs=rngs)
        sq_in_dim = 12 * cfg.history + 1 + 4 + 8 + 6
        self.token_projection = torch_linear(sq_in_dim, D, cfg, rngs=rngs)
        self.cond_to_sq = torch_linear(D, D, cfg, rngs=rngs, use_bias=False)

        if cfg.pos_encoding == "absolute":
            self.abs_pe = nnx.Param(_normal(0.02)(rngs.params(), (64, D), cfg.pdtype))
        if cfg.pos_encoding == "gab":
            gstd = math.sqrt(2.0 / (64 * 64 + cfg.gab_gen_size))
            self.gab_shared_weight = nnx.Param(
                _normal(gstd)(rngs.params(), (64 * 64, cfg.gab_gen_size), cfg.pdtype)
            )

        for i in range(cfg.num_blocks):
            setattr(self, f"block_{i}", EncoderBlockV2(cfg, rngs=rngs))
        self.final_norm = (
            rms_norm(D, cfg, rngs=rngs)
            if cfg.use_rms_norm
            else layer_norm(D, cfg, rngs=rngs)
        )
        if cfg.num_registers > 0:
            self.readout_pool = CrossAttentionPool(cfg, rngs=rngs)
        self.value_norm = layer_norm(D, cfg, rngs=rngs)
        self.ponder_norm = layer_norm(D, cfg, rngs=rngs)

        self.policy_head = BilinearMoveHead(
            cfg, with_attn_res=cfg.use_attn_policy_residual, rngs=rngs
        )
        if cfg.use_opponent_reply_head:
            self.reply_norm = rms_norm(D, cfg, rngs=rngs)
            self.reply_head = BilinearMoveHead(cfg, with_attn_res=False, rngs=rngs)

        self.fc_value_hid = torch_linear(D, hh, cfg, rngs=rngs)
        self.fc_value = torch_linear(hh, 3, cfg, rngs=rngs)

        self.fc_ponder_hid = torch_linear(D, hh, cfg, rngs=rngs)
        self.fc_ponder = torch_linear(hh, cfg.ponder_buckets, cfg, rngs=rngs)

    def _compute_ray_features(self, current_board: jax.Array) -> jax.Array:
        """current_board: int8 (B, 64) -> (B, 8, 64, 64) in compute_dtype (zero transposes)."""
        cdt = self.cfg.dtype
        occ = (current_board > 0).astype(cdt)
        ray_between = jnp.asarray(cu.RAY_BETWEEN, dtype=cdt)
        blockers = jnp.einsum(
            "bs,sp->bp", occ, ray_between, preferred_element_type=jnp.float32
        ).reshape(-1, 1, 64, 64)
        dirs = jnp.asarray(cu.RAY_DIR_MASK, dtype=jnp.float32)[None]
        clear = (blockers < 0.5).astype(cdt) * dirs.astype(cdt)
        pin_xray = ((blockers > 0.5) & (blockers < 1.5)).astype(cdt) * dirs.astype(cdt)
        return jnp.concatenate([clear, pin_xray], axis=1)

    def __call__(
        self,
        boards: jax.Array,
        self_elo: jax.Array,
        oppo_elo: jax.Array,
        *,
        rich_time: jax.Array | None = None,
        has_clock: jax.Array | None = None,
        board_meta: jax.Array | None = None,
        ep_square_idx: jax.Array | None = None,
        self_title: jax.Array | None = None,
        oppo_title: jax.Array | None = None,
        game_format: jax.Array | None = None,
        clk_left: jax.Array | None = None,
        deterministic: bool = True,
        rng: jax.Array | None = None,
        return_aux: bool = False,
    ):
        c = self.cfg
        cdt = c.dtype
        B, H, S = boards.shape
        assert H == c.history and S == 64, (boards.shape, c.history)

        if rich_time is None:
            rich_time = jnp.zeros((B, 6), jnp.float32)
        if has_clock is None:
            has_clock = jnp.zeros((B,), jnp.float32)
        if board_meta is None:
            board_meta = jnp.zeros((B, 8), jnp.float32)
        if ep_square_idx is None:
            ep_square_idx = jnp.full((B,), -1, jnp.int32)
        if self_title is None:
            self_title = jnp.zeros((B,), jnp.int32)
        if oppo_title is None:
            oppo_title = jnp.zeros((B,), jnp.int32)
        if game_format is None:
            game_format = jnp.zeros((B,), jnp.int32)

        cond, cond_low = self.cond_net(
            self_elo,
            oppo_elo,
            self_title,
            oppo_title,
            game_format,
            board_meta,
            rich_time,
            has_clock,
        )

        onehot = (boards[..., None] == jnp.arange(1, 13, dtype=boards.dtype)).astype(cdt)
        piece_tok = onehot.transpose(0, 2, 1, 3).reshape(B, 64, H * 12)
        ep_tok = (
            jnp.arange(64, dtype=jnp.int32)[None, :] == ep_square_idx[:, None]
        ).astype(cdt)[..., None]
        coords = jnp.broadcast_to(jnp.asarray(cu.SQ_COORDS, dtype=cdt)[None], (B, 64, 4))
        meta_tok = jnp.broadcast_to(board_meta[:, None, :].astype(cdt), (B, 64, 8))
        time_tok = jnp.broadcast_to(rich_time[:, None, :].astype(cdt), (B, 64, 6))

        sq_in = jnp.concatenate([piece_tok, ep_tok, coords, meta_tok, time_tok], axis=-1)
        x = self.token_projection(sq_in).astype(jnp.float32) + self.cond_to_sq(
            cond.astype(cdt)
        ).astype(jnp.float32)[:, None, :]

        if c.pos_encoding == "absolute":
            x = x + self.abs_pe.value.astype(jnp.float32)[None]

        gab_w = self.gab_shared_weight.value if c.pos_encoding == "gab" else None
        ray_feats = (
            self._compute_ray_features(boards[:, -1, :])
            if (c.pos_encoding == "gab" and c.use_ray_gab)
            else None
        )

        blk_rngs = (
            [None] * c.num_blocks
            if rng is None
            else list(jax.random.split(rng, c.num_blocks))
        )
        recent_attns: list[jax.Array] = []
        keep_from = max(0, c.num_blocks - 2)
        for i in range(c.num_blocks):
            blk: EncoderBlockV2 = getattr(self, f"block_{i}")
            if c.remat_blocks and not deterministic:
                blk_def, blk_state = nnx.split(blk)

                @jax.checkpoint
                def _ckpt_blk(b_st, x_in, cl_in, gw_in, rf_in, r_in):
                    m = nnx.merge(blk_def, b_st)
                    return m(x_in, cl_in, gw_in, rf_in, False, r_in)

                x, attn_sq = _ckpt_blk(
                    blk_state, x, cond_low, gab_w, ray_feats, blk_rngs[i]
                )
            elif c.remat_blocks:
                blk_def, blk_state = nnx.split(blk)

                @jax.checkpoint
                def _ckpt_blk_det(b_st, x_in, cl_in, gw_in, rf_in):
                    m = nnx.merge(blk_def, b_st)
                    return m(x_in, cl_in, gw_in, rf_in, True, None)

                x, attn_sq = _ckpt_blk_det(blk_state, x, cond_low, gab_w, ray_feats)
            else:
                x, attn_sq = blk(
                    x, cond_low, gab_w, ray_feats, deterministic, blk_rngs[i]
                )
            if c.use_attn_policy_residual and i >= keep_from:
                recent_attns.append(attn_sq)

        x_boards = self.final_norm(x)
        stacked_attn = (
            jnp.concatenate(recent_attns, axis=1) if recent_attns else None
        )
        logits_move = self.policy_head(x_boards, stacked_attn)

        if c.num_registers > 0:
            v_raw, p_raw = self.readout_pool(x_boards, cond)
            v_tok = self.value_norm(v_raw)
            p_tok = self.ponder_norm(p_raw)
        else:
            pooled = x_boards.mean(axis=1)
            v_tok = self.value_norm(pooled)
            p_tok = self.ponder_norm(pooled)

        logits_value = self.fc_value(jax.nn.silu(self.fc_value_hid(v_tok))).astype(
            jnp.float32
        )
        ponder_logits = self.fc_ponder(jax.nn.silu(self.fc_ponder_hid(p_tok))).astype(
            jnp.float32
        )

        edges = jnp.linspace(0.0, 1.0, c.ponder_buckets + 1, dtype=jnp.float32)
        centers = 0.5 * (edges[:-1] + edges[1:])
        ponder_effort = jnp.sum(
            jax.nn.softmax(ponder_logits, axis=-1) * centers[None, :], axis=-1
        )
        if clk_left is not None:
            ponder_sec = jnp.expm1(
                ponder_effort
                * jnp.log1p(jnp.maximum(clk_left.astype(jnp.float32), 1.0))
            )
        else:
            ponder_sec = ponder_effort * 10.0

        if not return_aux:
            return logits_move, logits_value, ponder_sec

        aux: dict[str, jax.Array] = {
            "ponder_logits": ponder_logits,
            "ponder_effort": ponder_effort,
        }
        if c.use_opponent_reply_head:
            mirror_idx = jnp.asarray(cu.MIRROR_SQ)
            x_opp = self.reply_norm(x_boards[:, mirror_idx, :])
            aux["reply_logits"] = self.reply_head(x_opp, None)
        return logits_move, logits_value, ponder_sec, aux
