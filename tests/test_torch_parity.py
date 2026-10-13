"""Forward-pass parity of the Flax NNX model against maia3.models.MAIA3Model (fp32)."""
from argparse import Namespace

import jax.numpy as jnp
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("maia3")
from maia3.models import MAIA3Model  # noqa: E402

from chessforge.models import build_model_config  # noqa: E402
from chessforge.models.convert_torch import convert  # noqa: E402


def torch_ns(c):
    return Namespace(
        history=c.history, include_time_info=c.include_time_info, dim_emb=c.dim_emb, dim_vit=c.dim_vit,
        num_blocks=c.num_blocks, num_heads=c.num_heads, mlp_ratio=c.mlp_ratio, dropout=0.0,
        head_hid_dim=c.head_hid_dim, use_gab=c.pos_encoding == "gab",
        use_relative_bias=c.pos_encoding == "relative", use_absolute_pe=c.pos_encoding == "absolute",
        gab_gen_size=c.gab_gen_size, gab_per_square_dim=c.gab_per_square_dim,
        gab_intermediate_dim=c.gab_intermediate_dim, use_rms_norm=c.use_rms_norm,
        omit_qkv_biases=c.omit_qkv_biases, activation=c.activation,
    )


@pytest.mark.parametrize("preset", ["maia3-5m", "maia3-23m", "maia3-5m-relative", "maia3-5m-absolute"])
def test_parity_nnx(preset):
    c = build_model_config(preset, {"compute_dtype": "float32"})
    tm = MAIA3Model(torch_ns(c)).eval()
    if c.pos_encoding == "relative":
        for blk in tm.transformer.layers:
            torch.nn.init.normal_(blk.self_attn.relative_bias.gate, std=0.5)
    if c.pos_encoding == "absolute":
        torch.nn.init.normal_(tm.abs_pe.bias, std=0.5)
    sd = {k: v.detach().numpy() for k, v in tm.state_dict().items()}

    # Convert PyTorch state_dict directly into an NNX Chessformer instance
    nnx_model = convert(sd, c)

    rng = np.random.default_rng(0)
    B = 4
    boards = rng.integers(0, 13, size=(B, c.history, 64)).astype(np.int8)
    se = rng.integers(500, 2900, size=B)
    oe = rng.integers(500, 2900, size=B)
    oh = (boards[..., None] == np.arange(1, 13)).astype(np.float32).transpose(0, 2, 1, 3).reshape(B, 64, -1)
    tokens = np.concatenate([oh, np.zeros((B, 64, 1), np.float32)], -1)  # + clk_ponder column
    with torch.no_grad():
        tl, tv, tp = tm(torch.from_numpy(tokens), torch.from_numpy(se), torch.from_numpy(oe))

    jl, jv, jp = nnx_model(
        jnp.asarray(boards),
        jnp.asarray(se, jnp.float32),
        jnp.asarray(oe, jnp.float32),
        deterministic=True,
    )
    np.testing.assert_allclose(np.asarray(jl), tl.numpy(), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(jv), tv.numpy(), atol=2e-4, rtol=2e-4)
    np.testing.assert_allclose(np.asarray(jp), tp.numpy(), atol=2e-4, rtol=2e-4)