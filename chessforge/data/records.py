"""Compact binary record for one training/eval position (ArrayRecord payload)."""
from __future__ import annotations

import functools

import numpy as np

FORMAT_VERSION = 2


@functools.lru_cache(maxsize=None)
def position_dtype(history: int) -> np.dtype:
    return np.dtype([
        ("boards", np.int8, (history, 64)),  # absolute codes, oldest -> current
        ("n_hist", np.int8),                 # real past boards available (<= history-1)
        ("stm", np.int8),                    # 0 white / 1 black to move (current board)
        ("move", np.int16),                  # target move index (side-to-move frame)
        ("self_elo", np.int16),
        ("oppo_elo", np.int16),
        ("value", np.int8),                  # 0 loss, 1 draw, 2 win (side to move)
        ("ply", np.int16),
        ("clk_left", np.float32),            # seconds on clock before move
        ("ponder", np.float32),              # seconds spent on the move (nan if unknown)
        ("base_time", np.int16),
        ("increment", np.int16),
        ("n_legal", np.int16),
    ])


def encode_position(*, boards, n_hist, stm, move, self_elo, oppo_elo, value, ply,
                    clk_left=0.0, ponder=np.nan, base_time=0, increment=0, legal=None) -> bytes:
    H = boards.shape[0]
    rec = np.zeros(1, position_dtype(H))
    rec["boards"][0] = boards
    rec["n_hist"] = n_hist
    rec["stm"] = stm
    rec["move"] = move
    rec["self_elo"] = self_elo
    rec["oppo_elo"] = oppo_elo
    rec["value"] = value
    rec["ply"] = min(ply, 32767)
    rec["clk_left"] = clk_left
    rec["ponder"] = ponder
    rec["base_time"] = min(base_time, 32767)
    rec["increment"] = min(increment, 32767)
    legal = np.zeros(0, np.int16) if legal is None else np.asarray(legal, np.int16)
    rec["n_legal"] = legal.size
    return bytes([FORMAT_VERSION, H]) + rec.tobytes() + legal.tobytes()


def decode_position(raw: bytes):
    if raw[0] != FORMAT_VERSION:
        raise ValueError(f"Unsupported record version {raw[0]}")
    dt = position_dtype(raw[1])
    rec = np.frombuffer(raw, dtype=dt, count=1, offset=2)[0]
    legal = np.frombuffer(raw, dtype=np.int16, offset=2 + dt.itemsize)
    return rec, legal