"""Compact binary record for training/eval positions (ArrayRecord payload, v2 & v3)."""

from __future__ import annotations

import functools

import numpy as np

LEGACY_FORMAT_VERSION = 2
FORMAT_VERSION = 3


@functools.lru_cache(maxsize=None)
def position_dtype_v2(history: int) -> np.dtype:
    return np.dtype(
        [
            ("boards", np.int8, (history, 64)),
            ("n_hist", np.int8),
            ("stm", np.int8),
            ("move", np.int16),
            ("self_elo", np.int16),
            ("oppo_elo", np.int16),
            ("value", np.int8),
            ("ply", np.int16),
            ("clk_left", np.float32),
            ("ponder", np.float32),
            ("base_time", np.int16),
            ("increment", np.int16),
            ("n_legal", np.int16),
        ]
    )


@functools.lru_cache(maxsize=None)
def position_dtype(history: int, version: int = FORMAT_VERSION) -> np.dtype:
    if version == LEGACY_FORMAT_VERSION:
        return position_dtype_v2(history)
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported record version {version}")
    return np.dtype(
        [
            ("boards", np.int8, (history, 64)),  # absolute codes, oldest -> current
            ("n_hist", np.int8),  # real past boards available (<= history-1)
            ("stm", np.int8),  # 0 white / 1 black to move (current board)
            ("move", np.int16),  # target move index (side-to-move frame)
            ("self_elo", np.int16),
            ("oppo_elo", np.int16),
            ("value", np.int8),  # 0 loss, 1 draw, 2 win (side to move)
            ("ply", np.int16),
            ("clk_left", np.float32),  # seconds on own clock before move
            ("opp_clk_left", np.float32),  # seconds on opponent clock
            ("ponder", np.float32),  # seconds spent on this move (nan if unknown)
            ("prev_opp_ponder", np.float32),  # seconds opponent spent on previous move
            ("base_time", np.int16),
            ("increment", np.int16),
            ("castling", np.uint8),  # 4-bit mask: own K, own Q, opp K, opp Q
            ("ep_file", np.int8),  # -1 if no legal EP, else 0..7
            ("rep_count", np.uint8),  # 0, 1, or 2+ prior occurrences in game
            ("self_title", np.uint8),  # 0..4 official human title tier
            ("oppo_title", np.uint8),  # 0..4 official human title tier
            ("game_format", np.uint8),  # 0=blitz, 1=rapid, 2=otb_classical
            ("has_clock", np.uint8),  # 1 if clock data valid, 0 if clockless OTB
            ("next_move", np.int16),  # -1 if none, else 0..4351 opponent reply (opp frame)
            ("sf_move", np.int16),  # -1 if unannotated, else Stockfish #1 move (stm frame)
            ("sf_cp_loss", np.int16),  # -1 if unannotated, else cp loss vs SF #1
            ("n_legal", np.int16),
        ]
    )


def upgrade_v2_to_v3(arr_v2: np.ndarray, history: int) -> np.ndarray:
    out = np.zeros(arr_v2.shape, dtype=position_dtype(history, FORMAT_VERSION))
    assert arr_v2.dtype.names is not None
    for name in arr_v2.dtype.names:
        out[name] = arr_v2[name]
    out["opp_clk_left"] = arr_v2["clk_left"]
    out["prev_opp_ponder"] = np.nan
    out["castling"] = 0
    out["ep_file"] = -1
    out["rep_count"] = 0
    out["self_title"] = 0
    out["oppo_title"] = 0
    out["game_format"] = 0
    out["has_clock"] = 1
    out["next_move"] = -1
    out["sf_move"] = -1
    out["sf_cp_loss"] = -1
    return out


def encode_position(
    *,
    boards,
    n_hist,
    stm,
    move,
    self_elo,
    oppo_elo,
    value,
    ply,
    clk_left=0.0,
    opp_clk_left=None,
    ponder=np.nan,
    prev_opp_ponder=np.nan,
    base_time=0,
    increment=0,
    castling=0,
    ep_file=-1,
    rep_count=0,
    self_title=0,
    oppo_title=0,
    game_format=0,
    has_clock=1,
    next_move=-1,
    sf_move=-1,
    sf_cp_loss=-1,
    legal=None,
) -> bytes:
    H = boards.shape[0]
    rec = np.zeros(1, position_dtype(H, FORMAT_VERSION))
    rec["boards"][0] = boards
    rec["n_hist"] = n_hist
    rec["stm"] = stm
    rec["move"] = move
    rec["self_elo"] = int(np.clip(self_elo, 0, 32767))
    rec["oppo_elo"] = int(np.clip(oppo_elo, 0, 32767))
    rec["value"] = value
    rec["ply"] = min(int(ply), 32767)
    rec["clk_left"] = clk_left
    rec["opp_clk_left"] = clk_left if opp_clk_left is None else opp_clk_left
    rec["ponder"] = ponder
    rec["prev_opp_ponder"] = prev_opp_ponder
    rec["base_time"] = min(max(int(base_time), 0), 32767)
    rec["increment"] = min(max(int(increment), 0), 32767)
    rec["castling"] = int(castling) & 0xF
    rec["ep_file"] = int(ep_file)
    rec["rep_count"] = min(max(int(rep_count), 0), 2)
    rec["self_title"] = int(self_title)
    rec["oppo_title"] = int(oppo_title)
    rec["game_format"] = int(game_format)
    rec["has_clock"] = int(bool(has_clock))
    rec["next_move"] = int(next_move)
    rec["sf_move"] = int(sf_move)
    rec["sf_cp_loss"] = min(max(int(sf_cp_loss), -1), 32767)
    legal_arr = np.zeros(0, np.int16) if legal is None else np.asarray(legal, np.int16)
    rec["n_legal"] = legal_arr.size
    return bytes([FORMAT_VERSION, H]) + rec.tobytes() + legal_arr.tobytes()


def encode_record_struct(rec_scalar: np.void, legal: np.ndarray) -> bytes:
    H = rec_scalar["boards"].shape[0]
    arr = np.asarray([rec_scalar], dtype=position_dtype(H, FORMAT_VERSION))
    legal_arr = np.asarray(legal, dtype=np.int16)
    arr["n_legal"] = legal_arr.size
    return bytes([FORMAT_VERSION, H]) + arr.tobytes() + legal_arr.tobytes()


def decode_position(raw: bytes):
    ver = raw[0]
    H = int(raw[1])
    if ver == LEGACY_FORMAT_VERSION:
        dt2 = position_dtype(H, LEGACY_FORMAT_VERSION)
        rec2 = np.frombuffer(raw, dtype=dt2, count=1, offset=2)
        rec = upgrade_v2_to_v3(rec2, H)[0]
        legal = np.frombuffer(raw, dtype=np.int16, offset=2 + dt2.itemsize)
        return rec, legal
    if ver != FORMAT_VERSION:
        raise ValueError(f"Unsupported record version {ver}")
    dt = position_dtype(H, FORMAT_VERSION)
    rec = np.frombuffer(raw, dtype=dt, count=1, offset=2)[0]
    legal = np.frombuffer(raw, dtype=np.int16, offset=2 + dt.itemsize)
    return rec, legal