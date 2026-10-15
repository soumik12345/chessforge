"""Chess encodings. Must stay byte-compatible with maia3/{utils,dataset}.py."""

from __future__ import annotations

import chess
import numpy as np

NUM_SQUARES = 64
PROMO_PIECES = "qrbn"
NUM_FROM_TO = 64 * 64
NUM_PROMOTIONS = 8 * 8 * 4
VOCAB_SIZE = NUM_FROM_TO + NUM_PROMOTIONS  # 4352
NUM_VALUE_CLASSES = 3  # [loss, draw, win] for side to move
LOSS, DRAW, WIN = 0, 1, 2

# absolute piece codes: 0 empty, 1..6 white P N B R Q K, 7..12 black
SWAP_COLOR = np.array([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], dtype=np.int8)
MIRROR_SQ = (np.arange(64) ^ 56).astype(
    np.int64
)  # python-chess board.mirror() flips ranks


def all_moves() -> list[str]:
    """Identical ordering to maia3.utils.get_all_possible_moves()."""
    moves = [
        chess.square_name(f) + chess.square_name(t)
        for f in range(64)
        for t in range(64)
    ]
    for ff in "abcdefgh":
        for tf in "abcdefgh":
            for p in PROMO_PIECES:
                moves.append(f"{ff}7{tf}8{p}")
    assert len(moves) == VOCAB_SIZE
    return moves


_MOVES = all_moves()
_MOVE_TO_IDX = {m: i for i, m in enumerate(_MOVES)}


def mirror_uci(uci: str) -> str:
    return uci[0] + str(9 - int(uci[1])) + uci[2] + str(9 - int(uci[3])) + uci[4:]


def move_index(uci: str) -> int:
    return _MOVE_TO_IDX[uci]


def index_to_uci(idx: int) -> str:
    return _MOVES[idx]


def stm_move_index(move: chess.Move, turn: bool) -> int:
    """Move index in the side-to-move frame (board mirrored when black moves)."""
    uci = move.uci()
    if turn == chess.BLACK:
        uci = mirror_uci(uci)
    return _MOVE_TO_IDX[uci]


def board_codes(board: chess.Board) -> np.ndarray:
    codes = np.zeros(64, np.int8)
    for sq, piece in board.piece_map().items():
        codes[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    return codes


def legal_move_indices(board: chess.Board) -> np.ndarray:
    out = []
    black = board.turn == chess.BLACK
    for m in board.legal_moves:
        uci = m.uci()
        if black:
            uci = mirror_uci(uci)
        idx = _MOVE_TO_IDX.get(uci)
        if idx is not None:
            out.append(idx)
    return np.asarray(sorted(out), np.int16)


def value_label(result: str, white_to_move: bool) -> int:
    if result == "1/2-1/2":
        return DRAW
    white_won = result == "1-0"
    return WIN if white_won == white_to_move else LOSS


def to_relative_codes(codes: np.ndarray, stm: int, n_hist: int) -> np.ndarray:
    """codes: int8 (H, 64) absolute boards, oldest -> current.

    Reproduces maia3: every board is tokenized from *its own* side-to-move
    perspective (mirrored if black to move), and missing history is padded by
    repeating the earliest available board's tokens.
    """
    H = codes.shape[0]
    n_hist = int(min(n_hist, H - 1))
    lag = (H - 1) - np.arange(H)
    black = (int(stm) ^ (lag & 1)) == 1
    out = codes.copy()
    if black.any():
        out[black] = SWAP_COLOR[codes[black][:, MIRROR_SQ]]
    first = H - 1 - n_hist
    if first > 0:
        out[:first] = out[first]
    return out
