"""Encoding parity vs the official maia3 package (pip install git+https://github.com/CSSLab/maia3)."""

import random
from argparse import Namespace
from collections import deque

import chess
import numpy as np
import pytest

from chessforge import chess_utils as cu

maia3 = pytest.importorskip("maia3")
from maia3.dataset import get_historical_tokens, tokenize_board  # noqa: E402
from maia3.utils import get_all_possible_moves, mirror_move  # noqa: E402


def test_vocab():
    assert cu.all_moves() == get_all_possible_moves()


def _onehot(rel):  # (H,64) -> (64, H*12)
    oh = (rel[..., None] == np.arange(1, 13)).astype(np.float32)
    return oh.transpose(1, 0, 2).reshape(64, -1)


@pytest.mark.parametrize("seed", range(5))
def test_history_tokens(seed):
    random.seed(seed)
    H = 8
    cfg = Namespace(history=H, include_time_info=False)
    board = chess.Board()
    hist_t, hist_codes = deque(maxlen=H), deque(maxlen=H)
    for ply in range(60):
        hist_t.append(tokenize_board(board))
        hist_codes.append(cu.board_codes(board))
        ref = get_historical_tokens(hist_t, cfg, 0, 0, 0, 0)[:, : 12 * H].numpy()
        codes = np.zeros((H, 64), np.int8)
        codes[H - len(hist_codes) :] = np.stack(hist_codes)
        rel = cu.to_relative_codes(
            codes, int(board.turn == chess.BLACK), len(hist_codes) - 1
        )
        np.testing.assert_array_equal(_onehot(rel), ref)
        moves = list(board.legal_moves)
        if not moves:
            break
        m = random.choice(moves)
        uci = m.uci() if board.turn == chess.WHITE else mirror_move(m.uci())
        assert cu.stm_move_index(m, board.turn) == get_all_possible_moves().index(uci)
        board.push(m)
