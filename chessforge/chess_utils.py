"""Chess encodings, official title tiers, and board geometry.

Stays byte-compatible with maia3/{utils,dataset}.py while adding v2 state & geometry.
"""

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

# Official human titles (FIDE + national + Lichess Master), strictly excluding BOT
TITLE_NONE = 0
TITLE_CANDIDATE_NATIONAL = 1  # CM, WCM, NM, WNM
TITLE_FIDE_MASTER = 2  # FM, WFM, LM
TITLE_INTL_MASTER = 3  # IM, WIM
TITLE_GRANDMASTER = 4  # GM, WGM
NUM_TITLE_TIERS = 5

TITLE_TO_TIER: dict[str, int] = {
    "CM": TITLE_CANDIDATE_NATIONAL,
    "WCM": TITLE_CANDIDATE_NATIONAL,
    "NM": TITLE_CANDIDATE_NATIONAL,
    "WNM": TITLE_CANDIDATE_NATIONAL,
    "FM": TITLE_FIDE_MASTER,
    "WFM": TITLE_FIDE_MASTER,
    "LM": TITLE_FIDE_MASTER,
    "IM": TITLE_INTL_MASTER,
    "WIM": TITLE_INTL_MASTER,
    "GM": TITLE_GRANDMASTER,
    "WGM": TITLE_GRANDMASTER,
}
OFFICIAL_TITLES = frozenset(TITLE_TO_TIER.keys())

# Game format / time-control categories
FORMAT_BLITZ = 0
FORMAT_RAPID = 1
FORMAT_OTB_CLASSICAL = 2
NUM_GAME_FORMATS = 3

# absolute piece codes: 0 empty, 1..6 white P N B R Q K, 7..12 black
SWAP_COLOR = np.array([0, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6], dtype=np.int8)
MIRROR_SQ = (np.arange(64) ^ 56).astype(
    np.int64
)  # python-chess board.mirror() flips ranks


def parse_title_tier(title: str | None) -> int:
    if not title:
        return TITLE_NONE
    return TITLE_TO_TIER.get(title.strip().upper(), TITLE_NONE)


def is_official_title(title: str | None) -> bool:
    return parse_title_tier(title) > TITLE_NONE


def infer_game_format(headers: dict) -> int:
    ev = headers.get("Event", "").lower()
    if "classical" in ev or "otb" in ev or "fide" in ev or "broadcast" in ev:
        return FORMAT_OTB_CLASSICAL
    if "rapid" in ev:
        return FORMAT_RAPID
    if "blitz" in ev or "bullet" in ev:
        return FORMAT_BLITZ
    tc = headers.get("TimeControl", "")
    if "+" in tc:
        try:
            base, inc = (int(x) for x in tc.split("+", 1))
            est = base + 40 * inc
            if est >= 1500:
                return FORMAT_OTB_CLASSICAL
            if est >= 600:
                return FORMAT_RAPID
            return FORMAT_BLITZ
        except ValueError:
            pass
    # Standard tournament PGNs without online Blitz/Rapid markers default to OTB classical
    if "lichess.org" not in headers.get("Site", "").lower() and "+" not in tc:
        return FORMAT_OTB_CLASSICAL
    return FORMAT_BLITZ


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


def castling_mask(board: chess.Board) -> int:
    """4-bit castling rights in side-to-move frame: bit0 own_K, bit1 own_Q, bit2 opp_K, bit3 opp_Q."""
    us, them = board.turn, not board.turn
    mask = 0
    if board.has_kingside_castling_rights(us):
        mask |= 1
    if board.has_queenside_castling_rights(us):
        mask |= 2
    if board.has_kingside_castling_rights(them):
        mask |= 4
    if board.has_queenside_castling_rights(them):
        mask |= 8
    return mask


def ep_file(board: chess.Board) -> int:
    """En-passant target file 0..7 if a legal EP capture exists, else -1."""
    if board.ep_square is not None and board.has_legal_en_passant():
        return chess.square_file(board.ep_square)
    return -1


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


def build_board_geometry():
    """Precompute exact ray-intervening masks and pairwise square relation indices.

    Returns:
      ray_between: float32 (64, 4096), entry [k, i*64+j] = 1 iff k lies strictly between i and j.
      ray_dir_mask: float32 (4, 64, 64) for [rank, file, diag, anti_diag] sliding pairs (i != j).
      pair_rel: int32 (64, 64) in 0..11 categorizing static chess move/attack geometry.
      sq_coords: float32 (64, 4) [file_norm, rank_norm, light_sq, center_proximity].
    """
    ray_between = np.zeros((64, 64 * 64), dtype=np.float32)
    ray_dir_mask = np.zeros((4, 64, 64), dtype=np.float32)
    pair_rel = np.zeros((64, 64), dtype=np.int32)
    sq_coords = np.zeros((64, 4), dtype=np.float32)

    for i in range(64):
        fi, ri = i & 7, i >> 3
        center_d = abs(fi - 3.5) + abs(ri - 3.5)
        sq_coords[i] = [
            fi / 7.0 - 0.5,
            ri / 7.0 - 0.5,
            float((fi + ri) & 1),
            1.0 - center_d / 7.0,
        ]
        for j in range(64):
            if i == j:
                pair_rel[i, j] = 1
                continue
            fj, rj = j & 7, j >> 3
            df, dr = fj - fi, rj - ri
            adf, adr = abs(df), abs(dr)
            cheb = max(adf, adr)

            if (adf == 1 and adr == 2) or (adf == 2 and adr == 1):
                pair_rel[i, j] = 2  # knight jump
                continue

            # Forward/backward pawn capture geometry (relative to stm pushing +rank)
            if adf == 1 and dr == 1:
                pair_rel[i, j] = 3  # own pawn attack / king forward-diag
            elif adf == 1 and dr == -1:
                pair_rel[i, j] = 4  # opp pawn attack / king backward-diag

            direction = -1
            if dr == 0:
                direction = 0  # rank
                if pair_rel[i, j] == 0:
                    pair_rel[i, j] = 5 if cheb == 1 else 6
            elif df == 0:
                direction = 1  # file
                if pair_rel[i, j] == 0:
                    pair_rel[i, j] = 7 if cheb == 1 else 8
            elif df == dr:
                direction = 2  # main diagonal
                if pair_rel[i, j] == 0:
                    pair_rel[i, j] = 9
            elif df == -dr:
                direction = 3  # anti-diagonal
                if pair_rel[i, j] == 0:
                    pair_rel[i, j] = 10

            if direction >= 0:
                ray_dir_mask[direction, i, j] = 1.0
                sf = 0 if df == 0 else (1 if df > 0 else -1)
                sr = 0 if dr == 0 else (1 if dr > 0 else -1)
                for step in range(1, cheb):
                    k = ((ri + step * sr) << 3) | (fi + step * sf)
                    ray_between[k, i * 64 + j] = 1.0

    return ray_between, ray_dir_mask, pair_rel, sq_coords


RAY_BETWEEN, RAY_DIR_MASK, PAIR_REL_INDEX, SQ_COORDS = build_board_geometry()
NUM_PAIR_RELATIONS = 12
