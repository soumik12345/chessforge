"""Streaming Lichess PGN parsing, Maia-3 game filtering/balancing, position extraction."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import re
import sys
from typing import Iterable, Iterator

import chess
import numpy as np
import zstandard

from .. import chess_utils as cu
from . import records

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
MOVE_RE = re.compile(r"([^\s{}()]+)\s*\{([^}]*)\}")
CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
NAG_RE = re.compile(r"\$\d+")
RESULTS = ("1-0", "0-1", "1/2-1/2")
NUM_ELO_BINS = 22  # <600, [600,700), ..., [2500,2600), >=2600


def strip_variations(movetext: str) -> str:
    """Remove parenthesized sidelines (...) outside {...} comments so MOVE_RE
    never matches annotated sideline moves in analyzed Lichess games."""
    if "(" not in movetext:
        return NAG_RE.sub("", movetext)
    out = []
    in_brace = False
    paren_depth = 0
    for ch in movetext:
        if ch == "{" and paren_depth == 0:
            in_brace = True
            out.append(ch)
        elif ch == "}" and in_brace:
            in_brace = False
            out.append(ch)
        elif not in_brace and ch == "(":
            paren_depth += 1
        elif not in_brace and ch == ")":
            paren_depth = max(0, paren_depth - 1)
        elif paren_depth == 0:
            out.append(ch)
    return NAG_RE.sub("", "".join(out))


def open_pgn_stream(path: str, compressed: bool | None = None) -> io.TextIOBase:
    raw = sys.stdin.buffer if path == "-" else open(path, "rb")
    if compressed is None:
        compressed = path == "-" or path.endswith(".zst")
    if compressed:
        raw = zstandard.ZstdDecompressor(max_window_size=2 ** 31).stream_reader(raw)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def iter_games(stream: Iterable[str]) -> Iterator[tuple[dict, str]]:
    """Yields (headers, movetext). Lichess DB puts the movetext on one line."""
    headers: dict = {}
    for line in stream:
        if line.startswith("["):
            m = HEADER_RE.match(line)
            if m:
                if m.group(1) == "Event" and headers:
                    headers = {}  # previous game had no movetext
                headers[m.group(1)] = m.group(2)
            continue
        s = line.strip()
        if not s:
            continue
        if headers:
            yield headers, s
            headers = {}


def elo_bin(avg_elo: float) -> int:
    if avg_elo < 600:
        return 0
    if avg_elo >= 2600:
        return NUM_ELO_BINS - 1
    return 1 + int((avg_elo - 600) // 100)


@dataclasses.dataclass
class GameFilter:
    game_type: str = "Blitz"       # case-sensitive Lichess Event marker
    require_rated: bool = True
    exclude_bots: bool = True
    require_clock: bool = True

    def __call__(self, h: dict, movetext: str) -> bool:
        ev = h.get("Event", "")
        if self.require_rated and "Rated" not in ev:
            return False
        if self.game_type and self.game_type not in ev:
            return False
        if h.get("Result") not in RESULTS:
            return False
        if h.get("Variant", "Standard") != "Standard" or "FEN" in h:
            return False
        try:
            int(h["WhiteElo"])
            int(h["BlackElo"])
            base, inc = h.get("TimeControl", "-").split("+")
            int(base)
            int(inc)
        except (KeyError, ValueError):
            return False
        if self.exclude_bots and "BOT" in (h.get("WhiteTitle", "").upper(), h.get("BlackTitle", "").upper()):
            return False
        if self.require_clock and "[%clk" not in movetext:
            return False
        return True


def balanced_games(games: Iterable[tuple[dict, str]], filt: GameFilter, *, chunk_games: int = 20_000,
                   per_bin: int = 10, chunk_counts: str = "raw", stats: dict | None = None):
    """Maia-3 balancing (App. A.1): per chunk of `chunk_games` games, accept games into
    22 average-Elo bins until each bin has `per_bin` games or the chunk is consumed."""
    counts = [0] * NUM_ELO_BINS
    seen, full = 0, False
    for h, mt in games:
        if stats is not None:
            stats["raw_games"] = stats.get("raw_games", 0) + 1
        ok = filt(h, mt)
        if chunk_counts == "raw" or ok:
            if seen == chunk_games:
                counts, seen, full = [0] * NUM_ELO_BINS, 0, False
            seen += 1
        if full or not ok:
            continue
        b = elo_bin((int(h["WhiteElo"]) + int(h["BlackElo"])) / 2)
        if counts[b] >= per_bin:
            continue
        counts[b] += 1
        if stats is not None:
            stats.setdefault("bin_games", [0] * NUM_ELO_BINS)[b] += 1
        full = all(c >= per_bin for c in counts)
        yield h, mt


def game_id(h: dict, movetext: str) -> str:
    site = h.get("Site", "")
    return site.rsplit("/", 1)[-1] if site else hashlib.md5(movetext.encode()).hexdigest()


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


@dataclasses.dataclass
class PositionConfig:
    store_history: int = 8
    positions_per_game: int = 32   # 0 => keep every eligible position
    skip_first_plies: int = 0      # Maia-3 training keeps the opening; ALLIE eval drops 10
    min_clock_seconds: float = 30.0
    clock_check_on_comment: bool = True  # True: break as soon as [%clk] < min_clock_seconds
    max_ply: int = 100_000
    store_legal: bool = True
    seed: int = 0


def _clock_seconds(comment: str):
    m = CLK_RE.search(comment)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def game_to_records(h: dict, movetext: str, cfg: PositionConfig) -> list[bytes]:
    cleaned = strip_variations(movetext)
    pairs = MOVE_RE.findall(cleaned)
    if not pairs:
        return []
    base, inc = (int(x) for x in h["TimeControl"].split("+"))
    elos = (int(h["WhiteElo"]), int(h["BlackElo"]))
    result = h["Result"]

    board = chess.Board()
    clocks = [float(base), float(base)]
    codes, move_objs, move_idx, clk_lefts, ponders = [], [], [], [], []
    for ply, (san, comment) in enumerate(pairs):
        if ply >= cfg.max_ply:
            break
        if min(clocks) < cfg.min_clock_seconds:
            break
        clk = _clock_seconds(comment)
        if clk is None:
            break
        if cfg.clock_check_on_comment and clk < cfg.min_clock_seconds:
            break
        try:
            move = board.parse_san(san.rstrip("?!+# "))
        except ValueError:
            break
        stm = ply & 1
        clk_before = clocks[stm]
        codes.append(cu.board_codes(board))
        move_objs.append(move)
        move_idx.append(cu.stm_move_index(move, board.turn))
        clk_lefts.append(clk_before)
        ponders.append(max(0.0, clk_before - clk + (inc if ply >= 2 else 0.0)))
        board.push(move)
        clocks[stm] = clk

    n = len(move_idx)
    eligible = np.arange(cfg.skip_first_plies, n)
    if eligible.size == 0:
        return []
    if cfg.positions_per_game and eligible.size > cfg.positions_per_game:
        rng = np.random.default_rng(stable_hash(game_id(h, movetext)) ^ cfg.seed)
        eligible = np.sort(rng.choice(eligible, cfg.positions_per_game, replace=False))
    selected = set(int(p) for p in eligible)

    legal_at = {}
    if cfg.store_legal:
        b = chess.Board()
        last = max(selected)
        for ply in range(last + 1):
            if ply in selected:
                legal_at[ply] = cu.legal_move_indices(b)
            b.push(move_objs[ply])

    codes_arr = np.stack(codes)
    H = cfg.store_history
    out = []
    for p in sorted(selected):
        lo = max(0, p - H + 1)
        window = np.zeros((H, 64), np.int8)
        window[H - (p - lo + 1):] = codes_arr[lo:p + 1]
        stm = p & 1
        out.append(records.encode_position(
            boards=window, n_hist=p - lo, stm=stm, move=move_idx[p],
            self_elo=elos[stm], oppo_elo=elos[1 - stm],
            value=cu.value_label(result, white_to_move=(stm == 0)),
            ply=p, clk_left=clk_lefts[p], ponder=ponders[p],
            base_time=base, increment=inc, legal=legal_at.get(p)))
    return out


class ShardedWriter:
    def __init__(self, out_dir: str, prefix: str, records_per_shard: int = 100_000):
        from array_record.python.array_record_module import ArrayRecordWriter
        self._cls = ArrayRecordWriter
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.rps = out_dir, prefix, records_per_shard
        self._w, self._n, self._shard, self.total = None, 0, 0, 0

    def write(self, payload: bytes):
        if self._w is None or self._n >= self.rps:
            self._rotate()
        self._w.write(payload)
        self._n += 1
        self.total += 1

    def _rotate(self):
        if self._w is not None:
            self._w.close()
        path = os.path.join(self.out_dir, f"{self.prefix}-{self._shard:05d}.array_record")
        self._w = self._cls(path, "group_size:1")
        self._shard += 1
        self._n = 0

    def close(self):
        if self._w is not None:
            self._w.close()
            self._w = None