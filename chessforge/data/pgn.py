"""Streaming Lichess & OTB PGN parsing, official-title filtering/balancing, position extraction."""

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
from array_record.python.array_record_module import ArrayRecordWriter

from chessforge import chess_utils as cu
from chessforge.data import records

HEADER_RE = re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
TOKEN_OR_COMMENT_RE = re.compile(r"\{([^}]*)\}|([^\s{}]+)")
CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
NAG_RE = re.compile(r"\$\d+")
MOVE_NUM_RE = re.compile(r"^\d+\.+$")
RESULTS = ("1-0", "0-1", "1/2-1/2")
RESULT_TOKENS = frozenset((*RESULTS, "*"))
NUM_ELO_BINS = 22  # <600, [600,700), ..., [2500,2600), >=2600


def strip_variations(movetext: str) -> str:
    """Remove parenthesized sidelines (...) outside {...} comments."""
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


def extract_moves_and_comments(movetext: str) -> list[tuple[str, str]]:
    """Extract (san, comment) pairs from either Lichess commented movetext or plain OTB PGNs."""
    cleaned = strip_variations(movetext)
    pairs: list[tuple[str, str]] = []
    for m in TOKEN_OR_COMMENT_RE.finditer(cleaned):
        comment, token = m.group(1), m.group(2)
        if comment is not None:
            if pairs:
                prev_san, prev_c = pairs[-1]
                pairs[-1] = (prev_san, f"{prev_c} {comment}" if prev_c else comment)
            continue
        if not token or token in RESULT_TOKENS or MOVE_NUM_RE.match(token):
            continue
        if "." in token:
            token = token.rsplit(".", 1)[-1]
            if not token or token in RESULT_TOKENS:
                continue
        pairs.append((token, ""))
    return pairs


def open_pgn_stream(path: str, compressed: bool | None = None) -> io.TextIOBase:
    raw = sys.stdin.buffer if path == "-" else open(path, "rb")
    if compressed is None:
        compressed = path == "-" or path.endswith(".zst")
    if compressed:
        raw = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(raw)
    return io.TextIOWrapper(raw, encoding="utf-8", errors="replace")


def iter_games(stream: Iterable[str]) -> Iterator[tuple[dict, str]]:
    """Yields (headers, movetext) for both single-line Lichess dumps and multi-line OTB PGNs."""
    headers: dict = {}
    movetext_lines: list[str] = []
    for line in stream:
        if line.startswith("["):
            m = HEADER_RE.match(line)
            if m:
                if movetext_lines and headers:
                    yield headers, " ".join(movetext_lines)
                    headers, movetext_lines = {}, []
                elif m.group(1) == "Event" and headers and not movetext_lines:
                    headers = {}
                headers[m.group(1)] = m.group(2)
                continue
        s = line.strip()
        if not s:
            if movetext_lines and headers:
                yield headers, " ".join(movetext_lines)
                headers, movetext_lines = {}, []
            continue
        if headers:
            movetext_lines.append(s)
    if headers and movetext_lines:
        yield headers, " ".join(movetext_lines)


def elo_bin(avg_elo: float) -> int:
    if avg_elo < 600:
        return 0
    if avg_elo >= 2600:
        return NUM_ELO_BINS - 1
    return 1 + int((avg_elo - 600) // 100)


def game_has_official_title(h: dict, require_both: bool = False) -> bool:
    wt = cu.is_official_title(h.get("WhiteTitle"))
    bt = cu.is_official_title(h.get("BlackTitle"))
    return (wt and bt) if require_both else (wt or bt)


def parse_time_control_header(tc: str) -> tuple[int, int, bool]:
    """Returns (base_seconds, increment_seconds, valid_plus_format)."""
    if "+" in tc:
        try:
            b, inc = tc.split("+", 1)
            return int(b), int(inc), True
        except ValueError:
            pass
    return 0, 0, False


@dataclasses.dataclass
class GameFilter:
    game_type: str = (
        "Blitz"  # comma-separated Event substrings (e.g. "Blitz,Rapid") or "" for OTB
    )
    require_rated: bool = True
    exclude_bots: bool = True
    exclude_rule_infractions: bool = True
    require_clock: bool = True
    min_elo: int = 0
    titled_mode: str = "none"  # none | any | both | above_elo
    titled_elo_threshold: int = 2350

    def __call__(self, h: dict, movetext: str) -> bool:
        ev = h.get("Event", "")
        if self.require_rated and "Rated" not in ev:
            return False
        if self.game_type:
            allowed = [g.strip() for g in self.game_type.split(",") if g.strip()]
            if allowed and not any(g in ev for g in allowed):
                return False
        if h.get("Result") not in RESULTS:
            return False
        if h.get("Variant", "Standard") != "Standard" or "FEN" in h:
            return False
        if (
            self.exclude_rule_infractions
            and "infraction" in h.get("Termination", "").lower()
        ):
            return False
        wt = h.get("WhiteTitle", "").strip().upper()
        bt = h.get("BlackTitle", "").strip().upper()
        if self.exclude_bots and ("BOT" in (wt, bt)):
            return False
        try:
            w_elo = int(h["WhiteElo"])
            b_elo = int(h["BlackElo"])
        except (KeyError, ValueError):
            return False
        avg_elo = 0.5 * (w_elo + b_elo)
        if self.min_elo and min(w_elo, b_elo) < self.min_elo:
            return False

        if self.titled_mode == "any" and not game_has_official_title(h, False):
            return False
        if self.titled_mode == "both" and not game_has_official_title(h, True):
            return False
        if (
            self.titled_mode == "above_elo"
            and max(w_elo, b_elo) >= self.titled_elo_threshold
            and not game_has_official_title(h, False)
        ):
            return False

        if self.require_clock:
            _, _, ok_tc = parse_time_control_header(h.get("TimeControl", "-"))
            if not ok_tc or "[%clk" not in movetext:
                return False
        return True


def balanced_games(
    games: Iterable[tuple[dict, str]],
    filt: GameFilter,
    *,
    chunk_games: int = 20_000,
    per_bin: int = 10,
    bin_caps: list[int] | None = None,
    always_keep_titled: bool = False,
    chunk_counts: str = "raw",
    stats: dict | None = None,
):
    """Per chunk of `chunk_games` games, accept up to `caps[b]` games per Elo bin.
    When `always_keep_titled=True`, games with any officially titled human player bypass bin caps."""
    caps = list(bin_caps) if bin_caps is not None else [per_bin] * NUM_ELO_BINS
    assert len(caps) == NUM_ELO_BINS
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
        if not ok:
            continue
        titled = always_keep_titled and game_has_official_title(h, False)
        b = elo_bin((int(h["WhiteElo"]) + int(h["BlackElo"])) / 2)
        if not titled:
            if full or counts[b] >= caps[b]:
                continue
            counts[b] += 1
            full = all(c >= cap for c, cap in zip(counts, caps))
        else:
            counts[b] = min(counts[b] + 1, caps[b])
            if stats is not None:
                stats["titled_kept"] = stats.get("titled_kept", 0) + 1
        if stats is not None:
            stats.setdefault("bin_games", [0] * NUM_ELO_BINS)[b] += 1
        yield h, mt


def game_id(h: dict, movetext: str) -> str:
    site = h.get("Site", "")
    if site and "lichess.org/" in site:
        return site.rsplit("/", 1)[-1]
    key = f"{h.get('Date', '')}|{h.get('White', '')}|{h.get('Black', '')}|{movetext[:128]}"
    return hashlib.md5(key.encode()).hexdigest()


def stable_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little")


@dataclasses.dataclass
class PositionConfig:
    store_history: int = 8
    positions_per_game: int = 32  # 0 => keep every eligible position
    skip_first_plies: int = 0
    min_clock_seconds: float = 30.0
    clock_check_on_comment: bool = False
    allow_no_clock: bool = False  # True for OTB / TWIC classical PGNs without [%clk]
    max_ply: int = 100_000
    store_legal: bool = True
    seed: int = 0


def _clock_seconds(comment: str):
    m = CLK_RE.search(comment)
    if not m:
        return None
    return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))


def game_to_records(h: dict, movetext: str, cfg: PositionConfig) -> list[bytes]:
    pairs = extract_moves_and_comments(movetext)
    if not pairs:
        return []
    base, inc, ok_tc = parse_time_control_header(h.get("TimeControl", ""))
    has_clk_stream = ok_tc and ("[%clk" in movetext)
    if not has_clk_stream and not cfg.allow_no_clock:
        return []

    elos = (int(h["WhiteElo"]), int(h["BlackElo"]))
    titles = (
        cu.parse_title_tier(h.get("WhiteTitle")),
        cu.parse_title_tier(h.get("BlackTitle")),
    )
    g_format = cu.infer_game_format(h)
    result = h["Result"]

    board = chess.Board()
    clocks = [float(base), float(base)]
    seen_keys: dict[tuple, int] = {}

    codes, move_objs, move_idx = [], [], []
    clk_lefts, opp_clk_lefts, ponders, prev_opp_ponders = [], [], [], []
    castlings, ep_files, rep_counts = [], [], []

    for ply, (san, comment) in enumerate(pairs):
        if ply >= cfg.max_ply:
            break
        if has_clk_stream:
            if min(clocks) < cfg.min_clock_seconds:
                break
            clk = _clock_seconds(comment)
            if clk is None:
                break
            if cfg.clock_check_on_comment and clk < cfg.min_clock_seconds:
                break
        else:
            clk = None

        try:
            move = board.parse_san(san.rstrip("?!+# "))
        except ValueError:
            break

        stm = ply & 1
        tkey = board._transposition_key()
        rep_c = seen_keys.get(tkey, 0)

        codes.append(cu.board_codes(board))
        castlings.append(cu.castling_mask(board))
        ep_files.append(cu.ep_file(board))
        rep_counts.append(min(rep_c, 2))
        move_objs.append(move)
        move_idx.append(cu.stm_move_index(move, board.turn))

        if has_clk_stream and clk is not None:
            clk_before = clocks[stm]
            clk_lefts.append(clk_before)
            opp_clk_lefts.append(clocks[1 - stm])
            prev_opp_ponders.append(ponders[-1] if ponders else np.nan)
            ponders.append(max(0.0, clk_before - clk + (inc if ply >= 2 else 0.0)))
            clocks[stm] = clk
        else:
            clk_lefts.append(0.0)
            opp_clk_lefts.append(0.0)
            prev_opp_ponders.append(np.nan)
            ponders.append(np.nan)

        if board.is_zeroing(move):
            seen_keys.clear()
        else:
            seen_keys[tkey] = rep_c + 1
        board.push(move)

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
        window[H - (p - lo + 1) :] = codes_arr[lo : p + 1]
        stm = p & 1
        next_mv = move_idx[p + 1] if (p + 1) < n else -1
        out.append(
            records.encode_position(
                boards=window,
                n_hist=p - lo,
                stm=stm,
                move=move_idx[p],
                self_elo=elos[stm],
                oppo_elo=elos[1 - stm],
                value=cu.value_label(result, white_to_move=(stm == 0)),
                ply=p,
                clk_left=clk_lefts[p],
                opp_clk_left=opp_clk_lefts[p],
                ponder=ponders[p],
                prev_opp_ponder=prev_opp_ponders[p],
                base_time=base if has_clk_stream else 0,
                increment=inc if has_clk_stream else 0,
                castling=castlings[p],
                ep_file=ep_files[p],
                rep_count=rep_counts[p],
                self_title=titles[stm],
                oppo_title=titles[1 - stm],
                game_format=g_format,
                has_clock=int(has_clk_stream),
                next_move=next_mv,
                legal=legal_at.get(p),
            )
        )
    return out


class ShardedWriter:
    def __init__(self, out_dir: str, prefix: str, records_per_shard: int = 100_000):
        self._cls = ArrayRecordWriter
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.prefix, self.rps = out_dir, prefix, records_per_shard
        self._w, self._n, self._shard, self.total = None, 0, 0, 0

    def write(self, payload: bytes):
        if self._w is None or self._n >= self.rps:
            self._rotate()
        assert self._w is not None
        self._w.write(payload)
        self._n += 1
        self.total += 1

    def _rotate(self):
        if self._w is not None:
            self._w.close()
        path = os.path.join(
            self.out_dir, f"{self.prefix}-{self._shard:05d}.array_record"
        )
        self._w = self._cls(path, "group_size:1")
        self._shard += 1
        self._n = 0

    def close(self):
        if self._w is not None:
            self._w.close()
            self._w = None
