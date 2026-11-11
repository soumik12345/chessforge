"""Annotate .array_record shards with Stockfish #1 move & human cp loss for Eval-AntiSF & DPO."""

from __future__ import annotations

import argparse
import glob
import multiprocessing as mp
import os

import chess
import chess.engine
import numpy as np
from array_record.python.array_record_module import ArrayRecordReader

from chessforge import chess_utils as cu
from chessforge.data import pgn, records

_ENG: chess.engine.SimpleEngine | None = None
_LIMIT: chess.engine.Limit | None = None


def _reconstruct_board(rec: np.void) -> chess.Board:
    board = chess.Board.empty()
    codes = rec["boards"][-1]
    for sq in range(64):
        c = int(codes[sq])
        if c == 0:
            continue
        color = chess.WHITE if c <= 6 else chess.BLACK
        ptype = c if c <= 6 else c - 6
        board.set_piece_at(sq, chess.Piece(ptype, color))
    board.turn = chess.BLACK if int(rec["stm"]) == 1 else chess.WHITE

    cmask = int(rec["castling"])
    us, them = board.turn, not board.turn
    cr = 0
    if cmask & 1:
        cr |= chess.BB_H1 if us == chess.WHITE else chess.BB_H8
    if cmask & 2:
        cr |= chess.BB_A1 if us == chess.WHITE else chess.BB_A8
    if cmask & 4:
        cr |= chess.BB_H8 if them == chess.BLACK else chess.BB_H1
    if cmask & 8:
        cr |= chess.BB_A8 if them == chess.BLACK else chess.BB_A1
    board.castling_rights = cr

    epf = int(rec["ep_file"])
    if 0 <= epf <= 7:
        ep_rank = 5 if board.turn == chess.WHITE else 2
        board.ep_square = chess.square(epf, ep_rank)
    return board


def _init_worker(sf_path: str, nodes: int, depth: int):
    global _ENG, _LIMIT
    _ENG = chess.engine.SimpleEngine.popen_uci(sf_path)
    _ENG.configure({"Threads": 1, "Hash": 16})
    _LIMIT = (
        chess.engine.Limit(nodes=nodes)
        if nodes > 0
        else chess.engine.Limit(depth=depth)
    )


def _annotate_raw(raw: bytes) -> bytes:
    assert _ENG is not None and _LIMIT is not None
    rec, legal = records.decode_position(raw)
    board = _reconstruct_board(rec)
    turn = board.turn
    hum_uci = cu.index_to_uci(int(rec["move"]))
    if turn == chess.BLACK:
        hum_uci = cu.mirror_uci(hum_uci)
    try:
        hum_move = chess.Move.from_uci(hum_uci)
        if hum_move not in board.legal_moves:
            return records.encode_record_struct(rec, legal)
        info_best = _ENG.analyse(board, _LIMIT)
        pv = info_best.get("pv", [])
        if not pv:
            return records.encode_record_struct(rec, legal)
        sf_best = pv[0]
        sf_idx = cu.stm_move_index(sf_best, turn)
        best_cp = info_best["score"].pov(turn).score(mate_score=10000) or 0
        if sf_best == hum_move:
            cp_loss = 0
        else:
            info_hum = _ENG.analyse(board, _LIMIT, root_moves=[hum_move])
            hum_cp = info_hum["score"].pov(turn).score(mate_score=10000) or 0
            cp_loss = max(0, best_cp - hum_cp)
        rec = rec.copy()
        rec["sf_move"] = sf_idx
        rec["sf_cp_loss"] = min(cp_loss, 30000)
    except Exception:  # noqa: BLE001
        pass
    return records.encode_record_struct(rec, legal)


def _iter_raw_records(paths: list[str]):
    for p in paths:
        r = ArrayRecordReader(p)
        try:
            for raw in r.read_all():
                yield raw
        finally:
            r.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--name", default="anti_sf")
    ap.add_argument("--stockfish", default="stockfish")
    ap.add_argument("--nodes", type=int, default=20_000, help="0 to use --depth")
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--records_per_shard", type=int, default=50_000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    a = ap.parse_args()

    paths = sorted(glob.glob(a.input_glob))
    if not paths:
        raise FileNotFoundError(f"No shards match {a.input_glob!r}")
    writer = pgn.ShardedWriter(a.output_dir, a.name, a.records_per_shard)
    with mp.Pool(
        a.workers,
        initializer=_init_worker,
        initargs=(a.stockfish, a.nodes, a.depth),
    ) as pool:
        for out_raw in pool.imap(_annotate_raw, _iter_raw_records(paths), chunksize=64):
            writer.write(out_raw)
    writer.close()
    print(f"Annotated {writer.total} positions into {os.path.abspath(a.output_dir)}")


if __name__ == "__main__":
    main()