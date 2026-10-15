"""Lichess monthly PGN(.zst) -> Maia-3 balanced training positions (ArrayRecord).

  curl -sfL https://database.lichess.org/standard/lichess_db_standard_rated_2023-01.pgn.zst \
    | python -m chessforge.cli.preprocess --input - --name 2023-01 --output_dir /data/2023-01
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

from chessforge.data import pgn

_CFG: pgn.PositionConfig | None = None
_VAL_PERMILLE = 0


def _init(cfg, val_permille):
    global _CFG, _VAL_PERMILLE
    _CFG, _VAL_PERMILLE = cfg, val_permille


def _work(item):
    h, mt = item
    is_val = pgn.stable_hash(pgn.game_id(h, mt)) % 1000 < _VAL_PERMILLE
    try:
        recs = pgn.game_to_records(h, mt, _CFG)
    except Exception:  # noqa: BLE001  malformed game
        recs = []
    return is_val, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="path to .pgn/.pgn.zst, or - for zstd stdin")
    ap.add_argument("--name", required=True, help="shard prefix, e.g. 2023-01")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--game_type", default="Blitz")
    ap.add_argument("--chunk_games", type=int, default=20_000)
    ap.add_argument("--per_bin", type=int, default=10)
    ap.add_argument("--chunk_counts", choices=["raw", "filtered"], default="raw")
    ap.add_argument("--positions_per_game", type=int, default=32)
    ap.add_argument("--store_history", type=int, default=8)
    ap.add_argument("--min_clock", type=float, default=30.0)
    ap.add_argument("--skip_first_plies", type=int, default=0)
    ap.add_argument("--no_legal", action="store_true")
    ap.add_argument("--val_permille", type=int, default=2)
    ap.add_argument("--records_per_shard", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    pos_cfg = pgn.PositionConfig(store_history=a.store_history, positions_per_game=a.positions_per_game,
                                 skip_first_plies=a.skip_first_plies, min_clock_seconds=a.min_clock,
                                 store_legal=not a.no_legal, seed=a.seed)
    filt = pgn.GameFilter(game_type=a.game_type)
    stats: dict = {}
    games = pgn.balanced_games(pgn.iter_games(pgn.open_pgn_stream(a.input)), filt,
                               chunk_games=a.chunk_games, per_bin=a.per_bin,
                               chunk_counts=a.chunk_counts, stats=stats)
    train_w = pgn.ShardedWriter(os.path.join(a.output_dir, "train"), f"{a.name}-train", a.records_per_shard)
    val_w = pgn.ShardedWriter(os.path.join(a.output_dir, "val"), f"{a.name}-val", a.records_per_shard)
    t0, n_games = time.time(), 0
    with mp.Pool(a.workers, initializer=_init, initargs=(pos_cfg, a.val_permille)) as pool:
        for is_val, recs in pool.imap(_work, games, chunksize=32):
            w = val_w if is_val else train_w
            for r in recs:
                w.write(r)
            n_games += 1
            if n_games % 50_000 == 0:
                print(f"[{a.name}] games={n_games} raw={stats.get('raw_games', 0)} "
                      f"train_pos={train_w.total} {time.time() - t0:.0f}s", flush=True)
    train_w.close()
    val_w.close()
    stats.update(name=a.name, selected_games=n_games, train_positions=train_w.total,
                 val_positions=val_w.total, seconds=time.time() - t0, args=vars(a))
    with open(os.path.join(a.output_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}))


if __name__ == "__main__":
    main()