"""Lichess Blitz/Rapid or OTB PGN(.zst) -> ArrayRecord training & val shards."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time

from chessforge.data import pgn

_CFG: pgn.PositionConfig | None = None
_VAL_PERMILLE = 0


def _init(cfg: pgn.PositionConfig, val_permille: int):
    global _CFG, _VAL_PERMILLE
    _CFG, _VAL_PERMILLE = cfg, val_permille


def _work(item):
    if _CFG is None:
        raise RuntimeError("Preprocessing worker has not been initialized")
    h, mt = item
    is_val = pgn.stable_hash(pgn.game_id(h, mt)) % 1000 < _VAL_PERMILLE
    try:
        recs = pgn.game_to_records(h, mt, _CFG)
    except Exception:  # noqa: BLE001
        recs = []
    return is_val, recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input", required=True, help="path to .pgn/.pgn.zst, or - for zstd stdin"
    )
    ap.add_argument("--name", required=True, help="shard prefix, e.g. 2024-06 or twic")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument(
        "--game_type",
        default="Blitz",
        help="Comma-separated Event markers (e.g. 'Blitz,Rapid'), or '' for OTB PGNs",
    )
    ap.add_argument(
        "--allow_unrated",
        action="store_true",
        help="Allow games without 'Rated' in Event header (required for OTB / TWIC PGNs)",
    )
    ap.add_argument(
        "--allow_no_clock",
        action="store_true",
        help="Allow clockless OTB games without TimeControl/[%%clk] comments",
    )
    ap.add_argument("--min_elo", type=int, default=0, help="Minimum player Elo floor")
    ap.add_argument(
        "--titled_mode",
        choices=["none", "any", "both", "above_elo"],
        default="none",
        help="Official human title filter (GM/WGM/IM/WIM/FM/WFM/CM/WCM/NM/WNM/LM)",
    )
    ap.add_argument("--titled_elo_threshold", type=int, default=2350)
    ap.add_argument(
        "--always_keep_titled",
        action="store_true",
        help="Never drop officially titled players due to per-chunk Elo bin caps",
    )
    ap.add_argument("--chunk_games", type=int, default=20_000)
    ap.add_argument("--per_bin", type=int, default=10)
    ap.add_argument(
        "--bin_caps",
        default="",
        help="Optional 22 comma-separated ints overriding per_bin across the 22 Elo bins",
    )
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

    pos_cfg = pgn.PositionConfig(
        store_history=a.store_history,
        positions_per_game=a.positions_per_game,
        skip_first_plies=a.skip_first_plies,
        min_clock_seconds=a.min_clock,
        allow_no_clock=a.allow_no_clock,
        store_legal=not a.no_legal,
        seed=a.seed,
    )
    filt = pgn.GameFilter(
        game_type=a.game_type,
        require_rated=not a.allow_unrated,
        require_clock=not a.allow_no_clock,
        min_elo=a.min_elo,
        titled_mode=a.titled_mode,
        titled_elo_threshold=a.titled_elo_threshold,
    )
    caps = [int(x) for x in a.bin_caps.split(",") if x.strip()] if a.bin_caps else None
    stats: dict = {}
    games = pgn.balanced_games(
        pgn.iter_games(pgn.open_pgn_stream(a.input)),
        filt,
        chunk_games=a.chunk_games,
        per_bin=a.per_bin,
        bin_caps=caps,
        always_keep_titled=a.always_keep_titled,
        chunk_counts=a.chunk_counts,
        stats=stats,
    )
    train_w = pgn.ShardedWriter(
        os.path.join(a.output_dir, "train"), f"{a.name}-train", a.records_per_shard
    )
    val_w = pgn.ShardedWriter(
        os.path.join(a.output_dir, "val"), f"{a.name}-val", a.records_per_shard
    )
    t0, n_games = time.time(), 0
    with mp.Pool(
        a.workers, initializer=_init, initargs=(pos_cfg, a.val_permille)
    ) as pool:
        for is_val, recs in pool.imap(_work, games, chunksize=32):
            w = val_w if is_val else train_w
            for r in recs:
                w.write(r)
            n_games += 1
            if n_games % 50_000 == 0:
                print(
                    f"[{a.name}] games={n_games} raw={stats.get('raw_games', 0)} "
                    f"titled={stats.get('titled_kept', 0)} train_pos={train_w.total} "
                    f"{time.time() - t0:.0f}s",
                    flush=True,
                )
    train_w.close()
    val_w.close()
    stats.update(
        name=a.name,
        selected_games=n_games,
        train_positions=train_w.total,
        val_positions=val_w.total,
        seconds=time.time() - t0,
        args=vars(a),
    )
    with open(os.path.join(a.output_dir, "stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps({k: v for k, v in stats.items() if k != "args"}))


if __name__ == "__main__":
    main()