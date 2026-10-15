"""PGN games -> eval positions with legal-move masks (ALLIE test set or reconstructions).

ALLIE protocol: drop first 10 (ply), truncate after first time a player has < 30 s.
Sanity check: the ALLIE test set should produce 884,049 positions.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os

import numpy as np

from chessforge.data import pgn

_CFG: pgn.PositionConfig | None = None


def _init(cfg: pgn.PositionConfig):
    global _CFG
    _CFG = cfg


def _work(item):
    if _CFG is None:
        raise RuntimeError("Evaluation worker has not been initialized")
    h, mt = item
    try:
        return pgn.game_to_records(h, mt, _CFG)
    except Exception:  # noqa: BLE001
        return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--name", default="eval")
    ap.add_argument("--skip_first_plies", type=int, default=10)
    ap.add_argument("--min_clock", type=float, default=30.0)
    ap.add_argument("--store_history", type=int, default=8)
    ap.add_argument(
        "--filter_game_type", default="", help="e.g. Blitz to apply rated-blitz filter"
    )
    ap.add_argument(
        "--sample_prob", type=float, default=1.0, help="reconstruct: keep games w.p."
    )
    ap.add_argument("--max_games", type=int, default=0)
    ap.add_argument(
        "--per_rating_bucket_cap", type=int, default=0, help="ALLIE-augmented style cap"
    )
    ap.add_argument(
        "--only_buckets",
        default="",
        help="comma list of rounded ratings, e.g. 600,700,2800",
    )
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 2))
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    cfg = pgn.PositionConfig(
        store_history=a.store_history,
        positions_per_game=0,
        skip_first_plies=a.skip_first_plies,
        min_clock_seconds=a.min_clock,
        store_legal=True,
    )
    filt = pgn.GameFilter(game_type=a.filter_game_type) if a.filter_game_type else None
    only = {int(x) for x in a.only_buckets.split(",") if x}
    rng = np.random.default_rng(a.seed)
    caps: dict[int, int] = {}

    def games():
        n = 0
        for h, mt in pgn.iter_games(pgn.open_pgn_stream(a.input)):
            if filt is not None and not filt(h, mt):
                continue
            if "TimeControl" not in h or "+" not in h["TimeControl"]:
                continue
            try:
                bucket = (
                    int(round((int(h["WhiteElo"]) + int(h["BlackElo"])) / 2 / 100))
                    * 100
                )
            except (KeyError, ValueError):
                continue
            if only and bucket not in only:
                continue
            if (
                a.per_rating_bucket_cap
                and caps.get(bucket, 0) >= a.per_rating_bucket_cap
            ):
                continue
            if a.sample_prob < 1.0 and rng.random() > a.sample_prob:
                continue
            caps[bucket] = caps.get(bucket, 0) + 1
            yield h, mt
            n += 1
            if a.max_games and n >= a.max_games:
                return

    w = pgn.ShardedWriter(a.output_dir, a.name, 100_000)
    n_games = 0
    with mp.Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for recs in pool.imap(_work, games(), chunksize=32):
            n_games += 1
            for r in recs:
                w.write(r)
    w.close()
    stats = {"games": n_games, "positions": w.total, "buckets": caps, "args": vars(a)}
    with open(os.path.join(a.output_dir, f"{a.name}_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)
    print(json.dumps({"games": n_games, "positions": w.total}))


if __name__ == "__main__":
    main()
