#!/usr/bin/env bash
set -euo pipefail
# 1) Preferred: the ALLIE authors' test games as PGN (see github.com/y0mingzhang/allie).
python -m chessforge.cli.prepare_eval --input "${ALLIE_TEST_PGN:?}" \
  --output_dir /mnt/maia3data/allie_test --skip_first_plies 10 --min_clock 30
# Expect ~884,049 positions. If not, try --skip_first_plies 20 (moves vs ply ambiguity).

# 2) ALLIE-AUGMENTED extremes (paper App. A.1): Aug/Sep 2025 blitz, rating buckets
#    under-represented in ALLIE, capped at 1000 games per bucket.
# curl -sfL https://database.lichess.org/standard/lichess_db_standard_rated_2025-08.pgn.zst | \
#   python -m chessforge.cli.prepare_eval --input - --output_dir /mnt/maia3data/allie_aug \
#     --filter_game_type Blitz --per_rating_bucket_cap 1000 --only_buckets 600,700,800,2600,2700,2800,2900