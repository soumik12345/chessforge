#!/usr/bin/env bash
# Run on a large CPU VM (e.g. c3-highcpu-176). Streams each month straight from
# Lichess (no raw-file storage), preprocesses, uploads to GCS.
set -euo pipefail
OUT=${OUT:-/mnt/disks/scratch/maia3}
BUCKET=${BUCKET:-gs://YOUR_BUCKET/maia3/data/train}
PAR=${PAR:-8}            # months in parallel
WORKERS=${WORKERS:-20}   # parse workers per month
export OUT BUCKET WORKERS

months() {
  for y in 2023 2024 2025; do
    for m in 01 02 03 04 05 06 07 08 09 10 11 12; do
      [[ "$y-$m" > "2025-07" ]] && continue
      echo "$y-$m"
    done
  done
}

months | xargs -P "$PAR" -I{} bash -c '
  set -euo pipefail
  curl -sfL "https://database.lichess.org/standard/lichess_db_standard_rated_{}.pgn.zst" \
    | python -m chessforge.cli.preprocess --input - --name {} --output_dir "$OUT/{}" --workers "$WORKERS"
  gcloud storage cp -r "$OUT/{}" "$BUCKET/"
  rm -rf "$OUT/{}"'