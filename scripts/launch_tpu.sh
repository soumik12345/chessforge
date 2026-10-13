#!/usr/bin/env bash
# 5M: v5litepod-8 (single host). 23M/79M: v5litepod-16/32 or v4-32.
set -euo pipefail
TPU=${TPU:-maia3}; ZONE=${ZONE:-us-west4-a}; ACCEL=${ACCEL:-v5litepod-16}
VERSION=${VERSION:-v2-alpha-tpuv5-lite}   # v4: tpu-ubuntu2204-base
BUCKET_NAME=${BUCKET_NAME:-YOUR_BUCKET}; CONFIG=${CONFIG:-configs/maia3_79m.yaml}

gcloud compute tpus tpu-vm create "$TPU" --zone "$ZONE" --accelerator-type "$ACCEL" --version "$VERSION"
gcloud compute tpus tpu-vm scp --recurse . "$TPU":~/chessforge --zone "$ZONE" --worker=all
gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --worker=all --command "
  set -e
  cd ~/chessforge
  pip install -U 'jax[tpu]' -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
  pip install -r requirements.txt && pip install -e .
  sudo mkdir -p /mnt/maia3data && sudo chown \$USER /mnt/maia3data
  gcsfuse --implicit-dirs --only-dir maia3/data $BUCKET_NAME /mnt/maia3data
"
gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --worker=all --command "
  cd ~/chessforge && nohup python -m chessforge.cli.train --config $CONFIG > train.log 2>&1 &"