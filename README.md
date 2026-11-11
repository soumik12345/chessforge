### Quick Launch Commands Across All Target Hardware

1. **Build Verified Titled + OTB Shards (All Official Titles Kept):**
   ```bash
   # Lichess Blitz + Rapid: keep >=1800 club anchor, require official title >=2350, NEVER cap titled games
   curl -sfL https://database.lichess.org/standard/lichess_db_standard_rated_2025-05.pgn.zst \
     | python -m chessforge.cli.preprocess \
         --input - --name 2025-05 --output_dir /mnt/maia3data/online_mix/2025-05 \
         --game_type "Blitz,Rapid" --min_elo 1800 \
         --titled_mode above_elo --titled_elo_threshold 2350 --always_keep_titled

   # OTB Classical / TWIC PGNs (multi-line, no [%clk], all officially titled players):
   python -m chessforge.cli.preprocess \
     --input /path/to/twic_2020_2025.pgn --name twic_otb \
     --output_dir /mnt/maia3data/pure_titled_and_otb/twic \
     --game_type "" --allow_unrated --allow_no_clock \
     --min_elo 2100 --titled_mode any --always_keep_titled --per_bin 500
   ```

2. **Colab TPU `v5e-1` (16GB HBM) or `v6e-1` (32GB HBM):**
   ```bash
   # On v5e-1 (16GB): remat_blocks=true, global_batch_size=2048, grad_accum_steps=2 (microbatch=1024)
   python -m chessforge.cli.train --config configs/chessforge_master_28m.yaml

   # On v6e-1 (32GB): disable remat and accum for maximum throughput!
   python -m chessforge.cli.train --config configs/chessforge_master_28m.yaml \
     --set model_overrides.remat_blocks=false optim.grad_accum_steps=1
   ```

3. **Single-Node Multi-GPU (e.g., 4x or 8x A100 / H100 / L4 / 4090) or Multi-Node `torchrun`:**
   ```bash
   # Single-node 4x or 8x GPU (JAX automatically shards across all visible GPUs):
   python -m chessforge.cli.train --config configs/chessforge_master_28m.yaml \
     --set optim.grad_accum_steps=1

   # Multi-node GPU via torchrun:
   torchrun --nnodes=2 --nproc_per_node=8 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29500 \
     -m chessforge.cli.train --config configs/chessforge_master_28m.yaml --set optim.grad_accum_steps=1
   ```

4. **Annotate Shards with Stockfish & Run Anti-Engine DPO + Play via UCI:**
   ```bash
   python -m chessforge.cli.annotate_anti_engine \
     --input_glob '/mnt/maia3data/pure_titled_and_otb/*/val/*.array_record' \
     --output_dir /mnt/maia3data/eval_titled_antisf --stockfish stockfish --nodes 20000

   python -m chessforge.cli.train --config configs/chessforge_master_dpo.yaml

   # Launch as a human-like UCI engine in Nibbler / CuteChess / Lichess-Bot:
   python -m chessforge.cli.uci \
     --config configs/chessforge_master_dpo.yaml --title GM --elo 2600 --format otb
   ```