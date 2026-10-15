# ChessForge

For the Maia-3 human-emulation model family (3M, 5M, 23M, 79M) and its ablations, `chessforge` is an exact architectural, numerical, and training-recipe reproduction of the official `maia3` PyTorch codebase and the ICLR 2026 paper.**

Here is a verification of how `chessforge` lines up with `maia3` and the paper, followed by a few subtle implementation details worth knowing:

#### What Matches 1:1
1. **Board Tokenization & Move Vocabulary (`chessforge/chess_utils.py` vs. `maia3/dataset.py`, `maia3/utils.py`)**
   * **Board representation:** Each board in the $H=8$ history window ($n=7$ past states + current state) is encoded from *its own* side-to-move perspective (rank-mirrored via `sq ^ 56` and color-swapped when Black is to move). Missing history at the start of a game is padded by repeating the earliest available board.
   * **Move indexing:** Identical 4,352-action vocabulary (`64 × 64 = 4096` from-to square pairs, followed by `8 × 8 × 4 = 256` rank-7-to-rank-8 promotions ordered `(from_file, to_file, q/r/b/n)`), with moves mirrored in UCI space when Black is to move.
   * `tests/test_encoding.py` directly verifies byte-for-byte parity against `maia3.dataset.get_historical_tokens` and `tokenize_board`.

2. **Model Architecture & Forward Pass (`chessforge/models/chessformer.py` vs. `maia3/models.py`)**
   * **Skill Conditioning:** Linearly interpolates between two learnable 128-d endpoints (`elo_embedding_low` and `elo_embedding_high`) over $[0, 5000]$ for both `self_elo` and `oppo_elo`, concatenating them with the $12 \times 8 = 96$ board channels to form 64 square tokens of dimension $352$.
   * **Geometric Attention Bias (GAB):** Reproduces both the average-pooling variant (`gab_per_square_dim=0` used in 3M/5M) and the per-square linear projection variant (`gab_per_square_dim=32` used in 23M/79M), exact erf-based GELU (`approximate=False`), standard `LayerNorm` inside GAB, and the shared `(4096, gab_gen_size)` bias template matrix.
   * **Post-Norm Transformer Trunk:** Uses `RMSNorm` with `eps = np.finfo(np.float32).eps` (matching PyTorch's `nn.RMSNorm(eps=None)` default) on the residual branches, `omit_qkv_biases=True`, followed by **two** separate standard `LayerNorm`s (`final_norm` corresponding to `transformer.norm` before the policy head, and `last_ln` after mean pooling before the value/ponder heads).
   * **Policy & Promotion Head:** Scaled dot-product between `proj_sq_from` and `proj_sq_to` ($\div \sqrt{d_{\text{head}}}$) plus `promo_bias_proj(sq_to[:, 56:64]) * sqrt(d_head)` broadcast over rank-7 source files.
   * **Weight Initializations:** Replicates PyTorch's default initializers (`U(-1/sqrt(fan_in), 1/sqrt(fan_in))` on linear layers, packed `(3D, D)` Xavier uniform $\sqrt{6/(4D)}$ on QKV projections, and Xavier normal on Elo embeddings and `gab_shared_weight`).
   * `tests/test_torch_parity.py` and `convert_torch.py` verify 1:1 parameter tree mapping and `< 2e-4` fp32 logit parity with `maia3.models.MAIA3Model`.

3. **Data Balancing & Training Hyperparameters (`pgn.py`, `losses.py`, `schedule.py` vs. Appendix A.1 & Table 4)**
   * **22-bin Elo balancing:** Chunks raw games into blocks of 20,000 and accepts up to 10 games per average-Elo bin (`<600`, twenty 100-point bins `600..2600`, `>=2600`), stopping early when all 22 bins reach 10 games.
   * **Position sampling:** Up to 32 random positions per game, discarding positions after a player's clock drops below 30 seconds, plus 5% random history-length masking during training (`history_mask_prob: 0.05`, Appendix E).
   * **Optimization:** AdamW (`lr=5e-5`, `min_lr=1e-5`, `weight_decay=1e-6`, `grad_clip_norm=3.5`), 1,000-step linear warmup followed by cyclic cosine annealing with 50,000-step cycles over 1,000,000 total steps, and `value_coef=0.1`.

#### Minor Nuances & Caveats to Be Aware Of
* **Paper text typo vs. released code (`head_hid_dim`):** Section 3.3 of the paper states the value head projects to dimension 128, but the official `maia3` code and released Hugging Face weights actually use `head_hid_dim = dim_vit` (`192, 256, 512, 1024`). `chessforge` follows the actual released code/checkpoints.
* **Unused Ponder Head (`ponder_coef=0.0`):** `MAIA3Model` includes `fc_ponder_hid` and `fc_ponder` layers, which `chessforge` instantiates for parameter parity. Since `ponder_coef` defaults to `0.0` (matching Table 4, which only lists `value_coefficient = 0.1`), those two layers receive zero gradients unless you set `ponder_coef > 0`.
* **ALLIE Test Set Move vs. Ply Ambiguity:** Section 4.1 says the ALLIE test set removes "the first 10 moves from each game". In chess codebases this can mean either 10 plies or 20 plies (10 full moves). When running `prepare_eval.py` on the ALLIE PGN, verify that the total position count equals **884,049** (toggle `--skip_first_plies 10` vs. `20` if needed).
* **Scope:** `chessforge` covers the **Maia-3 human-modeling suite** (Section 4). It does not include the Leela-CF 191M MCTS distillation pipeline (Section 5, which uses 112-dim Leela inputs, Nadam, and SWA) or the interpretability cross-layer transcoder (Section 6.2).

---

### How to Train It on TPU to Reproduce the Paper's Scores

To reproduce the Table 1 scores (**55.4%** for `maia3-5m`, **56.6%** for `maia3-23m`, **57.1%** for `maia3-79m` on the ALLIE test set), follow these steps:

#### Step 1: Prepare the ALLIE Evaluation Set & Verify the Pipeline First
Before spending TPU time, convert the ALLIE test set PGN into `.array_record` shards and verify that evaluating the released Hugging Face checkpoints reproduces Table 1:

```bash
# Convert ALLIE test PGN -> ArrayRecord shards
python -m chessforge.cli.prepare_eval \
  --input /path/to/allie_test.pgn \
  --output_dir /mnt/maia3data/allie_test \
  --skip_first_plies 10 \
  --min_clock 30
# Check output JSON: {"positions": 884049}. If not 884049, re-run with --skip_first_plies 20.

# Sanity-check evaluation using the official released Hugging Face weights in JAX:
python -m chessforge.cli.eval \
  --released maia3-79m \
  --test_glob '/mnt/maia3data/allie_test/*.array_record'
```
This should print `"move_acc": ~0.571`.

#### Step 2: Preprocess the Lichess Training Data (Jan 2023 – Jul 2025)
Run `scripts/build_train_data.sh` on a high-CPU VM (e.g., `c3-highcpu-176`) with a fast local scratch SSD. It streams all 31 monthly Lichess dumps (`2023-01` through `2025-07`), applies the rated-Blitz filter, 30s clock cutoff, 22-bin Elo balancing, and 32 positions/game sampling, and uploads `.array_record` shards to GCS:

```bash
export OUT=/mnt/disks/scratch/maia3
export BUCKET=gs://YOUR_BUCKET/maia3/data/train
export PAR=8        # process 8 months concurrently
export WORKERS=20   # multiprocessing workers per month
bash scripts/build_train_data.sh
```
> **CPU Tip:** In `chessforge/cli/preprocess.py`, PGN line scanning and header filtering happen in the main process while `python-chess` move parsing happens in the worker pool. Keeping `PAR=8` to `16` months running concurrently via `xargs -P` is key to saturating a large CPU machine.

Upload your prepared `allie_test` directory to `gs://YOUR_BUCKET/maia3/data/allie_test/` as well.

#### Step 3: Choose the Right TPU Topology & Avoid `gcsfuse` Random-Read Bottlenecks
In the paper (Appendix A.1 & Table 4):
* **3M / 5M & ablations** used 2 GPUs × 128 batch × 4 grad accum = **`global_batch_size: 1024`** for `1,000,000` steps.
  * **Recommended TPU:** `v5litepod-8` (single host, 8 chips → 128 positions/chip, no grad accum needed) or `v4-8` (single host, 4 chips → 256 positions/chip).
* **23M & 79M** used 8 GPUs × 128 batch × 4 grad accum = **`global_batch_size: 4096`** for `1,000,000` steps.
  * **Recommended TPU:** `v5litepod-16` (4 hosts, 16 chips → 256 positions/chip) or `v5litepod-32` / `v4-32` (128 positions/chip).

**Critical I/O Recommendation for Grain + ArrayRecord on TPU:**
`pygrain.ArrayRecordDataSource` with `ds.shuffle()` performs random `pread`s across all shards. Default `gcsfuse` without local file caching can severely bottleneck random reads. Do **one** of the following on your TPU workers:
1. **Use `chessforge`'s built-in local shard stager (simplest & fastest):** Skip `gcsfuse` and pass `gs://` globs directly alongside `data.local_cache_dir=/tmp/maia3_cache` (or a local NVMe mount). `pipeline._host_dataset` will automatically shard the file list across hosts (`files[pidx::pcount]`) and stage only that host's shards to local disk before training starts.
2. **Or enable `gcsfuse` file caching:** If using `gcsfuse`, mount with `--file-cache-max-size-mb=-1 --file-cache-cache-file-for-range-read=true --temp-dir=/tmp/gcsfuse_cache`.

#### Step 4: Launch Training on TPU

**1. Training `maia3-5m` (Single-host `v5litepod-8`, `global_batch_size=1024`):**
```bash
python -m chessforge.cli.train \
  --config configs/maia3_5m.yaml \
  --set \
    workdir=gs://YOUR_BUCKET/maia3/runs/maia3-5m \
    data.train_glob='gs://YOUR_BUCKET/maia3/data/train/*/train/*.array_record' \
    data.val_glob='gs://YOUR_BUCKET/maia3/data/train/*/val/*.array_record' \
    data.test_glob='gs://YOUR_BUCKET/maia3/data/allie_test/*.array_record' \
    data.local_cache_dir=/tmp/maia3_cache
```

**2. Training `maia3-23m` (Multi-host `v5litepod-16` or `v4-32`, `global_batch_size=4096`):**
Note that `maia3-23m` shares the 8-GPU (`global_batch_size=4096`) setup of `configs/maia3_79m.yaml`:
```bash
gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --worker=all --command "
  cd ~/chessforge && nohup python -m chessforge.cli.train \
    --config configs/maia3_79m.yaml \
    --set \
      model=maia3-23m \
      workdir=gs://YOUR_BUCKET/maia3/runs/maia3-23m \
      data.train_glob='gs://YOUR_BUCKET/maia3/data/train/*/train/*.array_record' \
      data.val_glob='gs://YOUR_BUCKET/maia3/data/train/*/val/*.array_record' \
      data.test_glob='gs://YOUR_BUCKET/maia3/data/allie_test/*.array_record' \
      data.local_cache_dir=/tmp/maia3_cache \
    > train.log 2>&1 &"
```

**3. Training `maia3-79m` (Multi-host `v5litepod-16` or `v4-32`, `global_batch_size=4096`):**
```bash
gcloud compute tpus tpu-vm ssh "$TPU" --zone "$ZONE" --worker=all --command "
  cd ~/chessforge && nohup python -m chessforge.cli.train \
    --config configs/maia3_79m.yaml \
    --set \
      workdir=gs://YOUR_BUCKET/maia3/runs/maia3-79m \
      data.train_glob='gs://YOUR_BUCKET/maia3/data/train/*/train/*.array_record' \
      data.val_glob='gs://YOUR_BUCKET/maia3/data/train/*/val/*.array_record' \
      data.test_glob='gs://YOUR_BUCKET/maia3/data/allie_test/*.array_record' \
      data.local_cache_dir=/tmp/maia3_cache \
    > train.log 2>&1 &"
```

#### Step 5: Evaluate Your Trained Checkpoints
During training, `Trainer.fit()` automatically evaluates on `val_glob` every 10,000 steps and on `test_glob` (ALLIE test set) every 50,000 steps and at step 1,000,000. You can also run standalone evaluation (in `float32` or `bfloat16`) at any checkpoint step:

```bash
python -m chessforge.cli.eval \
  --config configs/maia3_79m.yaml \
  --set workdir=gs://YOUR_BUCKET/maia3/runs/maia3-79m \
  --test_glob 'gs://YOUR_BUCKET/maia3/data/allie_test/*.array_record' \
  --step 1000000 \
  --out results_79m.json
```
This outputs `move_acc` (legal-move masked argmax accuracy, matching Table 1), `move_acc_ci95`, `perplexity` (matching Figure 7), `value_acc` / `value_nll` (matching Table 7), and the per-100-Elo breakdown (`per_rating`, matching Figures 2 and 6).