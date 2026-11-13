# Joint move and thinking-time ablations: J0–J4

This study tests whether modeling the relationship between human moves and
thinking time improves held-out move prediction beyond an independent effort
head or extra policy capacity. It follows the fixed-update, fixed-example
approach of the earlier [R1–R6 ablations](https://github.com/soumik12345/chessforge/blob/improvement-ablations/EXPERIMENTS.md#52-controlled-compute-matched-ablation-ladder-r1r6).
The definitions below describe the current implementation; they are not claims
of measured improvement. No completed J0–J4 results are recorded here yet.

## 1. Model and objectives

Let `x` denote the board and pre-move context, `m` the observed human move, and
`k` one of 16 effort buckets. The joint model predicts

```text
p(m, k | x) = p(k | x) p(m | k, x)
p(m | x) = sum_k p(k | x) p(m | k, x)
```

Each bucket's move distribution is normalized over legal moves before mixing.
A rank-16 residual conditions the existing policy head on effort; its gates
start at zero. The shared trunk, value head, and effort head are retained. The
ordinary move output is the predicted-effort marginal, never conditioned on the
observed future thinking time. See [prototype details](docs/joint_move_time.md).

The effort target is a smoothed histogram `q(k)` around

```text
u = clip(log1p(ponder_seconds) / log1p(max(pre_move_clock, 1)), 0, 1)
```

The histogram uses 16 equal-width buckets and Gaussian smoothing with
`sigma = 0.75 / 16`, renormalized over `[0, 1]`. Valid time supervision requires
finite nonnegative `ponder`, finite `clk_left > 0.5`, finite `has_clock > 0.5`,
and a finite positive position weight. Invalid time labels do not exclude an
otherwise valid example from move supervision.

Define the following losses, with position weights understood:

| Symbol | Definition | Averaging denominator |
| --- | --- | --- |
| `M` | Human-move NLL, `-log p(m | x)` | All position weight |
| `E` | Marginal effort CE, `-sum_k q(k) log p(k | x)` | Valid-clock position weight |
| `C` | Posterior effort CE, `-sum_k q(k) log p(k | m, x)` | All position weight; invalid-clock rows contribute zero |
| `V` | Three-class loss/draw/win cross-entropy | All position weight |
| `R` | Observed next human move cross-entropy, in the opponent frame | Weight of rows with a valid `next_move` |

The implementation floors training denominators at one. With ordinary unit
weights these definitions are the usual means; an all-invalid time microbatch
has zero time loss. The common supervised objective is `M + 0.10 V + 0.15 R`.
These experiments do not use DPO.

## 2. Ablation matrix

All five runs use the same V2 backbone, pre-move inputs, data, optimizer recipe,
batch size, and training budget. Change only the three settings below.

| Run | Description | `model_overrides.use_joint_move_time` | `joint_time_coef` | `ponder_coef` | Total objective |
| --- | --- | --- | --- | --- | --- |
| **J0** | V2 baseline without time supervision | `false` | `0.0` | `0.0` | `M + 0.10 V + 0.15 R` |
| **J1** | Independent move and effort heads | `false` | `0.0` | `1.0` | `M + E + 0.10 V + 0.15 R` |
| **J2** | Effort-conditioned policy without time labels | `true` | `0.0` | `0.0` | `M + 0.10 V + 0.15 R` |
| **J3** | Effort-conditioned policy with marginal effort supervision | `true` | `0.0` | `1.0` | `M + E + 0.10 V + 0.15 R` |
| **J4** | Full joint move/effort objective | `true` | `1.0` | `0.0` | `M + C + 0.10 V + 0.15 R` |

J0 and J1 retain the existing effort head and pre-move clock inputs; disabling
time supervision does not remove clock context. J2's effort prior is learned
through move prediction, but its latent buckets have no supervised thinking-time
meaning. J3 couples the move policy to effort while retaining separate move and
marginal-effort losses. At zero label smoothing and on valid-clock rows, J4's
`M + C` equals the smoothed joint NLL. This identity applies to those two terms,
not to the additional value/reply losses.

### Questions answered by each comparison

| Comparison | Question |
| --- | --- |
| J1 vs. J0 | Does the independent effort auxiliary task improve moves? |
| J2 vs. J0 | Does extra effort-conditioned policy capacity help without time labels? |
| J3 vs. J1 | Does conditioning the policy on effort help under the same loss form? |
| J4 vs. J3 | Does posterior effort supervision improve on marginal effort supervision? |
| J4 vs. J2 | Does assigning observed-time meaning to the joint buckets help? |
| J4 vs. J1 | Does the proposed method outperform the independent-head reference overall? |

J4 vs. J1 changes both architecture and objective. Use the intermediate controls
to interpret that difference. Compare held-out move NLL and accuracy rather than
total training loss: the latter includes different supervision across runs.

## 3. Shared training protocol

The starting recipe is [chessforge_joint_time_28m.yaml](configs/chessforge_joint_time_28m.yaml),
with the overrides below. The preset name is `28m`; the current J0 initialization
reports **35,229,229 parameters**. The rank-16 joint residual adds 12,608 parameters.
Use the logged count when reporting model size.

| Setting | Shared value |
| --- | --- |
| Model | `chessforge-v2-28m` |
| Precision / gradient checkpointing | `bfloat16` / `remat_blocks=true` |
| History / history-mask probability | 8 boards / `0.05` |
| Effort buckets / joint residual rank | 16 / 16 |
| Optimizer | AdamW; `b1=0.9`, `b2=0.999`, `eps=1e-8`, weight decay `1e-4` |
| Learning rate | WSD; peak `4e-4`, minimum `2e-6` |
| Schedule | 2,000 warmup updates, stable through update 8,000, final 20% cosine decay |
| Gradient clipping | Global norm threshold `1.0` |
| Training budget | 10,000 optimizer updates × 1,024 positions = **10.24M presentations** |
| Accumulation | 16; microbatch per device is `1024 / (device_count × 16)` |
| Evaluation batch size | 64 |
| Other loss settings | Value `0.10`, reply `0.15`, label smoothing `0`, DPO `0` |
| Legal masking | `mask_illegal_in_loss=true` |
| Curriculum | Disabled: `data.anneal_glob=""` |
| Logging / validation / checkpoint interval | 100 / 1,000 / 1,000 updates |
| Seeds | Screen with 42; repeat matched comparisons with 43 and 44 |

On one device the microbatch is 64. This is a matched-example/update study, not
a guarantee of equal FLOPs or wall time. Keep hardware and precision fixed and
report measured throughput. Train each variant from scratch; baseline and joint
parameter trees are incompatible for direct checkpoint restoration. Use a unique
workdir and match both `seed` and `data.shuffle_seed` across variants for each seed.

### Local data

| Role | Path relative to the repository root |
| --- | --- |
| Training | `data/cf_data/pool_B_titled/2025-06/train/*.array_record` |
| Validation | `data/cf_data/pool_B_titled/2025-06/val/*.array_record` |
| Held-out titled online | `data/cf_data/eval_titled_online/*.array_record` |
| Held-out OTB | `data/cf_data/eval_otb/*.array_record` |

Pool B's [preprocessing metadata](data/cf_data/pool_B_titled/2025-06/stats.json)
reports 4,083,779 training positions, 7,774 validation positions, required clock
streams, Blitz/Rapid games, and a 30-second cutoff. The directory contains 41
training shards and one validation shard. The training budget is approximately
2.51 passes over the recorded training positions, not 10.24M unique positions.
The startup position count is an estimate from buffered shards and can differ
from the preprocessing total.

Required clock streams do not by themselves prove that every time label is
valid. Audit the validity conditions in section 1 before interpreting the study
as fully clock-labeled. If clockless/invalid rows remain, `E` and `C` have
different denominators: the marginal effort loss averages valid labels, whereas
the posterior term includes all position weights. Exact mixed-data baseline
reweighting is not implemented. Report that difference rather than attributing
all changes solely to factorization.

The current J0 configuration retains `data.val_batches=50`, so each training-time
validation pass evaluates the first **3,200 positions**, not all 7,774. Keep that
subset fixed across J0–J4. Use standalone evaluation for full-split reporting;
changing the validation limit during the study would change comparability.

Keep test data out of training and experiment selection. Verify game-level
separation and duplicates across splits. The existing Pool B cutoff excludes
sub-30-second play, so it cannot establish time-pressure performance. A separate
dataset prepared with a lower cutoff is needed for that extension.

## 4. Launching J0–J4 and tracking runs

Run the following from the repository root. Select `ABLATION` and `SEED`; the
case statement sets the three ablation-specific overrides. This block starts one
run, not all five.

```bash
export TRACKIO_DIR="$PWD/trackio_logs"
ABLATION=J0
SEED=42

case "$ABLATION" in
  J0) USE_JOINT=false; JOINT_COEF=0.0; PONDER_COEF=0.0 ;;
  J1) USE_JOINT=false; JOINT_COEF=0.0; PONDER_COEF=1.0 ;;
  J2) USE_JOINT=true;  JOINT_COEF=0.0; PONDER_COEF=0.0 ;;
  J3) USE_JOINT=true;  JOINT_COEF=0.0; PONDER_COEF=1.0 ;;
  J4) USE_JOINT=true;  JOINT_COEF=1.0; PONDER_COEF=0.0 ;;
  *) echo "Choose J0, J1, J2, J3, or J4"; exit 1 ;;
esac

.venv/bin/python -m chessforge.cli.train \
  --config configs/chessforge_joint_time_28m.yaml \
  --set \
    workdir="$PWD/runs/${ABLATION}_s${SEED}" \
    trackio_project=chessforge \
    trackio_run_name="${ABLATION}_s${SEED}" \
    trackio_group=joint_move_time_ablations \
    trackio_space_id=geekyrakshit/chessforge-tracking \
    model_overrides.use_joint_move_time="$USE_JOINT" \
    model_overrides.joint_move_time_rank=16 \
    model_overrides.ponder_buckets=16 \
    joint_time_coef="$JOINT_COEF" \
    ponder_coef="$PONDER_COEF" \
    seed="$SEED" \
    data.shuffle_seed="$SEED" \
    data.train_glob='data/cf_data/pool_B_titled/2025-06/train/*.array_record' \
    data.val_glob='data/cf_data/pool_B_titled/2025-06/val/*.array_record' \
    'data.test_glob=""' \
    'data.anneal_glob=""' \
    data.global_batch_size=1024 \
    data.eval_batch_size=64 \
    data.val_batches=50 \
    optim.total_steps=10000 \
    optim.grad_accum_steps=16 \
    value_coef=0.10 \
    reply_coef=0.15 \
    label_smoothing=0.0 \
    dpo_coef=0.0 \
    log_every=100 \
    eval_every=1000 \
    ckpt_every=1000
```

The saved J0 config at `runs/J0_s42/config.json` records project `chessforge`,
run name `J0`, group `joint_move_time_ablations`, and Space
[`geekyrakshit/chessforge-tracking`](https://huggingface.co/spaces/geekyrakshit/chessforge-tracking).
The command above uses seed-qualified run names for future launches. The current
`run.sh` repeats tracking overrides; the last occurrence wins, explaining why
the saved run is named `J0` rather than `J0_s42`. Preserve the saved config as the
record of what that run actually used.

To use local-only tracking, omit `trackio_space_id` and ensure
`TRACKIO_SPACE_ID` is unset; the YAML default is `null`. Remote Space logging
requires an authenticated account with write access. To inspect the local project:

```bash
export TRACKIO_DIR="$PWD/trackio_logs"
.venv/bin/trackio show --project chessforge
```

Existing workdirs automatically resume training checkpoints. Trackio run resume
is separate: the trainer does not currently expose or pass Trackio's `resume`
option. Do not assume a restarted training process appends to the same tracking
run. Configuration, checkpoints, and iterator state live in the workdir; metric
tracking does not replace those artifacts. The trainer has periodic text logs,
not a progress bar.

## 5. Metric conventions and aggregation

Definitions are sourced from [training losses](chessforge/train/losses.py),
[trainer aggregation and logging](chessforge/train/trainer.py), and
[evaluation](chessforge/eval/evaluate.py). Accuracy and probability metrics are
fractions in `[0, 1]`; multiply by 100 for percentages. Cross-entropies, NLLs,
and entropies use natural logarithms and are in **nats**. Lower NLL is better;
higher accuracy is better. Counts, learning rate, and utilization are diagnostics,
not quality scores.

Only JAX process zero writes Trackio data. Chess metrics are logged against
optimizer-update `step`: `train/` every 100 updates; `val/` every 1,000 updates
and at the final update; `test/` only when `data.test_glob` is nonempty, at
`test_every` and at the final update. Tests are disabled in the J0–J4 launch.

Training losses are measured before each optimizer update. For ordinary training
metrics the trainer averages microbatch metrics across accumulation, then averages
update metrics across the logging window. These are not necessarily global
valid-label-weighted means if valid-label counts vary across microbatches. Joint
NLL is handled specially: its numerator and valid-clock weight are summed across
microbatches and updates, then divided once for the logging window. Padded rows
have zero evaluation weight. The intended experiments use unit weights for real
positions.

### 5.1 Training metrics (`train/`)

Loss component metrics below are reported **before multiplying by their configured
coefficients**. For example, a `reply_loss` of 2 contributes 0.30 to the total when
`reply_coef=0.15`.

| Exact Trackio key | Meaning and interpretation | Availability |
| --- | --- | --- |
| `train/loss` | Total optimized objective: `policy_loss + value_coef*value_loss + ponder_coef*ponder_loss + joint_time_coef*joint_time_loss + reply_coef*reply_loss + dpo_coef*dpo_loss`, including only enabled terms. Lower within the same objective; not directly comparable across J0–J4. | All runs |
| `train/policy_loss` | Weighted human-move cross-entropy, using legal moves in this recipe. At zero label smoothing it is move NLL. Joint variants use predicted-effort marginalization. Lower is better. | All runs |
| `train/value_loss` | Cross-entropy for the actual loss/draw/win label from the side-to-move perspective. Lower is better. | All runs |
| `train/reply_loss` | Cross-entropy for the opponent's actual next move, over the 4,352-move vocabulary without a reply legality mask. Only rows with `next_move >= 0` contribute. Lower is better. | All J0–J4 runs, when reply targets/head exist |
| `train/ponder_loss` | `E`: HL-Gauss cross-entropy for predicted marginal effort, normalized by valid-clock weight within each microbatch. Lower is better; this is not seconds error. | J1, J3 |
| `train/joint_time_loss` | `C`: posterior effort CE given the observed human move, with invalid-time rows zeroed and normalized over all position weight. Lower is better. It is still computed when its coefficient is zero. | J2, J3, J4 |
| `train/joint_nll` | Valid-clock-weighted mean `-sum_k q(k) log p(m,k|x)` across the logging window. Includes move and effort terms, but not value/reply losses. Lower is better. | J2, J3, J4; zero if no valid-clock weight |
| `train/joint_time_weight` | Sum of valid-clock position weights over the entire logging window, including all accumulated microbatches. With unit weights this is a count, not a fraction or per-update average. | J2, J3, J4 |
| `train/policy_acc` | Top-1 human-move match across the full unmasked 4,352-move vocabulary. For joint variants the unmasked mixture is recomputed. Higher is better. | All runs |
| `train/policy_acc_legal` | Top-1 human-move match restricted to legal moves, using bucket-wise legality before marginalization for joint variants. Higher is better. | All runs with legal masks |
| `train/value_acc` | Top-1 loss/draw/win accuracy. Higher is better. | All runs |
| `train/grad_norm` | Global L2 norm of the accumulated gradient before clipping and AdamW. Useful for instability and clipping diagnostics; neither minimizing nor maximizing it is the objective. | All runs |
| `train/lr` | Learning rate applied to the updates in the logging window; reported as their average. | All runs |
| `train/positions_per_sec` | `updates_in_window * global_batch_size / elapsed_wall_seconds`. The first window includes compilation; later windows can include preceding evaluation/checkpoint overhead because the timer resets before those operations. Not isolated model compute throughput. | All runs |
| `train/dpo_loss` | Preference loss `-log sigmoid(dpo_beta * margin)` on human/Stockfish disagreement rows. The margin is human-vs-Stockfish log odds minus frozen-reference log odds when supplied; otherwise the code subtracts a fixed 0.5. Lower is better for this preference objective. | Only outside this study when `dpo_coef > 0` and Stockfish annotations exist |

For DPO, the implementation accepts rows with a known differing Stockfish move
and either CP loss within `sf_max_cp_loss` or a negative (unknown) CP-loss sentinel.
It does not require a verified soundness label for that unknown-CP subset.
`joint_nll_sum` is an internal weighted numerator removed before Trackio logging.
There is no separate training `effort_given_move_nll` key.

Expected J0 training keys are `loss`, `policy_loss`, `value_loss`, `reply_loss`,
`policy_acc`, `policy_acc_legal`, `value_acc`, `grad_norm`, `lr`, and
`positions_per_sec`, all with the `train/` prefix. J0 has no time-loss or DPO keys.
Time metrics in J2 can be numerically computed against labels, but do not make
its unsupervised buckets calibrated time categories.

### 5.2 Validation and test metrics (`val/`, `test/`)

Every key in this table is prefixed with `val/` during training-time validation
or `test/` during training-time testing. Standalone evaluation uses the same
unprefixed names inside the JSON `overall` object.

| Key suffix | Meaning and interpretation | Availability |
| --- | --- | --- |
| `positions` | Sum of evaluation position weights, cast to integer. With unit weights, the number of real evaluated positions. Check it before comparing runs. | All |
| `move_acc` | Legal-masked top-1 human-move match. Higher is better. | All |
| `move_acc_top3` | Fraction where the human move appears among the top three legal-policy scores. Higher is better. | All |
| `move_acc_ci95` | Approximate binomial 95% confidence-interval **half-width**, `1.96*sqrt(acc*(1-acc)/n)`. Report `move_acc ± move_acc_ci95`. It assumes independent positions and is not a paired, game-clustered, or across-seed interval. | All |
| `move_acc_unmasked` | Top-1 human-move match without a legality mask over all 4,352 moves. Higher is better; it is not the fraction of predictions that are legal. | All |
| `perplexity` | `exp(mean legal move NLL)`. Lower is better. Recover move NLL in nats as `log(perplexity)`; the current summary does not expose a separate `move_nll` key. | All |
| `policy_entropy` | Mean entropy `-sum_m p(m) log p(m)` of the legal move distribution. Measures sharpness; no universally preferred direction. Low entropy with worsening held-out NLL can indicate overconfidence. | All |
| `value_nll` | Mean loss/draw/win NLL, with no `value_coef` scaling. Lower is better. Proper probabilistic score, not a complete calibration analysis. | All |
| `value_acc` | Top-1 loss/draw/win accuracy. Higher is better. | All |
| `anti_sf_positions` | Weighted number of eligible positions where the human move differs from the annotated Stockfish move. Eligibility includes CP loss within the threshold or unknown negative CP loss. | Only if eligible annotated positions exist |
| `anti_sf_human_acc` | Legal top-1 human-move match on that disagreement slice. Higher is better for matching humans. | Same slice |
| `anti_sf_engine_pick_rate` | Fraction of the disagreement slice whose legal top-1 prediction is Stockfish's differing top move. Lower means less Stockfish overlap, not necessarily better move quality. | Same slice |
| `anti_sf_human_pref_prob` | Mean `sigmoid(logit_human - logit_stockfish)`, equivalent to the human's probability after renormalizing over those two moves. Higher means stronger pairwise human preference; this is not top-1 accuracy. | Same slice |
| `joint_time_weight` | Total valid-clock evaluation position weight. Zero is reported when a joint model has no valid time labels. | J2, J3, J4 |
| `joint_nll` | Valid-clock-weighted mean smoothed joint move/effort NLL. Lower is better. Computed regardless of the training coefficient. | J2, J3, J4, only with positive valid-clock weight |
| `effort_given_move_nll` | Valid-clock-weighted mean posterior effort CE `-sum_k q(k) log p(k|m,x)`. Unlike training `joint_time_loss`, its denominator includes only valid time labels. Lower is better. | J2, J3, J4, only with positive valid-clock weight |

No AntiSF keys are emitted for an unannotated split or an empty disagreement
slice; their absence does not mean zero error. Clockless evaluation omits
`joint_nll` and `effort_given_move_nll`; a zero `joint_time_weight` explains why.
All time NLLs score discrete smoothed bucket targets, not a continuous-time
density or seconds error. Keep bucket count and target smoothing fixed when
comparing their values.

There are no automatically logged `val/loss`, `val/policy_loss`,
`val/ponder_loss`, or `val/reply_loss` metrics in the current evaluator. The
training and evaluation dictionaries are different.

### 5.3 Per-rating outputs and internal counters

The evaluator also returns `per_rating`, keyed by the nearest 100 points of
`game_elo = (self_elo + oppo_elo) / 2`, clipped to bins 0 through 3,100. Each
nonempty bin contains `n` (weighted count cast to integer), `acc` (legal move
accuracy), and `perplexity` (exponentiated mean move NLL). These are saved by
`chessforge.cli.eval --out`, but the trainer logs only `overall` to Trackio.
Per-rating outputs are not automatically Trackio charts, and no per-clock-regime
breakdowns are currently computed.

The evaluation step internally sums `n`, `correct`, `correct_top3`,
`correct_unmasked`, `nll`, `entropy`, `value_nll`, `value_correct`, `anti_sf_n`,
`anti_sf_human_pick`, `anti_sf_engine_pick`, `anti_sf_human_pref_prob`, `bin_n`,
`bin_correct`, and `bin_nll`, plus joint counters when applicable. These are
numerators/counts used to form the summary above, not additional Trackio metrics.
In particular, internal `nll` is a weighted sum; reported perplexity uses its mean.

### 5.4 Automatic Trackio system metrics

Trackio can collect host and GPU telemetry independently of chess metrics. The
installed integration uses Trackio's defaults: roughly 10-second sampling, stored
as timestamped **system metrics**, without `train/`, `val/`, or `test/` prefixes.
This cadence is independent of optimizer updates and `log_every`. CPU monitoring
requires `psutil`; NVIDIA GPU monitoring requires `nvidia-ml-py`/NVML and supported
hardware. Fields unavailable on a device/host are omitted. The J0 local tracking
database inspected while preparing this document contained CPU/memory/disk/network/
temperature telemetry; GPU telemetry was not present in that snapshot. A GPU
being busy in `nvidia-smi` does not imply Trackio is collecting GPU metrics.

Host readings describe the visible machine/container, not just the training
process. Use these for resource diagnosis, not as model-quality objectives.

| Exact key or key family | Meaning / units |
| --- | --- |
| `cpu/{i}/utilization` | Busy percentage of visible logical CPU core `i`, 0–100%. |
| `cpu/utilization` | Mean busy percentage across visible logical cores, 0–100%. |
| `cpu/frequency`, `cpu/frequency_max` | Current CPU frequency and reported maximum, MHz; static fields are typically collected on the first sample. |
| `cpu/count_logical`, `cpu/count_physical` | Visible logical and physical CPU counts. |
| `memory/used`, `memory/total`, `memory/available` | Used, total, and available host RAM, GiB. |
| `memory/percent` | Host RAM utilization, percent. |
| `swap/used`, `swap/total` | Used and total swap, GiB. |
| `swap/percent` | Swap utilization, percent. |
| `disk/read_mb_per_sec`, `disk/write_mb_per_sec` | Host disk read/write throughput, MiB/s despite the `mb` key spelling. |
| `disk/read_iops`, `disk/write_iops` | Host disk read/write operations per second. |
| `disk/read_bytes`, `disk/write_bytes` | Cumulative host disk bytes; fallback when rate baselines are unavailable. |
| `network/sent_mb_per_sec`, `network/recv_mb_per_sec` | Host network transmit/receive throughput, MiB/s. |
| `network/sent_bytes`, `network/recv_bytes` | Cumulative network byte counters; fallback when rate baselines are unavailable. |
| `temp/{label}` | Available named sensor temperature, degrees Celsius. Sensor labels vary by machine. |
| `battery/percent`, `battery/power_plugged` | Battery charge percent and AC-power flag (1 plugged in, 0 otherwise), if a battery exists. |

For NVIDIA telemetry, `{i}` is the GPU index used by Trackio:

| Exact key or key family | Meaning / units |
| --- | --- |
| `gpu/{i}/utilization` | GPU compute utilization, percent. |
| `gpu/{i}/memory_utilization` | NVML memory read/write activity, percent; distinct from occupied VRAM. |
| `gpu/{i}/allocated_memory`, `gpu/{i}/total_memory` | Device-wide used/total VRAM, GiB; not JAX-only allocations or peak memory. |
| `gpu/{i}/memory_usage` | Used VRAM divided by total VRAM, fraction 0–1. |
| `gpu/{i}/power`, `gpu/{i}/power_limit` | Current board power and enforced power limit, watts. |
| `gpu/{i}/power_percent` | Power as a percent of the enforced limit. |
| `gpu/{i}/temp` | GPU temperature, degrees Celsius. |
| `gpu/{i}/sm_clock`, `gpu/{i}/memory_clock` | SM and memory clock frequencies, MHz. |
| `gpu/{i}/fan_speed` | Fan speed, percent of the device's reported maximum. |
| `gpu/{i}/performance_state` | NVML performance state, e.g. P0 for highest performance. |
| `gpu/{i}/energy_consumed` | Cumulative GPU energy since Trackio's first energy reading, joules. |
| `gpu/{i}/pcie_tx`, `gpu/{i}/pcie_rx` | PCIe transmit/receive throughput, MiB/s. |
| `gpu/{i}/throttle_thermal` | 0/1 software thermal-throttling flag. |
| `gpu/{i}/throttle_power` | 0/1 software power-cap throttling flag. |
| `gpu/{i}/throttle_hw_slowdown` | 0/1 hardware slowdown flag. |
| `gpu/{i}/throttle_apps` | 0/1 application-clock restriction flag. |
| `gpu/{i}/corrected_memory_errors`, `gpu/{i}/uncorrected_memory_errors` | NVML volatile corrected/uncorrected ECC error counts. |
| `gpu/mean_utilization` | Mean utilization across GPUs reporting utilization, percent. |
| `gpu/total_allocated_memory` | Sum of used VRAM across monitored GPUs, GiB. |
| `gpu/total_power` | Sum of current board power across monitored GPUs, watts. |
| `gpu/max_temp` | Maximum temperature across monitored GPUs, degrees Celsius. |

This inventory follows the installed Trackio collectors; future Trackio versions
may add or change fields. Device telemetry does not provide an exact training-job
energy measurement or a directly logged peak-memory metric.

## 6. Final evaluation and interpretation

Use the final **10,000-step checkpoint** for the primary comparison. Evaluate the
same complete titled-online and OTB splits for every variant and seed. Preserve
the resolved `config.json`; do not reconstruct a checkpoint's model configuration
from a different ablation's YAML. Example after J0 completes:

```bash
mkdir -p results
.venv/bin/python -m chessforge.cli.eval \
  --config runs/J0_s42/config.json \
  --step 10000 \
  --batch_size 64 \
  --test_glob 'data/cf_data/eval_titled_online/*.array_record' \
  --out results/J0_s42_online.json

.venv/bin/python -m chessforge.cli.eval \
  --config runs/J0_s42/config.json \
  --step 10000 \
  --batch_size 64 \
  --test_glob 'data/cf_data/eval_otb/*.array_record' \
  --out results/J0_s42_otb.json
```

Standalone `cli.eval` prints `overall` and writes `overall` plus `per_rating` to
JSON. It does **not** initialize a Trackio run or send `test/` metrics to Trackio,
even when tracking fields are present in the saved config. To record those results
in Trackio would require an additional explicit logging step.

Use held-out move NLL (`log(perplexity)`) as the primary comparison and move
accuracy as the companion metric. Report online and OTB separately, along with
value NLL, top-3 accuracy, unmasked accuracy, AntiSF metrics where available, and
throughput on the same hardware. Report mean and standard deviation across seeds
and paired differences between matched seeds. The built-in `move_acc_ci95` is
not a significance test between models; paired game-level confidence intervals
require per-example results and game identifiers, which the aggregate evaluator
does not currently retain.

### Current measurement gaps

J1 does not receive joint/time metrics from the current evaluator because it has
no `conditional_move_logits`. J1 and J3 log marginal `train/ponder_loss`, but there
is no held-out marginal-effort CE metric. Fair time comparisons between J1 and J4
therefore require a shared evaluation extension. For an independent model, joint
NLL is move NLL plus marginal effort CE, and effort-given-move CE equals marginal
effort CE; compute both on exactly the same valid-clock subset as J4.

Time MAE/RMSE, time calibration/coverage, per-clock-regime breakdowns, reply
accuracy, legality rate, exact peak accelerator memory, and training-only step
latency are **not currently tracked**. Do not list them as measured results until
their evaluation/logging exists. Bucket NLL is not a substitute for seconds
calibration. Comparing different bucket counts requires additional care because
the discretization and target smoothing change.

For a later R6-style robustness study, repeat selected variants with the final
20% switched to Pool C's titled-online and OTB training shards. Give those runs
separate names and keep the curriculum identical across variants. That is a
separate experiment from J0–J4 above; clockless OTB rows reintroduce the effort-loss
weighting difference described in section 3.

### Results template

Record one row per variant, seed, and evaluation split. Dashes mean not yet
reported, not zero. Keep raw JSON beside the table and add across-seed summaries
after the matched runs finish.

| Run | Seed | Split | Move acc | Move NLL | Top-3 | Value NLL | Joint NLL | Effort-given-move NLL |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| J0 | 42 | Titled online / OTB (separate rows) | — | — | — | — | Not emitted | Not emitted |
| J1 | 42 | Titled online / OTB (separate rows) | — | — | — | — | Not emitted | Not emitted |
| J2 | 42 | Titled online / OTB (separate rows) | — | — | — | — | — | — |
| J3 | 42 | Titled online / OTB (separate rows) | — | — | — | — | — | — |
| J4 | 42 | Titled online / OTB (separate rows) | — | — | — | — | — | — |
