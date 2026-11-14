# Joint move and thinking-time ablations: J0–J4

This study tests whether modeling the relationship between human moves and
thinking time improves held-out move prediction beyond an independent effort
head or extra policy capacity. It follows the fixed-update, fixed-example
approach of the earlier [R1–R6 ablations](https://github.com/soumik12345/chessforge/blob/improvement-ablations/EXPERIMENTS.md#52-controlled-compute-matched-ablation-ladder-r1r6).
The definitions below describe the current implementation. Completed seed-42
J0–J3 validation results are recorded in [section 7](#7-j0j3-validation-results-seed-42).
J3 has the lowest final move NLL, while J1 has the highest top-1 accuracy;
the single-seed validation comparison does not establish a decisive winner.
J4 and held-out titled-online/OTB results remain unreported here.

## 1. Model and objectives

Let $x_n$ denote the board and pre-move context, $m_n$ the observed human
move, $t_n$ the observed thinking time in seconds, and $c_n$ the pre-move
clock. Let $k\in\{0,\ldots,K-1\}$ index the $K=16$ effort buckets.
The joint model predicts

$$
\begin{aligned}
p_\theta(m,k\mid x)
&=p_\theta(k\mid x)\,p_\theta(m\mid k,x),\\
\pi_\theta^{\mathrm{mix}}(m\mid x)
&=\sum_{k=0}^{K-1}p_\theta(k\mid x)\,p_\theta(m\mid k,x).
\end{aligned}
$$

For the legal move set $\mathcal{L}(x)$, each bucket is normalized separately
before mixing:

$$
p_\theta(m\mid k,x)
=\frac{\exp s_{\theta,k}(m,x)}
{\sum_{a\in\mathcal{L}(x)}\exp s_{\theta,k}(a,x)},
\qquad m\in\mathcal{L}(x).
$$

A rank-16 residual conditions the existing policy head on effort; its gates
start at zero. The shared trunk, value head, and effort head are retained.
The independent-head baseline instead uses the ordinary legal policy
$\pi_\theta^{\mathrm{ind}}(m\mid x)$, without the effort-conditioned residual.
Both policies receive the same pre-move context. Observed moves and future
thinking times are loss targets, not forward inputs. See
[prototype details](docs/joint_move_time.md).

### Effort targets and valid-label sets

The smoothed effort target $q_n(k)$ is centered on normalized effort

$$
u_n=\operatorname{clip}\!\left(
\frac{\log(1+t_n)}{\log(1+\max(c_n,1))},\,0,\,1
\right).
$$

Writing $\Phi$ for the standard normal CDF, the HL-Gauss histogram is

$$
\begin{aligned}
b_k&=\frac{k}{K},\qquad \sigma=\frac{0.75}{K},\\
h_{n,k}
&=\Phi\!\left(\frac{b_{k+1}-u_n}{\sigma}\right)
-\Phi\!\left(\frac{b_k-u_n}{\sigma}\right),\\
q_n(k)&=\frac{h_{n,k}}
{\max\!\left(\sum_{j=0}^{K-1}h_{n,j},10^{-8}\right)}.
\end{aligned}
$$

Valid time supervision requires finite nonnegative thinking time, a finite
pre-move clock greater than 0.5 seconds, a finite clock-presence flag greater
than 0.5, and a finite positive position weight. Invalid time labels do not
exclude an otherwise valid example from move supervision.

Let $\mathcal{B}$ be a training microbatch, $\mathcal{T}\subseteq\mathcal{B}$
the valid-clock rows, and $\mathcal{S}\subseteq\mathcal{B}$ the rows with a valid
next-move label. Let $w_n$ be the position weight after replacing nonfinite or
nonpositive weights with zero. Define

$$
\begin{aligned}
W&=\max\!\left(\sum_{n\in\mathcal{B}}w_n,1\right),\\
W_{\mathcal{T}}&=\max\!\left(\sum_{n\in\mathcal{T}}w_n,1\right),\\
W_{\mathcal{S}}&=\max\!\left(\sum_{n\in\mathcal{S}}w_n,1\right).
\end{aligned}
$$

### Loss components

The move loss for either legal policy $\pi$ is

$$
\mathcal{M}[\pi]
=-\frac{1}{W}\sum_{n\in\mathcal{B}}w_n\log\pi(m_n\mid x_n).
$$

The marginal effort loss and posterior effort loss are

$$
\begin{aligned}
\mathcal{E}
&=-\frac{1}{W_{\mathcal{T}}}
\sum_{n\in\mathcal{T}}w_n
\sum_{k=0}^{K-1}q_n(k)\log p_\theta(k\mid x_n),\\
\mathcal{C}
&=-\frac{1}{W}
\sum_{n\in\mathcal{T}}w_n
\sum_{k=0}^{K-1}q_n(k)\log p_\theta(k\mid m_n,x_n),\\
p_\theta(k\mid m_n,x_n)
&=\frac{p_\theta(k\mid x_n)\,p_\theta(m_n\mid k,x_n)}
{\pi_\theta^{\mathrm{mix}}(m_n\mid x_n)}.
\end{aligned}
$$

The distinction between $W_{\mathcal{T}}$ and $W$ matters on mixed-clock
microbatches. An empty $\mathcal{T}$ contributes zero to both time losses.

Let $y_n\in\{\mathrm{loss},\mathrm{draw},\mathrm{win}\}$ be the outcome from
the side-to-move perspective, and $r_n$ the actual next human move expressed
in the opponent's frame. The shared auxiliary losses are

$$
\begin{aligned}
\mathcal{V}
&=-\frac{1}{W}\sum_{n\in\mathcal{B}}w_n
\log p_\theta^{\mathrm{value}}(y_n\mid x_n),\\
\mathcal{R}
&=-\frac{1}{W_{\mathcal{S}}}\sum_{n\in\mathcal{S}}w_n
\log p_\theta^{\mathrm{reply}}(r_n\mid x_n),\\
\mathcal{A}&=0.10\,\mathcal{V}+0.15\,\mathcal{R}.
\end{aligned}
$$

The reply distribution covers the full move vocabulary. The symbols
$\mathcal{M},\mathcal{E},\mathcal{C},\mathcal{V},\mathcal{R}$ correspond to
the shorthand M, E, C, V, and R used elsewhere in this document. These are
zero-label-smoothing microbatch objectives; accumulation averages them as
described in section 5. DPO has coefficient zero for every J0–J4 experiment.

## 2. Ablation matrix

All five runs use the same V2 backbone, pre-move inputs, data, optimizer recipe,
batch size, and training budget. Change only the three settings below.

| Run | Description | `model_overrides.use_joint_move_time` | `joint_time_coef` | `ponder_coef` | Total objective |
| --- | --- | --- | --- | --- | --- |
| **J0** | V2 baseline without time supervision | `false` | `0.0` | `0.0` | $\mathcal{M}[\pi_\theta^{\mathrm{ind}}]+\mathcal{A}$ |
| **J1** | Independent move and effort heads | `false` | `0.0` | `1.0` | $\mathcal{M}[\pi_\theta^{\mathrm{ind}}]+\mathcal{E}+\mathcal{A}$ |
| **J2** | Effort-conditioned policy without time labels | `true` | `0.0` | `0.0` | $\mathcal{M}[\pi_\theta^{\mathrm{mix}}]+\mathcal{A}$ |
| **J3** | Effort-conditioned policy with marginal effort supervision | `true` | `0.0` | `1.0` | $\mathcal{M}[\pi_\theta^{\mathrm{mix}}]+\mathcal{E}+\mathcal{A}$ |
| **J4** | Full joint move/effort objective | `true` | `1.0` | `0.0` | $\mathcal{M}[\pi_\theta^{\mathrm{mix}}]+\mathcal{C}+\mathcal{A}$ |

J0 and J1 retain the existing effort head and pre-move clock inputs; disabling
time supervision does not remove clock context. J2's effort prior is learned
through move prediction, but its latent buckets have no supervised thinking-time
meaning. J3 couples the move policy to effort while retaining separate move and
marginal-effort losses. At zero label smoothing and on valid-clock rows, J4's
`M + C` equals the smoothed joint NLL. This identity applies to those two terms,
not to the additional value/reply losses.

### J0: Independent policy without time supervision

$$
\boxed{
\mathcal{L}_{\mathrm{J0}}(\theta)
=-\frac{1}{W}\sum_{n\in\mathcal{B}}w_n
\log\pi_\theta^{\mathrm{ind}}(m_n\mid x_n)
+\mathcal{A}
}
$$

The effort head receives no explicit effort-target loss and does not enter
the move policy. Pre-move clock features remain available to the shared trunk.

### J1: Independent move and effort supervision

The model represents conditional independence of move and effort given context:

$$
p_\theta^{\mathrm{ind}}(m,k\mid x)
=\pi_\theta^{\mathrm{ind}}(m\mid x)\,p_\theta(k\mid x).
$$

$$
\boxed{
\mathcal{L}_{\mathrm{J1}}(\theta)
=\mathcal{M}[\pi_\theta^{\mathrm{ind}}]+\mathcal{E}+\mathcal{A}
}
$$

Both heads share the trunk. The move head does not take an effort bucket as
input. On fully clock-labeled microbatches, the move and effort terms equal
the smoothed NLL of this independent joint distribution.

### J2: Latent effort-conditioned policy

$$
\boxed{
\mathcal{L}_{\mathrm{J2}}(\theta)
=-\frac{1}{W}\sum_{n\in\mathcal{B}}w_n
\log\!\left[
\sum_{k=0}^{K-1}p_\theta(k\mid x_n)\,p_\theta(m_n\mid k,x_n)
\right]+\mathcal{A}
}
$$

The buckets act as latent mixture components, with no loss assigning them the
observed thinking-time meaning. This isolates capacity and mixture effects.

### J3: Effort-conditioned policy with marginal effort supervision

$$
\boxed{
\mathcal{L}_{\mathrm{J3}}(\theta)
=\mathcal{M}[\pi_\theta^{\mathrm{mix}}]
+\mathcal{E}+\mathcal{A}
}
$$

Move prediction uses the effort mixture, while the time target supervises the
marginal $p_\theta(k\mid x)$. This does not generally equal the joint NLL of
$p_\theta(m,k\mid x)$ because it lacks the move-conditioned effort term.

### J4: Joint move and thinking-time supervision

$$
\boxed{
\mathcal{L}_{\mathrm{J4}}(\theta)
=\mathcal{M}[\pi_\theta^{\mathrm{mix}}]
+\mathcal{C}+\mathcal{A}
}
$$

For each valid-clock example, Bayes' rule gives the joint-loss identity:

$$
\begin{aligned}
&-\log\pi_\theta^{\mathrm{mix}}(m_n\mid x_n)
-\sum_{k=0}^{K-1}q_n(k)\log p_\theta(k\mid m_n,x_n)\\
&\qquad=-\sum_{k=0}^{K-1}q_n(k)\log p_\theta(m_n,k\mid x_n).
\end{aligned}
$$

Consequently, the exact mixed-clock microbatch objective is

$$
\boxed{
\begin{aligned}
\mathcal{L}_{\mathrm{J4}}(\theta)
=-\frac{1}{W}\Bigg[
&\sum_{n\in\mathcal{T}}w_n
\sum_{k=0}^{K-1}q_n(k)\log p_\theta(m_n,k\mid x_n)\\
+&\sum_{n\in\mathcal{B}\setminus\mathcal{T}}w_n
\log\pi_\theta^{\mathrm{mix}}(m_n\mid x_n)
\Bigg]+\mathcal{A}.
\end{aligned}
}
$$

Clock-labeled rows optimize smoothed joint likelihood; other rows optimize
marginal move likelihood. Observed thinking time is used only as a loss target.

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

Run these commands from the repository root (`/home/chessforge`). Each block
launches one seed-42 experiment with the local Pool B training/validation data,
a distinct workdir, and an explicit Trackio run name. On a single L4, run one
experiment at a time; wait for it to finish before launching the next. If J0 is
already running or complete, continue with J1 instead of starting a second J0.

### J0: V2 baseline without time supervision

```bash
export TRACKIO_DIR="$PWD/trackio_logs"

.venv/bin/python -m chessforge.cli.train \
  --config configs/chessforge_joint_time_28m.yaml \
  --set \
    workdir="$PWD/runs/J0_s42" \
    trackio_project=chessforge \
    trackio_run_name=J0_s42 \
    trackio_group=joint_move_time_ablations \
    trackio_space_id=geekyrakshit/chessforge-tracking \
    model_overrides.use_joint_move_time=false \
    model_overrides.joint_move_time_rank=16 \
    model_overrides.ponder_buckets=16 \
    joint_time_coef=0.0 \
    ponder_coef=0.0 \
    seed=42 \
    data.shuffle_seed=42 \
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

### J1: Independent move and effort heads

```bash
export TRACKIO_DIR="$PWD/trackio_logs"

.venv/bin/python -m chessforge.cli.train \
  --config configs/chessforge_joint_time_28m.yaml \
  --set \
    workdir="$PWD/runs/J1_s42" \
    trackio_project=chessforge \
    trackio_run_name=J1_s42 \
    trackio_group=joint_move_time_ablations \
    trackio_space_id=geekyrakshit/chessforge-tracking \
    model_overrides.use_joint_move_time=false \
    model_overrides.joint_move_time_rank=16 \
    model_overrides.ponder_buckets=16 \
    joint_time_coef=0.0 \
    ponder_coef=1.0 \
    seed=42 \
    data.shuffle_seed=42 \
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

### J2: Effort-conditioned policy without time labels

```bash
export TRACKIO_DIR="$PWD/trackio_logs"

.venv/bin/python -m chessforge.cli.train \
  --config configs/chessforge_joint_time_28m.yaml \
  --set \
    workdir="$PWD/runs/J2_s42" \
    trackio_project=chessforge \
    trackio_run_name=J2_s42 \
    trackio_group=joint_move_time_ablations \
    trackio_space_id=geekyrakshit/chessforge-tracking \
    model_overrides.use_joint_move_time=true \
    model_overrides.joint_move_time_rank=16 \
    model_overrides.ponder_buckets=16 \
    joint_time_coef=0.0 \
    ponder_coef=0.0 \
    seed=42 \
    data.shuffle_seed=42 \
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

### J3: Effort-conditioned policy with marginal effort supervision

```bash
export TRACKIO_DIR="$PWD/trackio_logs"

.venv/bin/python -m chessforge.cli.train \
  --config configs/chessforge_joint_time_28m.yaml \
  --set \
    workdir="$PWD/runs/J3_s42" \
    trackio_project=chessforge \
    trackio_run_name=J3_s42 \
    trackio_group=joint_move_time_ablations \
    trackio_space_id=geekyrakshit/chessforge-tracking \
    model_overrides.use_joint_move_time=true \
    model_overrides.joint_move_time_rank=16 \
    model_overrides.ponder_buckets=16 \
    joint_time_coef=0.0 \
    ponder_coef=1.0 \
    seed=42 \
    data.shuffle_seed=42 \
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

### J4: Full joint move/effort objective

```bash
export TRACKIO_DIR="$PWD/trackio_logs"

.venv/bin/python -m chessforge.cli.train \
  --config configs/chessforge_joint_time_28m.yaml \
  --set \
    workdir="$PWD/runs/J4_s42" \
    trackio_project=chessforge \
    trackio_run_name=J4_s42 \
    trackio_group=joint_move_time_ablations \
    trackio_space_id=geekyrakshit/chessforge-tracking \
    model_overrides.use_joint_move_time=true \
    model_overrides.joint_move_time_rank=16 \
    model_overrides.ponder_buckets=16 \
    joint_time_coef=1.0 \
    ponder_coef=0.0 \
    seed=42 \
    data.shuffle_seed=42 \
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

For seed 43 or 44, change `seed`, `data.shuffle_seed`, the `_s42` workdir suffix,
and the `_s42` Trackio run-name suffix together. Keep every other setting fixed.

The saved J0 config at `runs/J0_s42/config.json` records project `chessforge`,
run name `J0`, group `joint_move_time_ablations`, and Space
[`geekyrakshit/chessforge-tracking`](https://huggingface.co/spaces/geekyrakshit/chessforge-tracking).
The commands above use seed-qualified run names for future launches. The current
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

## 7. J0–J3 validation results (seed 42)

### Source and comparison scope

These results were retrieved on **2026-11-14** from project `chessforge` in
[`geekyrakshit/chessforge-tracking`](https://geekyrakshit-chessforge-tracking.hf.space).
The source runs are identified below; the two `J0-checkpoint-export` runs are
excluded.

| Variant | Trackio run name | Trackio run ID |
| --- | --- | --- |
| J0 | `J0` | `cca3cd3d7b8b46b7b0852c3f2aa35b49` |
| J1 | `J1_s42` | `b2c1dac4599e4a56ac6a21f864570ca5` |
| J2 | `J2_s42` | `3c9a8300495d46d7866f95ca61e80619` |
| J3 | `J3_s42` | `8b526994caa541f0ba67891c5e435e9e` |

The Space's read API at retrieval time used POST requests to
`/api/get_runs_for_project` and `/api/get_run_configs` with
`{"project": "chessforge"}`, and `/api/get_logs` and `/api/get_run_summary` with
the same project plus `run` and `run_id` from the table. Each run returned
**100 training records and 10 validation records**, ending at update 10,000.

The recorded configurations match the shared protocol in section 3: seed and
shuffle seed 42, batch size 1,024, accumulation 16, and 10,000 optimizer updates
(10.24M position presentations). All use the same Pool B titled June 2025
training and validation paths. Each validation pass reports **3,200 positions**
(`val_batches=50`, `eval_batch_size=64`), rather than the full 7,774-position
validation split. No `test/` metrics appear in these runs, and `test_glob` is
empty. These tables therefore report training-time validation, not the separate
held-out titled-online or OTB evaluations. Matching paths and configurations do
not verify dataset content hashes or training code revisions, which were not
recorded in the retrieved configurations.

### Final checkpoint metrics

All entries below use update **10,000**, not separately selected best checkpoints.
Move NLL is derived as `ln(val/perplexity)` using unrounded values. NLL and entropy
are in nats; accuracies are percentages. Move top-1 and top-3 use legality masking.

| Variant | Effort-conditioned policy | Time supervision | Move NLL ↓ | Perplexity ↓ | Move top-1 ↑ | Move top-3 ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| J0 | No | No | 1.39364 | 4.02948 | 54.500% | **82.719%** |
| J1 | No | Marginal effort | 1.39826 | 4.04813 | **55.344%** | 81.375% |
| J2 | Yes | No | 1.40150 | 4.06130 | 55.313% | 81.906% |
| J3 | Yes | Marginal effort | **1.38112** | **3.97937** | 54.719% | **82.719%** |

J0 and J2 still receive pre-move clock context; they lack time supervision, not
clock inputs. Total training loss is not comparable across these objectives.

| Variant | Value NLL ↓ | Value accuracy ↑ | Policy entropy | Unmasked move accuracy ↑ | Joint NLL ↓ | Effort-given-move NLL ↓ |
| --- | --- | --- | --- | --- | --- | --- |
| J0 | 0.70243 | 64.531% | 1.28800 | 2.969% | Not emitted | Not emitted |
| J1 | **0.69556** | 64.562% | 1.28353 | **4.938%** | Not emitted | Not emitted |
| J2 | 0.70400 | 64.375% | 1.29379 | 3.438% | 5.08869 | 3.68718 |
| J3 | 0.69783 | **64.844%** | 1.28901 | 3.250% | 3.39583 | 2.01471 |

Unmasked move accuracy is a human-move matching diagnostic over the full
vocabulary, not legality rate. The much higher legal-masked accuracies above
should not be interpreted as unmasked policy performance. Entropy has no
universally preferred direction.

### Ablation comparisons

Differences are candidate minus reference, computed before rounding. Negative
NLL or perplexity changes are improvements; positive accuracy changes are
improvements. Accuracy differences use percentage points (pp).

| Comparison | Move NLL change | Perplexity change | Top-1 change (pp) | Top-3 change (pp) |
| --- | --- | --- | --- | --- |
| J1 − J0 | +0.00462 | +0.463% | +0.844 | −1.344 |
| J2 − J0 | +0.00787 | +0.790% | +0.812 | −0.812 |
| J3 − J1 | −0.01713 | −1.699% | −0.625 | +1.344 |
| J3 − J2 | −0.02038 | −2.017% | −0.594 | +0.812 |
| J3 − J0 | −0.01251 | −1.244% | +0.219 | 0.000 |

- **Time supervision alone (J1 vs. J0):** top-1 improves, but move NLL and top-3
  worsen. The independent auxiliary effort task does not yield a consistent
  improvement in move prediction in this run.
- **Conditioning alone (J2 vs. J0):** top-1 improves, but move NLL and top-3
  worsen. Additional conditional capacity without time labels does not improve
  the primary probabilistic score at this budget.
- **Conditioning with supervision (J3 vs. J1):** move NLL and top-3 improve,
  while top-1 falls. This favors J3 for distribution quality, not for every
  move-matching metric.
- **Combined effect (J3 vs. J0):** perplexity improves 1.244%, top-3 ties, and
  top-1 gains only seven human-move matches out of 3,200. The pattern suggests
  that conditioning and supervision work better together, but requires
  replication before attributing a reliable benefit to their interaction.

### Learning dynamics and time prediction

J3 has the lowest move NLL at every recorded checkpoint from update 6,000 through
10,000. Its late advantage is therefore not confined to the final observation.
All four runs attain their lowest recorded validation NLL at update 10,000.

| Update | J0 move NLL | J1 move NLL | J2 move NLL | J3 move NLL |
| --- | --- | --- | --- | --- |
| 6,000 | 1.50206 | 1.50144 | 1.48950 | **1.48781** |
| 7,000 | 1.48361 | 1.46987 | 1.47313 | **1.46395** |
| 8,000 | 1.47249 | 1.47055 | 1.47655 | **1.46252** |
| 9,000 | 1.44216 | 1.44612 | 1.44008 | **1.42409** |
| 10,000 | 1.39364 | 1.39826 | 1.40150 | **1.38112** |

NLL improves by 0.07230–0.08139 nats across variants during the final 2,000
updates, which coincide with learning-rate decay. These results establish
performance at the fixed training budget, not convergence.

J2 and J3 both report valid time weight 3,200 on every validation pass. J3's final
effort-given-move NLL of 2.01471 is lower than J2's 3.68718, but J2's latent buckets
were never supervised to represent observed time. This comparison supports the
benefit of time labels; it does not establish that conditioning outperforms an
independent supervised time head. J1 has no comparable held-out time score.
Its final logging-window marginal effort loss is almost identical to J3's
(1.99213 vs. 1.99210), and training loss is not a substitute for that missing
evaluation. J3 uses `joint_time_coef=0`; posterior effort supervision remains
the separate J4 experiment. None of these bucket scores measures seconds error
or time calibration.

### Interpretation and next experiments

**J3 is the candidate to carry forward for probabilistic human-move modeling;
J1 leads top-1 accuracy.** Retain J0 and J1 as controls. This is a single-seed,
limited-validation comparison, with no evidence here for superiority in playing
strength or generalization to OTB games. The logged per-run accuracy intervals
have approximately ±1.72 percentage-point half-widths, assume independent
positions, and are not paired intervals for differences between models.
Aggregate logs cannot establish statistical significance across games or seeds.

Before declaring an improvement:

1. Evaluate the full validation split and the separate held-out titled-online
   and OTB splits, preserving the final-checkpoint comparison.
2. Repeat matched seeds 43 and 44; retain per-example predictions and game IDs
   for paired, game-level comparisons as well as across-seed summaries.
3. Add a common held-out marginal-effort score for J1 and J3, then run J4 to test
   whether posterior supervision improves on marginal supervision.

### Held-out results template

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
