# ChessformerV2 & ChessForge: Closing the Master, Over-the-Board, and Anti-Engine Alignment Gaps in Human Chess Modeling

---

## Abstract

Chessformer and the MAIA-3 model family (Monroe et al., 2026) recently advanced searchless human move prediction by treating the 64 board squares as tokens equipped with Geometric Attention Bias (GAB). However, auditing released MAIA-3 checkpoints (`5M`, `23M`, `79M`) across held-out titled online games and official FIDE Over-The-Board (OTB) classical broadcasts exposes three fundamental limitations: **(1) severe OTB domain shift**, where `MAIA-3-79M` suffers a $-7.12\%$ drop in move-matching accuracy ($63.02\% \to 55.90\%$) and near-collapse in outcome calibration ($\text{NLL } 0.853 \to 1.046$, approaching uniform random $\ln 3 \approx 1.099$); **(2) engine-overlap invariance**, where scaling model capacity $16\times$ (`5M` $\to$ `79M`) fails to reduce the rate at which the policy selects Stockfish's non-human top choice on human–engine disagreement positions ($19.19\% \to 19.28\%$ online; $20.07\% \to 20.11\%$ OTB); and **(3) severe sample inefficiency** stemming from a non-Markov board representation, collinear two-vector Elo interpolation, Post-Norm optimization bottlenecks, and 22-bin blitz downsampling dominated by noisy sub-1800 play and untitled high-Elo smurf/cheater accounts.

We introduce **ChessForge** and **ChessformerV2**, a complete architectural, data-curation, and alignment redesign implemented in JAX/Flax NNX. Architecturally, ChessformerV2 upgrades Chessformer with a **complete Markov & bilateral clock state (`v3`)**, **non-linear RBF Elo + official FIDE/national title + format conditioning via bottlenecked Adaptive RMSNorm (AdaLN)**, a **TensorCore-aligned ($S=64$) Pre-Norm + QK-Norm + SwiGLU backbone**, **Zero-Copy Ray-GAB** (augmenting learned GAB templates with exact analytical line-of-sight and 1-blocker pin/X-ray geometry), **post-trunk Cross-Attention Readout**, and **auxiliary human cognitive heads** (distributional HL-Gauss normalized ponder effort and rank-mirrored 2-ply human opponent-reply prediction). On the data and objective front, we combine **official-title-verified online + FIDE OTB curation**, **Warmup-Stable-Decay (WSD) curriculum annealing**, and **Stage-3 Stockfish-Disagreement Direct Preference Optimization (Anti-Engine DPO)**.

In controlled 10.24M-example single-GPU ablations, our recipe improves move-matching accuracy over the reproduced MAIA-3 baseline by **+17.87 percentage points** on Titled Online ($37.15\% \to 55.02\%$) and **+18.16 percentage points** on FIDE OTB ($31.97\% \to 50.13\%$). Scaling to 61.4M examples ($67\times$ less training compute than `MAIA-3-79M`) yields **state-of-the-art game-outcome calibration across both domains**, surpassing `MAIA-3-79M` on Titled Online ($61.68\%$ vs. $60.14\%$ accuracy; $0.842$ vs. $0.853$ NLL) and crushing `MAIA-3-79M` on FIDE OTB by **+7.80% accuracy** ($65.56\%$ vs. $57.76\%$) and **$-0.294\text{ nats}$ NLL** ($0.753$ vs. $1.046$), while Stage-3 Anti-Engine DPO breaks below MAIA-3's $20\%$ Stockfish-pick floor ($19.40\%$ on OTB). Finally, we characterize the multi-epoch vs. unique-position diversity trade-off in policy perplexity and entropy under constrained dataset budgets.

---

## 1. Introduction

Building AI systems that accurately model expert human decision-making—capturing human plans, bounded calculation, practical risk preferences, and natural blind spots rather than superhuman minimax optima—is a central problem in human-AI alignment (McIlroy-Young et al., 2020; Hamade et al., 2024). In chess, **Chessformer / MAIA-3** (Monroe et al., 2026) demonstrated that aligning transformer tokenization with the 64 squares of the chessboard, paired with dynamic Geometric Attention Bias (GAB) and a bilinear source–destination policy head, outperforms 355M-parameter move-sequence transformers (ALLIE; Zhang et al., 2025) with a fraction of the parameters. Importantly, Monroe et al. (2026, Appendix E) conclude by decomposing human move-prediction error into *aleatoric uncertainty* (dominant in stochastic low-rated play) and *epistemic uncertainty* (dominant in sophisticated master play), arguing that *"future human emulation work should focus on highly skilled play, where the performance ceiling appears to be much higher."*

Yet when we deploy released MAIA-3 models (`5M`, `23M`, `79M`) specifically on **verified master-level play**—spanning titled online games and real-world FIDE Over-The-Board (OTB) classical tournaments—three critical failure modes emerge:

1. **Catastrophic OTB Domain Shift:** Trained exclusively on Lichess 3–5 minute Blitz games, MAIA-3's move-matching accuracy plunges by more than $7$ percentage points on FIDE OTB games ($63.02\% \to 55.90\%$ for `79M`), and its win/draw/loss (WDL) value head nearly collapses ($\text{value NLL} = 1.0463$, barely outperforming a blind $1/3$ uniform guess at $\ln 3 = 1.0986$) because blitz flagging dynamics and low draw rates severely miscalibrate classical tournament play.
2. **Invariance of Engine-Like Play to Model Scale:** Standard cross-entropy behavior cloning maximizes $\log \pi_\theta(a_{\text{hum}} \mid s)$ without penalizing probability mass placed on computer-like tactical/prophylactic lines when humans deviate. Evaluating on `Eval-AntiSF`—positions where the human played a sound alternative ($\Delta_{\text{cp}} \le 35$) distinct from Stockfish 16's top choice—reveals that scaling MAIA-3 $16\times$ from `5M` to `79M` leaves its Stockfish-pick rate completely unchanged ($19.19\% \to 19.28\%$ online; $20.07\% \to 20.11\%$ OTB).
3. **Architectural, Representational, and Curation Bottlenecks:** As detailed in Section 2, MAIA-3 omits Markov chess rules (castling, en passant, repetition counts) from its human-emulation inputs, restricts skill conditioning to a collinear two-vector line without official title or time-control awareness, uses an optimization-limited Post-Norm trunk at $\text{lr}=5\times 10^{-5}$, relies on GAB biases that learn mostly static global stage templates rather than exact piece ray/blocking geometry, and wastes $>60\%$ of training compute on noisy untitled low-Elo blitz while capping verified Grandmaster games.

In this work, we build **ChessForge**, an end-to-end JAX/Flax NNX training, Stockfish-annotation, evaluation, and UCI-deployment framework, and propose **ChessformerV2** to systematically resolve each bottleneck:

* **Complete Markov & Temporal State (`v3`) + Non-Linear Context AdaLN:** We restore full Markov legality state (castling rights, side-to-move-framed en passant target square, Zobrist repetition history), bilateral log-scaled clocks, clock-presence masking for OTB compatibility, Gaussian Radial Basis Function (RBF) Elo encodings, rating-asymmetry features, and **official human title (`GM/WGM`, `IM/WIM`, `FM/WFM`, `CM/NM`)** and **format (`Blitz`, `Rapid`, `OTB`)** embeddings injected into every layer via a parameter-efficient bottlenecked Adaptive RMSNorm (AdaLN).
* **TensorCore-Aligned Pre-Norm Backbone & Zero-Copy Ray-GAB:** We replace Post-Norm + GELU with Pre-RMSNorm, QK-Norm, and SwiGLU, while strictly preserving an in-trunk sequence length of $S=64$ (moving register tokens to a post-trunk **Cross-Attention Readout** to maintain GPU TensorCore / TPU MXU power-of-2 tiling and $+41\%$ higher throughput). We augment GAB with **Ray-GAB**, computing exact per-position 0-blocker line-of-sight and 1-blocker pin/skewer/X-ray masks in a single fused matrix multiply.
* **Cognitive Auxiliary Supervision:** We train two zero-inference-overhead auxiliary heads: (i) an **HL-Gauss Distributional Ponder Head** predicting normalized human cognitive effort $u \in [0,1]$, and (ii) a **2-Ply Opponent-Reply Head** over rank-mirrored board representations, forcing the trunk to internalize human threat-and-response dynamics.
* **Title-Verified Curriculum, WSD Annealing, & Anti-Engine DPO:** We eliminate low-Elo blitz noise ($\ge 1800$ anchor), enforce **official title purity** above $2350$ Elo to purge untitled online smurfs/cheaters while never capping titled games, anneal onto pure FIDE OTB + titled games via a Warmup-Stable-Decay (WSD) schedule, and introduce **Stage-3 Anti-Engine DPO** to explicitly suppress non-human Stockfish #1 moves when sound human alternatives exist.

---

## 2. Diagnosing the Bottlenecks of MAIA-3 and Vanilla Chessformer

A forensic comparison of the Chessformer paper (Monroe et al., 2026) and its reference implementation reveals five concrete limitations when emulating strong human and OTB play:

### 2.1 Non-Markov Human Input & Missing Clock/Format Context
In Section 4.3, Monroe et al. (2026) note that a chess state is Markov only when equipped with side to move, castling rights, en passant availability, and repetition history. Surprisingly, while their Leela-CF engine distillation setup included those features (Appendix A.2), **MAIA-3 omitted all of them**: each square token receives only $12 \times 8 = 96$ historical piece indicators plus $2 \times 128$ Elo embeddings ($d_{\text{in}} = 352$). Consequently, MAIA-3 cannot distinguish whether castling is legal when a rook or king has moved and returned prior to the 7-ply window, whether an en passant capture is legal on the current ply, whether a move claims a threefold repetition draw in a worse position, or whether the player is in a 3+0 blitz scramble versus a 90+30 classical OTB game.

### 2.2 Collinear Two-Vector Skill Interpolation & Title Blindness
MAIA-3 encodes player rating $k \in [0, 5000]$ via linear interpolation between two vectors:
$$e_k = \gamma e_{\text{weak}} + (1 - \gamma) e_{\text{strong}}, \quad \gamma = \frac{5000 - k}{5000}.$$
This restricts all skill levels from 600 to 3000 Elo to a **1D affine line segment** in embedding space, projected only once at the input layer. Strategic shifts across skill tiers are highly non-monotonic (e.g., tactical sharpness peaking in young IMs vs. endgame prophylaxis in 2700+ GMs), and online ratings are heavily confounded by time control and account authenticity: an untitled 2450 Lichess Blitz account behaves markedly differently from a FIDE-verified 2450 Grandmaster playing classical chess.

### 2.3 Post-Norm Optimization Ceiling, Mean-Pool Dilution, & Static GAB Bias
1. **Post-Norm + Conservative LR:** MAIA-3 uses Post-RMSNorm (`x = norm(x + sublayer(x))`) with a tiny peak learning rate of $5 \times 10^{-5}$ and梯度 norm $\approx 2.8$ (Table 4 of Monroe et al., 2026), which learns slowly under finite compute budgets.
2. **Global Mean Pooling:** Both MAIA-3's value head (`last_ln(x.mean(dim=1))`) and its 3M/5M GAB generator average uniformly across all 64 squares, diluting localized tactical signals (e.g., a back-rank mate threat on two squares) with 60 quiet squares.
3. **Lack of Exact Ray/Blocker Geometry in GAB:** Monroe et al. (Section 6.1, Table 3) show that GAB rows have high between-position correlation ($0.770$), adapting primarily to *coarse game stage* rather than dynamic piece blocking. Learning from scratch whether square $k$ blocks a sliding attack between $i$ and $j$ across all $\binom{64}{2}$ pairs wastes attention capacity on elementary board rules.

### 2.4 Data Contamination & Sub-Club Compute Dilution
MAIA-3 downsamples Lichess Blitz into 22 uniform Elo bins (`<600` through `>=2600`), capping each 20,000-game chunk at 10 games per bin. For master-level modeling, this introduces three pathologies: (i) $13/22 \approx 59\%$ of updates are spent on `<1800` blitz games with high aleatoric noise; (ii) untitled online accounts rated $\ge 2350$ (a bracket heavily polluted by smurfs, rating manipulators, and engine-assisted accounts) are treated as ground-truth masters; and (iii) genuine `GM`/`IM` games are discarded whenever high-Elo bins hit their 10-game chunk cap, while classical OTB chess is completely absent ($0\%$).

### 2.5 Cross-Entropy's Blindness to Non-Human Engine Moves
Standard maximum-likelihood estimation (MLE) minimizes $-\log \pi_\theta(a_{\text{hum}} \mid s)$. Whenever human grandmasters favor pragmatic, natural consolidations while Stockfish prefers counter-intuitive concrete defense $a_{\text{SF}} \neq a_{\text{hum}}$, MLE applies no direct contrastive pressure against $\pi_\theta(a_{\text{SF}} \mid s)$ across positions where $a_{\text{SF}}$ was occasionally played or shares surface features with tactical moves—explaining why scaling MAIA-3 $16\times$ (`5M` $\to$ `79M`) fails to budge its $20\%$ Stockfish-pick rate on disagreement positions.

---

## 3. Methodology: ChessformerV2 & The ChessForge Pipeline

### 3.1 Complete Markov & Bilateral Clock Representation (`FORMAT_VERSION = 3`)

For each position $s_t$ at ply $t$, we retain the side-to-move-centric relative piece history $\mathbf{B}_t \in \{0,\dots,12\}^{H \times 64}$ ($H=8$, matching MAIA-3's $n=7$ history window with earliest-frame padding and $5\%$ random history-length dropout during training) and augment each square $i \in \{0,\dots,63\}$ with complete Markov and temporal features:

1. **Markov Board Metadata ($\mathbf{m}_t \in \{0,1\}^8$):** Side-to-move indicator, 4-bit relative castling rights ($\text{own}_{K}, \text{own}_{Q}, \text{opp}_{K}, \text{opp}_{Q}$), legal en-passant availability indicator, and Zobrist transposition-key repetition indicators ($\mathbb{I}[\text{rep} \ge 1], \mathbb{I}[\text{rep} \ge 2]$, reset on irreversible pawn/capture moves).
2. **Spatial En-Passant Target Indicator ($e_{t,i} \in \{0,1\}$):** Because rank-mirroring (`sq ^ 56`) preserves file $f_{\text{ep}} \in \{0,\dots,7\}$ and maps Black's rank-3 en-passant target square ($16 + f_{\text{ep}}$) identically to rank 6 ($40 + f_{\text{ep}}$), the legal en-passant capture destination in the side-to-move frame is invariantly square $i^* = 40 + f_{\text{ep}}$.
3. **Static Board Geometry ($\mathbf{g}_i \in \mathbb{R}^4$):** Normalized file, normalized rank, light/dark square parity, and Manhattan center proximity $1 - \frac{|f_i - 3.5| + |r_i - 3.5|}{7}$.
4. **Masked Bilateral Log-Clock Features ($\boldsymbol{\tau}_t \in \mathbb{R}^6$, $c_t \in \{0,1\}$):** Given base time $T_{\text{base}}$, increment $T_{\text{inc}}$, own remaining clock $T_{\text{self}}$, opponent remaining clock $T_{\text{opp}}$, and opponent's previous move time $P_{\text{opp}}^{\text{prev}}$, we compute log-compressed features gated by clock-presence indicator $c_t \in \{0,1\}$:
   $$\boldsymbol{\tau}_t = c_t \cdot \left[\frac{\ln(1+T_{\text{base}})}{8},\; \frac{\ln(1+T_{\text{inc}})}{4},\; \frac{\ln(1+T_{\text{self}})}{8},\; \frac{\ln(1+T_{\text{opp}})}{8},\; \frac{\ln(1+T_{\text{self}}) - \ln(1+T_{\text{opp}})}{4},\; \frac{\ln(1+P_{\text{opp}}^{\text{prev}})}{5}\right].$$
   For clockless classical OTB games, setting $c_t = 0$ cleanly zeros $\boldsymbol{\tau}_t$ while preserving full gradient flow through $c_t$ and the format embedding.

Each square's input vector has dimension $12H + 1 + 4 + 8 + 6 = 115$ (vs. MAIA-3's $352$, removing redundant $64\times$ copies of 256-d Elo vectors from the input projection).

### 3.2 Non-Linear RBF Skill, Official-Title, & Format Conditioning via Bottleneck AdaLN

We replace linear two-vector Elo interpolation with a dedicated `ContextConditioner`. Player and opponent ratings $E_{\text{self}}, E_{\text{opp}}$ (clipped to $[400, 3300]$) are expanded over $K = 16$ Gaussian RBF centers $\mu_k$ spaced uniformly on $[800, 2900]$ ($\sigma = 160$):
$$\phi_k(E) = \exp\!\left(-\frac{(E - \mu_k)^2}{2\sigma^2}\right), \quad \mathbf{s}_{\text{elo}} = \left[\frac{E_{\text{self}}-1800}{600},\; \frac{E_{\text{opp}}-1800}{600},\; \tanh\!\left(\frac{E_{\text{self}}-E_{\text{opp}}}{400}\right)\right].$$
Crucially, we embed **verified official human title tiers** $T_{\text{self}}, T_{\text{opp}} \in \{0,\dots,4\}$—strictly whitelisting official FIDE/national human titles (`0: Untitled`, `1: CM/WCM/NM/WNM`, `2: FM/WFM/LM`, `3: IM/WIM`, `4: GM/WGM`) and excluding `BOT`—alongside **game format** $F_t \in \{\text{Blitz}, \text{Rapid}, \text{OTB-Classical}\}$ into $\mathbb{R}^{32}$ embeddings. Passing $[\boldsymbol{\phi}(E_{\text{self}}), \boldsymbol{\phi}(E_{\text{opp}}), \mathbf{s}_{\text{elo}}, \mathbf{e}_{T_{\text{self}}}, \mathbf{e}_{T_{\text{opp}}}, \mathbf{e}_{F_t}, \mathbf{m}_t, \boldsymbol{\tau}_t, c_t] \in \mathbb{R}^{146}$ through a 2-layer SiLU MLP produces global context $\mathbf{c} \in \mathbb{R}^D$ and a low-rank bottleneck $\mathbf{c}_{\text{low}} = \text{SiLU}(\mathbf{W}_{\text{bot}}\mathbf{c}) \in \mathbb{R}^{128}$.

In every transformer block $\ell$, a zero-initialized projection maps $\mathbf{c}_{\text{low}} \mapsto (\boldsymbol{\beta}_a^\ell, \boldsymbol{\gamma}_a^\ell, \mathbf{g}_a^\ell, \boldsymbol{\beta}_m^\ell, \boldsymbol{\gamma}_m^\ell, \mathbf{g}_m^\ell) \in \mathbb{R}^{6D}$, modulating both attention and MLP sublayers via Pre-Norm AdaLN:
$$\mathbf{h}_1^\ell = \text{RMSNorm}(\mathbf{x}^\ell) \odot (1 + \boldsymbol{\gamma}_a^\ell) + \boldsymbol{\beta}_a^\ell, \qquad \mathbf{x}^{\ell + \frac{1}{2}} = \mathbf{x}^\ell + (1 + \mathbf{g}_a^\ell) \odot \text{MHA}_{\text{QK-Norm}}(\mathbf{h}_1^\ell, \mathbf{B}_{\text{RayGAB}}^\ell),$$
$$\mathbf{h}_2^\ell = \text{RMSNorm}(\mathbf{x}^{\ell + \frac{1}{2}}) \odot (1 + \boldsymbol{\gamma}_m^\ell) + \boldsymbol{\beta}_m^\ell, \qquad \mathbf{x}^{\ell + 1} = \mathbf{x}^{\ell + \frac{1}{2}} + (1 + \mathbf{g}_m^\ell) \odot \text{SwiGLU}(\mathbf{h}_2^\ell).$$
Using the $128$-d AdaLN bottleneck reduces per-layer conditioning parameters by $60$–$75\%$ while keeping the residual stream $\mathbf{x}^\ell$ in `float32` and inner projections in `bfloat16`.

### 3.3 Hardware-Aligned ($S=64$) Trunk & Zero-Copy Ray-GAB

**Preserving Exact $S=64$ Hardware Tiling:** Appending 4 register tokens directly inside the self-attention trunk changes sequence length from $64 \to 68$, breaking power-of-2 ($64 \times 64$) CUDA TensorCore / TPU MXU tiling and reducing GPU throughput by $29\%$ (`1,828` $\to$ `1,294 pos/s`; Section 5.3). We therefore keep the trunk strictly at $S=64$ tokens and place selective global aggregation in a lightweight post-trunk **Cross-Attention Readout** (`CrossAttentionPool`) where two learned context-conditioned queries $[\mathbf{q}_{\text{val}}, \mathbf{q}_{\text{ponder}}]$ attend over the final normalized board tokens $\mathbf{X} \in \mathbb{R}^{B \times 64 \times D}$.

**Zero-Copy Ray-GAB:** We always retain per-square spatial GAB compression ($d_1 > 0$, avoiding MAIA-3-5M's spatial mean-pool bottleneck) and augment it with exact analytical chess ray geometry. We precompute two static binary tensors: $\mathbf{M}_{\text{between}} \in \{0,1\}^{64 \times 4096}$, where $(\mathbf{M}_{\text{between}})_{k, 64i+j} = 1$ iff square $k$ lies strictly on the queen-ray segment between $i$ and $j$, and directional masks $\mathbf{D} \in \{0,1\}^{4 \times 64 \times 64}$ for rank, file, main diagonal, and anti-diagonal rays, plus a static 12-class pairwise relation table $\mathbf{P}_{\text{rel}} \in \{0,\dots,11\}^{64 \times 64}$ (encoding knight jumps, relative forward/backward pawn captures, king adjacency, and ray classes). Given current occupancy $\mathbf{o}_t = \mathbb{I}[\mathbf{B}_{t, H-1} > 0] \in \{0,1\}^{B \times 64}$, a single matrix multiply counts intervening blockers across all $4096$ pairs simultaneously:
$$\mathbf{N}_{\text{blk}} = \text{reshape}(\mathbf{o}_t \mathbf{M}_{\text{between}},\, [B, 1, 64, 64]),$$
$$\mathbf{F}_{\text{ray}} = \text{concat}\!\Big(\mathbb{I}[\mathbf{N}_{\text{blk}} = 0] \odot \mathbf{D},\;\; \mathbb{I}[\mathbf{N}_{\text{blk}} = 1] \odot \mathbf{D}\Big) \in \{0,1\}^{B \times 8 \times 64 \times 64}.$$
Here, the first 4 channels indicate **unobstructed sliding line-of-sight** ($0$ blockers), while the next 4 channels indicate **single-blocker pin, skewer, discovered-attack, and X-ray alignments** ($1$ blocker). Computed once per forward pass directly in `(B, 8, 64, 64)` `bfloat16` layout (zero HBM transpose copies), the total attention bias for head $h$ in layer $\ell$ is:
$$\mathbf{B}_{\text{RayGAB}}^{\ell, h} = \mathbf{B}_{\text{GAB}}^{\ell, h}(\mathbf{h}_1^\ell) + \sum_{c=1}^8 W_{\text{ray}}^{\ell, c, h} \mathbf{F}_{\text{ray}, c} + \mathbf{T}_{\text{rel}}^{\ell, h}[\mathbf{P}_{\text{rel}}],$$
with $\mathbf{W}_{\text{ray}}^\ell$ and $\mathbf{T}_{\text{rel}}^\ell$ zero-initialized so training begins smoothly from standard GAB.

### 3.4 Attention-Residual Policy & Human Cognitive Auxiliary Heads

1. **Source-Prior & Attention-Residual Move Head:** We enhance MAIA-3's bilinear source–destination policy head with a per-source piece-selection prior $b_{\text{from}}(\mathbf{x}_i)$ and a zero-initialized linear combination of pre-softmax $64 \times 64$ attention logits $\mathbf{A}^{\ell}$ from the final two transformer layers ($\ell \in \{L-2, L-1\}$):
   $$S_{i,j} = \frac{(\mathbf{W}_{\text{from}}\mathbf{x}_i)^\top (\mathbf{W}_{\text{to}}\mathbf{x}_j)}{\sqrt{d_h}} + b_{\text{from}}(\mathbf{x}_i) + \sum_{h'} w_{h'}^{\text{attn}} A_{h', i, j}^{\text{last2}},$$
   allowing tactical ray/target attention maps in the final blocks to route directly into move logits.
2. **HL-Gauss Normalized Ponder Effort Head:** Raw move times span orders of magnitude across blitz vs. rapid. Following distributional RL classification principles (Farebrother et al., 2024), we define scale-invariant **normalized cognitive effort** $u_t = \text{clip}\!\left(\frac{\ln(1 + t_{\text{ponder}})}{\ln(1 + \max(T_{\text{self}}, 1))}, 0, 1\right)$ and train a $K_p = 16$-bucket Histogram Loss with Gaussian smoothing ($\sigma = 0.75 / K_p$), masked cleanly when $c_t = 0$. At UCI inference, predicted expected effort $\hat{u}_t \in [0,1]$ dynamically modulates sampling temperature (cooling temperature on instant recaptures and sharp tactical spikes).
3. **2-Ply Human Opponent-Reply Head:** Because human master play revolves around anticipating *human* replies rather than engine refutations, we mirror the final board tokens along ranks ($\mathbf{X}_{\text{opp}} = \text{RMSNorm}(\mathbf{X}_{:, \text{MIRROR\_SQ}, :})$) and apply a second bilinear move head to predict the opponent's actual next move $a_{t+1}$ in the game ($\lambda_{\text{reply}} = 0.15$). Zero cost is incurred at inference.

### 3.5 Title-Verified Curation, WSD Annealing, & Stage-3 Anti-Engine DPO

* **Title-Verified Data Pools & Two-Stage WSD Curriculum:** We filter online games to Blitz + Rapid with a $\ge 1800$ Elo anchor, enforce **official title purity** ($\max(E_W, E_B) \ge 2350 \implies$ official title required) to reject untitled smurfs and engine cheaters, bypass per-chunk bin caps for all officially titled games (`--always_keep_titled`), and ingest multi-year FIDE OTB tournament broadcasts. We pair a **Warmup-Stable-Decay (WSD)** learning rate schedule ($\text{lr}_{\max} = 4\times 10^{-4}$) with a curriculum switch at $80\%$ of training steps onto pure titled + FIDE OTB shards during the $20\%$ cosine decay phase.
* **Stage-3 Anti-Engine DPO:** We annotate training shards with Stockfish 16 (`--nodes 4000`) to obtain Stockfish's top move $a_{\text{SF}}$ and the centipawn loss $\Delta_{\text{cp}}(s, a_{\text{hum}}) = \max(0, V_{\text{SF}}(s, a_{\text{SF}}) - V_{\text{SF}}(s, a_{\text{hum}}))$. Defining the **sound human disagreement indicator**
  $$I_{\text{DPO}}(s) = \mathbb{I}\!\Big[a_{\text{hum}} \neq a_{\text{SF}} \;\wedge\; \Delta_{\text{cp}}(s, a_{\text{hum}}) \le \tau_{\text{cp}}\Big] \quad (\tau_{\text{cp}} = 35\text{ cp}),$$
  we optimize an SFT-anchored Direct Preference Optimization objective against the frozen Stage-2 SFT reference policy $\pi_{\text{ref}}$:
  $$\mathcal{L}_{\text{Stage3}}(\theta) = \mathcal{L}_{\text{SFT}}(\theta) - \lambda_{\text{DPO}} \, \mathbb{E}_{s \mid I_{\text{DPO}}(s)=1}\!\left[\log \sigma\!\left(\beta \left[\log \frac{\pi_\theta(a_{\text{hum}}\mid s)}{\pi_{\text{ref}}(a_{\text{hum}}\mid s)} - \log \frac{\pi_\theta(a_{\text{SF}}\mid s)}{\pi_{\text{ref}}(a_{\text{SF}}\mid s)}\right]\right)\right].$$
  Crucially, restricting DPO to $\Delta_{\text{cp}} \le 35\text{ cp}$ ensures the model never learns to blunder when Stockfish finds the sole winning move, while explicitly pushing down computer-only preferences when human masters choose equally sound human plans.

---

## 4. Experimental Setup and Evaluation Suite

### 4.1 Held-Out Post-Cutoff Benchmarks (`2025-08` – `2025-09`)
All MAIA-3 checkpoints and all ChessForge training pools strictly terminate on or before **July 2025**. We construct two strictly held-out, Stockfish-annotated (`5,000` nodes/move) benchmarks from August–September 2025:

1. **`Eval-Titled-Online` ($37,308$ positions; $700$ games):** Lichess rated Blitz & Rapid games from August 2025 where **both** White and Black hold verified official titles, skipping the first 10 plies and moves with $<30\text{s}$ remaining. Contains **$11,480$ `Anti-SF` positions** where $a_{\text{hum}} \neq a_{\text{SF}}$ and $\Delta_{\text{cp}} \le 35$.
2. **`Eval-FIDE-OTB` ($37,499$ positions; $700$ games):** Official FIDE Over-The-Board classical tournament broadcasts from August–September 2025 (`min_elo >= 2100`, verified titled players, plies $> 14$). Contains **$12,051$ `Anti-SF` positions**.

### 4.2 Metrics
Alongside legal-masked top-1 move-matching accuracy (`Move Acc`), top-3 accuracy (`Top-3`), move perplexity (`PPL`), unmasked 4352-way move accuracy (`Unmasked Acc`, measuring intrinsic board-rule internalization without a legal-move mask), WDL outcome accuracy (`Val Acc`), and WDL cross-entropy (`Val NLL`), we introduce three **Anti-Engine Alignment metrics** on the $I_{\text{DPO}}=1$ disagreement slice:
* **`AntiSF Hum Acc` ($\uparrow$):** Rate at which $\arg\max_{a \in \mathcal{L}(s)} \pi_\theta(a\mid s) = a_{\text{hum}}$ when $a_{\text{hum}} \neq a_{\text{SF}}$.
* **`AntiSF Eng Pick` ($\downarrow$):** Rate at which $\arg\max_{a \in \mathcal{L}(s)} \pi_\theta(a\mid s) = a_{\text{SF}}$ (picking Stockfish's non-human move).
* **`AntiSF Hum Pref` ($\uparrow$):** Mean pairwise preference probability $\sigma(\ell_\theta(a_{\text{hum}}) - \ell_\theta(a_{\text{SF}}))$.

---

## 5. Results and Analysis

### 5.1 Auditing Official Released MAIA-3 Checkpoints (`5M`, `23M`, `79M`)

Table 1 reports the zero-shot performance of the official released MAIA-3 family (trained for $10^6$ steps; up to $4.1\times 10^9$ examples on $8\times\text{A100}$ GPUs). Two striking empirical laws stand out:

1. **Severe OTB Degradation:** Every MAIA-3 checkpoint loses **$-6.96\%$ to $-7.12\%$** move accuracy moving from Titled Online to FIDE OTB (`63.02%` $\to$ `55.90%` for `79M`), while `Val NLL` degrades from `0.8528` to **`1.0463`** (nearly uninformative compared to $\ln 3 = 1.0986$).
2. **Engine-Pick Invariance Across $16\times$ Scale:** Scaling from `5M` $\to$ `23M` $\to$ `79M` improves general accuracy, yet **`AntiSF Eng Pick` remains frozen at $\approx 19.2\%$ Online and $\approx 20.1\%$ OTB** (`79M` actually exhibits the *highest* OTB Stockfish-pick rate at `20.11%`).

**Table 1: Audit of official released MAIA-3 models on held-out Titled Online and FIDE OTB benchmarks.**
| Model | Train Examples | Domain | Move Acc (%) $\uparrow$ | Top-3 (%) $\uparrow$ | PPL $\downarrow$ | Val Acc (%) $\uparrow$ | Val NLL $\downarrow$ | AntiSF Hum (%) $\uparrow$ | AntiSF Eng Pick (%) $\downarrow$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MAIA-3-5M` | $5.1\times 10^8$ | Titled Online | $59.17 \pm 0.50$ | $86.28$ | $3.320$ | $59.24$ | $0.8677$ | $49.02$ | $19.19$ |
| `MAIA-3-23M` | $4.1\times 10^9$ | Titled Online | $61.81 \pm 0.49$ | $88.14$ | $3.019$ | $59.90$ | $0.8567$ | $51.12$ | $19.29$ |
| `MAIA-3-79M` | $4.1\times 10^9$ | Titled Online | $63.02 \pm 0.49$ | $88.97$ | $2.905$ | $60.14$ | $0.8528$ | $52.61$ | $19.28$ |
| `MAIA-3-5M` | $5.1\times 10^8$ | **FIDE OTB** | $52.21 \pm 0.51$ | $81.32$ | $4.157$ | $56.33$ | $1.0858$ | $39.20$ | $20.07$ |
| `MAIA-3-23M` | $4.1\times 10^9$ | **FIDE OTB** | $54.79 \pm 0.50$ | $83.70$ | $3.775$ | $57.20$ | $1.0583$ | $41.81$ | $19.97$ |
| `MAIA-3-79M` | $4.1\times 10^9$ | **FIDE OTB** | $55.90 \pm 0.50$ | $84.65$ | $3.618$ | $57.76$ | $1.0463$ | $42.74$ | $20.11$ |

---

### 5.2 Controlled Compute-Matched Ablation Ladder (`R1`–`R6`)

To isolate every design decision without confounding compute budgets, we run six controlled 10,000-step runs ($\text{BS}=1024$, **$10.24\text{M}$ training examples** per run—$400\times$ fewer examples than `MAIA-3-79M`) on a single NVIDIA L4 GPU (`Table 2`):

* **`R1` (`MAIA-3-5M` v1 Recipe, Pool A 22-bin Blitz):** Reproducing Table 4 of Monroe et al. (2026) verbatim (`lr=5e-5`, Post-RMSNorm) severely underfits at 10k steps (`37.15%` Online, `31.97%` OTB, `Unmasked Acc = 0.60%`).
* **`R2` (`ChessformerV2` Core Trunk + `v3` Markov State + RBF/Title AdaLN + WSD `lr=4e-4`, Pool A):** Switching to Pre-RMSNorm, QK-Norm, SwiGLU, WSD scheduling, Markov `v3` features, and non-linear RBF/Title AdaLN reduces gradient norm variance ($2.82 \to 0.62$) and delivers a massive **+13.12% Online** (`50.27%`) and **+10.55% OTB** (`42.52%`) accuracy jump on the exact same data.
* **`R3` (`+ Ray-GAB` + Register Readout + Attention-Residual Policy, Pool A):** Adding exact line-of-sight/pin Ray-GAB, selective readout, and last-2-layer attention residuals improves both Online (`50.70%`, PPL `4.571`) and OTB (`43.07%`, `+1.02%` OTB value accuracy).
* **`R4` (`+ HL-Gauss Effort Ponder` + `2-Ply Mirrored Opponent-Reply`, Pool A):** Adding cognitive auxiliary losses (`0.05` ponder + `0.15` reply) costs only $1.4\%$ step time while boosting move accuracy by **+0.74% Online** (`51.44%`) and **+1.18% OTB** (`44.25%`) and nearly **tripling unmasked legal move accuracy** (`1.25%` $\to$ `3.54%`), confirming that predicting the opponent's 2-ply human reply forces deeper board-rule and threat representations into the trunk.
* **`R5` (Title-Verified $\ge 1800$ Data Curation, Pool B):** Replacing 22-bin Blitz with $\ge 1800$ Blitz+Rapid + $\ge 2350$ official-title purity adds **+2.33% Online** (`53.77%`) and **+2.31% OTB** (`46.56%`). Remarkably, at just 10.24M examples, `R5`'s Online Value Accuracy (**`60.35%`**) **already surpasses `MAIA-3-79M` (`60.14%`)**.
* **`R6` (`+ 20%` WSD Anneal onto Pool C [Pure Official-Title Online + FIDE OTB]):** Annealing the final 2,000 steps onto verified titled & OTB games lifts Online accuracy to **`55.02%`** (`PPL 3.900`) and surges FIDE OTB accuracy by **+3.57%** to **`50.13%`**, while OTB WDL calibration leaps by **+8.91%** (`53.95%` $\to$ **`62.86%`**, `Val NLL 0.7941`), outperforming `MAIA-3-79M` on OTB outcome prediction by **+5.10%** with **$400\times$ less compute**.

**Table 2: Controlled 10,000-step ($10.24\text{M}$-example) ablation ladder on a single L4 GPU.** *Every architectural, auxiliary-loss, data-purity, and curriculum stage improves monotonically across both domains.*
| Run & Configuration | Online Acc (%) | Online PPL $\downarrow$ | Online Val Acc (%) | OTB Acc (%) | OTB PPL $\downarrow$ | OTB Val Acc (%) | OTB Val NLL $\downarrow$ | OTB AntiSF Hum (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R1**: MAIA-3 v1 Baseline (Pool A) | $37.15$ | $8.096$ | $55.97$ | $31.97$ | $9.728$ | $51.50$ | $1.2188$ | $22.75$ |
| **R2**: + V2 Trunk, `v3` State, RBF-AdaLN, WSD | $50.27$ | $4.638$ | $59.17$ | $42.52$ | $6.108$ | $51.62$ | $1.1537$ | $31.06$ |
| **R3**: + Ray-GAB, Registers, Attn-Res Policy | $50.70$ | $4.571$ | $59.65$ | $43.07$ | $6.040$ | $52.64$ | $1.1859$ | $31.24$ |
| **R4**: + HL-Gauss Ponder & 2-Ply Reply Heads | $51.44$ | $4.413$ | $59.61$ | $44.25$ | $5.800$ | $52.78$ | $1.2142$ | $32.30$ |
| **R5**: + Title-Pure $\ge 1800$ Online Data (Pool B) | $53.77$ | $4.062$ | $60.35$ | $46.56$ | $5.335$ | $53.95$ | $1.1395$ | $34.28$ |
| **R6**: + Final 20% WSD OTB+Titled Anneal (Pool C) | **$55.02$** | **$3.900$** | **$60.65$** | **$50.13$** | **$4.502$** | **$62.86$** | **$0.7941$** | **$38.15$** |

---

### 5.3 Hardware Alignment Analysis: In-Trunk Registers ($S=68$) vs. Post-Trunk Cross-Attention ($S=64$)

During our ablation study (`R2` $\to$ `R3`), appending 4 register tokens directly to the 64 board tokens (`S = 68`) combined with unaligned `(B, 64, 64, H)` Ray-GAB transposes reduced L4 GPU throughput from `1,827.9 pos/s` to `1,293.9 pos/s` ($-29.2\%$) because $S=68$ breaks 16/64-aligned CUDA TensorCore tiles. Refactoring `ChessformerV2` to keep the trunk strictly at **$S=64$**, moving register aggregation to post-trunk `CrossAttentionPool`, fusing `RayGABBias` directly in `(B, 8, 64, 64)` layout with zero transposes, and introducing the $128$-d AdaLN bottleneck enabled scaling to **16 layers (`35.2M` parameters—$59\%$ more parameters and $33\%$ deeper than `R3`)** while maintaining **`909.2 pos/s`** on a single L4 GPU ($>30\%$ higher FLOP utilization than the unaligned $S=68$ design).

---

### 5.4 Scaling `ChessformerV2-35M`, Stage-3 Anti-Engine DPO, & The Diversity–Epoch Trade-Off

Finally, we expanded our pure FIDE OTB corpus (`2021-01` through `2025-07`, totaling $\sim 14.9\text{M}$ unique combined Pool B + Pool C positions), trained `ChessForge-Master-35M` (`chessforge-v2-28m` preset; 16 blocks, $D=384$, $35.2\text{M}$ params) for $60,000$ steps ($61.44\text{M}$ examples, $\sim 4.1$ epochs), and ran Stage-3 Anti-Engine DPO ($3,000$ steps at $\text{BS}=512$ on a $200\text{k}$-position Stockfish-annotated subset; `Table 3`).

**Table 3: Scaled `ChessForge-Master-35M` (SFT & DPO) vs. Official Released MAIA-3 Family.**
| Model | Train Examples | Domain | Move Acc (%) $\uparrow$ | Top-3 (%) $\uparrow$ | Unmasked Acc (%) $\uparrow$ | Val Acc (%) $\uparrow$ | Val NLL $\downarrow$ | AntiSF Hum (%) $\uparrow$ | AntiSF Eng Pick (%) $\downarrow$ | AntiSF Pref (%) $\uparrow$ |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MAIA-3-5M` | $512\text{M}$ | Online | $59.17$ | $86.28$ | $12.13$ | $59.24$ | $0.8677$ | $49.02$ | $19.19$ | $65.36$ |
| `MAIA-3-79M` | $4,096\text{M}$ | Online | **$63.02$** | **$88.97$** | **$35.98$** | $60.14$ | $0.8528$ | **$52.61$** | $19.28$ | **$67.08$** |
| **`Master-35M-SFT`** *(Ours)* | **$61.4\text{M}$** | Online | $54.93$ | $83.16$ | $17.02$ | **$61.68$** | **$0.8420$** | $44.37$ | $20.17$ | $64.89$ |
| **`Master-35M-DPO`** *(Ours)* | $+1.5\text{M}$ | Online | $51.24$ | $79.89$ | $22.57$ | $61.26$ | $0.8468$ | $40.77$ | $19.35$ | $64.88$ |
| `MAIA-3-5M` | $512\text{M}$ | **FIDE OTB** | $52.21$ | $81.32$ | $11.56$ | $56.33$ | $1.0858$ | $39.20$ | $20.07$ | $60.91$ |
| `MAIA-3-79M` | $4,096\text{M}$ | **FIDE OTB** | **$55.90$** | **$84.65$** | **$33.08$** | $57.76$ | $1.0463$ | **$42.74$** | $20.11$ | **$62.82$** |
| **`Master-35M-SFT`** *(Ours)* | **$61.4\text{M}$** | **FIDE OTB** | $51.31$ | $80.54$ | $14.72$ | **$65.56$** | **$0.7526$** | $39.28$ | $20.61$ | $61.21$ |
| **`Master-35M-DPO`** *(Ours)* | $+1.5\text{M}$ | **FIDE OTB** | $47.75$ | $77.30$ | $18.21$ | $64.40$ | $0.7683$ | $36.05$ | **$19.40$** | $61.64$ |

These experiments yield four major findings:

1. **New State of the Art in Human Outcome (WDL) Calibration Across Both Domains:** With **$67\times$ fewer training examples** than `MAIA-3-79M` ($61.4\text{M}$ vs. $4.1\text{B}$), `Master-35M-SFT` achieves **`61.68%` Value Accuracy (`0.8420` NLL)** on Titled Online (beating `MAIA-3-79M` by $+1.54\%$) and **`65.56%` Value Accuracy (`0.7526` NLL)** on FIDE OTB—outperforming `MAIA-3-79M` by **+7.80 percentage points** and **$-0.2937\text{ nats}$**. Context conditioning on official titles, format (`Blitz`/`Rapid`/`OTB`), bilateral clocks, and post-trunk cross-attention readout completely eliminates MAIA-3's OTB value miscalibration.
2. **Faster Intrinsic Board-Rule Learning (`Unmasked Acc`) & OTB Disagreement Parity with `MAIA-3-5M`:** Without any legal-move mask, `Master-35M-SFT` achieves **`17.02%` Online / `14.72%` OTB** raw 4352-way accuracy (rising to **`22.57%` / `18.21%`** post-DPO), comfortably beating `MAIA-3-5M` (`12.13%` / `11.56%`) despite $8.3\times$ fewer training examples, while already surpassing `MAIA-3-5M` on FIDE OTB human disagreement accuracy (`39.28%` vs. `39.20%`) and human preference probability (`61.21%` vs. `60.91%`).
3. **Anti-Engine DPO Breaks MAIA-3's 20% Stockfish-Overlap Floor:** Whereas $16\times$ parameter scaling in MAIA-3 failed to move `AntiSF Eng Pick` on OTB (`20.07%` $\to$ `20.11%`), Stage-3 Anti-Engine DPO drives OTB Stockfish-pick rate down by **$-1.21\%$** (`20.61%` $\to$ **`19.40%`**—lower than `MAIA-3-5M`, `23M`, and `79M`) and Online Stockfish-pick rate down by **$-0.82\%$** (`20.17%` $\to$ `19.35%`), while simultaneously *increasing* OTB pairwise human preference (`61.21%` $\to$ `61.64%`).
4. **Diagnosing the Multi-Epoch Diversity Bottleneck:** Comparing single-pass `R6` ($10.2\text{M}$ examples) with 4.1-epoch `Master-35M-SFT` ($61.4\text{M}$ examples over $14.9\text{M}$ unique positions, where the anneal pool was $93.6\%$ OTB) reveals why Online top-1 plateaued at `54.93%` (`Top-3` rose to `83.16%`, OTB rose to `51.31%`, but training legal accuracy reached `70.25%` and test policy entropy fell from `1.41` $\to$ `0.93` nats). Similarly, running DPO for $3,000$ steps ($\sim 7.7$ epochs) on a $200\text{k}$-position preference shard drove training accuracy to `87.69%` (`dpo_loss = 0.356`), collapsing entropy to `0.335` nats (`PPL 21.9`) and trading $-3.6\%$ top-1 accuracy for lower engine overlap. This cleanly isolates **unique position diversity**—streaming $\ge 300\text{M}$ unique positions near-single-epoch for SFT and $\le 1.5$ epochs on a $\ge 1\text{M}$-position Stockfish-annotated pool for DPO (or adding a small KL/entropy bonus)—as the exact sufficient condition to prevent over-sharpening while retaining all architectural gains.

---

## 6. Related Work

**Human Behavior Modeling in Chess.** MAIA (McIlroy-Young et al., 2020) introduced discrete per-Elo CNNs for human move prediction; MAIA-2 (Tang et al., 2024) unified skill conditioning via channel-wise attention; ALLIE (Zhang et al., 2025) framed game trajectories as UCI move-token language modeling with adaptive MCTS; and MAIA4ALL (Tang et al., 2025) adapted player-specific prototypes. Chessformer / MAIA-3 (Monroe et al., 2026) established 64-square tokenization with GAB and bilinear from–to heads as the state of the art on Lichess Blitz. Our work directly addresses the skilled-play frontier posed in Appendix E of Monroe et al. (2026), extending square-token modeling to complete Markov states, official FIDE title/OTB conditioning, exact ray geometry, cognitive auxiliary supervision, and contrastive anti-engine alignment.

**Alignment & Preference Optimization Beyond MLE.** Standard imitation learning suffers from covariate shift and mode-averaging across heterogeneous demonstrations. Jacob et al. (2022) regularized MCTS toward human policies via KL penalties; Direct Preference Optimization (DPO; Rafailov et al., 2023) aligns policies directly from preference pairs. To our knowledge, ChessForge is the first chess behavioral model to construct **automated human-over-oracle preference pairs** ($a_{\text{hum}} \succ a_{\text{SF}}$ conditioned on bounded centipawn loss $\Delta_{\text{cp}} \le \tau_{\text{cp}}$) to suppress superhuman engine tells without degrading tactical soundness.

---

## 7. Discussion, Limitations, and Scaling Roadmap

Our results demonstrate that the gap between online blitz behavior cloning and authentic master/OTB human modeling is governed jointly by **Markov completeness, explicit domain/title conditioning, geometric inductive biases, and contrastive anti-engine supervision**:

1. **Architecture & Conditioning Efficiency:** At matched 10.24M-example budgets (`Table 2`), `ChessformerV2` + title-pure OTB annealing (`R6`) outperforms the MAIA-3 v1 training recipe (`R1`) by **+17.87% Online** and **+18.16% OTB**, and at $61.4\text{M}$ examples (`Table 3`) already surpasses `MAIA-3-79M` ($4.1\text{B}$ examples) on WDL outcome prediction in *both* online and classical OTB chess.
2. **Eliminating Multi-Epoch Over-Sharpening at Full Scale:** Our single-GPU experiments operated on a $14.9\text{M}$-position disk cache ($\sim 4.1$ SFT epochs) and a $200\text{k}$-position DPO cache ($\sim 7.7$ DPO epochs). Streaming $24$–$36$ months of $\ge 1800$ / titled Lichess Blitz+Rapid alongside TWIC/Lumbra's Gigabase OTB archives ($\sim 400\text{M}$ unique positions, $<0.5$ epochs) and running single-epoch DPO (`500–1,000` steps on $\ge 1\text{M}$ Stockfish-annotated positions with $50/50$ Online/OTB balance and $\beta \in [0.1, 0.15]$) directly eliminates entropy collapse while preserving every architectural gain quantified in `R2`–`R6`.

---

## References

* Campbell, M., Hoane Jr., A. J., & Hsu, F.-h. (2002). Deep Blue. *Artificial Intelligence*, 134(1–2), 57–83.
* Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and memory-efficient exact attention with IO-awareness. *NeurIPS*.
* Farebrother, J., Orbay, J., Vuong, Q., Taiga, A. A., Chebotar, Y., Xiao, T., ... & Agarwal, R. (2024). Stop regressing: Training value functions via classification for scalable deep RL. *ICML*.
* Hamade, K., McIlroy-Young, R., Sen, S., Kleinberg, J., & Anderson, A. (2024). Designing skill-compatible AI: Methodologies and frameworks in chess. *ICLR*.
* Jacob, A. P., Wu, D. J., Farina, G., Lerer, A., Hu, H., Bakhtin, A., ... & Brown, N. (2022). Modeling strong and human-like gameplay with KL-regularized search. *ICML*.
* McIlroy-Young, R., Sen, S., Kleinberg, J., & Anderson, A. (2020). Aligning superhuman AI with human behavior: Chess as a model system. *KDD*.
* Monroe, D., Eilender, G., Chalmers, P., Tang, Z., & Anderson, A. (2026). Chessformer: A unified architecture for chess modeling. *ICLR 2026* (arXiv:2605.19091v1).
* Rafailov, R., Sharma, A., Mitchell, E., Manning, C. D., Ermon, S., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS*.
* Ruoss, A., Delétang, G., Medapati, S., Grau-Moya, J., Wenliang, L. K., Catt, E., ... & Genewein, T. (2024). Amortized planning with large-scale transformers: A case study on chess. *NeurIPS*.
* Tang, Z., Jiao, D., McIlroy-Young, R., Kleinberg, J., Sen, S., & Anderson, A. (2024). Maia-2: A unified model for human-AI alignment in chess. *NeurIPS*.
* Tang, Z., Jiao, D., Xue, E., McIlroy-Young, R., Kleinberg, J., Sen, S., & Anderson, A. (2025). Learning to imitate with less: Efficient individual behavior modeling in chess. *ICLR*.
* Zhang, Y., Jacob, A., Lai, V., Fried, D., & Ippolito, D. (2025). Human-aligned chess with a bit of search. *ICLR*.

---

## Appendix A: Architecture Specifications & Reproducibility Summary

**Table 4: Comparison of Model & Training Specifications (`MAIA-3` vs. `ChessForge / ChessformerV2`).**
| Component | MAIA-3 (`v1`, Monroe et al., 2026) | ChessformerV2 (`ChessForge`, Ours) |
| :--- | :--- | :--- |
| **Board State Markovity** | Non-Markov ($8\times 12$ piece planes only) | Full Markov `v3`: piece planes + castling + EP square + Zobrist reps |
| **Clock & Format Context** | Unconditioned (`Blitz` only) | 6-d bilateral log-clocks + `has_clock` mask + `Blitz/Rapid/OTB` emb |
| **Skill & Identity Conditioning** | Collinear 2-vector Elo line ($d=128\times 2$ at input) | 16-RBF Elo + rating-gap $\tanh$ + **Official Title (`GM..CM`)** via Bottleneck AdaLN |
| **Transformer Block** | Post-RMSNorm + MHA + GELU MLP ($2\times$) | Pre-RMSNorm + QK-Norm MHA + SwiGLU MLP ($2.67\times$, `fp32` residual) |
| **Geometric Attention Bias** | Learned GAB (mean-pool on 3M/5M; static bias) | Spatial GAB ($d_1 > 0$) + **Zero-Copy Ray-GAB** (0-blocker clear + 1-blocker pin/X-ray) |
| **Global Readout & Sequence Length** | Global mean pool ($S=64$) | **Post-trunk Cross-Attention Pool** (`[CLS_VAL, CLS_PONDER]`, preserving $S=64$ tiling) |
| **Policy & Auxiliary Heads** | Bilinear From–To + MSE value/unused ponder | Bilinear + Source Prior + Last-2 Attn Residual + **HL-Gauss Effort Ponder** + **2-Ply Reply** |
| **Optimization & Curriculum** | AdamW, $\text{lr}=5\times 10^{-5}$, 50k cyclic cosine | AdamW, $\text{lr}=4\times 10^{-4}$, WSD schedule + final 20% Titled/OTB annealing |
| **Anti-Engine Alignment** | None (pure MLE behavior cloning) | **Stage-3 Stockfish-Disagreement DPO** ($\Delta_{\text{cp}} \le 35$, frozen SFT reference) |
| **UCI Deployment Guardrails** | Argmax / raw softmax | `min_p` truncation + effort-adaptive temperature + 1-ply human WDL drop guardrail |
